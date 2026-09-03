#!/usr/bin/env python3
"""Export existing native game receipts into the shared baseline source schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    canonical_source_episodes,
)
from motif_transfer.instrumented_import import import_native_source_batch  # noqa: E402
from motif_transfer.phase1_assets import read_jsonl  # noqa: E402


def _visible(state: dict) -> object:
    for key in ("observable_state", "observation", "structured_state"):
        if state.get(key) is not None:
            return state[key]
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, action="append", required=True,
        help="Existing native source evidence directory; repeat for multiple games",
    )
    parser.add_argument(
        "--source-domain", action="append",
        help="Alias paired positionally with --evidence; defaults to directory name",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aliases = args.source_domain or [path.name for path in args.evidence]
    if len(aliases) != len(args.evidence):
        raise SystemExit("--source-domain count must equal --evidence count")

    episodes = []
    for source_domain, evidence_dir in zip(aliases, args.evidence):
        # Episode-level termination is only recorded in episodes.jsonl; the last
        # transition's terminal flag is set by truncation too and cannot distinguish
        # a real stop from a step-budget cutoff.
        native = {
            str(row.get("episode_id")): row
            for row in read_jsonl(Path(evidence_dir) / "episodes.jsonl")
        }
        for episode in import_native_source_batch(evidence_dir):
            row = native.get(episode.episode_id, {})
            episodes.append({
                "episode_id": f"{source_domain}:{episode.episode_id}",
                "source_domain": str(source_domain),
                # Nullable on purpose: these games expose no official predicate, and
                # an absent outcome must not be exported as a failure.
                "official_success": episode.official_success,
                "terminated": bool(row.get("terminated", False)),
                "truncated": bool(row.get("truncated", False)),
                "total_reward": float(episode.total_reward),
                "steps": [
                    {
                        "receipt_id": row.transition.receipt_id,
                        "step": index,
                        "observation": _visible(dict(row.before.state)),
                        "action": row.action,
                        "next_observation": _visible(dict(row.after.state)),
                        "reward": row.reward,
                        "terminal": row.after.terminal,
                    }
                    for index, row in enumerate(episode.records)
                ],
            })
    body = {
        "schema_version": 1,
        "episodes": episodes,
        "source_evidence_directories": [str(path.resolve()) for path in args.evidence],
    }
    # Validate before writing and hash the exact canonical scientific content.
    canonical_source_episodes(body)
    body["source_payload_sha256"] = stable_hash({"episodes": episodes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_domains": sorted(set(aliases)), "episodes": len(episodes),
        "receipts": sum(len(row["steps"]) for row in episodes),
        "source_payload_sha256": body["source_payload_sha256"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
