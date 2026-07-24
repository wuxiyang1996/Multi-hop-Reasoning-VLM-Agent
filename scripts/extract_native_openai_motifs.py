#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

from motif_transfer.frozen_motif_agent import (
    FrozenJSONMotifAgent,
    OpenAICompatibleBackend,
    PromptCondition,
)
from motif_transfer.harness import DeterministicHarness
from motif_transfer.instrumented_import import import_native_source_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Untrusted GPT motif proposals over native source receipts"
    )
    parser.add_argument("evidence_dir")
    parser.add_argument("--endpoint", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--condition", choices=[row.value for row in PromptCondition], required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--key-name", default="OPENAI_API_KEY")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--supplemental-replays",
        help="hash-bound supplemental replay bundle to merge fail-closed",
    )
    parser.add_argument(
        "--supplemental-only",
        action="store_true",
        help="exclude base fixed-step replays from proposal and audit",
    )
    args = parser.parse_args()

    secrets = runpy.run_path(args.key_file)
    api_key = secrets.get(args.key_name)
    if not isinstance(api_key, str) or not api_key:
        raise RuntimeError(f"missing {args.key_name} in key file")
    os.environ["MOTIF_OPENAI_API_KEY"] = api_key

    backend = OpenAICompatibleBackend(
        args.endpoint,
        {"segment": args.model},
        api_key_env="MOTIF_OPENAI_API_KEY",
        json_mode=True,
    )
    agent = FrozenJSONMotifAgent(
        backend, condition=PromptCondition(args.condition)
    )
    harness = DeterministicHarness()
    rows = []
    usage_totals: dict[str, int] = {}
    for episode in import_native_source_batch(
        args.evidence_dir,
        args.supplemental_replays,
        include_base_replays=not args.supplemental_only,
    ):
        receipt_map = {
            row.transition.receipt_id: row.transition for row in episode.records
        } | {row.receipt_id: row for row in episode.replay_forks}
        try:
            candidates = agent.propose_source_motifs(
                episode.records, episode.replay_forks
            )
            audits = [harness.audit_motif(row, receipt_map) for row in candidates]
            error = None
        except Exception as exc:
            candidates, audits = (), []
            error = f"{type(exc).__name__}:{exc}"
        for key, value in backend.last_usage.items():
            if isinstance(value, int):
                usage_totals[key] = usage_totals.get(key, 0) + value
        rows.append({
            "episode_id": episode.episode_id,
            "game": episode.game,
            "records": len(episode.records),
            "replay_forks": len(episode.replay_forks),
            "import_gaps": list(episode.gaps),
            "candidates": [asdict(row) for row in candidates],
            "audits": [asdict(row) for row in audits],
            "model_error": error,
            "model_response_sha256": hashlib.sha256(
                backend.last_completion.encode()
            ).hexdigest(),
            "model_response_preview": backend.last_completion[:500],
            "usage": dict(backend.last_usage),
        })
    report = {
        "schema_version": 1,
        "authority": "UNTRUSTED_MOTIF_PROPOSAL_ONLY",
        "model_identity": dict(backend.identity),
        "prompt_condition": args.condition,
        "supplemental_replays": args.supplemental_replays,
        "supplemental_only": args.supplemental_only,
        "episodes": rows,
        "totals": {
            "episodes": len(rows),
            "candidates": sum(len(row["candidates"]) for row in rows),
            "accepted": sum(audit["accepted"] for row in rows for audit in row["audits"]),
            "model_errors": sum(row["model_error"] is not None for row in rows),
            "usage": usage_totals,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
