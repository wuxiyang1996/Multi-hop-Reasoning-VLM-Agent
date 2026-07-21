#!/usr/bin/env python3
"""Merge independently frozen per-game source-hypothesis artifacts by exact union."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = []
    games = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        unsigned = dict(payload)
        claimed = str(unsigned.pop("artifact_sha256", ""))
        if not claimed or _hash(unsigned) != claimed:
            raise SystemExit(f"source hypothesis artifact hash mismatch: {path}")
        if (
            payload.get("candidate_source") != "independent_untrusted_agents"
            or payload.get("full_observed_path_partition_required") is not True
            or payload.get("semantic_scoring") is not False
            or payload.get("ranking") is not False
            or payload.get("voting") is not False
        ):
            raise SystemExit(f"unsupported source hypothesis protocol: {path}")
        programs = list(payload.get("programs") or ())
        if not programs or any(int(row.get("n_qualified", 0)) < 1 for row in programs):
            raise SystemExit(f"input lacks a qualified hypothesis: {path}")
        input_games = {str(row["program"]["game"]) for row in programs}
        if len(input_games) != 1:
            raise SystemExit(f"input is not single-game: {path}")
        games.append(next(iter(input_games)))
        artifacts.append((path, payload, claimed))
    if len(games) != len(set(games)):
        raise SystemExit("duplicate source game in merge inputs")
    models = {str(payload.get("model")) for _, payload, _ in artifacts}
    if len(models) != 1:
        raise SystemExit("source hypothesis model mismatch")
    output = {
        "schema_version": 1,
        "candidate_source": "independent_untrusted_agents",
        "source_batch": "exact_union_of_frozen_per_game_artifacts",
        "model": next(iter(models)),
        "roles": list(artifacts[0][1]["roles"]),
        "full_observed_path_partition_required": True,
        "semantic_scoring": False,
        "ranking": False,
        "voting": False,
        "reused_proposal_receipts": all(
            bool(payload.get("reused_proposal_receipts"))
            for _, payload, _ in artifacts
        ),
        "api_calls_made_this_run": 0,
        "source_games": games,
        "input_artifacts": [{
            "path": str(path), "artifact_sha256": claimed,
        } for path, _, claimed in artifacts],
        "programs": [
            program for _, payload, _ in artifacts
            for program in payload["programs"]
        ],
        "proposal_receipts": [
            receipt for _, payload, _ in artifacts
            for receipt in payload.get("proposal_receipts") or ()
        ],
        "claim_limit": (
            "Exact set union only. All structures remain Agent hypotheses; no game or "
            "hypothesis was ranked by target data, reward, embedding, or semantic mapping."
        ),
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "games": games,
        "programs": len(output["programs"]),
        "qualified": sum(row["n_qualified"] for row in output["programs"]),
        "artifact_sha256": output["artifact_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
