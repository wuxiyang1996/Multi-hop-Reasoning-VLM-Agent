#!/usr/bin/env python3
"""Merge disjoint, outcome-blind Layer-B atomic-claim shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged atomic claims are immutable")
    cohort = json.loads(args.cohort.read_text())
    shards = [json.loads(path.read_text()) for path in args.shard]
    invariant_keys = (
        "schema_version", "status", "cohort_sha256",
        "base_grounding_report_sha256", "verifier_backend_sha256", "model",
        "frame_budget_per_task", "all_harness_arms_share_exact_receipts",
        "answer_read", "functional_program_read", "official_scene_graph_read",
        "source_controller_read",
    )
    reference = {key: shards[0].get(key) for key in invariant_keys}
    if any({key: shard.get(key) for key in invariant_keys} != reference for shard in shards):
        raise ValueError("atomic-claim shard invariants differ")
    if reference["status"] != "ATOMIC_VISUAL_CLAIMS_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("atomic-claim shard is not frozen")
    by_task = {}
    for shard in shards:
        for row in shard["rows"]:
            task_id = str(row["task_id"])
            if task_id in by_task:
                raise ValueError(f"duplicate atomic-claim task {task_id}")
            by_task[task_id] = row
    expected = [str(row["task_id"]) for row in cohort["rows"]]
    if set(by_task) != set(expected):
        raise ValueError("atomic-claim shards do not exactly cover cohort")
    body = {
        key: value for key, value in shards[0].items()
        if key not in {"rows", "positions", "reported_receipt_provider_cost_usd", "report_sha256"}
    }
    body.update({
        "positions": list(range(len(expected))),
        "rows": [by_task[task_id] for task_id in expected],
        "reported_receipt_provider_cost_usd": sum(
            float(shard["reported_receipt_provider_cost_usd"]) for shard in shards
        ),
        "merged_shard_paths": [str(path) for path in args.shard],
        "merged_shard_report_sha256s": [shard["report_sha256"] for shard in shards],
    })
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "rows": len(body["rows"]),
        "cost_usd": body["reported_receipt_provider_cost_usd"],
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
