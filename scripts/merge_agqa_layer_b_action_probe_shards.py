#!/usr/bin/env python3
"""Merge disjoint frozen SlowFast action-probe shards."""

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
        raise FileExistsError("merged action probe is immutable")
    cohort = json.loads(args.cohort.read_text()); shards = [json.loads(p.read_text()) for p in args.shard]
    keys = (
        "schema_version", "status", "source", "checkpoint_sha256", "classes_sha256",
        "ontology_sha256", "sampling", "temporal_views", "frame_presentation_budget",
        "answers_read", "official_program_read", "official_scene_graph_read",
    )
    reference = {key: shards[0].get(key) for key in keys}
    if any({key: shard.get(key) for key in keys} != reference for shard in shards):
        raise ValueError("action-probe shard invariants differ")
    by_task = {}
    for shard in shards:
        for row in shard["rows"]:
            task_id = str(row["task_id"])
            if task_id in by_task:
                raise ValueError(f"duplicate action-probe task {task_id}")
            by_task[task_id] = row
    expected = [str(row["task_id"]) for row in cohort["rows"]]
    if set(by_task) != set(expected):
        raise ValueError("action-probe shards do not exactly cover cohort")
    body = {key: value for key, value in shards[0].items() if key not in {"rows", "report_sha256"}}
    body.update({
        "rows": [by_task[task_id] for task_id in expected],
        "merged_shard_paths": [str(path) for path in args.shard],
        "merged_shard_report_sha256s": [shard["report_sha256"] for shard in shards],
    })
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(body["rows"]), "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
