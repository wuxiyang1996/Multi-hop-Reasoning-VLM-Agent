#!/usr/bin/env python3
"""Merge immutable answer-blind AGQA Query Grounder V4 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged V4 adjudication is immutable")
    grounding = json.loads(args.candidate_grounding.read_text())
    expected_ids = [str(row["task_id"]) for row in grounding["rows"]]
    shards = [json.loads(path.read_text()) for path in args.shard]
    if not shards:
        raise ValueError("no shards")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(shard.get(key) for shard in shards for key in forbidden):
        raise ValueError("shard violates authority boundary")
    invariants = (
        "model", "maximum_candidate_count", "maximum_presented_unique_frames",
        "shard_count", "cohort_sha256", "candidate_grounding_report_sha256",
        "protocol_file_sha256",
    )
    for key in invariants:
        if len({stable_hash(shard[key]) for shard in shards}) != 1:
            raise ValueError(f"shard invariant differs: {key}")
    coordinate_modes = {
        str(shard.get("coordinate_mode", "legacy_proxy")) for shard in shards
    }
    if len(coordinate_modes) != 1:
        raise ValueError("shard invariant differs: coordinate_mode")
    expected_shards = int(shards[0]["shard_count"])
    if len(shards) != expected_shards or {
        int(shard["shard_index"]) for shard in shards
    } != set(range(expected_shards)):
        raise ValueError("shard indices are incomplete")
    by_id = {}
    for shard in shards:
        for row in shard["rows"]:
            task_id = str(row["task_id"])
            if task_id in by_id:
                raise ValueError(f"duplicate task across shards: {task_id}")
            by_id[task_id] = row
    if set(by_id) != set(expected_ids):
        raise ValueError("merged task set differs from frozen candidate grounding")
    report = {
        "schema_version": "agqa-query-grounder-v4-qwen235-adjudication-v1",
        "status": "V4_ADJUDICATION_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "model": shards[0]["model"],
        "maximum_candidate_count": shards[0]["maximum_candidate_count"],
        "maximum_presented_unique_frames": shards[0]["maximum_presented_unique_frames"],
        "coordinate_mode": next(iter(coordinate_modes)),
        "rows": [by_id[task_id] for task_id in expected_ids],
        "provider_calls": sum(int(shard["provider_calls"]) for shard in shards),
        "reported_cost_usd": sum(float(shard["reported_cost_usd"]) for shard in shards),
        "cohort_sha256": shards[0]["cohort_sha256"],
        "candidate_grounding_report_sha256": shards[0]["candidate_grounding_report_sha256"],
        "protocol_file_sha256": shards[0]["protocol_file_sha256"],
        "shard_report_sha256s": [shard["report_sha256"] for shard in shards],
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "rows": len(report["rows"]),
        "provider_calls": report["provider_calls"],
        "reported_cost_usd": report["reported_cost_usd"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
