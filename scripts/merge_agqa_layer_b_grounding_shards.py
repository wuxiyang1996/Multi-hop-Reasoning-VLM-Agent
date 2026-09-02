#!/usr/bin/env python3
"""Merge disjoint outcome-blind Layer-B grounding shards in cohort order."""

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
        raise FileExistsError("merged grounding output is immutable")
    cohort = json.loads(args.cohort.read_text())
    shards = [json.loads(path.read_text()) for path in args.shard]
    if not shards:
        raise ValueError("no grounding shards")
    invariant_keys = (
        "schema_version", "status", "cohort_sha256", "semantic_runtime_sha256",
        "grounder_backend_sha256", "model", "frame_budget",
        "all_harness_arms_share_exact_receipts", "answer_read",
        "official_scene_graph_read", "functional_program_read", "source_controller_read",
    )
    reference = {key: shards[0].get(key) for key in invariant_keys}
    if reference["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("grounding shard not frozen")
    if reference["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("shard/cohort mismatch")
    if any({key: shard.get(key) for key in invariant_keys} != reference for shard in shards):
        raise ValueError("grounding shard invariants differ")
    by_position = {}
    for shard in shards:
        for row in shard["rows"]:
            position = int(row["cohort_position"])
            if position in by_position:
                raise ValueError(f"duplicate cohort position {position}")
            by_position[position] = row
    expected = set(range(len(cohort["rows"])))
    if set(by_position) != expected:
        raise ValueError(f"grounding shards do not cover cohort: missing={sorted(expected-set(by_position))[:8]}")
    rows = [by_position[position] for position in sorted(by_position)]
    for position, (row, public) in enumerate(zip(rows, cohort["rows"])):
        if int(row["cohort_position"]) != position or str(row["task_id"]) != str(public["task_id"]):
            raise ValueError("merged grounding order/task mismatch")
    body = {key: value for key, value in shards[0].items()
            if key not in {"rows", "selected_positions", "provider_calls",
                           "reported_receipt_provider_cost_usd", "incremental_provider_cost_usd",
                           "report_sha256", "pilot"}}
    body.update({
        "schema_version": "agqa-layer-b-qwen-grounding-merged-shards-v1",
        "rows": rows, "selected_positions": list(range(len(rows))),
        "cohort_rows_total": len(rows), "provider_calls": sum(int(s["provider_calls"]) for s in shards),
        "reported_receipt_provider_cost_usd": sum(float(s.get("reported_receipt_provider_cost_usd", 0)) for s in shards),
        "incremental_provider_cost_usd": sum(float(s.get("incremental_provider_cost_usd", 0)) for s in shards),
        "merged_shard_report_sha256s": [s["report_sha256"] for s in shards],
        "merged_shard_paths": [str(path) for path in args.shard],
        "pilot": False,
    })
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "provider_calls": body["provider_calls"],
                      "incremental_cost_usd": body["incremental_provider_cost_usd"],
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
