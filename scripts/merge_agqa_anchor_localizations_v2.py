#!/usr/bin/env python3
"""Merge complete answer-blind AGQA anchor-localization shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa_anchor_localizations_v2 import _artifact_status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged anchor-localization artifact is immutable")
    protocol = json.loads(args.protocol.read_text())
    shards = [json.loads(path.read_text()) for path in args.shard]
    if not shards:
        raise ValueError("no anchor-localization shards")
    count = int(shards[0]["shard_count"])
    indices = {int(row["shard_index"]) for row in shards}
    if len(shards) != count or indices != set(range(count)):
        raise ValueError("anchor-localization shards are incomplete")
    invariant_keys = (
        "schema_version", "status", "consumed_development_pilot", "model",
        "maximum_anchor_frames", "minimum_object_score",
        "bbox_coordinate_contract", "authority", "input_file_sha256s",
        "protocol_file_sha256", "collector_file_sha256",
        "runtime_amendment_file_sha256",
    )
    expected = {key: shards[0][key] for key in invariant_keys}
    if any(
        {key: shard[key] for key in invariant_keys} != expected
        for shard in shards
    ):
        raise ValueError("anchor-localization shard invariants differ")
    if expected["status"] != _artifact_status(
        bool(expected["consumed_development_pilot"])
    ):
        raise ValueError("anchor status and consumed-development label differ")
    if expected["protocol_file_sha256"] != _sha256(args.protocol):
        raise ValueError("anchor protocol hash differs from shard binding")
    authority = expected["authority"]
    if any(authority.get(key) for key in (
        "question_text_supplied_to_vlm", "root_query_predicate_supplied_to_vlm",
        "temporal_operator_supplied_to_vlm", "candidate_identity_supplied_to_vlm",
        "answer_read", "official_stsg_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("anchor localizations crossed their authority boundary")

    rows = []
    seen = set()
    for shard in sorted(shards, key=lambda row: int(row["shard_index"])):
        for row in shard["rows"]:
            task_id = str(row["task_id"])
            if task_id in seen:
                raise ValueError("duplicate task across anchor shards")
            seen.add(task_id)
            rows.append(row)
    task_count = int(protocol["development_cohort"]["task_count"])
    if len(rows) != task_count:
        raise ValueError(f"expected {task_count} tasks, received {len(rows)}")
    rows.sort(key=lambda row: str(row["task_id"]))
    report = {
        "schema_version": "agqa-answer-blind-anchor-localizations-v2",
        "status": expected["status"],
        "consumed_development_pilot": bool(expected["consumed_development_pilot"]),
        "model": expected["model"],
        "maximum_anchor_frames": expected["maximum_anchor_frames"],
        "minimum_object_score": expected["minimum_object_score"],
        "bbox_coordinate_contract": expected["bbox_coordinate_contract"],
        "authority": authority,
        "input_file_sha256s": expected["input_file_sha256s"],
        "collector_file_sha256": expected["collector_file_sha256"],
        "protocol_file_sha256": expected["protocol_file_sha256"],
        "runtime_amendment_file_sha256": expected[
            "runtime_amendment_file_sha256"
        ],
        "rows": rows,
        "provider_calls": sum(int(row["provider_calls"]) for row in shards),
        "reported_cost_usd": sum(
            float(row["reported_cost_usd"]) for row in shards
        ),
        "shard_file_sha256s": [_sha256(path) for path in args.shard],
        "shard_report_sha256s": [row["report_sha256"] for row in shards],
        "shard_count": count,
        "task_count": len(rows),
        "supported_task_count": sum(bool(row["anchor_intervals"]) for row in rows),
        "provider_error_task_count": sum(bool(row["provider_error"]) for row in rows),
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "tasks": len(rows),
        "supported": report["supported_task_count"],
        "provider_errors": report["provider_error_task_count"],
        "provider_calls": report["provider_calls"],
        "reported_cost_usd": report["reported_cost_usd"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
