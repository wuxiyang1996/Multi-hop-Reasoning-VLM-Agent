#!/usr/bin/env python3
"""Merge complete answer-blind binary event-verifier shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


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
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged binary event verification is immutable")
    shards = [json.loads(path.read_text()) for path in args.shard]
    candidate = json.loads(args.candidate_grounding.read_text())
    protocol_sha = _sha256(args.protocol)
    if not shards:
        raise ValueError("no binary-verifier shards")
    count = int(shards[0]["shard_count"])
    if len(shards) != count or {int(row["shard_index"]) for row in shards} != set(range(count)):
        raise ValueError("binary-verifier shards are incomplete")
    invariant_keys = (
        "schema_version", "status", "model", "maximum_frames", "cohort_sha256",
        "candidate_grounding_report_sha256", "protocol_file_sha256",
        "question_text_read", "answer_read", "official_scene_graph_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
        "alternative_candidate_selection_allowed",
        "candidate_label_emitted_as_answer", "consumed_development_pilot",
        "supported_evidence_requires_same_frame_person_and_candidate_tracks",
    )
    expected = {key: shards[0][key] for key in invariant_keys}
    if expected["protocol_file_sha256"] != protocol_sha:
        raise ValueError("binary-verifier protocol hash differs")
    if expected["candidate_grounding_report_sha256"] != candidate["report_sha256"]:
        raise ValueError("binary verifier and candidate grounding differ")
    if any({key: row[key] for key in invariant_keys} != expected for row in shards):
        raise ValueError("binary-verifier shard invariants differ")
    if any(expected[key] for key in (
        "question_text_read", "answer_read", "official_scene_graph_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
        "alternative_candidate_selection_allowed",
        "candidate_label_emitted_as_answer",
    )):
        raise ValueError("binary verifier crossed its authority boundary")
    rows = []
    seen = set()
    for shard in sorted(shards, key=lambda row: int(row["shard_index"])):
        for row in shard["rows"]:
            task_id = str(row["task_id"])
            if task_id in seen:
                raise ValueError("duplicate task in binary-verifier shards")
            seen.add(task_id)
            rows.append(row)
    order = {str(row["task_id"]): index for index, row in enumerate(candidate["rows"])}
    if set(order) != seen:
        raise ValueError("binary verifier does not cover every candidate-grounding row")
    rows.sort(key=lambda row: order[str(row["task_id"])])
    attempted = [row for row in rows if row["status"] != "ABSTAIN_NO_EVENT_CANDIDATE"]
    report = {
        **{key: value for key, value in expected.items() if key not in {
            "schema_version", "status",
        }},
        "schema_version": "agqa-answer-blind-binary-event-verifier-v1",
        "status": "BINARY_EVENT_VERIFICATION_ANSWER_BLIND",
        "rows": rows, "task_count": len(rows),
        "attempted_task_count": len(attempted),
        "provider_calls": sum(int(row["provider_calls"]) for row in shards),
        "reported_cost_usd": sum(float(row["reported_cost_usd"]) for row in shards),
        "provider_and_contract_success_fraction": (
            sum(row.get("provider_error") is None for row in attempted) / len(attempted)
            if attempted else 1.0
        ),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("SUPPORTED", "REFUTED", "UNKNOWN", "ABSTAIN_NO_EVENT_CANDIDATE")
        },
        "shard_file_sha256s": [_sha256(path) for path in args.shard],
        "shard_report_sha256s": [row["report_sha256"] for row in shards],
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "tasks": report["task_count"],
        "status_counts": report["status_counts"],
        "provider_and_contract_success_fraction": report["provider_and_contract_success_fraction"],
        "reported_cost_usd": report["reported_cost_usd"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
