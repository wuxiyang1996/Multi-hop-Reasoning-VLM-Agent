#!/usr/bin/env python3
"""Split selected AGQA2 rows into public-runtime and evaluator-only files.

This one-time freezer is allowed to read official QA annotations.  It selects
only task IDs that were frozen previously, verifies their public hashes, and
writes disjoint files.  The runtime file contains no answer, functional
program, or program-derived scene-graph grounding.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterator, TextIO
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


def iter_top_level_object(handle: TextIO, chunk_size: int = 1024 * 1024) -> Iterator[tuple[str, Any]]:
    """Incrementally decode a large JSON object without materializing it."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def refill() -> None:
        nonlocal buffer, position, eof
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = handle.read(chunk_size)
        if chunk:
            buffer += chunk
        else:
            eof = True

    refill()
    while position < len(buffer) and buffer[position].isspace():
        position += 1
    if position >= len(buffer) or buffer[position] != "{":
        raise ValueError("AGQA annotation is not a top-level JSON object")
    position += 1
    while True:
        while True:
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError("truncated AGQA annotation")
            refill()
        if buffer[position] == "}":
            return
        while True:
            try:
                key, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                refill()
        position = end
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            refill()
        if buffer[position] != ":":
            raise ValueError("malformed AGQA object separator")
        position += 1
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            try:
                value, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                refill()
        position = end
        yield str(key), value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip"),
    )
    parser.add_argument(
        "--member", default="AGQA_balanced/train_balanced.txt",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/agqa2_full_distribution_v62_manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/agqa2_oracle_query_mdp_v1",
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=REPO / "runs/agqa2_full_distribution_v62/base_report.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = {str(row["task_id"]): row for row in manifest["samples"]}
    selected = {}
    with zipfile.ZipFile(args.archive) as archive, archive.open(args.member) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8")
        for task_id, row in iter_top_level_object(text):
            if task_id in expected:
                selected[task_id] = row
                if len(selected) == len(expected):
                    break
    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(f"selected AGQA tasks missing from official annotation: {missing[:5]}")

    public_rows = []
    evaluator_rows = []
    hash_matches = True
    for task_id in sorted(selected):
        row = selected[task_id]
        frozen = expected[task_id]
        question = str(row["question"])
        question_sha = stable_hash(question)
        hash_matches &= question_sha == str(frozen["question_sha256"])
        public_rows.append({
            "task_id": task_id, "video_id": str(row["video_id"]),
            "question": question, "question_sha256": question_sha,
            "answer_type": str(row.get("ans_type", "")),
            "semantic": str(row.get("semantic", "")),
            "structural": str(row.get("structural", "")),
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        })
        evaluator_rows.append({
            "task_id": task_id, "answer": row["answer"],
            "functional_program_sha256": stable_hash(row.get("program")),
            "sg_grounding_sha256": stable_hash(row.get("sg_grounding")),
        })
    public_body = {
        "schema_version": "agqa2-oracle-query-public-cohort-v1",
        "role": "CONSUMED_DIAGNOSTIC_RUNTIME_ONLY",
        "source_manifest_sha256": manifest["manifest_sha256"],
        "rows": public_rows,
    }
    evaluator_body = {
        "schema_version": "agqa2-oracle-query-evaluator-only-v1",
        "role": "EVALUATOR_ONLY_OPEN_AFTER_PREDICTIONS_FREEZE",
        "rows": evaluator_rows,
    }
    base_report = json.loads(args.base_report.read_text())
    direct_by_id = {str(row["task_id"]): row for row in base_report["rows"]}
    if set(direct_by_id) != set(selected):
        raise ValueError("frozen direct prediction identities do not match cohort")
    if not all(
        row.get("direct_call_started_after_all_typed_receipts_froze") is True
        and row.get("official_answer_first_read_after_all_runtime_rows_froze") is True
        for row in direct_by_id.values()
    ):
        raise ValueError("direct prediction report lacks a valid authority boundary")
    direct_body = {
        "schema_version": "agqa2-oracle-query-frozen-direct-predictions-v1",
        "role": "RUNTIME_BASELINE_NO_OUTCOME_FIELDS",
        "rows": [{
            "task_id": task_id,
            "prediction": str(direct_by_id[task_id]["direct_response"]),
            "runtime_receipt_sha256": str(
                direct_by_id[task_id]["runtime_receipt_sha256"]
            ),
        } for task_id in sorted(direct_by_id)],
    }
    receipt_body = {
        "schema_version": "agqa2-oracle-query-cohort-freeze-v1",
        "status": "PASSED" if hash_matches else "FAILED",
        "selected_rows": len(selected),
        "question_hashes_match_frozen_manifest": hash_matches,
        "runtime_file_contains_answer": False,
        "runtime_file_contains_functional_program": False,
        "runtime_file_contains_sg_grounding": False,
        "public_sha256": stable_hash(public_body),
        "evaluator_sha256": stable_hash(evaluator_body),
        "direct_predictions_sha256": stable_hash(direct_body),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "public_cohort.json").write_text(
        json.dumps(public_body | {"artifact_sha256": stable_hash(public_body)}, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "evaluator_only.json").write_text(
        json.dumps(evaluator_body | {"artifact_sha256": stable_hash(evaluator_body)}, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "direct_predictions.json").write_text(
        json.dumps(direct_body | {"artifact_sha256": stable_hash(direct_body)}, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "freeze_receipt.json").write_text(
        json.dumps(receipt_body | {"receipt_sha256": stable_hash(receipt_body)}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt_body, indent=2, sort_keys=True))
    if not hash_matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
