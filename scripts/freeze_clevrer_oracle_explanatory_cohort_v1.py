#!/usr/bin/env python3
"""Freeze disjoint CLEVRER public runtime and evaluator-only artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


def _artifact(body):
    return body | {"artifact_sha256": stable_hash(body)}


def main() -> None:
    split = json.loads((REPO / "configs/clevrer_sokoban_proof_v14_splits.json").read_text())
    task_ids = split["benchmarks"]["clevrer"]["family_roles"]["explanatory"]["reserve"]
    official = json.loads(Path(
        "/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official/executor/data/validation.json"
    ).read_text())
    scenes = {int(row["scene_index"]): row for row in official}
    old_report = json.loads((
        REPO / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json"
    ).read_text())
    old = {row["sample_id"]: row for row in old_report["rows"]}
    public_rows, evaluator_rows, baseline_rows = [], [], []
    for task_id in task_ids:
        match = re.fullmatch(r"video_(\d+)\.mp4\.Q(\d+)", task_id)
        scene, question_id = int(match.group(1)), int(match.group(2))
        question = next(
            row for row in scenes[scene]["questions"]
            if int(row["question_id"]) == question_id
        )
        public_rows.append({
            "task_id": task_id, "scene_index": scene, "family": "explanatory",
            "question": str(question["question"]),
            "choices": [str(row["choice"]) for row in question["choices"]],
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
        })
        evaluator_rows.append({
            "task_id": task_id,
            "gold": "".join(
                "1" if row["answer"] == "correct" else "0"
                for row in question["choices"]
            ),
            "official_question_program_sha256": stable_hash(question["program"]),
            "official_choice_programs_sha256": stable_hash(
                [row["program"] for row in question["choices"]]
            ),
        })
        baseline_rows.append({
            "task_id": task_id,
            "prediction": old[task_id]["conditions"][
                "neural_only_explicit_relation"
            ]["answer"],
        })
    output = REPO / "runs/clevrer_oracle_explanatory_v1"
    output.mkdir(parents=True, exist_ok=True)
    for name, body in {
        "public_cohort.json": {
            "schema_version": "clevrer-oracle-explanatory-public-v1",
            "role": "CONSUMED_DIAGNOSTIC_RUNTIME_ONLY", "rows": public_rows,
        },
        "evaluator_only.json": {
            "schema_version": "clevrer-oracle-explanatory-evaluator-v1",
            "role": "OPEN_AFTER_PREDICTIONS_FREEZE", "rows": evaluator_rows,
        },
        "baseline_predictions.json": {
            "schema_version": "clevrer-oracle-explanatory-baseline-v1",
            "role": "FROZEN_RUNTIME_BASELINE", "rows": baseline_rows,
        },
    }.items():
        artifact = _artifact(body)
        (output / name).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASSED", "tasks": len(public_rows),
        "runtime_answer_or_program_fields": False,
    }, indent=2))


if __name__ == "__main__":
    main()
