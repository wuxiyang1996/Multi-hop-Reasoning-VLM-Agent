#!/usr/bin/env python3
"""Execute CLEVRER explanatory questions on the official factual graph."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
EXECUTOR_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official/executor"
)
sys.path.insert(0, str(EXECUTOR_ROOT))

from executor import Executor  # type: ignore  # noqa: E402
from simulation import Simulation  # type: ignore  # noqa: E402
from motif_transfer.clevrer_oracle_query_mdp import (  # noqa: E402
    ClevrerOracleExecutionReceipt, official_annotation_to_factual_prediction,
)
from motif_transfer.clevrer_query_compiler import (  # noqa: E402
    compile_choice, compile_question,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import sha256_file  # noqa: E402


def _read(path):
    value = json.loads(path.read_text())
    body = dict(value); claimed = body.pop("artifact_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def main() -> None:
    cohort = REPO / "runs/clevrer_oracle_explanatory_v1"
    public = _read(cohort / "public_cohort.json")
    baseline = _read(cohort / "baseline_predictions.json")
    baseline_by_id = {row["task_id"]: row["prediction"] for row in baseline["rows"]}
    archive_path = Path(
        "/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official/"
        "official_annotations/annotation_validation.zip"
    )
    archive_sha = sha256_file(archive_path)
    runtime_rows = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as temp:
        for row in public["rows"]:
            scene = int(row["scene_index"])
            name = (
                f"annotation_{scene // 1000 * 1000:05d}-"
                f"{(scene // 1000 + 1) * 1000:05d}/annotation_{scene:05d}.json"
            )
            annotation = json.load(archive.open(name))
            payload = official_annotation_to_factual_prediction(
                annotation, question_family=row["family"],
            )
            # Legacy Simulation ignores this authority metadata; keep it out of
            # the native payload while retaining the runtime receipt below.
            payload.pop("authority")
            path = Path(temp) / f"{scene}.json"
            path.write_text(json.dumps(payload))
            executor = Executor(Simulation(str(path), use_event_ann=True))
            question_program = compile_question(row["question"], row["family"])
            choice_programs = [
                compile_choice(choice, row["family"]) for choice in row["choices"]
            ]
            answers = [
                executor.run(choice + question_program) for choice in choice_programs
            ]
            prediction = "".join("1" if answer == "yes" else "0" for answer in answers)
            receipt = ClevrerOracleExecutionReceipt.create(
                task_id=row["task_id"], scene_index=scene, family=row["family"],
                official_graph_sha256=stable_hash([archive_sha, scene]),
                question_program=question_program, choice_programs=choice_programs,
                prediction=prediction,
            )
            receipt.validate()
            runtime_rows.append({
                "task_id": row["task_id"], "oracle_prediction": prediction,
                "baseline_prediction": baseline_by_id[row["task_id"]],
                "receipt_sha256": receipt.receipt_sha256,
                "runtime_answer_read": False,
                "runtime_functional_program_read": False,
            })
    runtime_body = {
        "schema_version": "clevrer-oracle-explanatory-runtime-v1",
        "rows": runtime_rows,
    }
    runtime = runtime_body | {"artifact_sha256": stable_hash(runtime_body)}
    (cohort / "runtime_predictions.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n"
    )
    evaluator = _read(cohort / "evaluator_only.json")
    gold = {row["task_id"]: row["gold"] for row in evaluator["rows"]}
    oracle_correct = sum(
        row["oracle_prediction"] == gold[row["task_id"]] for row in runtime_rows
    )
    baseline_correct = sum(
        row["baseline_prediction"] == gold[row["task_id"]] for row in runtime_rows
    )
    gates = {
        "all_120_explanatory_tasks_executed": len(runtime_rows) == 120,
        "zero_runtime_answer_program_read": all(
            not row["runtime_answer_read"] and not row["runtime_functional_program_read"]
            for row in runtime_rows
        ),
        "official_factual_executor_perfect_on_consumed_diagnostic": oracle_correct == 120,
        "official_factual_executor_improves_neural_grounding": oracle_correct > baseline_correct,
        "predictive_counterfactual_not_claimed": True,
    }
    body = {
        "schema_version": "clevrer-oracle-explanatory-consumed-audit-v1",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": (
            "Consumed explanatory-only grounding diagnostic. Official factual graph "
            "does not authorize predictive or counterfactual claims."
        ),
        "tasks": len(runtime_rows),
        "neural_only_correct": baseline_correct,
        "neural_only_accuracy": baseline_correct / len(runtime_rows),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / len(runtime_rows),
        "gates": gates,
        "lineage": {
            "official_archive_sha256": archive_sha,
            "public_sha256": public["artifact_sha256"],
            "runtime_sha256": runtime["artifact_sha256"],
            "evaluator_sha256": evaluator["artifact_sha256"],
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / "docs/results/clevrer_oracle_explanatory_v1_consumed.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
