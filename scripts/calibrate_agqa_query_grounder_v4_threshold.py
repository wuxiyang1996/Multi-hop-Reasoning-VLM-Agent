#!/usr/bin/env python3
"""Select one global V4 adjudicator confidence threshold on train development."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

from motif_transfer.agqa_query_object_grounder import canonical_object_label
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object
from scripts.calibrate_agqa_query_grounder_v3_threshold import select_threshold, wilson_lower


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V4 calibration output is immutable")
    adjudication = json.loads(args.adjudication.read_text())
    protocol = json.loads(args.protocol.read_text())
    if adjudication.get("status") != "V4_ADJUDICATION_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("adjudication did not freeze before development outcomes")
    if any(adjudication.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("adjudication crossed the authority boundary")
    wanted = {str(row["task_id"]) for row in adjudication["rows"]}
    answers = {}
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            if str(task_id) in wanted:
                answers[str(task_id)] = canonical_object_label(str(row["answer"]))
                if len(answers) == len(wanted):
                    break
    if set(answers) != wanted:
        raise ValueError("development answers are incomplete")

    predictions = []
    for row in adjudication["rows"]:
        selected = row.get("selected_candidate")
        predicted = canonical_object_label(str(selected["label"])) if selected else None
        task_id = str(row["task_id"])
        candidates = {
            canonical_object_label(str(candidate["label"]))
            for candidate in row.get("candidates", ())
        }
        predictions.append({
            "task_id": task_id, "predicted": predicted,
            "confidence": float(row["confidence"]),
            "correct": predicted == answers[task_id] if predicted is not None else False,
            "gold_in_candidate_pool": answers[task_id] in candidates,
        })

    selection = protocol["threshold_selection"]
    curve = []
    for threshold in selection["candidate_grid"]:
        selected = [
            row for row in predictions
            if row["predicted"] is not None and row["confidence"] >= float(threshold)
        ]
        correct = sum(bool(row["correct"]) for row in selected)
        total = len(selected)
        curve.append({
            "threshold": float(threshold), "supported": total,
            "correct": correct, "precision": correct / total if total else 0.0,
            "precision_wilson_95_lower": wilson_lower(correct, total),
            "coverage": total / len(predictions) if predictions else 0.0,
        })
    chosen = select_threshold(curve, selection["constraints"])
    body = {
        "schema_version": "agqa-query-grounder-v4-global-threshold-calibration-v1",
        "status": "V4_GLOBAL_THRESHOLD_QUALIFIED_ON_DEVELOPMENT" if chosen else "V4_CALIBRATION_FAILED",
        "adjudication_report_sha256": adjudication["report_sha256"],
        "development_tasks": len(predictions),
        "bound_tasks": sum(row["predicted"] is not None for row in predictions),
        "all_row_top1_correct": sum(row["correct"] for row in predictions),
        "all_row_top1_accuracy": sum(row["correct"] for row in predictions) / len(predictions),
        "gold_in_candidate_pool": sum(row["gold_in_candidate_pool"] for row in predictions),
        "gold_in_candidate_pool_fraction": sum(row["gold_in_candidate_pool"] for row in predictions) / len(predictions),
        "curve": curve, "selected": chosen,
        "selection_objective": selection["objective"],
        "selection_constraints": selection["constraints"],
        "one_global_threshold": True,
        "predicate_role_slice_or_task_specific_thresholds_used": False,
        "answers_read_for_development_calibration_only": True,
        "official_scene_graph_or_program_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "development_tasks": len(predictions),
        "all_row_top1_accuracy": body["all_row_top1_accuracy"],
        "gold_in_candidate_pool_fraction": body["gold_in_candidate_pool_fraction"],
        "selected": chosen, "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
