#!/usr/bin/env python3
"""Evaluate the confirmed Sokoban effect guard on consumed TIR forks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import stable_hash  # noqa: E402
from motif_transfer.candidate_transfer_experiment import (  # noqa: E402
    candidate_calibration_predictions,
    nested_cross_fitted_candidate_predictions,
    receipt_answer_slots,
)
from motif_transfer.sokoban_effect_program import (  # noqa: E402
    select_option,
    validate_effect_program,
)


CONDITIONS = (
    "raw_target_only",
    "null_skill_same_harness",
    "authentic_sokoban_effect_skill",
    "commit_availability_control",
    "inverted_effect_control",
    "position_prior_control",
    "target_oracle_skill",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("claim_boundary") != "CONSUMED_TIR_DEVELOPMENT_ONLY":
        raise SystemExit("this evaluator cannot read fresh target data")
    source_receipt_path = (REPO / config["source"]["compact_receipt"]).resolve()
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    receipt_body = dict(source_receipt)
    claimed_receipt = str(receipt_body.pop("compact_receipt_sha256", ""))
    if stable_hash(receipt_body) != claimed_receipt:
        raise SystemExit("source compact receipt hash mismatch")
    source_path = (REPO / source_receipt["artifact"]["path"]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if _sha256(source_path) != source_receipt["artifact"]["file_sha256"]:
        raise SystemExit("source artifact file changed")
    validate_effect_program(source)
    if not source_receipt["fresh_confirmation"]["source_gate_passed"]:
        raise SystemExit("source confirmation is not qualified")

    receipts_path = (REPO / config["target"]["receipts"]).resolve()
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    expected_ids = list(map(str, config["splits"]["consumed_development"]))
    if [str(row["sample_id"]) for row in receipts] != expected_ids:
        raise SystemExit("consumed TIR receipt split/order mismatch")
    slots = receipt_answer_slots(receipts)
    grounding = config["target_grounder"]
    calibrated, _, _ = candidate_calibration_predictions(
        receipts, seed=int(grounding["belief_seed"]),
    )
    predictions = nested_cross_fitted_candidate_predictions(
        receipts,
        belief_seed=int(grounding["belief_seed"]),
        candidate_seed=int(grounding["candidate_seed"]),
        hidden_units=int(grounding["candidate_hidden_units"]),
        epochs=int(grounding["candidate_epochs"]),
    )
    threshold = float(config["policy"]["positive_effect_threshold"])
    fallback_threshold = float(config["policy"]["fallback_commit_threshold"])
    traces = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        gold = slots.index(str(receipt["gold_answer"]))
        before = calibrated[(sample_id, "BASE")]
        candidates = []
        for candidate in receipt["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            candidates.append({
                "candidate_id": candidate_id,
                "planner_score": float(candidate["planner_score"]),
                "predicted_effects": predictions[(sample_id, candidate_id)],
                "after_belief": calibrated[(sample_id, candidate_id)],
                "wrapper_tool": str(candidate["wrapper_receipt"]["tool"]),
            })
        neural_best = max(
            candidates, key=lambda row: row["predicted_effects"][2]
        )
        planner_best = max(candidates, key=lambda row: row["planner_score"])
        predicates = {
            "commit_available": bool(candidates),
            "direct_progress_available": (
                float(neural_best["predicted_effects"][2]) > threshold
            ),
            "assignment_improvement_available": False,
            "regression_observed": False,
            "deadlock_observed": False,
        }
        baseline_index = int(np.argmax(before))
        null_test = float(np.max(before)) < fallback_threshold
        for condition in CONDITIONS:
            selected_candidate = None
            if condition == "raw_target_only":
                selected_option = None
                committed = baseline_index
            elif condition == "null_skill_same_harness":
                selected_option = None
                selected_candidate = planner_best if null_test else None
                committed = (
                    int(np.argmax(planner_best["after_belief"]))
                    if null_test else baseline_index
                )
            elif condition == "target_oracle_skill":
                selected_option = "TARGET_ORACLE"
                correct = [
                    row for row in candidates
                    if int(np.argmax(row["after_belief"])) == gold
                ]
                selected_candidate = correct[0] if correct else None
                committed = gold if correct else baseline_index
            else:
                source_condition = {
                    "authentic_sokoban_effect_skill": "authentic_effect_guard",
                    "commit_availability_control": "commit_availability_only",
                    "inverted_effect_control": "inverted_effect_guard",
                    "position_prior_control": "position_occupancy_prior",
                }[condition]
                selected_option = select_option(source_condition, predicates)
                # The source COMMIT role is an effectful intervention; TIR
                # realizes it as TEST. POSITION realizes as target COMMIT.
                selected_candidate = (
                    neural_best if selected_option == "COMMIT" else None
                )
                committed = (
                    int(np.argmax(neural_best["after_belief"]))
                    if selected_candidate is not None else baseline_index
                )
            body = {
                "sample_id": sample_id,
                "condition": condition,
                "source_selected_option": selected_option,
                "target_native_action": (
                    "TEST" if selected_candidate is not None else "COMMIT"
                ),
                "selected_candidate_id": (
                    selected_candidate["candidate_id"]
                    if selected_candidate is not None else None
                ),
                "selected_wrapper_tool": (
                    selected_candidate["wrapper_tool"]
                    if selected_candidate is not None else None
                ),
                "baseline_answer": slots[baseline_index],
                "committed_answer": slots[committed],
                "gold_answer_evaluator_only": slots[gold],
                "correct_evaluator_only": committed == gold,
                "effect_predicates": predicates,
            }
            traces.append(body | {"trace_sha256": stable_hash(body)})

    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in CONDITIONS
    }
    summaries = {
        condition: {
            "tasks": len(rows),
            "successes": sum(row["correct_evaluator_only"] for row in rows),
            "success_rate": sum(row["correct_evaluator_only"] for row in rows)
            / len(rows),
            "tests": sum(row["target_native_action"] == "TEST" for row in rows),
        }
        for condition, rows in by_condition.items()
    }
    authentic = {
        row["sample_id"]: row
        for row in by_condition["authentic_sokoban_effect_skill"]
    }
    paired = {}
    for comparator in (
        "null_skill_same_harness", "commit_availability_control",
        "inverted_effect_control", "position_prior_control",
    ):
        other = {row["sample_id"]: row for row in by_condition[comparator]}
        delta = [
            int(authentic[sample_id]["correct_evaluator_only"])
            - int(other[sample_id]["correct_evaluator_only"])
            for sample_id in expected_ids
        ]
        paired[comparator] = {
            "wins": sum(value > 0 for value in delta),
            "losses": sum(value < 0 for value in delta),
            "ties": sum(value == 0 for value in delta),
            "net_wins": sum(delta),
        }
    gates = {
        "shared_sokoban_source_qualified": True,
        "cross_fitted_target_grounding": True,
        "authentic_action_contrast": any(
            authentic[sample_id]["target_native_action"]
            != next(row for row in by_condition["null_skill_same_harness"]
                    if row["sample_id"] == sample_id)["target_native_action"]
            for sample_id in expected_ids
        ),
        "authentic_success_gain_over_null": paired[
            "null_skill_same_harness"
        ]["net_wins"] > 0,
        "authentic_zero_negative_transfer": paired[
            "null_skill_same_harness"
        ]["losses"] == 0,
        "authentic_strictly_beats_source_controls": all(
            summaries["authentic_sokoban_effect_skill"]["successes"]
            > summaries[name]["successes"]
            for name in (
                "commit_availability_control", "inverted_effect_control",
                "position_prior_control",
            )
        ),
    }
    body = {
        "schema_version": "sokoban-tir-effect-development-v5",
        "status": (
            "CONSUMED_MECHANISM_GATE_PASSED" if all(gates.values())
            else "CONSUMED_MECHANISM_GATE_FAILED"
        ),
        "claim_boundary": config["claim_boundary"],
        "source_artifact_sha256": source["artifact_sha256"],
        "source_confirmation_sha256": source_receipt[
            "fresh_confirmation"
        ]["report_sha256"],
        "target_receipts_file_sha256": _sha256(receipts_path),
        "tasks": expected_ids,
        "mapping": {
            "source_COMMIT": "target_TEST_best_target_neural_candidate",
            "source_POSITION": "target_COMMIT_baseline_answer",
        },
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "traces": traces,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": body["status"], "summaries": summaries,
        "paired": paired, "gates": gates, "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
