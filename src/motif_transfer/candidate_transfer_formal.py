"""Frozen evaluation for parameterized TEST/COMMIT transfer artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .active_video_transfer import (
    CalibrationRow,
    CandidateEffectGrounder,
    GroundedCandidateIntervention,
    SoftmaxCalibrationHead,
    build_source_value_models,
    choose_candidate_action,
    exact_binomial_two_sided,
    normalized_entropy,
    normalized_probabilities,
    stable_hash,
)
from .candidate_transfer_experiment import receipt_answer_slots


SOURCE_CONDITIONS = (
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
)
FORMAL_CONDITIONS = (
    "target_only",
    *SOURCE_CONDITIONS,
    "target_native_information_gain",
    "target_native_candidate_uplift",
)


def validate_frozen_artifact(
    artifact: Mapping[str, Any],
    *,
    controlled_config: Mapping[str, Any],
) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError("frozen target grounder artifact hash mismatch")
    if artifact.get("role") != (
        "TARGET_NATIVE_PARAMETERIZED_INTERVENTION_GROUNDER_ADAPTATION_ONLY"
    ):
        raise ValueError("artifact has the wrong training authority")
    if str(artifact["source_config_sha256"]) != stable_hash(controlled_config):
        raise ValueError("frozen source config hash mismatch")
    if artifact.get("source_value_model", {}).get("kind") != "frozen_source_prior":
        raise ValueError("formal evaluator forbids target-residual source models")
    SoftmaxCalibrationHead.from_dict(artifact["belief_calibration_head"])
    CandidateEffectGrounder.from_dict(artifact["candidate_effect_grounder"])


def _calibrated(
    head: SoftmaxCalibrationHead,
    probabilities: Mapping[str, float],
    *,
    slots: Sequence[str],
    prefix_length: int,
    planner_score: float,
) -> np.ndarray:
    row = CalibrationRow(
        sample_id="formal",
        prefix_length=prefix_length,
        max_tests=1,
        mean_planner_score=float(planner_score),
        raw_probabilities=tuple(normalized_probabilities(
            probabilities, answer_slots=slots,
        )),
        answer_index=0,
    )
    return head.predict(row.features())


def evaluate_frozen_candidate_transfer(
    receipts: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
    controlled_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen neural grounding and source models without target fitting."""

    validate_frozen_artifact(artifact, controlled_config=controlled_config)
    expected_ids = tuple(map(str, config["splits"]["qualification"]))
    observed_ids = tuple(str(row["sample_id"]) for row in receipts)
    if observed_ids != expected_ids:
        raise ValueError("formal receipt order/coverage differs from frozen split")
    training_ids = set(map(str, artifact["training_sample_ids"]))
    overlap = sorted(training_ids.intersection(observed_ids))
    if overlap:
        raise ValueError(f"adaptation/qualification leakage: {overlap}")

    slots = receipt_answer_slots(receipts)
    artifact_slots = tuple(map(str, artifact.get("answer_slots", slots)))
    if artifact_slots != slots:
        raise ValueError("artifact/target answer-slot contract mismatch")
    calibration = SoftmaxCalibrationHead.from_dict(
        artifact["belief_calibration_head"]
    )
    grounder = CandidateEffectGrounder.from_dict(
        artifact["candidate_effect_grounder"]
    )
    source_models = build_source_value_models(
        controlled_config,
        seed=int(config["source"]["model_seed"]),
        objective_test_cost=float(artifact["target_objective_test_cost"]),
    )
    policy = config["policy"]
    traces: list[dict[str, Any]] = []
    baseline_correct: dict[str, bool] = {}
    oracle_correct: dict[str, bool] = {}

    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        gold = slots.index(str(receipt["gold_answer"]))
        before = _calibrated(
            calibration,
            receipt["baseline"]["answer"]["probabilities"],
            slots=slots,
            prefix_length=0,
            planner_score=0.0,
        )
        baseline_correct[sample_id] = int(np.argmax(before)) == gold
        grounded: list[GroundedCandidateIntervention] = []
        candidate_index: dict[str, Mapping[str, Any]] = {}
        candidate_beliefs: dict[str, np.ndarray] = {}
        for candidate in receipt["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            candidate_index[candidate_id] = candidate
            planner_score = float(candidate["planner_score"])
            information_gain, confidence_gain, quality_gain = grounder.predict(
                before,
                planner_score=planner_score,
                descriptor=tuple(map(float, candidate["descriptor"])),
            )
            positive_quality = max(0.0, quality_gain)
            grounded.append(GroundedCandidateIntervention(
                candidate_id=candidate_id,
                planner_score=planner_score,
                predicted_information_gain=max(
                    information_gain,
                    positive_quality * normalized_entropy(before),
                ),
                predicted_confidence_gain=max(
                    confidence_gain,
                    positive_quality * (1.0 - float(np.max(before))),
                ),
                predicted_answer_quality_gain=quality_gain,
                predicted_outcome_balance=(
                    1.0 - 2.0 * abs(planner_score - 0.5)
                ),
            ))
            candidate_beliefs[candidate_id] = _calibrated(
                calibration,
                candidate["answer"]["probabilities"],
                slots=slots,
                prefix_length=1,
                planner_score=planner_score,
            )
        oracle_correct[sample_id] = any(
            int(np.argmax(value)) == gold for value in candidate_beliefs.values()
        )

        for condition in FORMAL_CONDITIONS:
            decision = choose_candidate_action(
                before,
                condition=condition,
                candidates=grounded,
                source_models=source_models,
                uncertainty_scale=float(policy["uncertainty_scale"]),
                decision_margin=float(policy["decision_margin"]),
                fallback_commit_threshold=float(policy["fallback_commit_threshold"]),
                target_quality_threshold=float(policy["target_quality_threshold"]),
                information_gain_threshold=float(policy["information_gain_threshold"]),
            )
            selected_id = (
                str(decision.candidate_id) if decision.candidate_id is not None else None
            )
            if decision.kind == "TEST":
                committed_index = int(np.argmax(candidate_beliefs[selected_id]))
                selected_tool = str(
                    candidate_index[selected_id]["wrapper_receipt"]["tool"]
                )
            else:
                committed_index = int(decision.answer_index)
                selected_tool = None
            trace_body = {
                "sample_id": sample_id,
                "family": str(receipt.get("family") or ""),
                "condition": condition,
                "decision": decision.__dict__,
                "selected_candidate_id": selected_id,
                "selected_wrapper_tool": selected_tool,
                "baseline_answer": slots[int(np.argmax(before))],
                "committed_answer": slots[committed_index],
                "gold_answer_evaluator_only": slots[gold],
                "correct_evaluator_only": committed_index == gold,
                "grounded_candidates": [row.__dict__ for row in grounded],
            }
            traces.append(trace_body | {"trace_sha256": stable_hash(trace_body)})

    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in FORMAL_CONDITIONS
    }
    conditions = {
        condition: {
            "samples": len(rows),
            "successes": sum(row["correct_evaluator_only"] for row in rows),
            "accuracy": sum(row["correct_evaluator_only"] for row in rows) / len(rows),
            "tests": sum(row["decision"]["kind"] == "TEST" for row in rows),
            "action_changes_vs_baseline": sum(
                row["committed_answer"] != row["baseline_answer"] for row in rows
            ),
        }
        for condition, rows in by_condition.items()
    }
    authentic_rows = {
        row["sample_id"]: row for row in by_condition["authentic_source_plus_target"]
    }
    comparisons: dict[str, Any] = {}
    for comparator in (
        "target_only", "shuffled_source_plus_target", "source_marginal_plus_target",
    ):
        other = {row["sample_id"]: row for row in by_condition[comparator]}
        wins = losses = 0
        contrasts = 0
        for sample_id in expected_ids:
            a = authentic_rows[sample_id]
            b = other[sample_id]
            delta = int(a["correct_evaluator_only"]) - int(b["correct_evaluator_only"])
            wins += delta > 0
            losses += delta < 0
            contrasts += (
                a["decision"]["kind"], a["selected_candidate_id"]
            ) != (
                b["decision"]["kind"], b["selected_candidate_id"]
            )
        comparisons[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(expected_ids) - wins - losses,
            "net_wins": wins - losses,
            "action_contrast_samples": contrasts,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }

    gates_spec = config["formal_gates"]
    authentic = conditions["authentic_source_plus_target"]
    gates = {
        "receipt_matrix_complete": len(receipts) == len(expected_ids),
        "adaptation_qualification_disjoint": not overlap,
        "authentic_action_contrast": comparisons["target_only"][
            "action_contrast_samples"
        ] >= int(gates_spec["minimum_authentic_action_contrasts"]),
        "authentic_success_gain_over_target_only": comparisons["target_only"][
            "net_wins"
        ] > 0,
        "authentic_zero_negative_transfer": comparisons["target_only"][
            "losses"
        ] == 0,
        "authentic_strictly_beats_source_controls": all(
            authentic["successes"] > conditions[name]["successes"]
            for name in (
                "shuffled_source_plus_target", "source_marginal_plus_target",
            )
        ),
        "oracle_candidate_headroom": sum(oracle_correct.values()) >= authentic[
            "successes"
        ],
    }
    body = {
        "schema_version": "frozen-candidate-transfer-formal-v1",
        "status": "FORMAL_PASS" if all(gates.values()) else "FORMAL_FAIL",
        "claim_boundary": str(config["claim_boundary"]),
        "artifact_sha256": artifact["artifact_sha256"],
        "source_config_sha256": artifact["source_config_sha256"],
        "adaptation_sample_ids": sorted(training_ids),
        "qualification_sample_ids": list(expected_ids),
        "conditions": conditions,
        "paired_comparisons": comparisons,
        "baseline_accuracy": sum(baseline_correct.values()) / len(expected_ids),
        "oracle_candidate_accuracy": sum(oracle_correct.values()) / len(expected_ids),
        "gates": gates,
        "traces": traces,
    }
    body["report_sha256"] = stable_hash(body)
    return body


__all__ = [
    "FORMAL_CONDITIONS",
    "evaluate_frozen_candidate_transfer",
    "validate_frozen_artifact",
]
