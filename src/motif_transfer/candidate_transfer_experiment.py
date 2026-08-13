"""Shared adaptation-only evaluator for parameterized visual interventions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .active_video_transfer import (
    ANSWER_SLOTS,
    CandidateEffectGrounder,
    CandidateEffectRow,
    CalibrationRow,
    GroundedCandidateIntervention,
    add_target_residual_to_source_models,
    build_source_value_models,
    candidate_action_features,
    choose_candidate_action,
    fit_calibration_head,
    fit_candidate_effect_grounder,
    normalized_entropy,
    normalized_probabilities,
    source_test_feature_support,
    stable_hash,
)
from .controlled_exploration_transfer import AbstractAction, MatchedValueExample


def receipt_answer_slots(
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Resolve and validate one experiment-wide native answer vocabulary."""

    if not receipts:
        raise ValueError("receipts cannot be empty")
    declared = receipts[0].get("answer_slots")
    if declared is None:
        declared = tuple(
            (receipts[0].get("baseline") or {}).get("answer", {})
            .get("probabilities", {})
            .keys()
        ) or ANSWER_SLOTS
    slots = tuple(map(str, declared))
    if len(slots) < 2 or len(set(slots)) != len(slots):
        raise ValueError("receipt answer_slots must contain unique native slots")
    for receipt in receipts:
        row_slots = tuple(map(str, receipt.get("answer_slots", slots)))
        if row_slots != slots:
            raise ValueError("all receipts must share the same answer_slots")
        if str(receipt["gold_answer"]) not in slots:
            raise ValueError("gold answer is outside receipt answer_slots")
    return slots


def _candidate_calibration_rows(
    receipts: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, CalibrationRow]]:
    answer_slots = receipt_answer_slots(receipts)
    indexed: list[tuple[str, str, CalibrationRow]] = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        gold_index = answer_slots.index(str(receipt["gold_answer"]))
        indexed.append((sample_id, "BASE", CalibrationRow(
            sample_id=f"{sample_id}::BASE",
            prefix_length=0,
            max_tests=1,
            mean_planner_score=0.0,
            raw_probabilities=tuple(normalized_probabilities(
                receipt["baseline"]["answer"]["probabilities"],
                answer_slots=answer_slots,
            )),
            answer_index=gold_index,
        )))
        for candidate in receipt["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            indexed.append((sample_id, candidate_id, CalibrationRow(
                sample_id=f"{sample_id}::{candidate_id}",
                prefix_length=1,
                max_tests=1,
                mean_planner_score=float(candidate["planner_score"]),
                raw_probabilities=tuple(normalized_probabilities(
                    candidate["answer"]["probabilities"],
                    answer_slots=answer_slots,
                )),
                answer_index=gold_index,
            )))
    return indexed


def candidate_calibration_predictions(
    receipts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    Any,
]:
    """Return leave-one-task-out and full slot-symmetric beliefs."""

    indexed = _candidate_calibration_rows(receipts)
    sample_ids = sorted({sample_id for sample_id, _, _ in indexed})
    if len(sample_ids) < 3:
        raise ValueError("belief cross-fitting requires at least three samples")
    cross_fitted: dict[tuple[str, str], np.ndarray] = {}
    for fold, held_out in enumerate(sample_ids):
        training = [row for sample_id, _, row in indexed if sample_id != held_out]
        head = fit_calibration_head(training, seed=seed + fold)
        for sample_id, key, row in indexed:
            if sample_id == held_out:
                cross_fitted[(sample_id, key)] = head.predict(row.features())
    full_head = fit_calibration_head([row for _, _, row in indexed], seed=seed)
    full = {
        (sample_id, key): full_head.predict(row.features())
        for sample_id, key, row in indexed
    }
    return cross_fitted, full, full_head


def calibration_predictions_excluding_sample(
    receipts: Sequence[Mapping[str, Any]],
    *,
    excluded_sample_id: str,
    seed: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Fit one calibration head with the held-out task completely absent."""

    indexed = _candidate_calibration_rows(receipts)
    training = [
        row for sample_id, _, row in indexed
        if sample_id != excluded_sample_id
    ]
    head = fit_calibration_head(training, seed=seed)
    return {
        (sample_id, key): head.predict(row.features())
        for sample_id, key, row in indexed
    }


def nested_cross_fitted_candidate_predictions(
    receipts: Sequence[Mapping[str, Any]],
    *,
    belief_seed: int,
    candidate_seed: int,
    hidden_units: int,
    epochs: int,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Predict each task after removing it from both learned target heads."""

    answer_slots = receipt_answer_slots(receipts)
    sample_ids = sorted({str(receipt["sample_id"]) for receipt in receipts})
    if len(sample_ids) < 3:
        raise ValueError("nested candidate cross-fitting needs three samples")
    output: dict[tuple[str, str], tuple[float, float, float]] = {}
    for fold, held_out in enumerate(sample_ids):
        training_receipts = [
            receipt for receipt in receipts
            if str(receipt["sample_id"]) != held_out
        ]
        calibration_head = fit_calibration_head(
            [row for _, _, row in _candidate_calibration_rows(training_receipts)],
            seed=belief_seed + fold,
        )
        predictions: dict[tuple[str, str], np.ndarray] = {}
        for receipt in receipts:
            sample_id = str(receipt["sample_id"])
            baseline = CalibrationRow(
                sample_id=f"{sample_id}::BASE",
                prefix_length=0,
                max_tests=1,
                mean_planner_score=0.0,
                raw_probabilities=tuple(normalized_probabilities(
                    receipt["baseline"]["answer"]["probabilities"],
                    answer_slots=answer_slots,
                )),
                answer_index=0,
            )
            predictions[(sample_id, "BASE")] = calibration_head.predict(
                baseline.features()
            )
            for candidate in receipt["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                row = CalibrationRow(
                    sample_id=f"{sample_id}::{candidate_id}",
                    prefix_length=1,
                    max_tests=1,
                    mean_planner_score=float(candidate["planner_score"]),
                    raw_probabilities=tuple(normalized_probabilities(
                        candidate["answer"]["probabilities"],
                        answer_slots=answer_slots,
                    )),
                    answer_index=0,
                )
                predictions[(sample_id, candidate_id)] = (
                    calibration_head.predict(row.features())
                )
        effect_rows = candidate_effect_rows(
            receipts, calibrated_predictions=predictions,
        )
        model = fit_candidate_effect_grounder(
            [row for row in effect_rows if row.sample_id != held_out],
            seed=candidate_seed + fold,
            hidden_units=hidden_units,
            epochs=epochs,
        )
        for row in effect_rows:
            if row.sample_id == held_out:
                output[(row.sample_id, row.candidate_id)] = model.predict(
                    row.current_belief,
                    planner_score=row.planner_score,
                    descriptor=row.descriptor,
                )
    return output


def candidate_effect_rows(
    receipts: Sequence[Mapping[str, Any]],
    *,
    calibrated_predictions: Mapping[tuple[str, str], np.ndarray] | None = None,
) -> list[CandidateEffectRow]:
    answer_slots = receipt_answer_slots(receipts)
    rows: list[CandidateEffectRow] = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        gold_index = answer_slots.index(str(receipt["gold_answer"]))
        before = (
            calibrated_predictions[(sample_id, "BASE")]
            if calibrated_predictions is not None
            else normalized_probabilities(
                receipt["baseline"]["answer"]["probabilities"],
                answer_slots=answer_slots,
            )
        )
        for candidate in receipt["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            after = (
                calibrated_predictions[(sample_id, candidate_id)]
                if calibrated_predictions is not None
                else normalized_probabilities(
                    candidate["answer"]["probabilities"],
                    answer_slots=answer_slots,
                )
            )
            quality = math.log(float(after[gold_index])) - math.log(
                float(before[gold_index])
            )
            rows.append(CandidateEffectRow(
                sample_id=sample_id,
                candidate_id=candidate_id,
                current_belief=tuple(map(float, before)),
                planner_score=float(candidate["planner_score"]),
                descriptor=tuple(map(float, candidate["descriptor"])),
                information_gain=normalized_entropy(before) - normalized_entropy(after),
                confidence_gain=float(np.max(after) - np.max(before)),
                answer_quality_gain=float(np.clip(quality / 4.0, -1.0, 1.0)),
            ))
    return rows


def leave_one_sample_candidate_predictions(
    rows: Sequence[CandidateEffectRow],
    *,
    seed: int,
    hidden_units: int,
    epochs: int = 1800,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    output: dict[tuple[str, str], tuple[float, float, float]] = {}
    sample_ids = sorted({row.sample_id for row in rows})
    if len(sample_ids) < 3:
        raise ValueError("candidate cross-fitting requires at least three samples")
    for fold, sample_id in enumerate(sample_ids):
        train = [row for row in rows if row.sample_id != sample_id]
        model = fit_candidate_effect_grounder(
            train, seed=seed + fold, hidden_units=hidden_units, epochs=epochs,
        )
        for row in rows:
            if row.sample_id != sample_id:
                continue
            output[(row.sample_id, row.candidate_id)] = model.predict(
                row.current_belief,
                planner_score=row.planner_score,
                descriptor=row.descriptor,
            )
    return output


def _condition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "tests": int(sum(row["decision"]["kind"] == "TEST" for row in rows)),
        "mean_answer_quality_gain": float(np.mean([
            row["realized_answer_quality_gain"] for row in rows
        ])),
        "action_changes_vs_baseline": int(sum(
            row["committed_answer"] != row["baseline_answer"] for row in rows
        )),
    }


def _target_residual_examples(
    receipts: Sequence[Mapping[str, Any]],
    *,
    calibrated: Mapping[tuple[str, str], np.ndarray],
    grounded_by_sample: Mapping[str, Sequence[GroundedCandidateIntervention]],
    excluded_sample_id: str,
    included_sample_ids: set[str] | None,
    test_cost: float,
) -> tuple[MatchedValueExample, ...]:
    """Compile matched target forks to abstract TEST/COMMIT value rows."""

    answer_slots = receipt_answer_slots(receipts)
    output: list[MatchedValueExample] = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        if sample_id == excluded_sample_id:
            continue
        if included_sample_ids is not None and sample_id not in included_sample_ids:
            continue
        belief = calibrated[(sample_id, "BASE")]
        grounded = tuple(grounded_by_sample[sample_id])
        tests, commits = candidate_action_features(
            belief, candidates=grounded, remaining_test_fraction=1.0,
        )
        gold_index = answer_slots.index(str(receipt["gold_answer"]))
        candidate_index = {
            str(candidate["candidate_id"]): candidate
            for candidate in receipt["candidates"]
        }
        for action_index, (candidate, features) in enumerate(zip(grounded, tests)):
            after = calibrated[(sample_id, candidate.candidate_id)]
            success = int(np.argmax(after)) == gold_index
            output.append(MatchedValueExample(
                state_id=sample_id,
                action=AbstractAction(
                    "TEST", action_index, f"target_test_{action_index}",
                ),
                features=features,
                value=float(success) - test_cost,
            ))
            if candidate.candidate_id not in candidate_index:
                raise AssertionError("grounded candidate/receipt drift")
        for answer_index, features in enumerate(commits):
            output.append(MatchedValueExample(
                state_id=sample_id,
                action=AbstractAction(
                    "COMMIT", answer_index, f"target_commit_{answer_index}",
                ),
                features=features,
                value=float(answer_index == gold_index),
            ))
    return tuple(output)


def evaluate_candidate_adaptation(
    receipts: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    controlled_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-fit the target grounder and evaluate one matched intervention.

    Every condition sees the same candidate receipt set.  Source conditions may
    choose a different parameterized TEST candidate or COMMIT, but never see a
    target tool name, coordinate, timestamp, answer text, or gold label.
    """

    answer_slots = receipt_answer_slots(receipts)
    grounder_config = config["target_grounder"]
    direct_applicability = (
        str(grounder_config["kind"])
        == "target_native_outcome_blind_neural_applicability"
    )
    calibrated, full_calibrated, calibration_head = candidate_calibration_predictions(
        receipts, seed=int(grounder_config["belief_seed"]),
    )
    rows = candidate_effect_rows(
        receipts, calibrated_predictions=calibrated,
    )
    full_rows = candidate_effect_rows(
        receipts, calibrated_predictions=full_calibrated,
    )
    objective_test_cost = float(
        config["source"].get(
            "target_objective_test_cost",
            controlled_config["domain"]["test_cost"],
        )
    )
    predicted_balances: dict[tuple[str, str], float] = {}
    if direct_applicability:
        support = source_test_feature_support(
            controlled_config, objective_test_cost=objective_test_cost,
        )
        cross_fitted = {}
        for receipt in receipts:
            sample_id = str(receipt["sample_id"])
            for candidate in receipt["candidates"]:
                grounding = candidate.get("outcome_blind_applicability")
                if not isinstance(grounding, Mapping):
                    raise ValueError(
                        "candidate is missing outcome-blind applicability receipt"
                    )
                information_score = float(grounding[
                    "expected_information_gain"
                ])
                answer_change = float(grounding[
                    "expected_answer_change_probability"
                ])
                balance = float(grounding["outcome_balance"])
                cross_fitted[(sample_id, str(candidate["candidate_id"]))] = (
                    information_score * support["maximum_information_gain"],
                    answer_change * support["maximum_confidence_gain"],
                    information_score * answer_change * balance,
                )
                predicted_balances[(sample_id, str(candidate["candidate_id"]))] = (
                    balance
                )
        full_grounder_payload: dict[str, Any] = {
            "kind": "target_native_outcome_blind_neural_applicability",
            "source_feature_support": support,
            "forbidden_inputs": [
                "gold_answer", "baseline.answer", "candidate.answer",
                "post_intervention_evidence",
            ],
        }
    else:
        cross_fitted = nested_cross_fitted_candidate_predictions(
            receipts,
            belief_seed=int(grounder_config["belief_seed"]),
            candidate_seed=int(grounder_config["candidate_seed"]),
            hidden_units=int(grounder_config["candidate_hidden_units"]),
            epochs=int(grounder_config.get("candidate_epochs", 1800)),
        )
        full_grounder = fit_candidate_effect_grounder(
            full_rows,
            seed=int(grounder_config["candidate_seed"]),
            hidden_units=int(grounder_config["candidate_hidden_units"]),
            epochs=int(grounder_config.get("candidate_epochs", 1800)),
        )
        full_grounder_payload = full_grounder.as_dict()
    base_source_models = build_source_value_models(
        controlled_config,
        seed=int(config["source"]["model_seed"]),
        objective_test_cost=objective_test_cost,
    )
    policy = config["policy"]
    traces: list[dict[str, Any]] = []
    row_index = {(row.sample_id, row.candidate_id): row for row in rows}
    candidate_indices: dict[str, dict[str, Mapping[str, Any]]] = {}
    grounded_by_sample: dict[str, list[GroundedCandidateIntervention]] = {}
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        before = calibrated[(sample_id, "BASE")]
        candidate_index = {
            str(candidate["candidate_id"]): candidate
            for candidate in receipt["candidates"]
        }
        candidate_indices[sample_id] = candidate_index
        grounded = []
        for candidate_id, candidate in candidate_index.items():
            information_gain, confidence_gain, quality_gain = cross_fitted[
                (sample_id, candidate_id)
            ]
            planner_score = float(candidate["planner_score"])
            positive_quality = max(0.0, quality_gain)
            if direct_applicability:
                symbolic_confidence_gain = confidence_gain
                symbolic_information_gain = information_gain
            else:
                symbolic_confidence_gain = max(
                    confidence_gain,
                    positive_quality * (1.0 - float(np.max(before))),
                )
                symbolic_information_gain = max(
                    information_gain,
                    positive_quality * normalized_entropy(before),
                )
            grounded.append(GroundedCandidateIntervention(
                candidate_id=candidate_id,
                planner_score=planner_score,
                predicted_information_gain=symbolic_information_gain,
                predicted_confidence_gain=symbolic_confidence_gain,
                predicted_answer_quality_gain=quality_gain,
                predicted_outcome_balance=predicted_balances.get(
                    (sample_id, candidate_id),
                    1.0 - 2.0 * abs(planner_score - 0.5),
                ),
            ))
        grounded_by_sample[sample_id] = grounded

    use_target_residual = bool(config["source"].get(
        "use_target_residual_value_ensemble", False,
    ))
    source_models_by_sample: dict[str, Mapping[str, Any]] = {}
    target_residual_state_counts: dict[str, int] = {}
    target_residual_support_ids: dict[str, list[str]] = {}
    if use_target_residual:
        residual_config = config["source"]["target_residual"]
        full_strength_states = int(residual_config["full_strength_states"])
        maximum_scale = float(residual_config["maximum_scale"])
        support_task_count = residual_config.get("support_task_count")
        support_seed = int(residual_config.get("support_seed", 0))
        for fold, receipt in enumerate(receipts):
            held_out = str(receipt["sample_id"])
            available_support = [
                str(row["sample_id"]) for row in receipts
                if str(row["sample_id"]) != held_out
            ]
            included_support = None
            if support_task_count is not None:
                count = int(support_task_count)
                if not 0 < count <= len(available_support):
                    raise ValueError("invalid target residual support_task_count")
                available_support.sort(key=lambda sample_id: stable_hash({
                    "support_seed": support_seed,
                    "held_out": held_out,
                    "candidate_support_id": sample_id,
                }))
                included_support = set(available_support[:count])
            fold_calibrated = calibration_predictions_excluding_sample(
                receipts,
                excluded_sample_id=held_out,
                seed=int(grounder_config["belief_seed"]) + fold,
            )
            target_examples = _target_residual_examples(
                receipts,
                calibrated=fold_calibrated,
                grounded_by_sample=grounded_by_sample,
                excluded_sample_id=held_out,
                included_sample_ids=included_support,
                test_cost=objective_test_cost,
            )
            state_count = len({row.state_id for row in target_examples})
            residual_scale = maximum_scale * min(
                1.0, state_count / max(1, full_strength_states),
            )
            source_models_by_sample[held_out] = (
                add_target_residual_to_source_models(
                    controlled_config,
                    base_source_models,
                    target_examples,
                    seed=int(config["source"]["model_seed"]) + 1000 + fold * 7,
                    residual_scale=residual_scale,
                )
            )
            target_residual_state_counts[held_out] = state_count
            target_residual_support_ids[held_out] = sorted(
                included_support or set(available_support)
            )

    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        gold_index = answer_slots.index(str(receipt["gold_answer"]))
        before = calibrated[(sample_id, "BASE")]
        candidate_index = candidate_indices[sample_id]
        grounded = grounded_by_sample[sample_id]
        source_models = source_models_by_sample.get(
            sample_id, base_source_models,
        )
        for condition in config["conditions"]:
            decision = choose_candidate_action(
                before,
                condition=condition,
                candidates=grounded,
                source_models=source_models,
                uncertainty_scale=float(policy["uncertainty_scale"]),
                decision_margin=float(policy["decision_margin"]),
                fallback_commit_threshold=float(policy[
                    "fallback_commit_threshold"
                ]),
                target_quality_threshold=float(policy["target_quality_threshold"]),
                information_gain_threshold=float(policy["information_gain_threshold"]),
            )
            selected = None
            after = before
            realized_quality = 0.0
            if decision.kind == "TEST":
                selected = candidate_index[str(decision.candidate_id)]
                after = calibrated[(sample_id, str(decision.candidate_id))]
                effect = row_index[(sample_id, str(decision.candidate_id))]
                realized_quality = effect.answer_quality_gain
                committed = int(np.argmax(after))
            else:
                committed = int(decision.answer_index)
            traces.append({
                "sample_id": sample_id,
                "family": str(receipt.get("family") or ""),
                "condition": condition,
                "baseline_answer": answer_slots[int(np.argmax(before))],
                "committed_answer": answer_slots[committed],
                "gold_answer": answer_slots[gold_index],
                "correct": committed == gold_index,
                "selected_candidate_id": (
                    str(selected["candidate_id"]) if selected is not None else None
                ),
                "selected_wrapper_tool": (
                    str(selected["wrapper_receipt"]["tool"])
                    if selected is not None else None
                ),
                "realized_answer_quality_gain": float(realized_quality),
                "decision": decision.__dict__,
                "grounded_candidates": [row.__dict__ for row in grounded],
            })

    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in config["conditions"]
    }
    conditions = {
        condition: _condition_summary(condition_rows)
        for condition, condition_rows in by_condition.items()
    }
    baseline_correct = []
    selector_correct = []
    planner_correct = []
    oracle_correct = []
    candidate_effect_ranges = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        gold = answer_slots.index(str(receipt["gold_answer"]))
        before = calibrated[(sample_id, "BASE")]
        candidates = list(receipt["candidates"])
        baseline_correct.append(int(np.argmax(before)) == gold)
        selected = max(
            candidates,
            key=lambda candidate: cross_fitted[(
                sample_id, str(candidate["candidate_id"])
            )][2],
        )
        planner = max(candidates, key=lambda candidate: float(candidate["planner_score"]))
        candidate_rows = [
            row_index[(sample_id, str(candidate["candidate_id"]))]
            for candidate in candidates
        ]
        oracle = max(candidate_rows, key=lambda row: row.answer_quality_gain)
        selector_correct.append(int(np.argmax(calibrated[
            (sample_id, str(selected["candidate_id"]))
        ])) == gold)
        planner_correct.append(int(np.argmax(calibrated[
            (sample_id, str(planner["candidate_id"]))
        ])) == gold)
        oracle_candidate = next(
            candidate for candidate in candidates
            if str(candidate["candidate_id"]) == oracle.candidate_id
        )
        oracle_correct.append(int(np.argmax(calibrated[
            (sample_id, str(oracle_candidate["candidate_id"]))
        ])) == gold)
        candidate_effect_ranges.append(
            max(row.answer_quality_gain for row in candidate_rows)
            - min(row.answer_quality_gain for row in candidate_rows)
        )

    signatures = {
        condition: {
            row["sample_id"]: (
                row["decision"]["kind"], row["selected_candidate_id"],
            )
            for row in condition_rows
        }
        for condition, condition_rows in by_condition.items()
    }
    authentic_contrast = sum(
        signatures["authentic_source_plus_target"][sample_id]
        != signatures["target_only"][sample_id]
        for sample_id in signatures["target_only"]
    )
    authentic_control_contrast = {
        control: sum(
            signatures["authentic_source_plus_target"][sample_id]
            != signatures[control][sample_id]
            for sample_id in signatures[control]
        )
        for control in (
            "shuffled_source_plus_target", "source_marginal_plus_target",
        )
    }
    baseline_accuracy = float(np.mean(baseline_correct))
    selector_accuracy = float(np.mean(selector_correct))
    planner_accuracy = float(np.mean(planner_correct))
    oracle_accuracy = float(np.mean(oracle_correct))
    minimum_contrasts = int(config["development_preflight"][
        "minimum_authentic_action_contrasts"
    ])
    requirements = config["development_preflight"]
    gates = {
        "all_receipts_complete": len(receipts) == len(config["splits"]["adaptation"]),
        "matched_candidate_effect_identifiable": int(sum(
            value > 1e-6 for value in candidate_effect_ranges
        )) >= int(config["development_preflight"][
            "minimum_samples_with_candidate_effect_variation"
        ]),
        "oracle_candidate_headroom_positive": (
            oracle_accuracy > baseline_accuracy
            if bool(requirements["require_oracle_candidate_headroom"])
            else True
        ),
        "cross_fitted_selector_positive_response": (
            selector_accuracy > baseline_accuracy
            if bool(requirements["require_cross_fitted_positive_response"])
            else True
        ),
        "cross_fitted_selector_not_below_planner": (
            selector_accuracy >= planner_accuracy
            if bool(requirements["require_cross_fitted_positive_response"])
            else True
        ),
        "authentic_source_action_contrast": authentic_contrast >= minimum_contrasts,
        "authentic_differs_from_each_source_control": (
            all(
            value >= minimum_contrasts
            for value in authentic_control_contrast.values()
            ) if bool(requirements["require_source_control_contrast"]) else True
        ),
        "authentic_source_improves_baseline_accuracy": (
            conditions["authentic_source_plus_target"]["accuracy"]
            > baseline_accuracy
            if bool(requirements["require_authentic_source_accuracy_superiority"])
            else True
        ),
        "authentic_source_improves_target_only_accuracy": (
            conditions["authentic_source_plus_target"]["accuracy"]
            > conditions["target_only"]["accuracy"]
            if bool(requirements["require_authentic_source_accuracy_superiority"])
            else True
        ),
        "authentic_source_improves_each_source_control_accuracy": (
            all(
                conditions["authentic_source_plus_target"]["accuracy"]
                > conditions[control]["accuracy"]
                for control in (
                    "shuffled_source_plus_target", "source_marginal_plus_target",
                )
            ) if bool(requirements["require_source_control_contrast"]) else True
        ),
    }
    artifact = {
        "schema_version": 1,
        "role": "TARGET_NATIVE_PARAMETERIZED_INTERVENTION_GROUNDER_ADAPTATION_ONLY",
        "answer_slots": list(answer_slots),
        "belief_calibration_head": calibration_head.as_dict(),
        "candidate_effect_grounder": full_grounder_payload,
        "training_sample_ids": sorted({row.sample_id for row in rows}),
        "source_config_sha256": stable_hash(controlled_config),
        "target_objective_test_cost": objective_test_cost,
        "source_value_model": {
            "kind": (
                "frozen_source_prior_plus_leave_one_task_out_target_residual"
                if use_target_residual else "frozen_source_prior"
            ),
            "target_residual_state_counts": target_residual_state_counts,
            "target_residual_support_ids": target_residual_support_ids,
        },
        "wrapper_contract": dict(config["wrapper"]),
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    report = {
        "schema_version": 1,
        "answer_slots": list(answer_slots),
        "status": (
            "ADAPTATION_PREFLIGHT_PASS" if all(gates.values())
            else "ADAPTATION_PREFLIGHT_FAIL"
        ),
        "claim_boundary": (
            "Adaptation-only parameterized intervention preflight; formal "
            "qualification and held-out outcomes remain unread."
        ),
        "baseline_accuracy": baseline_accuracy,
        "oracle_candidate_accuracy": oracle_accuracy,
        "cross_fitted_selector_accuracy": selector_accuracy,
        "selector_evaluation_kind": (
            "outcome_blind_neural_applicability"
            if direct_applicability else "nested_cross_fitted_effect_mlp"
        ),
        "planner_top_candidate_accuracy": planner_accuracy,
        "cross_fitted_mean_baseline_map_confidence": float(np.mean([
            np.max(calibrated[(str(receipt["sample_id"]), "BASE")])
            for receipt in receipts
        ])),
        "samples_with_candidate_effect_variation": int(sum(
            value > 1e-6 for value in candidate_effect_ranges
        )),
        "source_value_model_kind": (
            "frozen_source_prior_plus_leave_one_task_out_target_residual"
            if use_target_residual else "frozen_source_prior"
        ),
        "authentic_vs_target_action_contrast_samples": authentic_contrast,
        "authentic_vs_source_control_action_contrast_samples": (
            authentic_control_contrast
        ),
        "conditions_cross_fitted": conditions,
        "gates": gates,
        "policy_traces": traces,
    }
    report["report_sha256"] = stable_hash(report)
    return report, artifact


__all__ = [
    "candidate_effect_rows",
    "evaluate_candidate_adaptation",
    "leave_one_sample_candidate_predictions",
    "nested_cross_fitted_candidate_predictions",
]
