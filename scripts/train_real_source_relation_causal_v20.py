#!/usr/bin/env python3
"""Train and calibrate V20 target-native causal relation grounding and utility."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.neurosymbolic_transfer_contract import (  # noqa: E402
    CAUSAL_EFFECT_SCORE_SEMANTICS,
)
from motif_transfer.real_source_relation_causal_v20 import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    UTILITY_FEATURE_NAMES,
    action_causal_features,
    linear_probability,
    linear_value,
    utility_features,
)
from motif_transfer.relation_edge_value_v13 import fork_utility  # noqa: E402


UTILITY_L2_GRID = (0.1, 1.0, 10.0, 100.0)
CONFORMAL_ALPHA = 0.1


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def _grounded_scores(features: Mapping[str, float], source: bool) -> dict[str, float]:
    if source:
        return {
            "policy": float(features["source_policy"]),
            "completion": float(features["source_completion"]),
            "binding": float(features["source_binding"]),
            "applicability": float(features["source_applicability"]),
        }
    return {
        "policy": float(features["fallback_policy"]),
        "completion": float(features["source_completion"] - features["completion_margin"]),
        "binding": float(features["source_binding"] - features["binding_margin"]),
        "applicability": float(
            features["source_applicability"] - features["applicability_margin"]
        ),
    }


def _fork_row(fork: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    source = fork["branches"]["SOURCE_EDGE"]
    fallback = fork["branches"]["TARGET_ABSTAIN"]
    base = dict(fork["features"])
    ledger = source["fork_ledger_before"]
    history = tuple(map(str, fork["prefix_actions"]))
    step = int(fork["fork_step"])
    source_action = str(source["source_action"])
    fallback_action = str(source["control_action"])
    source_action_features = action_causal_features(
        action=source_action,
        grounded_scores=_grounded_scores(base, True),
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
    )
    fallback_action_features = action_causal_features(
        action=fallback_action,
        grounded_scores=_grounded_scores(base, False),
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
    )
    source_success = bool(source["official_success"])
    fallback_success = bool(fallback["official_success"])
    utility = fork_utility(
        source_success=source_success,
        control_success=fallback_success,
        source_steps=int(source["steps"]),
        control_steps=int(fallback["steps"]),
        source_completed_fraction=float(source["completed_fraction"]),
        control_completed_fraction=float(fallback["completed_fraction"]),
        max_steps=max_steps,
    )
    goal_spec = ledger["goal_spec"]
    return {
        "fork_id": str(fork["fork_id"]),
        "role": str(fork["role"]),
        "task_id": str(fork["task_id"]),
        "task_family": str(fork["task_family"]),
        "target_receptacle_type": str(goal_spec["target_receptacle_type"]),
        "base_features": base,
        "source_action_features": source_action_features,
        "fallback_action_features": fallback_action_features,
        "source_effect_label": int(
            source["fork_target_effect_receipt"] == "RELATE_SLOT_CLOSED"
        ),
        "fallback_effect_label": int(
            fallback["fork_target_effect_receipt"] == "RELATE_SLOT_CLOSED"
        ),
        "source_success": source_success,
        "fallback_success": fallback_success,
        "success_delta": int(source_success) - int(fallback_success),
        "source_steps": int(source["steps"]),
        "fallback_steps": int(fallback["steps"]),
        "utility": float(utility),
    }


def _fit_logistic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x = []
    y = []
    for row in rows:
        for prefix in ("source", "fallback"):
            features = row[f"{prefix}_action_features"]
            x.append([float(features[name]) for name in ACTION_FEATURE_NAMES])
            y.append(int(row[f"{prefix}_effect_label"]))
    array = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("V20 causal effect training needs both event labels")
    means = array.mean(axis=0)
    scales = array.std(axis=0)
    scales[scales < 1e-12] = 1.0
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=200812,
        solver="lbfgs",
    ).fit((array - means) / scales, labels)
    return {
        "schema_version": "target-causal-relation-logistic-head-v20",
        "feature_names": list(ACTION_FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(model.intercept_[0]),
        "weights": model.coef_[0].tolist(),
        "training_action_rows": len(labels),
        "training_positive_rows": int(np.sum(labels)),
        "optimizer_iterations": int(model.n_iter_[0]),
        "optimizer_converged": bool(model.n_iter_[0] < model.max_iter),
    }


def _with_effect_probabilities(
    rows: Sequence[Mapping[str, Any]], effect_head: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        row = dict(raw)
        source_p = linear_probability(effect_head, row["source_action_features"])
        fallback_p = linear_probability(effect_head, row["fallback_action_features"])
        row["source_effect_probability"] = source_p
        row["fallback_effect_probability"] = fallback_p
        row["utility_features"] = utility_features(
            base_features=row["base_features"],
            source_effect_probability=source_p,
            fallback_effect_probability=fallback_p,
            source_action_features=row["source_action_features"],
            fallback_action_features=row["fallback_action_features"],
        )
        output.append(row)
    return output


def _fit_ridge(rows: Sequence[Mapping[str, Any]], l2: float) -> dict[str, Any]:
    x = np.asarray([
        [float(row["utility_features"][name]) for name in UTILITY_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    y = np.asarray([float(row["utility"]) for row in rows], dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(rows)), (x - means) / scales])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "schema_version": "target-incremental-utility-ridge-head-v20",
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "l2": float(l2),
        "training_rows": len(rows),
    }


def _select_l2(rows: Sequence[Mapping[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    groups = sorted({str(row["target_receptacle_type"]) for row in rows})
    results = []
    for l2 in UTILITY_L2_GRID:
        squared = []
        group_rmse = {}
        for group in groups:
            training = [row for row in rows if row["target_receptacle_type"] != group]
            testing = [row for row in rows if row["target_receptacle_type"] == group]
            if not training or not testing:
                continue
            model = _fit_ridge(training, l2)
            errors = [
                linear_value(model, row["utility_features"]) - float(row["utility"])
                for row in testing
            ]
            squared.extend(error * error for error in errors)
            group_rmse[group] = float(np.sqrt(np.mean(np.square(errors))))
        results.append({
            "l2": l2,
            "group_count": len(group_rmse),
            "rmse": float(np.sqrt(np.mean(squared))),
            "maximum_group_rmse": max(group_rmse.values()),
            "group_rmse": group_rmse,
        })
    selected = min(results, key=lambda row: (
        row["maximum_group_rmse"], row["rmse"], -row["l2"]
    ))
    return float(selected["l2"]), results


def _effect_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = []
    probabilities = []
    source_labels = []
    source_predictions = []
    for row in rows:
        for prefix in ("source", "fallback"):
            labels.append(int(row[f"{prefix}_effect_label"]))
            probabilities.append(float(row[f"{prefix}_effect_probability"]))
        source_labels.append(int(row["source_effect_label"]))
        source_predictions.append(float(row["source_effect_probability"]))
    labels_a = np.asarray(labels)
    probs_a = np.asarray(probabilities)
    predictions = probs_a >= 0.5
    recalls = []
    for label in (0, 1):
        mask = labels_a == label
        recalls.append(float(np.mean(predictions[mask] == label)))
    return {
        "action_rows": len(labels),
        "positive_rows": int(np.sum(labels_a)),
        "brier_score": float(np.mean((probs_a - labels_a) ** 2)),
        "balanced_accuracy_at_0p5": float(np.mean(recalls)),
        "source_event_recall_at_0p5": float(np.mean([
            prediction >= 0.5 if label else prediction < 0.5
            for label, prediction in zip(source_labels, source_predictions)
        ])),
        "probability_range": [float(np.min(probs_a)), float(np.max(probs_a))],
    }


def _conformal_quantile(errors: Sequence[float], alpha: float) -> float:
    if not errors:
        raise ValueError("conformal calibration requires at least one error")
    values = np.sort(np.asarray(errors, dtype=np.float64))
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - alpha)))
    return float(values[rank - 1])


def _partition_calibration(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Outcome-blind split for utility conformal calibration vs qualification."""
    ordered = sorted(
        rows,
        key=lambda row: stable_hash({
            "fork_id": str(row["fork_id"]),
            "partition": "V20_CONFORMAL_VS_QUALIFICATION",
        }),
    )
    return ordered[::2], ordered[1::2]


def _selection(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any], q: float) -> dict[str, Any]:
    evaluated = []
    for raw in rows:
        prediction = linear_value(model, raw["utility_features"])
        lower = prediction - q
        evaluated.append(dict(raw) | {
            "predicted_utility": prediction,
            "conformal_lower_bound": lower,
            "admitted": lower > 0.0,
        })
    selected = [row for row in evaluated if row["admitted"]]
    return {
        "rows": evaluated,
        "selected": len(selected),
        "selected_success_wins": sum(row["success_delta"] > 0 for row in selected),
        "selected_success_losses": sum(row["success_delta"] < 0 for row in selected),
        "selected_success_delta": sum(int(row["success_delta"]) for row in selected),
        "selected_utility": float(sum(float(row["utility"]) for row in selected)),
        "selected_positive_utility": sum(row["utility"] > 1e-12 for row in selected),
        "selected_negative_utility": sum(row["utility"] < -1e-12 for row in selected),
        "selected_task_ids": sorted(str(row["task_id"]) for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-report", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 candidate: {args.output}")
    report = _read(args.fork_report)
    report_hash = _validate_hash(report, "report_sha256")
    if report.get("status") != (
        "MATCHED_CAUSAL_ADAPTATION_CALIBRATION_FORKS_COMPLETE"
    ) or not report.get("all_matched_invariants_passed"):
        raise SystemExit("V20 causal forks are incomplete")
    source = _read(args.source_summary)
    if source.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise SystemExit("V20 source typed gate did not pass")
    expected_ir = report["plan"]
    plan = _read(Path(str(expected_ir["path"])))
    _validate_hash(plan, "plan_sha256")
    parent = _read(Path(str(plan["parent_candidate"]["path"])))
    _validate_hash(parent, "candidate_sha256")
    if source["effect_ir"]["ir_sha256"] != parent["slot_source_ir"]["parent_ir_sha256"]:
        raise SystemExit("V20 source summary and executable target IR differ")
    max_steps = int(report["max_steps"])
    rows = [_fork_row(row, max_steps) for row in report["forks"]]
    adaptation = [row for row in rows if row["role"] == "causal_adaptation"]
    calibration = [row for row in rows if row["role"] == "causal_calibration"]
    effect_head = _fit_logistic(adaptation)
    adaptation = _with_effect_probabilities(adaptation, effect_head)
    calibration = _with_effect_probabilities(calibration, effect_head)
    conformal_calibration, target_qualification = _partition_calibration(
        calibration
    )
    selected_l2, l2_audit = _select_l2(adaptation)
    utility_head = _fit_ridge(adaptation, selected_l2)
    calibration_predictions = [
        linear_value(utility_head, row["utility_features"])
        for row in conformal_calibration
    ]
    q = _conformal_quantile([
        prediction - float(row["utility"])
        for prediction, row in zip(
            calibration_predictions, conformal_calibration
        )
    ], CONFORMAL_ALPHA)
    effect_metrics = _effect_metrics(target_qualification)
    selection = _selection(target_qualification, utility_head, q)
    role_counts = Counter(row["role"] for row in rows)
    utility_signs = Counter(
        "positive" if row["utility"] > 1e-12 else
        "negative" if row["utility"] < -1e-12 else "neutral"
        for row in rows
    )
    gates = {
        "minimum_adaptation_forks": len(adaptation) >= 24,
        "minimum_calibration_forks": len(calibration) >= 12,
        "minimum_six_conformal_calibration_forks": (
            len(conformal_calibration) >= 6
        ),
        "minimum_six_target_qualification_forks": (
            len(target_qualification) >= 6
        ),
        "both_causal_event_labels_in_adaptation": (
            len({
                row[key]
                for row in adaptation
                for key in ("source_effect_label", "fallback_effect_label")
            }) == 2
        ),
        "effect_balanced_accuracy_at_least_0p80": (
            effect_metrics["balanced_accuracy_at_0p5"] >= 0.8
        ),
        "effect_brier_at_most_0p20": effect_metrics["brier_score"] <= 0.2,
        "source_event_recall_at_least_0p90": (
            effect_metrics["source_event_recall_at_0p5"] >= 0.9
        ),
        "minimum_four_calibration_admissions": selection["selected"] >= 4,
        "calibration_selected_success_delta_positive": (
            selection["selected_success_delta"] > 0
        ),
        "zero_calibration_selected_success_losses": (
            selection["selected_success_losses"] == 0
        ),
        "calibration_selected_utility_positive": selection["selected_utility"] > 0,
        "source_edge_replication_passed": (
            source["edge_replication_gate"]["status"]
            == "EDGE_REPLICATION_GATE_PASSED"
        ),
        "source_effect_value_passed": (
            source["effect_value_gate"]["status"] == "EFFECT_VALUE_GATE_PASSED"
        ),
        "all_matched_fork_invariants": bool(report["all_matched_invariants_passed"]),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "real-source-relation-causal-candidate-v20",
        "status": (
            "TARGET_CAUSAL_AND_UTILITY_GATE_PASSED"
            if passed else "TARGET_CAUSAL_OR_UTILITY_GATE_FAILED_STOP"
        ),
        "claim_boundary": (
            "REAL_SOURCE_TYPED_BIND_TO_RELATE_GRAPH; TARGET_NATIVE_MATCHED_"
            "COUNTERFACTUAL_CAUSAL_EFFECT_AND_INCREMENTAL_UTILITY; TRAIN_AND_"
            "CALIBRATION_ONLY; DEVELOPMENT_CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "fork_report": {
            "path": str(args.fork_report.resolve()),
            "file_sha256": _sha256(args.fork_report),
            "report_sha256": report_hash,
        },
        "source_summary": {
            "path": str(args.source_summary.resolve()),
            "file_sha256": _sha256(args.source_summary),
            "ir_sha256": source["effect_ir"]["ir_sha256"],
            "matched_source_forks": int(source["matched_forks"]),
            "supporting_source_tasks": source["edge_replication_gate"][
                "supporting_source_tasks"
            ],
            "supporting_simulator_families": source["edge_replication_gate"][
                "supporting_simulator_families"
            ],
        },
        "parent_candidate": plan["parent_candidate"],
        "role_counts": dict(role_counts),
        "utility_sign_counts": dict(utility_signs),
        "target_causal_effect_head": effect_head,
        "score_contract": {
            "score_semantics": CAUSAL_EFFECT_SCORE_SEMANTICS,
            "predicted_successor_event": "RELATE_SLOT_CLOSED",
            "causal_successor_effect_certified": passed,
            "counterfactual_action_supervision": True,
            "probability_estimation": (
                "TARGET_ADAPTATION_LOGISTIC_WITH_DISJOINT_TARGET_QUALIFICATION"
            ),
            "entity_conditioned_action_binding": True,
            "successor_event_prediction": True,
            "reward_success_completion_fields_consumed_by_effect_head": False,
        },
        "target_incremental_utility_head": utility_head,
        "utility_model_selection": {
            "authority": "ADAPTATION_LEAVE_ONE_TARGET_RECEPTACLE_OUT_ONLY",
            "l2_grid": list(UTILITY_L2_GRID),
            "selected_l2": selected_l2,
            "audit": l2_audit,
        },
        "conformal": {
            "alpha": CONFORMAL_ALPHA,
            "overprediction_error_quantile": q,
            "calibration_rows": len(conformal_calibration),
            "admission": "PREDICTED_UTILITY_MINUS_QUANTILE_STRICTLY_POSITIVE",
        },
        "calibration_partition": {
            "authority": (
                "OUTCOME_BLIND_STABLE_HASH_OF_FROZEN_FORK_ID_BEFORE_"
                "TRAINER_READS_ANY_OUTCOME"
            ),
            "full_role_rows": len(calibration),
            "conformal_rows": len(conformal_calibration),
            "qualification_rows": len(target_qualification),
            "conformal_fork_ids": sorted(
                str(row["fork_id"]) for row in conformal_calibration
            ),
            "qualification_fork_ids": sorted(
                str(row["fork_id"]) for row in target_qualification
            ),
            "disjoint": not (
                {str(row["fork_id"]) for row in conformal_calibration}
                & {str(row["fork_id"]) for row in target_qualification}
            ),
        },
        "target_calibration": {
            "authority": "DISJOINT_TARGET_QUALIFICATION_ONLY",
            "effect": effect_metrics,
            "utility_selection": {
                key: value for key, value in selection.items() if key != "rows"
            },
        },
        "support_contract": {
            "status": (
                "TYPED_SOURCE_TARGET_CAUSAL_SUPPORT_PASSED"
                if passed else "TYPED_SOURCE_TARGET_CAUSAL_SUPPORT_FAILED"
            ),
            "source_conformal_reused": False,
            "typed_event_semantics_shared_and_target_calibrated": passed,
            "source_matched_forks": int(source["matched_forks"]),
            "target_adaptation_forks": len(adaptation),
            "target_calibration_forks": len(calibration),
            "target_conformal_calibration_forks": len(
                conformal_calibration
            ),
            "target_qualification_forks": len(target_qualification),
        },
        "gates": gates,
        "development_authorized": passed,
        "confirmation_authorized": False,
        "development_or_confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    candidate = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": candidate["status"],
        "candidate_sha256": candidate["candidate_sha256"],
        "role_counts": dict(role_counts),
        "utility_sign_counts": dict(utility_signs),
        "effect_metrics": effect_metrics,
        "calibration_selection": {
            key: value for key, value in selection.items() if key != "rows"
        },
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
