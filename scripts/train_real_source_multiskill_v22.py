#!/usr/bin/env python3
"""Train V22 effect-conditioned target grounding and selective utility heads."""

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
from motif_transfer.real_source_multiskill_causal_v22 import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    RECEIPT_BY_EFFECT,
    UTILITY_FEATURE_NAMES,
    action_causal_features,
    utility_features,
)
from motif_transfer.real_source_relation_causal_v20 import (  # noqa: E402
    linear_probability,
    linear_value,
)
from motif_transfer.relation_edge_value_v13 import fork_utility  # noqa: E402


L2_GRID = (1.0, 10.0, 100.0, 1000.0)
UTILITY_THRESHOLDS = (-0.10, -0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
EFFECT_MARGIN_THRESHOLDS = (0.0, 0.25, 0.50, 0.75)


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
    effect = str(fork["requested_source_effect"])
    required_property = str(fork["required_property"])
    source_features = action_causal_features(
        action=str(source["source_action"]),
        grounded_scores=_grounded_scores(base, True),
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
        requested_effect=effect,
        required_property=required_property,
    )
    fallback_features = action_causal_features(
        action=str(source["control_action"]),
        grounded_scores=_grounded_scores(base, False),
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
        requested_effect=effect,
        required_property=required_property,
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
    expected_receipt = RECEIPT_BY_EFFECT[effect]
    return {
        "fork_id": str(fork["fork_id"]),
        "role": str(fork["role"]),
        "task_id": str(fork["task_id"]),
        "task_family": str(fork["task_family"]),
        "requested_effect": effect,
        "required_property": required_property,
        "base_features": base,
        "source_action_features": source_features,
        "fallback_action_features": fallback_features,
        "source_effect_label": int(
            source["fork_target_effect_receipt"] == expected_receipt
        ),
        "fallback_effect_label": int(
            fallback["fork_target_effect_receipt"] == expected_receipt
        ),
        "source_success": source_success,
        "fallback_success": fallback_success,
        "success_delta": int(source_success) - int(fallback_success),
        "utility": float(utility),
    }


def _fit_logistic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x, y = [], []
    for row in rows:
        for prefix in ("source", "fallback"):
            features = row[f"{prefix}_action_features"]
            x.append([float(features[name]) for name in ACTION_FEATURE_NAMES])
            y.append(int(row[f"{prefix}_effect_label"]))
    array = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("V22 effect head requires positive and negative events")
    means = array.mean(axis=0)
    scales = array.std(axis=0)
    scales[scales < 1e-12] = 1.0
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=220812,
        solver="lbfgs",
    ).fit((array - means) / scales, labels)
    return {
        "schema_version": "target-typed-successor-logistic-head-v22",
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


def _with_probabilities(
    rows: Sequence[Mapping[str, Any]], head: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        row = dict(raw)
        source_p = linear_probability(head, row["source_action_features"])
        fallback_p = linear_probability(head, row["fallback_action_features"])
        row["source_effect_probability"] = source_p
        row["fallback_effect_probability"] = fallback_p
        row["utility_features"] = utility_features(
            base_features=row["base_features"],
            requested_effect=row["requested_effect"],
            required_property=row["required_property"],
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
        "schema_version": "target-incremental-utility-ridge-head-v22",
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "l2": float(l2),
        "training_rows": len(rows),
    }


def _sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    return float(sum(math.comb(total, i) for i in range(wins, total + 1)) / 2**total)


def _selection_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
    utility_threshold: float,
    effect_margin_threshold: float,
) -> dict[str, Any]:
    selected = [
        row for row, prediction in zip(rows, predictions)
        if prediction > utility_threshold and (
            float(row["source_effect_probability"])
            - float(row["fallback_effect_probability"])
        ) > effect_margin_threshold
    ]
    wins = sum(int(row["success_delta"]) > 0 for row in selected)
    losses = sum(int(row["success_delta"]) < 0 for row in selected)
    return {
        "utility_threshold": float(utility_threshold),
        "minimum_causal_effect_margin": float(effect_margin_threshold),
        "selected": len(selected),
        "success_wins": wins,
        "success_losses": losses,
        "success_delta": wins - losses,
        "one_sided_exact_sign_p": _sign_p(wins, losses),
        "selected_utility": float(sum(float(row["utility"]) for row in selected)),
        "selected_by_effect": dict(Counter(
            str(row["requested_effect"]) for row in selected
        )),
        "selected_by_property": dict(Counter(
            str(row["required_property"]) for row in selected
        )),
        "selected_fork_ids": sorted(str(row["fork_id"]) for row in selected),
    }


def _effect_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels, probabilities = [], []
    per_effect = {}
    for row in rows:
        for prefix in ("source", "fallback"):
            labels.append(int(row[f"{prefix}_effect_label"]))
            probabilities.append(float(row[f"{prefix}_effect_probability"]))
    labels_a = np.asarray(labels, dtype=np.int64)
    probs_a = np.asarray(probabilities, dtype=np.float64)
    predictions = probs_a >= 0.5
    recalls = [
        float(np.mean(predictions[labels_a == label] == label))
        for label in (0, 1) if np.any(labels_a == label)
    ]
    for effect in sorted({str(row["requested_effect"]) for row in rows}):
        subset = [row for row in rows if row["requested_effect"] == effect]
        per_effect[effect] = {
            "forks": len(subset),
            "source_receipt_recall": float(np.mean([
                float(row["source_effect_probability"]) >= 0.5
                if int(row["source_effect_label"]) else
                float(row["source_effect_probability"]) < 0.5
                for row in subset
            ])),
        }
    return {
        "action_rows": len(labels),
        "positive_rows": int(labels_a.sum()),
        "balanced_accuracy_at_0p5": float(np.mean(recalls)),
        "brier_score": float(np.mean((probs_a - labels_a) ** 2)),
        "per_effect": per_effect,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V22 candidate: {args.output}")
    report = _read(args.fork_report)
    report_hash = _validate_hash(report, "report_sha256")
    if report.get("status") != "MATCHED_TYPED_ADAPTATION_CALIBRATION_FORKS_COMPLETE":
        raise SystemExit("V22 fork report is incomplete")
    if not report.get("all_matched_invariants_passed"):
        raise SystemExit("V22 matched-fork invariants failed")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if report["plan"]["plan_sha256"] == "":
        raise SystemExit("V22 missing fork plan lineage")
    rows = [_fork_row(row, int(report["max_steps"])) for row in report["forks"]]
    adaptation = [row for row in rows if row["role"] == "causal_adaptation"]
    calibration = [row for row in rows if row["role"] == "causal_calibration"]
    effect_head = _fit_logistic(adaptation)
    adaptation = _with_probabilities(adaptation, effect_head)
    calibration = _with_probabilities(calibration, effect_head)
    effect_metrics = _effect_metrics(calibration)
    audit = []
    eligible = []
    for l2 in L2_GRID:
        utility_head = _fit_ridge(adaptation, l2)
        predictions = [
            linear_value(utility_head, row["utility_features"])
            for row in calibration
        ]
        mse = float(np.mean([
            (prediction - float(row["utility"])) ** 2
            for row, prediction in zip(calibration, predictions)
        ]))
        for threshold in UTILITY_THRESHOLDS:
            for margin in EFFECT_MARGIN_THRESHOLDS:
                metrics = _selection_metrics(
                    calibration, predictions, threshold, margin
                )
                gates = {
                    "minimum_selected": metrics["selected"] >= 8,
                    "minimum_wins": metrics["success_wins"] >= 2,
                    "success_delta_positive": metrics["success_delta"] > 0,
                    "selected_utility_positive": metrics["selected_utility"] > 0.0,
                    "both_effects_selected": set(metrics["selected_by_effect"]) >= {
                        "MUTATE", "RELATE"
                    },
                }
                cell = {
                    "l2": l2,
                    "calibration_mse": mse,
                    "metrics": metrics,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
                audit.append(cell)
                if cell["passed"]:
                    eligible.append(cell)
    causal_gates = {
        "balanced_accuracy_at_least_0p80": (
            effect_metrics["balanced_accuracy_at_0p5"] >= 0.80
        ),
        "brier_at_most_0p20": effect_metrics["brier_score"] <= 0.20,
        "both_effects_have_calibration_support": set(effect_metrics["per_effect"]) >= {
            "MUTATE", "RELATE"
        },
        "per_effect_source_recall_at_least_0p75": all(
            row["source_receipt_recall"] >= 0.75
            for row in effect_metrics["per_effect"].values()
        ),
    }
    if not all(causal_gates.values()) or not eligible:
        selected = None
        passed = False
        utility_head = _fit_ridge(adaptation, 100.0)
    else:
        selected = min(eligible, key=lambda cell: (
            -int(cell["metrics"]["success_delta"]),
            int(cell["metrics"]["success_losses"]),
            -int(cell["metrics"]["success_wins"]),
            -int(cell["metrics"]["selected"]),
            float(cell["calibration_mse"]),
            float(cell["l2"]),
        ))
        utility_head = _fit_ridge(adaptation, float(selected["l2"]))
        passed = True
    body = {
        "schema_version": "real-source-multiskill-candidate-v22",
        "status": (
            "PROSPECTIVE_REQUALIFICATION_AUTHORIZED"
            if passed else "V22_CAUSAL_OR_UTILITY_GATE_FAILED_STOP"
        ),
        "claim_boundary": (
            "SOURCE_TYPED_BIND_TO_MUTATE_AND_BIND_TO_RELATE_STRUCTURE; "
            "TARGET_NATIVE_EFFECT_CONDITIONED_NEURAL_SUCCESSOR_AND_UTILITY_"
            "HEADS; ADAPTATION_CALIBRATION_ONLY; PROSPECTIVE_"
            "REQUALIFICATION_AND_FUTURE_FINAL_SPLITS_UNREAD"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "file_sha256": _sha256(args.manifest),
            "manifest_sha256": manifest_hash,
        },
        "fork_report": {
            "path": str(args.fork_report.resolve()),
            "file_sha256": _sha256(args.fork_report),
            "report_sha256": report_hash,
        },
        "parent_candidate": manifest["parent_candidate"],
        "source_summary": manifest["source_summary"],
        "transfer_scope": {
            "allowed_source_effects": manifest["allowed_source_effects"],
            "active_required_properties": manifest["active_required_properties"],
        },
        "target_typed_successor_head": effect_head,
        "target_typed_successor_calibration_metrics": effect_metrics,
        "target_incremental_utility_head": utility_head,
        "selective_risk_calibration": {
            "authority": "DISJOINT_V22_CAUSAL_CALIBRATION_ONLY",
            "selected_l2": None if selected is None else float(selected["l2"]),
            "admission_threshold": (
                1e9 if selected is None else
                float(selected["metrics"]["utility_threshold"])
            ),
            "minimum_causal_effect_margin": (
                1e9 if selected is None else
                float(selected["metrics"]["minimum_causal_effect_margin"])
            ),
            "selected_calibration_metrics": (
                None if selected is None else selected["metrics"]
            ),
            "audit": audit,
        },
        "causal_gates": causal_gates,
        "selective_utility_cell_passed": selected is not None,
        "prospective_requalification_authorized": passed,
        "future_development_authorized": False,
        "confirmation_authorized": False,
        "prospective_requalification_read_or_run": False,
        "future_development_confirmation_read_or_run": False,
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
        "effect_metrics": effect_metrics,
        "causal_gates": causal_gates,
        "selected_cell": selected,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
