#!/usr/bin/env python3
"""Train a small OOF neural risk model from all opened matched relation forks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_source_relation_causal_v20 import (  # noqa: E402
    UTILITY_FEATURE_NAMES,
)
from motif_transfer.real_source_relation_neural_risk_v24 import (  # noqa: E402
    neural_value,
)
from train_real_source_relation_causal_v20 import (  # noqa: E402
    _fork_row,
    _with_effect_probabilities,
)


HIDDEN_GRID = ((4,), (8,), (16,))
ALPHA = 10.0
ADMISSION_FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.30)
FOLD_COUNT = 5
COHORTS = ("v20_adaptation_calibration", "v21_requalification", "v23_development")


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


def _fold(task_id: str) -> int:
    return int(hashlib.sha256(
        f"v24:{task_id}".encode("utf-8")
    ).hexdigest(), 16) % FOLD_COUNT


def _model(hidden: tuple[int, ...]) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="lbfgs",
        alpha=ALPHA,
        max_iter=5000,
        random_state=240812,
    )


def _fit(
    x: np.ndarray, y: np.ndarray, hidden: tuple[int, ...]
) -> tuple[StandardScaler, MLPRegressor]:
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = _model(hidden).fit(scaler.transform(x), y)
    return scaler, model


def _serialize(
    scaler: StandardScaler, model: MLPRegressor
) -> dict[str, Any]:
    if len(model.coefs_) != 2 or len(model.intercepts_) != 2:
        raise ValueError("V24 serialization expects one hidden layer")
    return {
        "schema_version": "target-neural-utility-mlp-v24",
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "input_weights": model.coefs_[0].tolist(),
        "hidden_bias": model.intercepts_[0].tolist(),
        "output_weights": model.coefs_[1][:, 0].tolist(),
        "output_bias": float(model.intercepts_[1][0]),
        "activation": "relu",
        "hidden_layer_sizes": list(model.hidden_layer_sizes),
        "alpha": float(model.alpha),
        "solver": str(model.solver),
        "training_iterations": int(model.n_iter_),
        "training_loss": float(model.loss_),
    }


def _sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    return float(sum(math.comb(total, i) for i in range(wins, total + 1)) / 2**total)


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, fraction: float
) -> dict[str, Any]:
    selected_count = max(1, round(len(rows) * fraction))
    order = np.argsort(-predictions, kind="stable")
    selected_indices = list(map(int, order[:selected_count]))
    selected = [rows[index] for index in selected_indices]
    wins = sum(int(row["success_delta"]) > 0 for row in selected)
    losses = sum(int(row["success_delta"]) < 0 for row in selected)
    by_cohort = {}
    for cohort in COHORTS:
        subset = [row for row in selected if row["cohort"] == cohort]
        cohort_wins = sum(int(row["success_delta"]) > 0 for row in subset)
        cohort_losses = sum(int(row["success_delta"]) < 0 for row in subset)
        by_cohort[cohort] = {
            "selected": len(subset),
            "success_wins": cohort_wins,
            "success_losses": cohort_losses,
            "success_delta": cohort_wins - cohort_losses,
        }
    cutoff = float(predictions[order[selected_count - 1]])
    next_value = (
        float(predictions[order[selected_count]])
        if selected_count < len(rows) else cutoff - 1.0
    )
    return {
        "admission_fraction": float(fraction),
        "selected": selected_count,
        "success_wins": wins,
        "success_losses": losses,
        "success_delta": wins - losses,
        "one_sided_exact_sign_p": _sign_p(wins, losses),
        "selected_incremental_utility": float(sum(
            float(row["utility"]) for row in selected
        )),
        "selected_by_cohort": by_cohort,
        "cohorts_with_positive_delta": sum(
            cell["success_delta"] > 0 for cell in by_cohort.values()
        ),
        "cohorts_with_negative_delta": sum(
            cell["success_delta"] < 0 for cell in by_cohort.values()
        ),
        "oof_cutoff_midpoint": (cutoff + next_value) / 2.0,
        "selected_fork_ids": sorted(str(row["fork_id"]) for row in selected),
    }


def _threshold(predictions: np.ndarray, fraction: float) -> float:
    selected = max(1, round(len(predictions) * fraction))
    ordered = np.sort(predictions)[::-1]
    if selected == len(ordered):
        return float(ordered[-1] - 1.0)
    return float((ordered[selected - 1] + ordered[selected]) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v20-report", type=Path, required=True)
    parser.add_argument("--v21-report", type=Path, required=True)
    parser.add_argument("--v23-report", type=Path, required=True)
    parser.add_argument("--v21-candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V24 candidate: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    inherited = _read(args.v21_candidate)
    inherited_hash = _validate_hash(inherited, "candidate_sha256")
    inputs = []
    rows = []
    for cohort, path, expected in (
        (COHORTS[0], args.v20_report,
         "MATCHED_CAUSAL_ADAPTATION_CALIBRATION_FORKS_COMPLETE"),
        (COHORTS[1], args.v21_report, "UTILITY_REQUALIFICATION_FAILED_STOP"),
        (COHORTS[2], args.v23_report, "V23_DEVELOPMENT_TRANSFER_GATE_FAILED_STOP"),
    ):
        report = _read(path)
        report_hash = _validate_hash(report, "report_sha256")
        if report.get("status") != expected:
            raise SystemExit(f"V24 unexpected input status for {cohort}")
        if any(not all(row["invariants"].values()) for row in report["forks"]):
            raise SystemExit(f"V24 input invariant failed for {cohort}")
        cohort_rows = []
        for raw in report["forks"]:
            fork = dict(raw)
            fork.setdefault("features", fork["branches"]["SOURCE_EDGE"]["features"])
            cohort_rows.append(_fork_row(fork, 60))
        cohort_rows = _with_effect_probabilities(
            cohort_rows, inherited["target_causal_effect_head"]
        )
        for row in cohort_rows:
            row["cohort"] = cohort
        rows.extend(cohort_rows)
        inputs.append({
            "cohort": cohort,
            "path": str(path.resolve()),
            "file_sha256": _sha256(path),
            "report_sha256": report_hash,
            "fork_count": len(cohort_rows),
        })
    if len(rows) != 271 or len({str(row["task_id"]) for row in rows}) != len(rows):
        raise SystemExit("V24 expected 271 task-disjoint opened forks")
    x = np.asarray([
        [float(row["utility_features"][name]) for name in UTILITY_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    y = np.asarray([float(row["utility"]) for row in rows], dtype=np.float64)
    folds = np.asarray([_fold(str(row["task_id"])) for row in rows], dtype=np.int64)
    audit = []
    eligible = []
    for hidden in HIDDEN_GRID:
        predictions = np.zeros(len(rows), dtype=np.float64)
        for fold in range(FOLD_COUNT):
            training = folds != fold
            testing = ~training
            scaler, model = _fit(x[training], y[training], hidden)
            predictions[testing] = model.predict(scaler.transform(x[testing]))
        mse = float(np.mean((predictions - y) ** 2))
        for fraction in ADMISSION_FRACTIONS:
            metrics = _metrics(rows, predictions, fraction)
            gates = {
                "minimum_oof_selected": metrics["selected"] >= 24,
                "minimum_oof_success_wins": metrics["success_wins"] >= 5,
                "zero_oof_success_losses": metrics["success_losses"] == 0,
                "oof_sign_test_at_most_0p05": (
                    metrics["one_sided_exact_sign_p"] <= 0.05
                ),
                "oof_incremental_utility_positive": (
                    metrics["selected_incremental_utility"] > 0.0
                ),
                "at_least_two_positive_cohorts": (
                    metrics["cohorts_with_positive_delta"] >= 2
                ),
                "no_negative_cohort": metrics["cohorts_with_negative_delta"] == 0,
            }
            cell = {
                "hidden_layer_sizes": list(hidden),
                "alpha": ALPHA,
                "oof_utility_mse": mse,
                "metrics": metrics,
                "gates": gates,
                "passed": all(gates.values()),
            }
            audit.append(cell)
            if cell["passed"]:
                eligible.append(cell)
    if not eligible:
        raise SystemExit("V24 has no OOF neural risk candidate; fail closed")
    selected = min(eligible, key=lambda cell: (
        int(cell["metrics"]["success_losses"]),
        -int(cell["metrics"]["success_wins"]),
        float(cell["metrics"]["admission_fraction"]),
        sum(map(int, cell["hidden_layer_sizes"])),
        float(cell["oof_utility_mse"]),
    ))
    hidden = tuple(map(int, selected["hidden_layer_sizes"]))
    scaler, model = _fit(x, y, hidden)
    serialized = _serialize(scaler, model)
    serialized["training_rows"] = len(rows)
    full_predictions = model.predict(scaler.transform(x))
    threshold = _threshold(
        full_predictions, float(selected["metrics"]["admission_fraction"])
    )
    checks = np.asarray([
        neural_value(serialized, row["utility_features"]) for row in rows
    ])
    if not np.allclose(checks, full_predictions, atol=1e-10, rtol=1e-10):
        raise RuntimeError("V24 serialized neural model changed predictions")
    folds_count = Counter(map(int, folds.tolist()))
    body = {
        "schema_version": "real-source-relation-neural-risk-candidate-v24",
        "status": "V24_SEALED_CONFIRMATION_AUTHORIZED",
        "claim_boundary": (
            "POST_V23_MODEL_DEVELOPMENT_FROM_ALL_OPENED_MATCHED_FORKS; "
            "FIVE_FOLD_TASK_HASH OOF_SMALL_MLP; SOURCE_TYPED_BIND_TO_RELATE_"
            "STRUCTURE_PLUS_TARGET_NATIVE_NEURAL_SUCCESSOR_AND_RISK_"
            "GROUNDING; V20_SEALED_CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "file_sha256": _sha256(args.manifest),
            "manifest_sha256": manifest_hash,
        },
        "opened_matched_fork_reports": inputs,
        "inherited_v21_candidate": {
            "path": str(args.v21_candidate.resolve()),
            "file_sha256": _sha256(args.v21_candidate),
            "candidate_sha256": inherited_hash,
        },
        "parent_candidate": inherited["parent_candidate"],
        "source_summary": inherited["source_summary"],
        "target_causal_effect_head": inherited["target_causal_effect_head"],
        "target_causal_effect_metrics": inherited["target_causal_effect_metrics"],
        "target_incremental_utility_head": inherited[
            "target_incremental_utility_head"
        ],
        "selective_risk_calibration": inherited["selective_risk_calibration"],
        "target_neural_utility_mlp": serialized,
        "neural_risk_calibration": {
            "authority": "FIVE_FOLD_TASK_HASH_OOF_ALL_OPENED_TARGET_FORKS",
            "target": "PAIRED_INCREMENTAL_UTILITY",
            "fold_function": "sha256(v24:task_id) modulo 5",
            "fold_counts": dict(folds_count),
            "hidden_grid": [list(value) for value in HIDDEN_GRID],
            "fixed_alpha": ALPHA,
            "admission_fraction_grid": list(ADMISSION_FRACTIONS),
            "selected_hidden_layer_sizes": list(hidden),
            "selected_admission_fraction": float(
                selected["metrics"]["admission_fraction"]
            ),
            "admission_threshold": threshold,
            "minimum_causal_effect_margin": 0.0,
            "selected_oof_metrics": selected["metrics"],
            "selection_order": (
                "ZERO_LOSSES_THEN_MAX_WINS_THEN_MIN_ADMISSION_FRACTION_"
                "THEN_MIN_HIDDEN_UNITS_THEN_MIN_OOF_MSE"
            ),
            "audit": audit,
        },
        "score_contract": {
            "source_structure": "BIND--CARRIER_BOUND-->RELATE",
            "target_native_causal_successor_grounding": True,
            "target_native_neural_risk_grounding": True,
            "source_actions_coordinates_or_oracle_consumed": False,
            "target_outcome_fields_consumed_at_live_inference": False,
        },
        "confirmation_gates": {
            "minimum_opportunities": 12,
            "minimum_primary_admissions": 5,
            "minimum_success_wins": 5,
            "one_sided_exact_sign_alpha": 0.05,
            "success_delta_strictly_positive": True,
            "selected_incremental_utility_strictly_positive": True,
            "source_event_recall_at_least": 0.90,
            "loss_count_strictly_less_than_always_source": True,
            "net_delta_strictly_greater_than_lexical": True,
            "net_delta_strictly_greater_than_late_step": True,
            "net_delta_strictly_greater_than_v20_selective": True,
            "all_exact_state_fork_invariants": True,
        },
        "oof_development_gate_passed": True,
        "confirmation_authorized": True,
        "sealed_confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
        "sklearn_version": sklearn.__version__,
    }
    candidate = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": candidate["candidate_sha256"],
        "status": candidate["status"],
        "training_rows": len(rows),
        "selected_hidden_layer_sizes": list(hidden),
        "selected_admission_fraction": selected["metrics"]["admission_fraction"],
        "admission_threshold": threshold,
        "selected_oof_metrics": selected["metrics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
