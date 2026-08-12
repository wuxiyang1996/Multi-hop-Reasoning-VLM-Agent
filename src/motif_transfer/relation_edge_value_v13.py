"""Target-native value features for matched relation-edge forks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "step_fraction",
    "remaining_budget_fraction",
    "after_v12_step_threshold",
    "remaining_slots_fraction",
    "completed_slots_fraction",
    "target_policy_ratio",
    "source_policy",
    "fallback_policy",
    "policy_margin",
    "source_completion",
    "completion_margin",
    "source_binding",
    "binding_margin",
    "source_applicability",
    "applicability_margin",
    "realization_score",
    "native_action_count_fraction",
    "fallback_position",
    "fallback_bind",
    "fallback_mutate",
    "fallback_relate",
    "fallback_other",
)
RIDGE_L2 = 1.0
ADMISSION_THRESHOLD = 0.0
EFFICIENCY_WEIGHT = 0.1
PROGRESS_WEIGHT = 0.01


def extract_relation_edge_features(
    *,
    decision: Mapping[str, Any],
    grounded: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[str, Any],
    step: int,
    max_steps: int,
    native_action_count: int,
) -> dict[str, float]:
    """Extract outcome-blind features available before the fork action."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    source_action = str(decision["action"])
    fallback_action = str(decision["fallback_action"])
    if source_action not in grounded or fallback_action not in grounded:
        raise ValueError("source and fallback actions must be grounded")
    source = grounded[source_action]
    fallback = grounded[fallback_action]
    required = max(int(ledger.get("required_count", 0)), 1)
    remaining = int(decision["slot_state"]["remaining_slots"])
    completed = int(decision["slot_state"]["completed_count"])
    fallback_effect = str(decision.get("fallback_effect", "OTHER")).upper()
    known_effects = {"POSITION", "BIND", "MUTATE", "RELATE"}
    values = {
        "step_fraction": float(step) / max_steps,
        "remaining_budget_fraction": float(max_steps - step) / max_steps,
        "after_v12_step_threshold": float(step >= 9),
        "remaining_slots_fraction": float(remaining) / required,
        "completed_slots_fraction": float(completed) / required,
        "target_policy_ratio": float(decision["target_policy_ratio"]),
        "source_policy": float(source["policy"]),
        "fallback_policy": float(fallback["policy"]),
        "policy_margin": float(source["policy"]) - float(fallback["policy"]),
        "source_completion": float(source["completion"]),
        "completion_margin": (
            float(source["completion"]) - float(fallback["completion"])
        ),
        "source_binding": float(source["binding"]),
        "binding_margin": (
            float(source["binding"]) - float(fallback["binding"])
        ),
        "source_applicability": float(source["applicability"]),
        "applicability_margin": (
            float(source["applicability"])
            - float(fallback["applicability"])
        ),
        "realization_score": float(decision["best_realization_score"]),
        "native_action_count_fraction": min(
            float(native_action_count) / 50.0, 1.0
        ),
        "fallback_position": float(fallback_effect == "POSITION"),
        "fallback_bind": float(fallback_effect == "BIND"),
        "fallback_mutate": float(fallback_effect == "MUTATE"),
        "fallback_relate": float(fallback_effect == "RELATE"),
        "fallback_other": float(fallback_effect not in known_effects),
    }
    if tuple(values) != FEATURE_NAMES:
        raise RuntimeError("feature order drifted from frozen V13 schema")
    return values


def fork_utility(
    *,
    source_success: bool,
    control_success: bool,
    source_steps: int,
    control_steps: int,
    source_completed_fraction: float,
    control_completed_fraction: float,
    max_steps: int,
) -> float:
    """Success-primary utility with bounded efficiency/progress tie breaks."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return (
        int(source_success)
        - int(control_success)
        + EFFICIENCY_WEIGHT * (control_steps - source_steps) / max_steps
        + PROGRESS_WEIGHT
        * (source_completed_fraction - control_completed_fraction)
    )


def fit_ridge_value_head(
    rows: Sequence[Mapping[str, Any]],
    *,
    l2: float = RIDGE_L2,
) -> dict[str, Any]:
    """Fit a deterministic standardized linear value head."""
    if not rows:
        raise ValueError("ridge value head requires training rows")
    if l2 <= 0.0:
        raise ValueError("ridge penalty must be positive")
    x = np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES]
         for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row["utility"]) for row in rows], dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (x - means) / scales
    design = np.column_stack([np.ones(len(rows)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    return {
        "schema_version": "relation-edge-linear-value-head-v13",
        "feature_names": list(FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "l2": float(l2),
        "admission_threshold": ADMISSION_THRESHOLD,
        "training_rows": len(rows),
    }


def predict_relation_edge_value(
    model: Mapping[str, Any], features: Mapping[str, float]
) -> float:
    """Predict incremental fork value from a frozen linear head."""
    if tuple(model["feature_names"]) != FEATURE_NAMES:
        raise ValueError("value-head feature schema mismatch")
    vector = np.asarray(
        [float(features[name]) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if not (len(vector) == len(means) == len(scales) == len(weights)):
        raise ValueError("value-head dimensions do not match")
    return float(
        float(model["intercept"])
        + ((vector - means) / scales) @ weights
    )


__all__ = [
    "ADMISSION_THRESHOLD",
    "EFFICIENCY_WEIGHT",
    "FEATURE_NAMES",
    "PROGRESS_WEIGHT",
    "RIDGE_L2",
    "extract_relation_edge_features",
    "fit_ridge_value_head",
    "fork_utility",
    "predict_relation_edge_value",
]
