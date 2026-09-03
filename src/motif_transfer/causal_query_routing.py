"""Intervention-grounded source values for causal-query representation routing.

The source task is a controlled causal game with two matched interventions:
consume an explicit learned relation edge, or recompute the relation from a
predicted trajectory.  Explicit edges are usually valuable for factual and
predictive queries, while an intervention can shift their support and make a
fresh trajectory-derived relation safer.  The model learns this state-action
value relation from source-only matched values.

Target domains provide their own neural representations for the two actions.
Only anonymous causal-query state features cross the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np


ACTIONS = ("USE_EXPLICIT_RELATION", "DERIVE_FROM_TRAJECTORY")
FEATURE_NAMES = (
    "is_explicit_relation_action",
    "intervention_active",
    "future_query",
    "explicit_relation_reliability",
    "trajectory_reliability",
    "predicted_intervention_shift",
    "remaining_compute_fraction",
    "explicit_x_intervention",
    "selected_representation_reliability",
)


def _seed(payload: object) -> int:
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


@dataclass(frozen=True)
class CausalQueryState:
    state_id: str
    intervention_active: bool
    future_query: bool
    explicit_relation_reliability: float
    trajectory_reliability: float
    predicted_intervention_shift: float
    remaining_compute_fraction: float = 1.0

    def validate(self) -> None:
        if not self.state_id:
            raise ValueError("causal query state_id cannot be empty")
        for name in (
            "explicit_relation_reliability",
            "trajectory_reliability",
            "predicted_intervention_shift",
            "remaining_compute_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RoutingValueRow:
    state_id: str
    action: str
    features: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class RoutingValueModel:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("causal routing feature shape mismatch")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
        return design @ np.asarray(self.coefficients, dtype=np.float64)


def action_features(state: CausalQueryState, action: str) -> tuple[float, ...]:
    state.validate()
    if action not in ACTIONS:
        raise ValueError(f"unsupported causal routing action: {action}")
    explicit = float(action == "USE_EXPLICIT_RELATION")
    selected_reliability = (
        state.explicit_relation_reliability
        if explicit
        else state.trajectory_reliability
    )
    features = (
        explicit,
        float(state.intervention_active),
        float(state.future_query),
        float(state.explicit_relation_reliability),
        float(state.trajectory_reliability),
        float(state.predicted_intervention_shift),
        float(state.remaining_compute_fraction),
        explicit * float(state.intervention_active),
        float(selected_reliability),
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("causal routing feature contract drift")
    return features


def exact_source_values(state: CausalQueryState) -> dict[str, float]:
    """Return matched source action values under the controlled causal game."""

    state.validate()
    intervention = float(state.intervention_active)
    future = float(state.future_query)
    shift = state.predicted_intervention_shift
    explicit = (
        state.explicit_relation_reliability
        + 0.04 * future
        - intervention * shift
    )
    trajectory = (
        state.trajectory_reliability
        - 0.025
        + intervention * (0.08 + 0.15 * shift)
    )
    return {
        "USE_EXPLICIT_RELATION": float(np.clip(explicit, 0.0, 1.0)),
        "DERIVE_FROM_TRAJECTORY": float(np.clip(trajectory, 0.0, 1.0)),
    }


def generate_source_rows(
    *,
    seed: int,
    state_count: int,
    namespace: str,
) -> tuple[RoutingValueRow, ...]:
    if state_count <= 0:
        raise ValueError("source state_count must be positive")
    rng = np.random.default_rng(_seed((seed, namespace)))
    rows = []
    for index in range(state_count):
        state = CausalQueryState(
            state_id=f"{namespace}:{index}",
            intervention_active=bool(rng.integers(0, 2)),
            future_query=bool(rng.integers(0, 2)),
            explicit_relation_reliability=float(rng.uniform(0.72, 0.95)),
            trajectory_reliability=float(rng.uniform(0.56, 0.80)),
            predicted_intervention_shift=float(rng.uniform(0.38, 0.72)),
            remaining_compute_fraction=float(rng.uniform(0.35, 1.0)),
        )
        values = exact_source_values(state)
        for action in ACTIONS:
            rows.append(RoutingValueRow(
                state.state_id,
                action,
                action_features(state, action),
                values[action],
            ))
    return tuple(rows)


def shuffled_value_rows(
    rows: Sequence[RoutingValueRow], *, seed: int,
) -> tuple[RoutingValueRow, ...]:
    grouped: dict[str, list[RoutingValueRow]] = {}
    for row in rows:
        grouped.setdefault(row.state_id, []).append(row)
    output = []
    for state_id, group in sorted(grouped.items()):
        if {row.action for row in group} != set(ACTIONS):
            raise ValueError("each source state must contain both matched actions")
        rng = np.random.default_rng(_seed((seed, state_id, "shuffle")))
        values = np.asarray([row.value for row in group], dtype=np.float64)
        values = values[rng.permutation(len(values))]
        output.extend(
            RoutingValueRow(row.state_id, row.action, row.features, float(value))
            for row, value in zip(group, values)
        )
    return tuple(output)


def marginal_value_rows(
    rows: Sequence[RoutingValueRow],
) -> tuple[RoutingValueRow, ...]:
    means = {
        action: float(np.mean([row.value for row in rows if row.action == action]))
        for action in ACTIONS
    }
    return tuple(
        RoutingValueRow(row.state_id, row.action, row.features, means[row.action])
        for row in rows
    )


def fit_routing_model(
    rows: Sequence[RoutingValueRow], *, ridge_alpha: float,
) -> RoutingValueModel:
    if not rows or ridge_alpha < 0:
        raise ValueError("routing fit requires rows and nonnegative ridge_alpha")
    matrix = np.asarray([row.features for row in rows], dtype=np.float64)
    labels = np.asarray([row.value for row in rows], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
    penalty = np.eye(design.shape[1]) * float(ridge_alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ labels,
    )
    return RoutingValueModel(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(map(float, coefficients)),
    )


def build_source_models(config: Mapping[str, object]) -> dict[str, RoutingValueModel]:
    rows = generate_source_rows(
        seed=int(config["seed"]),
        state_count=int(config["train_states"]),
        namespace="source-train",
    )
    controls = {
        "authentic_source_router": rows,
        "shuffled_source_router": shuffled_value_rows(
            rows, seed=int(config["control_seed"]),
        ),
        "source_marginal_router": marginal_value_rows(rows),
    }
    return {
        name: fit_routing_model(values, ridge_alpha=float(config["ridge_alpha"]))
        for name, values in controls.items()
    }


def select_action(state: CausalQueryState, model: RoutingValueModel) -> str:
    values = model.predict([action_features(state, action) for action in ACTIONS])
    return ACTIONS[int(np.argmax(values))]


def source_gate_report(config: Mapping[str, object]) -> dict[str, object]:
    models = build_source_models(config)
    rows = generate_source_rows(
        seed=int(config["heldout_seed"]),
        state_count=int(config["heldout_states"]),
        namespace="source-heldout",
    )
    grouped: dict[str, list[RoutingValueRow]] = {}
    for row in rows:
        grouped.setdefault(row.state_id, []).append(row)
    correct = {name: 0 for name in models}
    action_counts = {
        name: {action: 0 for action in ACTIONS} for name in models
    }
    for group in grouped.values():
        true_action = max(group, key=lambda row: row.value).action
        features = {row.action: row.features for row in group}
        for name, model in models.items():
            values = model.predict([features[action] for action in ACTIONS])
            selected = ACTIONS[int(np.argmax(values))]
            correct[name] += int(selected == true_action)
            action_counts[name][selected] += 1
    count = len(grouped)
    accuracy = {name: value / count for name, value in correct.items()}
    minimum_accuracy = float(config["minimum_authentic_accuracy"])
    minimum_margin = float(config["minimum_control_margin"])
    passed = (
        accuracy["authentic_source_router"] >= minimum_accuracy
        and all(
            accuracy["authentic_source_router"] - accuracy[name] >= minimum_margin
            for name in ("shuffled_source_router", "source_marginal_router")
        )
        and all(action_counts["authentic_source_router"][action] > 0 for action in ACTIONS)
    )
    return {
        "status": "SOURCE_CAUSAL_ROUTING_GATE_PASSED" if passed else "SOURCE_CAUSAL_ROUTING_GATE_FAILED",
        "heldout_states": count,
        "selection_accuracy": accuracy,
        "selected_action_counts": action_counts,
        "minimum_authentic_accuracy": minimum_accuracy,
        "minimum_control_margin": minimum_margin,
    }


__all__ = [
    "ACTIONS",
    "CausalQueryState",
    "FEATURE_NAMES",
    "RoutingValueModel",
    "action_features",
    "build_source_models",
    "exact_source_values",
    "generate_source_rows",
    "select_action",
    "source_gate_report",
]
