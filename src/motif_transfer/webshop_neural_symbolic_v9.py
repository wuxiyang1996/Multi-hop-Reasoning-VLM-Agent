"""Target-native neural grounding for state-dependent TEST/COMMIT transfer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .active_video_transfer import build_source_value_models
from .controlled_exploration_transfer import FEATURE_NAMES, ValueEnsemble


TARGET_FEATURE_NAMES = (
    "is_click",
    "is_scroll_or_history",
    "is_commit",
    "is_constraint",
    "is_goal_constraint",
    "is_paired_constraint",
    "is_navigation",
    "is_noop",
    "goal_overlap",
    "visible_goal_constraint_satisfied",
    "visible_goal_constraint_unsatisfied",
    "prior_action_had_no_effect",
    "step_fraction",
)
OUTCOME_NAMES = ("state_changed", "terminated", "reward", "prerequisite_progress")
SOURCE_CONDITIONS = (
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
)


def _tokens(text: str) -> set[str]:
    stop = {"and", "for", "the", "with", "than", "under", "looking", "want", "buy"}
    return {
        token for token in re.findall(r"[a-z0-9.]+", text.lower())
        if len(token) > 1 and token not in stop
    }


def visible_goal_constraint_status(axtree: str, goal: str) -> tuple[bool, bool]:
    """Ground visible radio constraints against the target goal only."""

    goal_tokens = _tokens(goal)
    satisfied = False
    unsatisfied = False
    pattern = re.compile(
        r"radio\s+'([^']+)'[^\n]*checked\s*=\s*['\"]?(true|false|1|0)",
        re.IGNORECASE,
    )
    for label, raw_checked in pattern.findall(axtree):
        if not (_tokens(label) & goal_tokens):
            continue
        checked = raw_checked.lower() in {"true", "1"}
        satisfied = satisfied or checked
        unsatisfied = unsatisfied or not checked
    return satisfied, unsatisfied


def target_features(
    semantics: Mapping[str, Any],
    *,
    visible_satisfied: bool,
    visible_unsatisfied: bool,
    prior_no_effect: bool,
    step_index: int,
    maximum_steps: int,
) -> tuple[float, ...]:
    verb = str(semantics.get("verb") or "")
    row = (
        float(verb == "click"),
        float(verb in {"scroll", "go_back", "go_forward"}),
        float(bool(semantics.get("is_commit"))),
        float(bool(semantics.get("is_constraint"))),
        float(bool(semantics.get("is_goal_constraint"))),
        float(semantics.get("paired_constraint_bid") is not None),
        float(bool(semantics.get("is_navigation"))),
        float(bool(semantics.get("is_noop"))),
        float(semantics.get("goal_overlap") or 0.0),
        float(visible_satisfied),
        float(visible_unsatisfied),
        float(prior_no_effect),
        step_index / max(1, maximum_steps - 1),
    )
    if len(row) != len(TARGET_FEATURE_NAMES):
        raise AssertionError("WebShop V9 target feature contract drift")
    return row


@dataclass(frozen=True)
class OutcomeRow:
    features: tuple[float, ...]
    outcomes: tuple[float, ...]


@dataclass(frozen=True)
class TargetOutcomeMLP:
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(TARGET_FEATURE_NAMES):
            raise ValueError("WebShop V9 target feature shape mismatch")
        normalized = (matrix - np.asarray(self.input_mean)) / np.asarray(self.input_scale)
        hidden = np.tanh(
            normalized @ np.asarray(self.hidden_weights) + np.asarray(self.hidden_bias)
        )
        logits = hidden @ np.asarray(self.output_weights) + np.asarray(self.output_bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "target_native_webshop_outcome_mlp",
            "feature_names": list(TARGET_FEATURE_NAMES),
            "outcome_names": list(OUTCOME_NAMES),
            "input_mean": list(self.input_mean),
            "input_scale": list(self.input_scale),
            "hidden_weights": [list(row) for row in self.hidden_weights],
            "hidden_bias": list(self.hidden_bias),
            "output_weights": [list(row) for row in self.output_weights],
            "output_bias": list(self.output_bias),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetOutcomeMLP":
        if tuple(payload["feature_names"]) != TARGET_FEATURE_NAMES:
            raise ValueError("WebShop V9 grounder feature contract mismatch")
        if tuple(payload["outcome_names"]) != OUTCOME_NAMES:
            raise ValueError("WebShop V9 grounder outcome contract mismatch")
        return cls(
            tuple(map(float, payload["input_mean"])),
            tuple(map(float, payload["input_scale"])),
            tuple(tuple(map(float, row)) for row in payload["hidden_weights"]),
            tuple(map(float, payload["hidden_bias"])),
            tuple(tuple(map(float, row)) for row in payload["output_weights"]),
            tuple(map(float, payload["output_bias"])),
        )


def fit_target_outcome_mlp(
    rows: Sequence[OutcomeRow],
    *,
    seed: int,
    hidden_units: int = 12,
    epochs: int = 2200,
    learning_rate: float = 0.02,
    l2: float = 0.01,
) -> TargetOutcomeMLP:
    if not rows:
        raise ValueError("target outcome rows cannot be empty")
    matrix = np.asarray([row.features for row in rows], dtype=np.float64)
    labels = np.asarray([row.outcomes for row in rows], dtype=np.float64)
    if matrix.shape[1] != len(TARGET_FEATURE_NAMES):
        raise ValueError("target outcome feature width mismatch")
    if labels.shape != (len(rows), len(OUTCOME_NAMES)):
        raise ValueError("target outcome label width mismatch")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    inputs = (matrix - mean) / scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, math.sqrt(2 / inputs.shape[1]), (inputs.shape[1], hidden_units))
    b1 = np.zeros(hidden_units)
    w2 = rng.normal(0.0, math.sqrt(2 / hidden_units), (hidden_units, len(OUTCOME_NAMES)))
    b2 = np.zeros(len(OUTCOME_NAMES))
    parameters = [w1, b1, w2, b2]
    first = [np.zeros_like(value) for value in parameters]
    second = [np.zeros_like(value) for value in parameters]
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        hidden = np.tanh(inputs @ parameters[0] + parameters[1])
        logits = hidden @ parameters[2] + parameters[3]
        predictions = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        logit_gradient = (predictions - labels) / len(labels)
        hidden_gradient = (logit_gradient @ parameters[2].T) * (1.0 - hidden**2)
        gradients = [
            inputs.T @ hidden_gradient + l2 * parameters[0],
            np.sum(hidden_gradient, axis=0),
            hidden.T @ logit_gradient + l2 * parameters[2],
            np.sum(logit_gradient, axis=0),
        ]
        for index, gradient in enumerate(gradients):
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
            second[index] = beta2 * second[index] + (1.0 - beta2) * gradient**2
            parameters[index] -= learning_rate * (
                first[index] / (1.0 - beta1**epoch)
            ) / (np.sqrt(second[index] / (1.0 - beta2**epoch)) + 1e-8)
    return TargetOutcomeMLP(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(tuple(map(float, row)) for row in parameters[0]),
        tuple(map(float, parameters[1])),
        tuple(tuple(map(float, row)) for row in parameters[2]),
        tuple(map(float, parameters[3])),
    )


@dataclass(frozen=True)
class TransferDecision:
    selected_index: int
    abstract_kind: str
    source_abstained: bool
    source_test_value: float | None
    source_commit_value: float | None
    reason: str


def _source_features(
    *,
    predicted_test_progress: float,
    predicted_commit_reward: float,
    predicted_test_change: float,
    visible_satisfied: bool,
    visible_unsatisfied: bool,
    remaining_fraction: float,
    repeated_test: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # Official WebShop reward can be high for a purchase that violates one
    # visible constraint.  It is an outcome estimate, not readiness belief.
    # A checked=false goal constraint is direct target-native evidence that the
    # state is not ready to commit.
    all_visible_constraints_satisfied = visible_satisfied and not visible_unsatisfied
    confidence = float(np.clip(
        1.0 if all_visible_constraints_satisfied else 0.0 if visible_unsatisfied
        else predicted_commit_reward,
        0.0,
        1.0,
    ))
    gain = float(np.clip(predicted_test_progress * (1.0 - confidence), 0.0, 1.0))
    test = (
        1.0,
        gain,
        gain,
        1.0 - 2.0 * abs(float(predicted_test_change) - 0.5),
        confidence,
        1.0 - confidence,
        remaining_fraction,
        0.0,
        float(repeated_test),
    )
    commit = (
        0.0, 0.0, 0.0, 0.0, confidence, 1.0 - confidence,
        remaining_fraction, confidence, 0.0,
    )
    if len(test) != len(FEATURE_NAMES):
        raise AssertionError("WebShop V9 source feature contract drift")
    return test, commit


def choose_transfer_action(
    *,
    condition: str,
    predictions: np.ndarray,
    semantics: Sequence[Mapping[str, Any]],
    source_models: Mapping[str, ValueEnsemble],
    visible_satisfied: bool,
    visible_unsatisfied: bool,
    prior_no_effect: bool,
    remaining_fraction: float,
    previous_action: str | None,
    candidates: Sequence[str],
    uncertainty_scale: float,
    decision_margin: float,
) -> TransferDecision:
    if len(candidates) != len(semantics) or predictions.shape != (
        len(candidates), len(OUTCOME_NAMES)
    ):
        raise ValueError("candidate prediction alignment mismatch")
    if condition == "target_only":
        return TransferDecision(0, "TARGET", True, None, None, "target_rank_zero")

    commit_indices = [index for index, row in enumerate(semantics) if row["is_commit"]]
    test_indices = [
        index for index, row in enumerate(semantics)
        if not row["is_commit"] and not row["is_noop"]
    ]
    if not commit_indices or not test_indices:
        return TransferDecision(0, "TARGET", True, None, None, "missing_test_or_commit")
    changed, terminated, reward, progress = range(len(OUTCOME_NAMES))
    best_commit = max(
        commit_indices,
        key=lambda index: (predictions[index, reward] - predictions[index, terminated], -index),
    )
    best_test = max(
        test_indices,
        key=lambda index: (
            predictions[index, progress],
            predictions[index, changed] - predictions[index, terminated],
            -index,
        ),
    )
    if condition == "target_native_myopic":
        selected = max(
            range(len(candidates)),
            key=lambda index: (predictions[index, reward], -predictions[index, terminated], -index),
        )
        return TransferDecision(
            selected,
            "COMMIT" if semantics[selected]["is_commit"] else "TEST",
            False,
            None,
            None,
            "maximum_predicted_immediate_reward",
        )
    if condition not in SOURCE_CONDITIONS:
        raise ValueError(f"unknown WebShop V9 condition: {condition}")

    # Applicability is grounded in observed target state.  Before all visible
    # constraints are satisfied, transfer may act only after a real no-effect transition and
    # only when the neural grounder identifies a progress-making TEST.
    all_visible_constraints_satisfied = visible_satisfied and not visible_unsatisfied
    if not all_visible_constraints_satisfied and (
        not prior_no_effect
        or predictions[best_test, progress] <= 0.5
        or predictions[best_test, changed] <= 0.5
    ):
        return TransferDecision(
            0, "TARGET", True, None, None, "no_grounded_progress_test",
        )
    model = source_models[condition]
    test_features, commit_features = _source_features(
        predicted_test_progress=float(predictions[best_test, progress]),
        predicted_commit_reward=float(predictions[best_commit, reward]),
        predicted_test_change=float(predictions[best_test, changed]),
        visible_satisfied=visible_satisfied,
        visible_unsatisfied=visible_unsatisfied,
        remaining_fraction=remaining_fraction,
        repeated_test=candidates[best_test] == previous_action,
    )
    means, deviations = model.predict((test_features, commit_features))
    gap = float(means[0] - means[1])
    uncertainty = uncertainty_scale * math.sqrt(float(deviations[0] ** 2 + deviations[1] ** 2))
    if gap - uncertainty > decision_margin:
        return TransferDecision(
            best_test, "TEST", False, float(means[0]), float(means[1]), "source_prefers_test",
        )
    if -gap - uncertainty > decision_margin:
        return TransferDecision(
            best_commit, "COMMIT", False, float(means[0]), float(means[1]), "source_prefers_commit",
        )
    return TransferDecision(
        0, "TARGET", True, float(means[0]), float(means[1]), "source_uncertain",
    )


__all__ = [
    "OUTCOME_NAMES",
    "SOURCE_CONDITIONS",
    "TARGET_FEATURE_NAMES",
    "OutcomeRow",
    "TargetOutcomeMLP",
    "TransferDecision",
    "build_source_value_models",
    "choose_transfer_action",
    "fit_target_outcome_mlp",
    "target_features",
    "visible_goal_constraint_status",
]
