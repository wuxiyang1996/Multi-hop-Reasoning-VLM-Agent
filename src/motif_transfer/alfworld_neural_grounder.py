from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .controlled_exploration_transfer import RidgeValueModel, ValueEnsemble


ACTION_VERBS = (
    "go", "open", "close", "examine", "look", "inventory", "take", "put",
    "move", "use", "clean", "heat", "cool", "slice", "toggle", "help",
)
TEST_VERBS = frozenset({"go", "open", "examine", "look", "inventory"})
EXCLUDED_VERBS = frozenset({"help"})
TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.lower()))


def _verb(action: str) -> str:
    tokens = _tokens(action)
    return tokens[0] if tokens else ""


def action_role(action: str) -> str:
    verb = _verb(action)
    if verb in EXCLUDED_VERBS:
        return "EXCLUDE"
    return "TEST" if verb in TEST_VERBS else "COMMIT"


def _hash_bin(value: str, bins: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % bins


def grounder_features(
    *,
    goal: str,
    observation: str,
    action: str,
    step: int,
    action_history: Sequence[str],
    feature_bins: int,
) -> np.ndarray:
    if feature_bins < 16:
        raise ValueError("feature_bins must be at least 16")
    goal_tokens = set(_tokens(goal))
    observation_tokens = set(_tokens(observation))
    action_tokens = _tokens(action)
    action_set = set(action_tokens)
    verb = _verb(action)
    one_hot = [float(verb == candidate) for candidate in ACTION_VERBS]
    lexical = np.zeros(feature_bins, dtype=np.float64)
    for token in action_tokens:
        lexical[_hash_bin(f"action:{token}", feature_bins)] += 1.0
        if token in goal_tokens:
            lexical[_hash_bin(f"goal-action:{token}", feature_bins)] += 1.0
        if token in observation_tokens:
            lexical[_hash_bin(f"obs-action:{token}", feature_bins)] += 1.0
    norm = float(np.linalg.norm(lexical))
    if norm > 0:
        lexical /= norm
    content = action_set - {verb, "to", "from", "in", "on", "with", "the"}
    repeats = sum(1 for previous in action_history if previous == action)
    previous_verb = _verb(action_history[-1]) if action_history else ""
    scalars = [
        len(action_set & goal_tokens) / max(1, len(action_set)),
        len(action_set & observation_tokens) / max(1, len(action_set)),
        float(bool(content & goal_tokens)),
        float(bool(content & observation_tokens)),
        min(int(step), 100) / 100.0,
        min(repeats, 5) / 5.0,
        float(previous_verb == verb and bool(previous_verb)),
        float(action in action_history[-3:]),
        float(action_role(action) == "TEST"),
        float(action_role(action) == "COMMIT"),
    ]
    return np.asarray([*one_hot, *scalars, *lexical.tolist()], dtype=np.float64)


def mlp_score(features: np.ndarray, artifact: Mapping[str, Any]) -> float:
    value = np.asarray(features, dtype=np.float64)
    for layer_index, layer in enumerate(artifact["layers"]):
        weights = np.asarray(layer["weights"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        value = value @ weights + bias
        if layer_index < len(artifact["layers"]) - 1:
            activation = str(artifact["hidden_activation"])
            if activation == "tanh":
                value = np.tanh(value)
            elif activation == "relu":
                value = np.maximum(value, 0.0)
            else:
                raise ValueError(f"unsupported hidden activation: {activation}")
    scalar = float(np.ravel(value)[0])
    if scalar >= 0:
        return 1.0 / (1.0 + math.exp(-scalar))
    exponential = math.exp(scalar)
    return exponential / (1.0 + exponential)


def score_native_actions(
    *,
    goal: str,
    observation: str,
    native_actions: Sequence[str],
    step: int,
    action_history: Sequence[str],
    artifact: Mapping[str, Any],
) -> dict[str, float]:
    feature_bins = int(artifact["feature_bins"])
    return {
        str(action): mlp_score(
            grounder_features(
                goal=goal,
                observation=observation,
                action=str(action),
                step=step,
                action_history=action_history,
                feature_bins=feature_bins,
            ),
            artifact,
        )
        for action in native_actions
        if action_role(str(action)) != "EXCLUDE"
    }


def target_symbolic_features(
    *,
    actions: Sequence[str],
    scores: Mapping[str, float],
    step: int,
    max_steps: int,
    action_history: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    candidates = [action for action in actions if action in scores]
    if not candidates:
        raise ValueError("no grounded candidate actions")
    raw = np.asarray([max(float(scores[action]), 1e-8) for action in candidates])
    probabilities = raw / np.sum(raw)
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    entropy_scale = max(math.log(len(probabilities)), 1e-8)
    normalized_entropy = entropy / entropy_scale if len(probabilities) > 1 else 0.0
    map_confidence = float(np.max(probabilities))
    result = {}
    for action, probability in zip(candidates, probabilities):
        role = action_role(action)
        repeat_fraction = min(action_history.count(action), max_steps) / max_steps
        if role == "TEST":
            information_gain = normalized_entropy * float(probability)
            confidence_gain = float(probability) * (1.0 - map_confidence)
            balance = 1.0 - 2.0 * abs(float(probability) - 0.5)
            candidate_probability = 0.0
            is_test = 1.0
        else:
            information_gain = 0.0
            confidence_gain = 0.0
            balance = 0.0
            candidate_probability = float(probability)
            is_test = 0.0
        result[action] = (
            is_test,
            information_gain,
            confidence_gain,
            balance,
            map_confidence,
            normalized_entropy,
            max(0, max_steps - step) / max_steps,
            candidate_probability,
            repeat_fraction,
        )
    return result


def deserialize_value_ensemble(payload: Mapping[str, Any]) -> ValueEnsemble:
    return ValueEnsemble(tuple(
        RidgeValueModel(
            feature_mean=tuple(map(float, row["feature_mean"])),
            feature_scale=tuple(map(float, row["feature_scale"])),
            coefficients=tuple(map(float, row["coefficients"])),
        )
        for row in payload["models"]
    ))


def choose_grounded_action(
    *,
    actions: Sequence[str],
    grounder_scores: Mapping[str, float],
    symbolic_features: Mapping[str, tuple[float, ...]],
    source_model: ValueEnsemble | None,
    uncertainty_scale: float,
    decision_margin: float,
) -> dict[str, Any]:
    candidates = [action for action in actions if action in grounder_scores]
    if not candidates:
        raise ValueError("no candidates have neural-grounder scores")
    fallback = max(candidates, key=lambda action: (grounder_scores[action], action))
    if source_model is None:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "source_admitted": False,
            "changed_action": False,
            "changed_role": False,
            "diagnostic": "TARGET_ONLY",
        }
    tests = [action for action in candidates if action_role(action) == "TEST"]
    commits = [action for action in candidates if action_role(action) == "COMMIT"]
    if not tests or not commits:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "source_admitted": False,
            "changed_action": False,
            "changed_role": False,
            "diagnostic": "ROLE_COMPARISON_UNAVAILABLE",
        }
    feature_matrix = [symbolic_features[action] for action in candidates]
    means, deviations = source_model.predict(feature_matrix)
    indices = {action: index for index, action in enumerate(candidates)}
    best_test = max(tests, key=lambda action: means[indices[action]])
    best_commit = max(commits, key=lambda action: means[indices[action]])
    test_index = indices[best_test]
    commit_index = indices[best_commit]
    gap = float(means[test_index] - means[commit_index])
    uncertainty = float(uncertainty_scale) * math.sqrt(
        float(deviations[test_index] ** 2 + deviations[commit_index] ** 2)
    )
    if gap - uncertainty > decision_margin:
        selected = best_test
        diagnostic = "SOURCE_SELECTED_TEST"
    elif -gap - uncertainty > decision_margin:
        selected = best_commit
        diagnostic = "SOURCE_SELECTED_COMMIT"
    else:
        selected = fallback
        diagnostic = "SOURCE_ABSTAINED_TO_TARGET"
    admitted = selected != fallback or diagnostic != "SOURCE_ABSTAINED_TO_TARGET"
    return {
        "action": selected,
        "fallback_action": fallback,
        "source_admitted": admitted,
        "changed_action": selected != fallback,
        "changed_role": action_role(selected) != action_role(fallback),
        "diagnostic": diagnostic,
        "test_commit_gap": gap,
        "ensemble_uncertainty": uncertainty,
    }


__all__ = [
    "ACTION_VERBS",
    "action_role",
    "choose_grounded_action",
    "deserialize_value_ensemble",
    "grounder_features",
    "mlp_score",
    "score_native_actions",
    "target_symbolic_features",
]
