"""Target-native causal/action and utility models for real-source relation transfer."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .relation_edge_value_v13 import (
    FEATURE_NAMES as V13_FEATURE_NAMES,
    extract_relation_edge_features,
)


ACTION_FEATURE_NAMES = (
    "step_fraction",
    "remaining_slots_fraction",
    "completed_slots_fraction",
    "action_policy",
    "action_completion",
    "action_binding",
    "action_applicability",
    "action_exact_repeat_fraction",
    "goal_object_token_match",
    "goal_receptacle_token_match",
    "verb_move",
    "verb_take",
    "verb_open",
    "verb_close",
    "verb_examine_or_look",
    "verb_navigation",
)

UTILITY_FEATURE_NAMES = (
    *V13_FEATURE_NAMES,
    "source_causal_effect_probability",
    "fallback_causal_effect_probability",
    "causal_effect_margin",
    "source_exact_repeat_fraction",
    "fallback_exact_repeat_fraction",
    "source_goal_object_match",
    "source_goal_receptacle_match",
    "fallback_goal_object_match",
    "fallback_goal_receptacle_match",
)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value).lower()))


def action_causal_features(
    *,
    action: str,
    grounded_scores: Mapping[str, float],
    ledger: Mapping[str, Any],
    history: Sequence[str],
    step: int,
    max_steps: int,
) -> dict[str, float]:
    """Encode pre-action target evidence for a typed RELATE successor event."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    goal_spec = ledger.get("goal_spec", {})
    goal_object = set(_tokens(str(goal_spec.get("goal_object_type", ""))))
    receptacle = set(_tokens(str(goal_spec.get("target_receptacle_type", ""))))
    action_tokens = set(_tokens(action))
    verb = next(iter(_tokens(action)), "")
    required = max(int(goal_spec.get("required_count", 1)), 1)
    completed = len(ledger.get("completed_objects", ()))
    remaining = max(0, required - completed)
    repeats = sum(str(row) == str(action) for row in history)
    values = {
        "step_fraction": min(max(float(step) / max_steps, 0.0), 1.0),
        "remaining_slots_fraction": min(float(remaining) / required, 1.0),
        "completed_slots_fraction": min(float(completed) / required, 1.0),
        "action_policy": float(grounded_scores.get("policy", 0.0)),
        "action_completion": float(grounded_scores.get("completion", 0.0)),
        "action_binding": float(grounded_scores.get("binding", 0.0)),
        "action_applicability": float(grounded_scores.get("applicability", 0.0)),
        "action_exact_repeat_fraction": min(repeats, 4) / 4.0,
        "goal_object_token_match": float(bool(goal_object & action_tokens)),
        "goal_receptacle_token_match": float(bool(receptacle & action_tokens)),
        "verb_move": float(verb == "move"),
        "verb_take": float(verb == "take"),
        "verb_open": float(verb == "open"),
        "verb_close": float(verb == "close"),
        "verb_examine_or_look": float(verb in {"examine", "look"}),
        "verb_navigation": float(verb in {"go", "walk"}),
    }
    if tuple(values) != ACTION_FEATURE_NAMES:
        raise RuntimeError("V20 causal action feature order drift")
    return values


def linear_probability(
    model: Mapping[str, Any], features: Mapping[str, float]
) -> float:
    names = tuple(map(str, model["feature_names"]))
    vector = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if not (len(vector) == len(means) == len(scales) == len(weights)):
        raise ValueError("V20 probability-head dimensions mismatch")
    score = float(model["intercept"]) + ((vector - means) / scales) @ weights
    scalar = float(score)
    if scalar >= 0:
        return 1.0 / (1.0 + math.exp(-scalar))
    exponential = math.exp(scalar)
    return exponential / (1.0 + exponential)


def utility_features(
    *,
    base_features: Mapping[str, float],
    source_effect_probability: float,
    fallback_effect_probability: float,
    source_action_features: Mapping[str, float],
    fallback_action_features: Mapping[str, float],
) -> dict[str, float]:
    values = {name: float(base_features[name]) for name in V13_FEATURE_NAMES}
    values.update({
        "source_causal_effect_probability": float(source_effect_probability),
        "fallback_causal_effect_probability": float(fallback_effect_probability),
        "causal_effect_margin": (
            float(source_effect_probability) - float(fallback_effect_probability)
        ),
        "source_exact_repeat_fraction": float(
            source_action_features["action_exact_repeat_fraction"]
        ),
        "fallback_exact_repeat_fraction": float(
            fallback_action_features["action_exact_repeat_fraction"]
        ),
        "source_goal_object_match": float(
            source_action_features["goal_object_token_match"]
        ),
        "source_goal_receptacle_match": float(
            source_action_features["goal_receptacle_token_match"]
        ),
        "fallback_goal_object_match": float(
            fallback_action_features["goal_object_token_match"]
        ),
        "fallback_goal_receptacle_match": float(
            fallback_action_features["goal_receptacle_token_match"]
        ),
    })
    if tuple(values) != UTILITY_FEATURE_NAMES:
        raise RuntimeError("V20 utility feature order drift")
    return values


def linear_value(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    names = tuple(map(str, model["feature_names"]))
    vector = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if not (len(vector) == len(means) == len(scales) == len(weights)):
        raise ValueError("V20 utility-head dimensions mismatch")
    return float(float(model["intercept"]) + ((vector - means) / scales) @ weights)


def score_relation_decision(
    *,
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
    grounded: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[str, Any],
    history: Sequence[str],
    step: int,
    max_steps: int,
    native_action_count: int,
) -> dict[str, Any]:
    """Score a live symbolic edge using only evidence available pre-action."""
    source_action = str(decision["action"])
    fallback_action = str(decision["fallback_action"])
    if source_action not in grounded or fallback_action not in grounded:
        raise ValueError("candidate actions must both have target-native scores")
    base = extract_relation_edge_features(
        decision=decision,
        grounded=grounded,
        ledger=ledger,
        step=step,
        max_steps=max_steps,
        native_action_count=native_action_count,
    )
    source_features = action_causal_features(
        action=source_action,
        grounded_scores=grounded[source_action],
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
    )
    fallback_features = action_causal_features(
        action=fallback_action,
        grounded_scores=grounded[fallback_action],
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
    )
    effect_head = candidate["target_causal_effect_head"]
    source_probability = linear_probability(effect_head, source_features)
    fallback_probability = linear_probability(effect_head, fallback_features)
    value_features = utility_features(
        base_features=base,
        source_effect_probability=source_probability,
        fallback_effect_probability=fallback_probability,
        source_action_features=source_features,
        fallback_action_features=fallback_features,
    )
    predicted_utility = linear_value(
        candidate["target_incremental_utility_head"], value_features
    )
    if "selective_risk_calibration" in candidate:
        admission_threshold = float(
            candidate["selective_risk_calibration"]["admission_threshold"]
        )
        admission_authority = "OUT_OF_FOLD_SELECTIVE_RISK_THRESHOLD"
    else:
        admission_threshold = float(
            candidate["conformal"]["overprediction_error_quantile"]
        )
        admission_authority = "CONFORMAL_OVERPREDICTION_QUANTILE"
    lower_bound = predicted_utility - admission_threshold
    return {
        "source_action": source_action,
        "fallback_action": fallback_action,
        "source_action_features": source_features,
        "fallback_action_features": fallback_features,
        "source_causal_effect_probability": source_probability,
        "fallback_causal_effect_probability": fallback_probability,
        "causal_effect_margin": source_probability - fallback_probability,
        "utility_features": value_features,
        "predicted_incremental_utility": predicted_utility,
        "admission_threshold": admission_threshold,
        "admission_authority": admission_authority,
        "conformal_lower_bound": lower_bound,
        "admitted": lower_bound > 0.0,
    }


__all__ = [
    "ACTION_FEATURE_NAMES",
    "UTILITY_FEATURE_NAMES",
    "action_causal_features",
    "linear_probability",
    "linear_value",
    "score_relation_decision",
    "utility_features",
]
