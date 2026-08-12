"""Effect-conditioned target-native models for V22 neural-symbolic transfer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .real_source_relation_causal_v20 import (
    ACTION_FEATURE_NAMES as BASE_ACTION_FEATURE_NAMES,
    action_causal_features as base_action_causal_features,
    linear_probability,
    linear_value,
)
from .relation_edge_value_v13 import FEATURE_NAMES as BASE_UTILITY_FEATURE_NAMES


EFFECTS = ("BIND", "MUTATE", "RELATE")
PROPERTIES = ("CLEAN", "HEAT")
ACTION_FEATURE_NAMES = (
    *BASE_ACTION_FEATURE_NAMES,
    *(f"requested_{name.lower()}" for name in EFFECTS),
    *(f"required_{name.lower()}" for name in PROPERTIES),
)
UTILITY_FEATURE_NAMES = (
    *BASE_UTILITY_FEATURE_NAMES,
    *(f"requested_{name.lower()}" for name in EFFECTS),
    *(f"required_{name.lower()}" for name in PROPERTIES),
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
RECEIPT_BY_EFFECT = {
    "BIND": "BIND_INSTANCE",
    "MUTATE": "MUTATE_REQUIRED_PROPERTY",
    "RELATE": "RELATE_SLOT_CLOSED",
}


def action_causal_features(
    *,
    action: str,
    grounded_scores: Mapping[str, float],
    ledger: Mapping[str, Any],
    history: Sequence[str],
    step: int,
    max_steps: int,
    requested_effect: str,
    required_property: str,
) -> dict[str, float]:
    if requested_effect not in EFFECTS:
        raise ValueError(f"unsupported V22 effect: {requested_effect}")
    if required_property not in PROPERTIES:
        raise ValueError(f"unsupported V22 property: {required_property}")
    values = base_action_causal_features(
        action=action,
        grounded_scores=grounded_scores,
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
    )
    values.update({
        **{
            f"requested_{name.lower()}": float(requested_effect == name)
            for name in EFFECTS
        },
        **{
            f"required_{name.lower()}": float(required_property == name)
            for name in PROPERTIES
        },
    })
    if tuple(values) != ACTION_FEATURE_NAMES:
        raise RuntimeError("V22 action feature order drift")
    return values


def utility_features(
    *,
    base_features: Mapping[str, float],
    requested_effect: str,
    required_property: str,
    source_effect_probability: float,
    fallback_effect_probability: float,
    source_action_features: Mapping[str, float],
    fallback_action_features: Mapping[str, float],
) -> dict[str, float]:
    values = {
        name: float(base_features[name]) for name in BASE_UTILITY_FEATURE_NAMES
    }
    values.update({
        **{
            f"requested_{name.lower()}": float(requested_effect == name)
            for name in EFFECTS
        },
        **{
            f"required_{name.lower()}": float(required_property == name)
            for name in PROPERTIES
        },
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
        raise RuntimeError("V22 utility feature order drift")
    return values


def score_multiskill_decision(
    *,
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
    grounded: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[str, Any],
    history: Sequence[str],
    step: int,
    max_steps: int,
    base_features: Mapping[str, float],
) -> dict[str, Any]:
    """Score an authentic typed source edge from pre-action target evidence."""
    source_action = str(decision["action"])
    fallback_action = str(decision["fallback_action"])
    requested_effect = str(decision["requested_source_effect"])
    required_property = str(decision["required_property"])
    source_features = action_causal_features(
        action=source_action,
        grounded_scores=grounded[source_action],
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
        requested_effect=requested_effect,
        required_property=required_property,
    )
    fallback_features = action_causal_features(
        action=fallback_action,
        grounded_scores=grounded[fallback_action],
        ledger=ledger,
        history=history,
        step=step,
        max_steps=max_steps,
        requested_effect=requested_effect,
        required_property=required_property,
    )
    head = candidate["target_typed_successor_head"]
    source_probability = linear_probability(head, source_features)
    fallback_probability = linear_probability(head, fallback_features)
    value_features = utility_features(
        base_features=base_features,
        requested_effect=requested_effect,
        required_property=required_property,
        source_effect_probability=source_probability,
        fallback_effect_probability=fallback_probability,
        source_action_features=source_features,
        fallback_action_features=fallback_features,
    )
    predicted_utility = linear_value(
        candidate["target_incremental_utility_head"], value_features
    )
    threshold = float(candidate["selective_risk_calibration"][
        "admission_threshold"
    ])
    effect_margin_threshold = float(candidate["selective_risk_calibration"][
        "minimum_causal_effect_margin"
    ])
    return {
        "requested_effect": requested_effect,
        "required_property": required_property,
        "source_action": source_action,
        "fallback_action": fallback_action,
        "source_action_features": source_features,
        "fallback_action_features": fallback_features,
        "source_causal_effect_probability": source_probability,
        "fallback_causal_effect_probability": fallback_probability,
        "causal_effect_margin": source_probability - fallback_probability,
        "utility_features": value_features,
        "predicted_incremental_utility": predicted_utility,
        "admission_threshold": threshold,
        "minimum_causal_effect_margin": effect_margin_threshold,
        "admitted": bool(
            predicted_utility > threshold
            and source_probability - fallback_probability > effect_margin_threshold
        ),
    }


__all__ = [
    "ACTION_FEATURE_NAMES",
    "EFFECTS",
    "PROPERTIES",
    "RECEIPT_BY_EFFECT",
    "UTILITY_FEATURE_NAMES",
    "action_causal_features",
    "score_multiskill_decision",
    "utility_features",
]
