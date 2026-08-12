"""Bind source intervention structure to oracle-free target neural grounding."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .hierarchical_skill_transfer import FEATURE_NAMES, OPTION_NAMES
from .oracle_free_target_grounder import score_native_actions
from .pairwise_option_advantage import (
    PairwiseAdvantageEnsemble,
    choose_option_against_fallback,
)


FORBIDDEN_FEATURE_INDICES = (
    *range(5, 11),
    13,
    15,
    18,
    19,
    20,
    21,
    22,
)


def target_option_features(
    *,
    option: str,
    neural_effect_probability: float,
    representative_action: str,
    action_history: Sequence[str],
    step: int,
    max_steps: int,
) -> tuple[float, ...]:
    """Map target neural evidence into the transferable source feature slots.

    Non-grounded source fields are fixed to the same zero for every option, so
    they cancel in the pairwise controller rather than receiving guessed values.
    """
    if option not in OPTION_NAMES:
        raise ValueError("unknown transferable option")
    effect = min(max(float(neural_effect_probability), 0.0), 1.0)
    repeats = Counter(map(str, action_history))
    repeat_fraction = min(repeats[str(representative_action)], 4) / 4.0
    remaining = max(0, int(max_steps) - int(step)) / max(1, int(max_steps))
    values = (
        *(float(option == name) for name in OPTION_NAMES),
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0,
        effect,
        effect,
        0.0,
        remaining,
        0.0,
        repeat_fraction,
        1.0 - effect,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("target/source pairwise feature contract drift")
    if any(values[index] != 0.0 for index in FORBIDDEN_FEATURE_INDICES):
        raise AssertionError("forbidden target option feature was populated")
    return tuple(values)


def ground_target_options(
    *,
    goal: str,
    observation: str,
    native_actions: Sequence[str],
    step: int,
    max_steps: int,
    action_history: Sequence[str],
    target_grounder: Mapping[str, Any],
) -> dict[str, Any]:
    scored = score_native_actions(
        goal=goal,
        observation=observation,
        native_actions=native_actions,
        step=step,
        action_history=action_history,
        artifact=target_grounder,
    )
    if not scored:
        raise ValueError("oracle-free target grounder excluded every action")
    ranked_actions = sorted(
        scored,
        key=lambda action: (
            -float(scored[action]["policy_probability"]),
            action,
        ),
    )
    fallback_action = ranked_actions[0]
    representatives: dict[str, str] = {}
    for action in ranked_actions:
        representatives.setdefault(str(scored[action]["option"]), action)
    features = {
        option: target_option_features(
            option=option,
            neural_effect_probability=float(scored[action]["policy_probability"]),
            representative_action=action,
            action_history=action_history,
            step=step,
            max_steps=max_steps,
        )
        for option, action in representatives.items()
    }
    return {
        "fallback_action": fallback_action,
        "fallback_option": str(scored[fallback_action]["option"]),
        "representative_actions": representatives,
        "option_features": features,
        "action_scores": scored,
    }


def source_shadow_decision(
    grounded: Mapping[str, Any],
    *,
    model: PairwiseAdvantageEnsemble,
    conformal_error: float,
) -> dict[str, Any]:
    if len(grounded["option_features"]) == 1:
        fallback_option = str(grounded["fallback_option"])
        fallback_action = str(grounded["fallback_action"])
        comparison = {
            "option": fallback_option,
            "predicted_advantage": 0.0,
            "ensemble_deviation": 0.0,
            "conformal_lower_bound": 0.0,
        }
        return {
            "option": fallback_option,
            "action": fallback_action,
            "fallback_option": fallback_option,
            "source_admitted": False,
            "comparison": comparison,
            "all_comparisons": [],
        }
    decision = choose_option_against_fallback(
        model,
        grounded["option_features"],
        fallback_option=str(grounded["fallback_option"]),
        conformal_error=float(conformal_error),
    )
    option = str(decision["option"])
    action = str(grounded["representative_actions"][option])
    return decision | {"action": action}


def within_state_target_effect_permutation(
    grounded: Mapping[str, Any],
) -> dict[str, Any]:
    """Rotate target neural effect receipts while preserving option identities."""
    features = grounded["option_features"]
    options = sorted(features)
    donors = options[1:] + options[:1]
    permuted = {}
    for option, donor in zip(options, donors):
        values = list(features[option])
        donor_values = features[donor]
        values[11] = donor_values[11]
        values[12] = donor_values[12]
        values[17] = donor_values[17]
        permuted[option] = tuple(values)
    return dict(grounded) | {"option_features": permuted}


__all__ = [
    "FORBIDDEN_FEATURE_INDICES",
    "ground_target_options",
    "source_shadow_decision",
    "target_option_features",
    "within_state_target_effect_permutation",
]
