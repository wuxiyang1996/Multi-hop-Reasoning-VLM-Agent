"""Target-native action grounding without a hand-written workflow oracle.

The representation deliberately has no required-option, task-progress, inventory,
completion, reward, or success input.  The only shared symbol is the coarse action
option; everything else is lexical target evidence or observable action history.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .alfworld_hierarchical_grounder import (
    OPTION_NAMES,
    action_option,
    action_verb,
    mlp_probability,
    tokens,
)
from .neurosymbolic_transfer_contract import target_score_contract


DENSE_FEATURE_NAMES = (
    *(f"action_option_{name.lower()}" for name in OPTION_NAMES),
    "goal_action_token_overlap",
    "observation_action_token_overlap",
    "exact_action_repeat_fraction",
    "same_previous_action_verb",
    "normalized_step",
)

FORBIDDEN_SEMANTIC_INPUTS = (
    "required_option",
    "workflow_progress",
    "held_object",
    "transformed_object",
    "completion_label",
    "reward",
    "official_success",
)


def _hash_bin(value: str, bins: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % bins


def policy_features(
    *,
    goal: str,
    observation: str,
    action: str,
    step: int,
    action_history: Sequence[str],
    feature_bins: int,
) -> np.ndarray:
    """Encode only target-native lexical evidence and observable history."""
    if feature_bins < 16:
        raise ValueError("feature_bins must be at least 16")
    option = action_option(action)
    if option == "EXCLUDE":
        raise ValueError("excluded native action has no transfer option")
    action_values = tokens(action)
    action_set = set(action_values)
    goal_set = set(tokens(goal))
    observation_set = set(tokens(observation))
    lexical = np.zeros(feature_bins, dtype=np.float64)
    for token in action_values:
        lexical[_hash_bin(f"action:{token}", feature_bins)] += 1.0
        if token in goal_set:
            lexical[_hash_bin(f"goal-action:{token}", feature_bins)] += 1.0
        if token in observation_set:
            lexical[_hash_bin(f"observation-action:{token}", feature_bins)] += 1.0
    norm = float(np.linalg.norm(lexical))
    if norm:
        lexical /= norm
    repeat = sum(str(previous) == str(action) for previous in action_history)
    scalars = (
        len(action_set & goal_set) / max(1, len(action_set)),
        len(action_set & observation_set) / max(1, len(action_set)),
        min(repeat, 8) / 8.0,
        float(
            bool(action_history)
            and action_verb(str(action_history[-1])) == action_verb(action)
        ),
        min(max(int(step), 0), 180) / 180.0,
    )
    return np.asarray((
        *(float(option == name) for name in OPTION_NAMES),
        *scalars,
        *lexical.tolist(),
    ), dtype=np.float64)


def score_native_actions(
    *,
    goal: str,
    observation: str,
    native_actions: Sequence[str],
    step: int,
    action_history: Sequence[str],
    artifact: Mapping[str, Any],
) -> dict[str, dict[str, float | str]]:
    if artifact.get("required_option_or_workflow_features_used") is not False:
        raise ValueError("target grounder does not certify the oracle-free contract")
    feature_bins = int(artifact["feature_bins"])
    head = artifact["policy_head"]
    contract = target_score_contract(artifact)
    result: dict[str, dict[str, float | str]] = {}
    for raw_action in native_actions:
        action = str(raw_action)
        option = action_option(action)
        if option == "EXCLUDE":
            continue
        features = policy_features(
            goal=goal,
            observation=observation,
            action=action,
            step=step,
            action_history=action_history,
            feature_bins=feature_bins,
        )
        score = mlp_probability(features, head)
        result[action] = {
            "option": option,
            "score": score,
            "score_semantics": str(contract["score_semantics"]),
            # Kept only so frozen V17/V18 artifacts remain inspectable.  A
            # caller must not reinterpret this alias as an intervention effect.
            "policy_probability": score,
        }
    return result


__all__ = [
    "DENSE_FEATURE_NAMES",
    "FORBIDDEN_SEMANTIC_INPUTS",
    "policy_features",
    "score_native_actions",
]
