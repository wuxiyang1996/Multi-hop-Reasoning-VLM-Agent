"""Target-native ALFWorld grounding for the Phase-3 anonymous typed IR.

The source program never sees ALFWorld verbs, objects, observations, or task
success.  Four target-native neural heads expose the unchanged source effect
vocabulary.  Current qualified artifacts supervise those heads with matched
development-only option interventions at transition 1/4/8 and executable
transition persistence; legacy artifacts used expert-continuation labels and
are retained as failed controls.  A separately trained target policy realizes
the selected target-native option as a concrete action.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .alfworld_hierarchical_grounder import (
    action_option,
    grounder_features,
    mlp_probability,
)
from .contracts import stable_hash


ARTIFACT_VERSION = "PHASE3_ALFWORLD_TYPED_GROUNDER_V1"
EFFECT_TYPES = (
    "EFFECT_BY_TRANSITION_1",
    "EFFECT_BY_TRANSITION_4",
    "EFFECT_BY_TRANSITION_8",
    "EXECUTABLE_TRANSITION_PERSISTENCE",
)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("Phase-3 ALFWorld grounder hash mismatch")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported Phase-3 ALFWorld grounder version")
    if tuple(artifact.get("effect_types", ())) != EFFECT_TYPES:
        raise ValueError("Phase-3 ALFWorld effect vocabulary changed")
    if not artifact.get("required_option_masked_for_every_head"):
        raise ValueError("target stage leaked into a neural grounding head")
    if artifact.get("formal_success_read_for_training_or_qualification") is not False:
        raise ValueError("formal success is forbidden during target grounding")
    heads = artifact.get("typed_effect_heads")
    if not isinstance(heads, Mapping) or set(heads) != set(EFFECT_TYPES):
        raise ValueError("typed target effect heads are incomplete")


def masked_features(
    *, goal: str, observation: str, action: str, step: int,
    action_history: Sequence[str], feature_bins: int,
) -> Any:
    """Build the target neural input without a symbolic required-stage hint."""

    return grounder_features(
        goal=goal,
        observation=observation,
        action=str(action),
        required_option="SEARCH",
        step=int(step),
        action_history=tuple(map(str, action_history)),
        feature_bins=int(feature_bins),
        mask_required_option=True,
    )


def score_actions(
    *, goal: str, observation: str, native_actions: Sequence[str], step: int,
    action_history: Sequence[str], artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Ground native actions into the unchanged four-field source vocabulary."""

    feature_bins = int(artifact["feature_bins"])
    policy_exponent = float(artifact["policy_support_exponent"])
    policy_head = artifact["target_policy_head"]
    effect_heads = artifact["typed_effect_heads"]
    rows: dict[str, dict[str, Any]] = {}
    for raw_action in native_actions:
        action = str(raw_action)
        if action_option(action) == "EXCLUDE":
            continue
        features = masked_features(
            goal=goal,
            observation=observation,
            action=action,
            step=step,
            action_history=action_history,
            feature_bins=feature_bins,
        )
        policy = mlp_probability(features, policy_head)
        horizon = {
            effect: mlp_probability(features, effect_heads[effect])
            for effect in EFFECT_TYPES
        }
        typed = {
            effect: min(1.0, max(0.0, value * policy ** policy_exponent))
            for effect, value in horizon.items()
        }
        rows[action] = {
            "target_policy_probability": policy,
            "target_horizon_probabilities": horizon,
            "typed_effect_probabilities": typed,
            "action_sha256": stable_hash({"target_native_action": action}),
        }
    return rows


__all__ = [
    "ARTIFACT_VERSION", "EFFECT_TYPES", "masked_features", "score_actions",
    "validate_artifact",
]
