"""Inference for a target-native ALFWorld grounder with stage inputs masked."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .alfworld_hierarchical_grounder import (
    action_option,
    grounder_features,
    infer_required_option,
    mlp_probability,
)
from .contracts import stable_hash


ARTIFACT_VERSION = "ALFWORLD_MASKED_EFFECT_GROUNDER_V2"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError("masked target grounder hash mismatch")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported masked target grounder version")
    if not artifact.get("required_option_masked_for_every_head"):
        raise ValueError("target stage input was not masked")


def score_actions(
    *, goal: str, observation: str, native_actions: Sequence[str], step: int,
    action_history: Sequence[str], artifact: Mapping[str, Any],
) -> dict[str, dict[str, float | str]]:
    """Score actions without exposing target stage to any neural head.

    Long-running callers validate the frozen artifact once before their loop;
    avoiding a full artifact hash at every environment step is intentional.
    """
    diagnostic_required = infer_required_option(
        goal=goal, native_actions=native_actions, action_history=action_history,
    )
    result = {}
    for action in map(str, native_actions):
        option = action_option(action)
        if option == "EXCLUDE":
            continue
        # SEARCH is a dummy required option: all required-option coordinates,
        # including option==required, are zeroed by mask_required_option.
        features = grounder_features(
            goal=goal,
            observation=observation,
            action=action,
            required_option="SEARCH",
            step=step,
            action_history=action_history,
            feature_bins=int(artifact["feature_bins"]),
            mask_required_option=True,
        )
        result[action] = {
            "option": option,
            "required_option": diagnostic_required,
            "applicability": mlp_probability(
                features, artifact["applicability_head"],
            ),
            "binding": mlp_probability(features, artifact["binding_head"]),
            "completion": mlp_probability(features, artifact["completion_head"]),
            "policy": mlp_probability(features, artifact["policy_head"]),
        }
    return result


__all__ = ["ARTIFACT_VERSION", "score_actions", "validate_artifact"]
