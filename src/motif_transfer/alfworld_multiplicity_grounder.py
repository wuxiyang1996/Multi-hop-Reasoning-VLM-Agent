"""Identity-aware ALFWorld grounding for multiplicity goals.

The V1 workflow state counted PLACE events.  That is sufficient for singleton
goals, but it aliases repeatedly placing the same object with placing two
distinct objects.  This extension replays target-native object identities and
represents the goal predicate as a set of distinct bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .alfworld_hierarchical_grounder import (
    OPTION_NAMES,
    action_option,
    action_verb,
    grounder_features,
    mlp_probability,
    parse_goal,
    tokens,
)


@dataclass(frozen=True)
class MultiplicityWorkflowStatus:
    held: bool
    transformed: bool
    placed_count: int
    progress_fraction: float
    held_object_id: str
    placed_object_ids: tuple[str, ...]
    required_count: int

    @property
    def remaining_count(self) -> int:
        return max(0, self.required_count - self.placed_count)


def _object_id(action: str, target_object: str) -> str:
    values = tokens(action)
    for index, value in enumerate(values):
        if value != target_object:
            continue
        suffix = values[index + 1] if index + 1 < len(values) else ""
        return f"{target_object}:{suffix}" if suffix.isdigit() else target_object
    return ""


def _mentions_destination(action: str, destination: str) -> bool:
    return bool(destination) and destination in tokens(action)


def workflow_status(
    goal: str, action_history: Sequence[str]
) -> MultiplicityWorkflowStatus:
    spec = parse_goal(goal)
    held_id = ""
    transformed_ids: set[str] = set()
    placed_ids: set[str] = set()
    for action in action_history:
        verb = action_verb(action)
        object_id = _object_id(action, spec.target_object)
        if verb == "take" and object_id:
            held_id = object_id
            # Taking an object back from the goal receptacle reverses a
            # previously established goal predicate.
            if _mentions_destination(action, spec.destination):
                placed_ids.discard(object_id)
        elif (
            verb in {"clean", "heat", "cool", "use", "toggle"}
            and held_id
            and (spec.look_task or object_id == held_id)
        ):
            transformed_ids.add(held_id)
        elif (
            verb in {"move", "put"}
            and held_id
            and object_id == held_id
            and _mentions_destination(action, spec.destination)
        ):
            placed_ids.add(held_id)
            held_id = ""
    placed_count = min(len(placed_ids), spec.count)
    base = placed_count / spec.count
    held_credit = 0.35 / spec.count if held_id else 0.0
    transformed_credit = (
        0.30 / spec.count if held_id and held_id in transformed_ids else 0.0
    )
    return MultiplicityWorkflowStatus(
        held=bool(held_id),
        transformed=bool(held_id and held_id in transformed_ids),
        placed_count=placed_count,
        progress_fraction=min(1.0, base + held_credit + transformed_credit),
        held_object_id=held_id,
        placed_object_ids=tuple(sorted(placed_ids)),
        required_count=spec.count,
    )


def candidate_effect(goal: str, history: Sequence[str], action: str) -> dict[str, Any]:
    before = workflow_status(goal, history)
    after = workflow_status(goal, (*history, action))
    spec = parse_goal(goal)
    object_id = _object_id(action, spec.target_object)
    reverses = (
        action_verb(action) == "take"
        and object_id in before.placed_object_ids
        and _mentions_destination(action, spec.destination)
    )
    return {
        "object_id": object_id,
        "distinct_placed_before": before.placed_count,
        "distinct_placed_after": after.placed_count,
        "remaining_count_before": before.remaining_count,
        "symbolic_progress_delta": after.progress_fraction - before.progress_fraction,
        "reverses_completed_binding": reverses,
        "establishes_new_binding": (
            after.placed_count > before.placed_count and bool(object_id)
        ),
    }


def infer_required_option(
    *, goal: str, native_actions: Sequence[str], action_history: Sequence[str]
) -> str:
    spec = parse_goal(goal)
    status = workflow_status(goal, action_history)
    if status.remaining_count == 0:
        return "VERIFY"
    candidates = [str(action) for action in native_actions]
    if not status.held:
        can_take_unfilled_target = any(
            action_option(action) == "ACQUIRE"
            and (object_id := _object_id(action, spec.target_object))
            and object_id not in status.placed_object_ids
            and not candidate_effect(goal, action_history, action)[
                "reverses_completed_binding"
            ]
            for action in candidates
        )
        return "ACQUIRE" if can_take_unfilled_target else "SEARCH"
    if spec.transform != "none" and not status.transformed:
        expected_verbs = {"use", "toggle"} if spec.look_task else {spec.transform}
        can_transform = any(
            action_verb(action) in expected_verbs
            and (spec.look_task or _object_id(action, spec.target_object) == status.held_object_id)
            for action in candidates
        )
        return "TRANSFORM" if can_transform else "SEARCH"
    can_place_held = any(
        action_option(action) == "PLACE"
        and _object_id(action, spec.target_object) == status.held_object_id
        and _mentions_destination(action, spec.destination)
        for action in candidates
    )
    return "PLACE" if can_place_held else "SEARCH"


def _features(
    *, goal: str, observation: str, action: str, required_option: str,
    step: int, action_history: Sequence[str], feature_bins: int,
    mask_required_option: bool = False,
) -> np.ndarray:
    values = grounder_features(
        goal=goal,
        observation=observation,
        action=action,
        required_option=required_option,
        step=step,
        action_history=action_history,
        feature_bins=feature_bins,
        mask_required_option=mask_required_option,
    ).copy()
    # Preserve the frozen neural feature contract while replacing the three
    # aliased V1 state scalars with identity-aware target-native state.
    status = workflow_status(goal, action_history)
    values[20] = status.progress_fraction
    values[21] = float(status.held)
    values[22] = float(status.transformed)
    return values


def score_actions(
    *, goal: str, observation: str, native_actions: Sequence[str], step: int,
    action_history: Sequence[str], artifact: Mapping[str, Any],
) -> dict[str, dict[str, float | str | bool]]:
    required = infer_required_option(
        goal=goal, native_actions=native_actions, action_history=action_history,
    )
    result: dict[str, dict[str, float | str | bool]] = {}
    for raw_action in native_actions:
        action = str(raw_action)
        option = action_option(action)
        if option == "EXCLUDE":
            continue
        features = _features(
            goal=goal,
            observation=observation,
            action=action,
            required_option=required,
            step=step,
            action_history=action_history,
            feature_bins=int(artifact["feature_bins"]),
        )
        neural_binding = mlp_probability(features, artifact["binding_head"])
        neural_completion = mlp_probability(features, artifact["completion_head"])
        neural_applicability = mlp_probability(features, artifact["applicability_head"])
        effect = candidate_effect(goal, action_history, action)
        reverses = bool(effect["reverses_completed_binding"])
        row: dict[str, float | str | bool] = {
            "option": option,
            "required_option": required,
            "applicability": 0.0 if reverses else neural_applicability,
            "binding": 0.0 if reverses else neural_binding,
            "completion": 0.0 if reverses else neural_completion,
            "neural_applicability": neural_applicability,
            "neural_binding": neural_binding,
            "neural_completion": neural_completion,
            **effect,
        }
        if "policy_head" in artifact:
            policy_features = _features(
                goal=goal,
                observation=observation,
                action=action,
                required_option=required,
                step=step,
                action_history=action_history,
                feature_bins=int(artifact["feature_bins"]),
                mask_required_option=True,
            )
            row["policy"] = mlp_probability(policy_features, artifact["policy_head"])
        result[action] = row
    return result


__all__ = [
    "MultiplicityWorkflowStatus",
    "candidate_effect",
    "infer_required_option",
    "score_actions",
    "workflow_status",
]
