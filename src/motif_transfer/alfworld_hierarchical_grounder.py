from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


OPTION_NAMES = ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "VERIFY")


TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
DETERMINERS = frozenset({"a", "an", "the", "some", "two"})
TRANSFORM_WORDS = frozenset({"clean", "hot", "heat", "cool", "cold"})
EXCLUDED_VERBS = frozenset({"help"})


def tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(str(text).lower()))


def action_verb(action: str) -> str:
    values = tokens(action)
    return values[0] if values else ""


@dataclass(frozen=True)
class GoalSpec:
    target_object: str
    destination: str
    transform: str
    count: int
    look_task: bool


def _after(values: Sequence[str], marker: str) -> str:
    try:
        index = values.index(marker) + 1
    except ValueError:
        return ""
    while index < len(values) and (
        values[index] in DETERMINERS or values[index] in TRANSFORM_WORDS
    ):
        index += 1
    return values[index] if index < len(values) else ""


def parse_goal(goal: str) -> GoalSpec:
    values = list(tokens(goal))
    look_task = "desklamp" in values and ("look" in values or "examine" in values)
    if "examine" in values:
        target = _after(values, "examine")
    elif "look" in values and "at" in values:
        target = _after(values[values.index("at") :], "at")
    elif "find" in values:
        target = _after(values, "find")
    elif any(word in values for word in ("clean", "heat", "cool")) and "and" in values:
        operation = next(word for word in ("clean", "heat", "cool") if word in values)
        target = _after(values, operation)
    else:
        target = _after(values, "put")
    destination = "desklamp" if look_task else ""
    if not destination:
        positions = [index for index, value in enumerate(values) if value in {"in", "on"}]
        if positions and positions[-1] + 1 < len(values):
            destination = values[positions[-1] + 1]
    if look_task:
        transform = "light"
    elif "clean" in values:
        transform = "clean"
    elif "heat" in values or "hot" in values:
        transform = "heat"
    elif "cool" in values or "cold" in values:
        transform = "cool"
    else:
        transform = "none"
    if not target:
        raise ValueError(f"could not parse ALFWorld goal target: {goal!r}")
    return GoalSpec(target, destination, transform, 2 if "two" in values else 1, look_task)


def action_option(action: str) -> str:
    verb = action_verb(action)
    if verb in EXCLUDED_VERBS:
        return "EXCLUDE"
    if verb in {"go", "look", "open", "close", "examine", "inventory"}:
        return "SEARCH"
    if verb == "take":
        return "ACQUIRE"
    if verb in {"clean", "heat", "cool", "slice", "use", "toggle"}:
        return "TRANSFORM"
    if verb in {"move", "put"}:
        return "PLACE"
    return "EXCLUDE"


def _mentions(action: str, entity: str) -> bool:
    return bool(entity) and entity in tokens(action)


@dataclass(frozen=True)
class WorkflowStatus:
    held: bool
    transformed: bool
    placed_count: int
    progress_fraction: float


def workflow_status(goal: str, action_history: Sequence[str]) -> WorkflowStatus:
    spec = parse_goal(goal)
    held = False
    transformed = False
    placed = 0
    for action in action_history:
        verb = action_verb(action)
        if verb == "take" and _mentions(action, spec.target_object):
            held = True
        elif (
            verb in {"clean", "heat", "cool", "use", "toggle"}
            and held
            and (spec.look_task or _mentions(action, spec.target_object))
        ):
            transformed = True
        elif (
            verb in {"move", "put"}
            and held
            and _mentions(action, spec.target_object)
            and _mentions(action, spec.destination)
        ):
            placed += 1
            held = False
            transformed = False
    base = min(placed, spec.count) / spec.count
    partial = (0.35 if held else 0.0) + (0.30 if transformed else 0.0)
    progress = min(1.0, base + partial / spec.count)
    return WorkflowStatus(held, transformed, placed, progress)


def infer_required_option(
    *, goal: str, native_actions: Sequence[str], action_history: Sequence[str]
) -> str:
    spec = parse_goal(goal)
    status = workflow_status(goal, action_history)
    if status.placed_count >= spec.count:
        return "VERIFY"
    candidates = [str(action) for action in native_actions]
    if not status.held:
        can_take_target = any(
            action_option(action) == "ACQUIRE" and _mentions(action, spec.target_object)
            for action in candidates
        )
        return "ACQUIRE" if can_take_target else "SEARCH"
    if spec.transform != "none" and not status.transformed:
        expected_verbs = {"use", "toggle"} if spec.look_task else {spec.transform}
        can_transform = any(
            action_verb(action) in expected_verbs
            and (spec.look_task or _mentions(action, spec.target_object))
            for action in candidates
        )
        return "TRANSFORM" if can_transform else "SEARCH"
    can_place = any(
        action_option(action) == "PLACE"
        and _mentions(action, spec.target_object)
        and _mentions(action, spec.destination)
        for action in candidates
    )
    return "PLACE" if can_place else "SEARCH"


def completion_label(
    *,
    goal: str,
    before_native_actions: Sequence[str],
    action_history: Sequence[str],
    action: str,
    after_native_actions: Sequence[str],
    official_success_after: bool,
) -> int:
    if official_success_after:
        return 1
    required_before = infer_required_option(
        goal=goal, native_actions=before_native_actions, action_history=action_history,
    )
    required_after = infer_required_option(
        goal=goal,
        native_actions=after_native_actions,
        action_history=(*action_history, action),
    )
    if required_before == "SEARCH":
        return int(required_after != "SEARCH")
    return int(action_option(action) == required_before and required_after != required_before)


def goal_binding_label(goal: str, action: str) -> int:
    return int(_mentions(action, parse_goal(goal).target_object))


def _hash_bin(value: str, bins: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % bins


def grounder_features(
    *,
    goal: str,
    observation: str,
    action: str,
    required_option: str,
    step: int,
    action_history: Sequence[str],
    feature_bins: int,
    mask_required_option: bool = False,
) -> np.ndarray:
    if required_option not in OPTION_NAMES:
        raise ValueError("unknown required option")
    spec = parse_goal(goal)
    option = action_option(action)
    action_values = tokens(action)
    action_set = set(action_values)
    observation_set = set(tokens(observation))
    goal_set = set(tokens(goal))
    verb = action_verb(action)
    repeat = sum(previous == action for previous in action_history)
    status = workflow_status(goal, action_history)
    lexical = np.zeros(feature_bins, dtype=np.float64)
    for token in action_values:
        lexical[_hash_bin(f"action:{token}", feature_bins)] += 1.0
        if token in goal_set:
            lexical[_hash_bin(f"goal-action:{token}", feature_bins)] += 1.0
        if token in observation_set:
            lexical[_hash_bin(f"obs-action:{token}", feature_bins)] += 1.0
    norm = float(np.linalg.norm(lexical))
    if norm:
        lexical /= norm
    scalars = (
        float(option == required_option),
        float(spec.target_object in action_set),
        float(spec.destination in action_set),
        float(spec.target_object in observation_set),
        float(spec.destination in observation_set),
        len(action_set & goal_set) / max(1, len(action_set)),
        len(action_set & observation_set) / max(1, len(action_set)),
        min(repeat, 8) / 8.0,
        float(bool(action_history) and action_verb(action_history[-1]) == verb),
        min(step, 180) / 180.0,
        status.progress_fraction,
        float(status.held),
        float(status.transformed),
        float(spec.count == 2),
        float(spec.transform == "clean"),
        float(spec.transform == "heat"),
        float(spec.transform == "cool"),
        float(spec.transform == "light"),
    )
    return np.asarray((
        *(float(option == name) for name in OPTION_NAMES),
        *(
            (0.0 for _ in OPTION_NAMES)
            if mask_required_option
            else (float(required_option == name) for name in OPTION_NAMES)
        ),
        *( (0.0, *scalars[1:]) if mask_required_option else scalars ),
        *lexical.tolist(),
    ), dtype=np.float64)


def mlp_probability(features: np.ndarray, artifact: Mapping[str, Any]) -> float:
    value = np.asarray(features, dtype=np.float64)
    for layer_index, layer in enumerate(artifact["layers"]):
        weights = np.asarray(layer["weights"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        value = value @ weights + bias
        if layer_index < len(artifact["layers"]) - 1:
            activation = str(artifact["hidden_activation"])
            value = np.tanh(value) if activation == "tanh" else np.maximum(value, 0.0)
    scalar = float(np.ravel(value)[0])
    if scalar >= 0:
        return 1.0 / (1.0 + math.exp(-scalar))
    exponential = math.exp(scalar)
    return exponential / (1.0 + exponential)


def score_actions(
    *,
    goal: str,
    observation: str,
    native_actions: Sequence[str],
    step: int,
    action_history: Sequence[str],
    artifact: Mapping[str, Any],
) -> dict[str, dict[str, float | str]]:
    required = infer_required_option(
        goal=goal, native_actions=native_actions, action_history=action_history,
    )
    result = {}
    for action in native_actions:
        option = action_option(str(action))
        if option == "EXCLUDE":
            continue
        features = grounder_features(
            goal=goal,
            observation=observation,
            action=str(action),
            required_option=required,
            step=step,
            action_history=action_history,
            feature_bins=int(artifact["feature_bins"]),
        )
        result[str(action)] = {
            "option": option,
            "required_option": required,
            "applicability": mlp_probability(features, artifact["applicability_head"]),
            "binding": mlp_probability(features, artifact["binding_head"]),
            "completion": mlp_probability(features, artifact["completion_head"]),
        }
        if "policy_head" in artifact:
            policy_features = grounder_features(
                goal=goal,
                observation=observation,
                action=str(action),
                required_option=required,
                step=step,
                action_history=action_history,
                feature_bins=int(artifact["feature_bins"]),
                mask_required_option=True,
            )
            result[str(action)]["policy"] = mlp_probability(
                policy_features, artifact["policy_head"],
            )
    return result


__all__ = [
    "GoalSpec",
    "OPTION_NAMES",
    "WorkflowStatus",
    "action_option",
    "completion_label",
    "goal_binding_label",
    "grounder_features",
    "infer_required_option",
    "mlp_probability",
    "parse_goal",
    "score_actions",
    "workflow_status",
]
