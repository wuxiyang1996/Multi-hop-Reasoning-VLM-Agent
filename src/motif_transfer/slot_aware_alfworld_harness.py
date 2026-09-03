"""Slot-aware target grounding for intervention-grounded source structure."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .alfworld_hierarchical_grounder import action_verb
from .contracts import stable_hash
from .parameterized_alfworld_harness import PROPERTY_CLASSES
from .typed_alfworld_harness import target_effect, validate_typed_effect_ir


CONDITIONS = (
    "target_only",
    "authentic_slot_ir",
    "edge_permuted_ir",
    "property_permuted_router",
)
RECEIPTS = (
    "IGNORE",
    "BIND_INSTANCE",
    "MUTATE_REQUIRED_PROPERTY",
    "RELATE_SLOT_CLOSED",
    "RELATE_NO_PROGRESS",
    "LIGHT_SLOT_CLOSED",
)

_TAKE = re.compile(
    r"^take (?P<object>[a-z]+) (?P<object_id>\d+) from "
    r"(?P<receptacle>[a-z]+) (?P<receptacle_id>\d+)$"
)
_MOVE = re.compile(
    r"^move (?P<object>[a-z]+) (?P<object_id>\d+) to "
    r"(?P<receptacle>[a-z]+) (?P<receptacle_id>\d+)$"
)
_PROPERTY = re.compile(
    r"^(?P<verb>clean|heat|cool) (?P<object>[a-z]+) "
    r"(?P<object_id>\d+) with (?P<tool>[a-z]+) (?P<tool_id>\d+)$"
)
_USE = re.compile(r"^use (?P<tool>[a-z]+) (?P<tool_id>\d+)$")
_VISIBLE_LOCATION = re.compile(
    r"(?:on|in) the (?P<receptacle>[a-z]+) "
    r"(?P<receptacle_id>\d+).*?you see (?P<objects>[^.]+)",
    re.IGNORECASE,
)


def parameterize_slot_source_ir(source_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the real-source effect graph to target-native goal slots."""
    validate_typed_effect_ir(source_ir)
    edges = {
        (str(row["from"]), str(row["to"])): row
        for row in source_ir["edges"]
    }
    mutate = edges[("BIND", "MUTATE")]
    relate = edges[("BIND", "RELATE")]
    body = {
        "schema_version": "slot-aware-real-source-effect-ir-v8",
        "parent_ir_sha256": str(source_ir["ir_sha256"]),
        "induction_split": str(source_ir["induction_split"]),
        "validation_splits": list(source_ir["validation_splits"]),
        "nodes": [
            {
                "effect": "BIND",
                "roles": ["goal_object_instance", "unsatisfied_goal_slot"],
            },
            {
                "effect": "ACHIEVE_UNARY_GOAL",
                "roles": [
                    "goal_object_instance",
                    "required_property",
                    "unsatisfied_goal_slot",
                ],
            },
            {
                "effect": "RELATE",
                "roles": [
                    "goal_object_instance",
                    "goal_receptacle",
                    "unsatisfied_goal_slot",
                ],
            },
        ],
        "edges": [
            {
                "from": "BIND(goal_object_instance, unsatisfied_goal_slot)",
                "to": (
                    "ACHIEVE_UNARY_GOAL(goal_object_instance, "
                    "required_property, unsatisfied_goal_slot)"
                ),
                "guard": "TARGET_NATIVE_SLOT_REQUIRES_UNARY_PROPERTY",
                "supporting_source_tasks": list(
                    mutate["supporting_source_tasks"]
                ),
            },
            {
                "from": "BIND(goal_object_instance, unsatisfied_goal_slot)",
                "to": (
                    "RELATE(goal_object_instance, goal_receptacle, "
                    "unsatisfied_goal_slot)"
                ),
                "guard": "TARGET_NATIVE_SLOT_READY_FOR_RELATION",
                "supporting_source_tasks": list(
                    relate["supporting_source_tasks"]
                ),
            },
            {
                "from": (
                    "ACHIEVE_UNARY_GOAL(goal_object_instance, "
                    "required_property, unsatisfied_goal_slot)"
                ),
                "to": (
                    "RELATE(goal_object_instance, goal_receptacle, "
                    "unsatisfied_goal_slot)"
                ),
                "guard": "OBSERVED_REQUIRED_PROPERTY_POSTCONDITION",
                "supporting_source_tasks": list(
                    mutate["supporting_source_tasks"]
                ),
            },
        ],
        "execution_authority": "SYMBOLIC_EFFECT_ROUTING_ONLY",
        "target_grounding": (
            "TARGET_NATIVE_NEURAL_PROPERTY_AND_ACTION_GROUNDING_WITH_"
            "OBSERVED_SLOT_POSTCONDITIONS"
        ),
        "monitor_state": [
            "goal_object_instance",
            "required_count",
            "observed_properties",
            "observed_target_relation",
            "completed_goal_slots",
        ],
        "prohibited_runtime_fields": sorted(set(map(
            str, source_ir["prohibited_runtime_fields"]
        )) | {
            "source_action_ordinal",
            "environment_id",
            "source_task_id",
            "official_success_for_action_selection",
        }),
        "source_lineage": list(source_ir["source_lineage"]),
    }
    return body | {"ir_sha256": stable_hash(body)}


def validate_slot_source_ir(source_ir: Mapping[str, Any]) -> None:
    body = dict(source_ir)
    claimed = str(body.pop("ir_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError("slot-aware source IR hash mismatch")
    if source_ir.get("schema_version") != "slot-aware-real-source-effect-ir-v8":
        raise ValueError("unsupported slot-aware source IR")
    effects = {str(row["effect"]) for row in source_ir.get("nodes", ())}
    if effects != {"BIND", "ACHIEVE_UNARY_GOAL", "RELATE"}:
        raise ValueError("slot-aware source IR has wrong node set")
    roles = {
        str(role)
        for row in source_ir.get("nodes", ())
        for role in row.get("roles", ())
    }
    if not {
        "goal_object_instance",
        "unsatisfied_goal_slot",
        "required_property",
        "goal_receptacle",
    }.issubset(roles):
        raise ValueError("slot-aware source IR lacks required target roles")


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def parse_goal_spec(goal: str, *, required_property: str) -> dict[str, Any]:
    """Parse ALFWorld target roles without consulting task paths or outcomes."""
    value = _normalized(goal)
    patterns = (
        (
            "RELATE",
            re.compile(
                r"^(?:find|put) two (?P<object>[a-z]+) "
                r"(?:and put them )?(?:in|on) (?P<receptacle>[a-z]+)$"
            ),
            2,
        ),
        (
            "RELATE",
            re.compile(
                r"^(?:clean|heat|cool) (?:some|a|the) (?P<object>[a-z]+) "
                r"and put it (?:in|on) (?P<receptacle>[a-z]+)$"
            ),
            1,
        ),
        (
            "RELATE",
            re.compile(
                r"^put (?:a|some|the) (?:(?:clean|hot|cool|cold) )?"
                r"(?P<object>[a-z]+) (?:in|on) (?P<receptacle>[a-z]+)$"
            ),
            1,
        ),
    )
    for kind, pattern, count in patterns:
        match = pattern.match(value)
        if match:
            return {
                "kind": kind,
                "goal_object_type": match.group("object"),
                "target_receptacle_type": match.group("receptacle"),
                "required_count": count,
                "required_property": required_property,
            }
    look = re.match(
        r"^(?:look at|examine) (?:the )?(?P<object>[a-z]+) "
        r"(?:under|with) (?:the )?(?P<tool>[a-z]+)$",
        value,
    )
    if look:
        return {
            "kind": "LIGHT",
            "goal_object_type": look.group("object"),
            "target_receptacle_type": look.group("tool"),
            "required_count": 1,
            "required_property": required_property,
        }
    raise ValueError(f"unsupported target goal syntax: {goal!r}")


def _object_key(object_type: str, object_id: str | int) -> str:
    return f"{object_type.lower()} {int(object_id)}"


def initialize_slot_ledger(
    goal: str,
    *,
    required_property: str,
    initial_observation: str = "",
) -> dict[str, Any]:
    ledger = {
        "schema_version": "target-native-slot-ledger-v8",
        "goal_spec": parse_goal_spec(
            goal, required_property=required_property
        ),
        "carried_object": None,
        "observed_properties": {},
        "observed_locations": {},
        "completed_objects": [],
        "successful_effect_counts": {
            "BIND": 0,
            "MUTATE": 0,
            "RELATE": 0,
        },
        "reopened_completed_slots": 0,
        "failed_postconditions": 0,
    }
    return reconcile_visible_target_objects(ledger, initial_observation)


def _property_satisfied(ledger: Mapping[str, Any], object_key: str) -> bool:
    required = str(ledger["goal_spec"]["required_property"])
    if required == "NONE":
        return True
    return required in set(map(
        str, ledger["observed_properties"].get(object_key, ())
    ))


def _refresh_completed(ledger: dict[str, Any]) -> None:
    spec = ledger["goal_spec"]
    completed = set(map(str, ledger["completed_objects"]))
    if spec["kind"] == "RELATE":
        for object_key, location in ledger["observed_locations"].items():
            object_type = object_key.split(" ", 1)[0]
            if (
                object_type == spec["goal_object_type"]
                and str(location).split(" ", 1)[0]
                == spec["target_receptacle_type"]
                and _property_satisfied(ledger, object_key)
            ):
                completed.add(object_key)
        completed = {
            object_key for object_key in completed
            if ledger["observed_locations"].get(object_key, "").split(" ", 1)[0]
            == spec["target_receptacle_type"]
            and _property_satisfied(ledger, object_key)
        }
    ledger["completed_objects"] = sorted(completed)


def reconcile_visible_target_objects(
    ledger: Mapping[str, Any], observation: str
) -> dict[str, Any]:
    """Use visible target-native relations as monotone positive evidence."""
    result = deepcopy(dict(ledger))
    spec = result["goal_spec"]
    if spec["kind"] != "RELATE":
        return result
    for match in _VISIBLE_LOCATION.finditer(observation):
        if match.group("receptacle").lower() != spec["target_receptacle_type"]:
            continue
        location = _object_key(
            match.group("receptacle"), match.group("receptacle_id")
        )
        object_pattern = re.compile(
            rf"\b{re.escape(spec['goal_object_type'])} (?P<id>\d+)\b",
            re.IGNORECASE,
        )
        for object_match in object_pattern.finditer(match.group("objects")):
            object_key = _object_key(
                spec["goal_object_type"], object_match.group("id")
            )
            result["observed_locations"][object_key] = location
    _refresh_completed(result)
    return result


def slot_state(ledger: Mapping[str, Any]) -> dict[str, Any]:
    spec = ledger["goal_spec"]
    completed = tuple(map(str, ledger["completed_objects"]))
    required_count = int(spec["required_count"])
    return {
        "goal_object_type": str(spec["goal_object_type"]),
        "target_receptacle_type": str(spec["target_receptacle_type"]),
        "required_property": str(spec["required_property"]),
        "required_count": required_count,
        "completed_count": min(len(completed), required_count),
        "remaining_slots": max(required_count - len(completed), 0),
        "completed_objects": list(completed),
        "carried_object": ledger.get("carried_object"),
        "reopened_completed_slots": int(
            ledger.get("reopened_completed_slots", 0)
        ),
    }


def _successful_postcondition(action: str, after_observation: str) -> bool:
    action_value = _normalized(action)
    after = _normalized(after_observation)
    take = _TAKE.match(action_value)
    if take:
        expected = (
            f"you pick up the {take.group('object')} {take.group('object_id')} "
            f"from the {take.group('receptacle')} {take.group('receptacle_id')}"
        )
        return expected in after
    move = _MOVE.match(action_value)
    if move:
        expected = (
            f"you move the {move.group('object')} {move.group('object_id')} "
            f"to the {move.group('receptacle')} {move.group('receptacle_id')}"
        )
        return expected in after
    mutation = _PROPERTY.match(action_value)
    if mutation:
        expected = (
            f"you {mutation.group('verb')} the {mutation.group('object')} "
            f"{mutation.group('object_id')} using the {mutation.group('tool')} "
            f"{mutation.group('tool_id')}"
        )
        return expected in after
    use = _USE.match(action_value)
    if use:
        expected = (
            f"you turn on the {use.group('tool')} {use.group('tool_id')}"
        )
        return expected in after
    return False


def observe_target_transition(
    ledger: Mapping[str, Any],
    *,
    action: str,
    after_observation: str,
) -> tuple[dict[str, Any], str]:
    """Advance target state only after an observed native postcondition."""
    result = deepcopy(dict(ledger))
    action_value = _normalized(action)
    if not _successful_postcondition(action_value, after_observation):
        if action_verb(action_value) in {
            "take", "move", "clean", "heat", "cool", "use"
        }:
            result["failed_postconditions"] += 1
        return result, "IGNORE"
    spec = result["goal_spec"]
    completed_before = set(map(str, result["completed_objects"]))
    take = _TAKE.match(action_value)
    if take:
        object_key = _object_key(take.group("object"), take.group("object_id"))
        result["carried_object"] = object_key
        result["observed_locations"].pop(object_key, None)
        if object_key in completed_before:
            result["reopened_completed_slots"] += 1
            result["completed_objects"] = sorted(
                completed_before - {object_key}
            )
        if take.group("object") == spec["goal_object_type"]:
            result["successful_effect_counts"]["BIND"] += 1
            return result, "BIND_INSTANCE"
        return result, "IGNORE"
    mutation = _PROPERTY.match(action_value)
    if mutation:
        object_key = _object_key(
            mutation.group("object"), mutation.group("object_id")
        )
        property_name = mutation.group("verb").upper()
        properties = set(map(
            str, result["observed_properties"].get(object_key, ())
        ))
        properties.add(property_name)
        result["observed_properties"][object_key] = sorted(properties)
        _refresh_completed(result)
        if (
            object_key == result.get("carried_object")
            and mutation.group("object") == spec["goal_object_type"]
            and property_name == spec["required_property"]
        ):
            result["successful_effect_counts"]["MUTATE"] += 1
            return result, "MUTATE_REQUIRED_PROPERTY"
        return result, "IGNORE"
    use = _USE.match(action_value)
    if use:
        carried = result.get("carried_object")
        if (
            spec["kind"] == "LIGHT"
            and carried
            and str(carried).split(" ", 1)[0] == spec["goal_object_type"]
            and use.group("tool") == spec["target_receptacle_type"]
        ):
            properties = set(map(
                str, result["observed_properties"].get(carried, ())
            ))
            properties.add("LIGHT")
            result["observed_properties"][carried] = sorted(properties)
            result["completed_objects"] = [str(carried)]
            result["successful_effect_counts"]["MUTATE"] += 1
            return result, "LIGHT_SLOT_CLOSED"
        return result, "IGNORE"
    move = _MOVE.match(action_value)
    if move:
        object_key = _object_key(move.group("object"), move.group("object_id"))
        location = _object_key(
            move.group("receptacle"), move.group("receptacle_id")
        )
        result["carried_object"] = None
        result["observed_locations"][object_key] = location
        _refresh_completed(result)
        completed_after = set(map(str, result["completed_objects"]))
        if object_key in completed_after - completed_before:
            result["successful_effect_counts"]["RELATE"] += 1
            return result, "RELATE_SLOT_CLOSED"
        if move.group("object") == spec["goal_object_type"]:
            return result, "RELATE_NO_PROGRESS"
    return result, "IGNORE"


def _policy_score(
    action: str, row: Mapping[str, Any], history: Sequence[str]
) -> float:
    return float(row.get("policy", row["applicability"])) / (
        1.0 + history.count(action)
    )


def _realization_score(
    action: str, row: Mapping[str, Any], history: Sequence[str]
) -> float:
    return (
        float(row["applicability"])
        * (0.20 + 0.80 * float(row["completion"]))
        * (0.25 + 0.75 * float(row["binding"]))
        / (1.0 + history.count(action))
    )


def _permuted_property(property_name: str) -> str:
    permutation = {
        "NONE": "CLEAN",
        "CLEAN": "HEAT",
        "HEAT": "COOL",
        "COOL": "LIGHT",
        "LIGHT": "NONE",
    }
    return permutation[property_name]


def condition_required_property(
    property_probabilities: Mapping[str, float], condition: str
) -> tuple[str, str, float]:
    authentic = max(
        PROPERTY_CLASSES,
        key=lambda name: (float(property_probabilities.get(name, 0.0)), name),
    )
    required = (
        _permuted_property(authentic)
        if condition == "property_permuted_router"
        else authentic
    )
    return required, authentic, float(property_probabilities.get(authentic, 0.0))


def _requested_effect(ledger: Mapping[str, Any]) -> str | None:
    state = slot_state(ledger)
    spec = ledger["goal_spec"]
    carried = state["carried_object"]
    if not carried:
        return "BIND" if state["remaining_slots"] > 0 else None
    if str(carried).split(" ", 1)[0] != spec["goal_object_type"]:
        return None
    if not _property_satisfied(ledger, str(carried)):
        return "MUTATE"
    if spec["kind"] == "LIGHT":
        return None
    return "RELATE"


def _action_matches_effect(
    action: str,
    *,
    effect: str,
    ledger: Mapping[str, Any],
) -> tuple[bool, str | None]:
    value = _normalized(action)
    spec = ledger["goal_spec"]
    completed = set(map(str, ledger["completed_objects"]))
    if effect == "BIND":
        match = _TAKE.match(value)
        if not match or match.group("object") != spec["goal_object_type"]:
            return False, None
        object_key = _object_key(match.group("object"), match.group("object_id"))
        from_target = match.group("receptacle") == spec["target_receptacle_type"]
        protected = object_key in completed or (
            spec["required_property"] == "NONE" and from_target
        )
        return not protected, object_key
    if effect == "MUTATE":
        carried = ledger.get("carried_object")
        if not carried:
            return False, None
        if spec["kind"] == "LIGHT":
            match = _USE.match(value)
            return bool(
                match and match.group("tool") == spec["target_receptacle_type"]
            ), str(carried)
        match = _PROPERTY.match(value)
        if not match:
            return False, None
        object_key = _object_key(match.group("object"), match.group("object_id"))
        return bool(
            object_key == carried
            and match.group("verb").upper() == spec["required_property"]
        ), object_key
    if effect == "RELATE":
        match = _MOVE.match(value)
        if not match:
            return False, None
        object_key = _object_key(match.group("object"), match.group("object_id"))
        return bool(
            object_key == ledger.get("carried_object")
            and match.group("receptacle") == spec["target_receptacle_type"]
            and _property_satisfied(ledger, object_key)
        ), object_key
    return False, None


def _would_reopen_completed_slot(
    action: str, ledger: Mapping[str, Any]
) -> bool:
    match = _TAKE.match(_normalized(action))
    if not match:
        return False
    spec = ledger["goal_spec"]
    if match.group("object") != spec["goal_object_type"]:
        return False
    object_key = _object_key(match.group("object"), match.group("object_id"))
    if object_key in set(map(str, ledger["completed_objects"])):
        return True
    return bool(
        spec["required_property"] == "NONE"
        and match.group("receptacle") == spec["target_receptacle_type"]
    )


def choose_slot_aware_action(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    history: Sequence[str],
    ledger: Mapping[str, Any],
    source_ir: Mapping[str, Any],
    property_probabilities: Mapping[str, float],
    minimum_property_confidence: float,
    minimum_role_binding: float,
    minimum_realization_score: float,
    minimum_target_policy_ratio: float,
    allowed_source_effects: Sequence[str] = ("BIND", "MUTATE", "RELATE"),
    active_required_properties: Sequence[str] = PROPERTY_CLASSES,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown slot-aware condition: {condition}")
    if not grounded:
        raise ValueError("slot-aware Harness received no target-native actions")
    validate_slot_source_ir(source_ir)
    required, authentic, confidence = condition_required_property(
        property_probabilities, condition
    )
    if str(ledger["goal_spec"]["required_property"]) != required:
        raise ValueError("slot ledger property does not match Harness condition")
    allowed_effects = set(map(str, allowed_source_effects))
    if not allowed_effects or not allowed_effects.issubset(
        {"BIND", "MUTATE", "RELATE"}
    ):
        raise ValueError("invalid slot-aware source-effect scope")
    scope_active = required in set(map(str, active_required_properties))
    raw_fallback = max(
        grounded,
        key=lambda action: (
            _policy_score(action, grounded[action], history), action
        ),
    )
    safe_actions = [
        action for action in grounded
        if not _would_reopen_completed_slot(action, ledger)
    ]
    safe_fallback = max(
        safe_actions or list(grounded),
        key=lambda action: (
            _policy_score(action, grounded[action], history), action
        ),
    )
    safety_enabled = bool(
        condition != "target_only"
        and scope_active
        and confidence >= minimum_property_confidence
    )
    fallback = safe_fallback if safety_enabled else raw_fallback
    fallback_effect = target_effect(str(grounded[fallback]["option"]))
    state = slot_state(ledger)
    common = {
        "fallback_action": fallback,
        "raw_target_fallback_action": raw_fallback,
        "slot_safety_shielded": fallback != raw_fallback,
        "slot_safety_enabled": safety_enabled,
        "transfer_scope_active": scope_active,
        "allowed_source_effects": sorted(allowed_effects),
        "fallback_effect": fallback_effect,
        "slot_state": state,
        "required_property": required,
        "authentic_required_property": authentic,
        "property_confidence": confidence,
    }
    if condition == "target_only":
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_ONLY",
        }
    if confidence < minimum_property_confidence:
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_PROPERTY_ROUTER_ABSTAINED",
        }
    if not scope_active:
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TRANSFER_SCOPE_TARGET_ABSTENTION",
        }
    requested = _requested_effect(ledger)
    if requested is None:
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "requested_source_effect": None,
            "diagnostic": "NO_UNSATISFIED_SLOT_TARGET_ABSTENTION",
        }
    if condition == "edge_permuted_ir":
        requested = {
            "BIND": "MUTATE",
            "MUTATE": "RELATE",
            "RELATE": "BIND",
        }[requested]
    if requested not in allowed_effects:
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "requested_source_effect": requested,
            "diagnostic": "UNCLAIMED_SOURCE_EFFECT_TARGET_ABSTENTION",
        }
    candidates = []
    protected = 0
    for action, row in grounded.items():
        effect = target_effect(str(row["option"]))
        if effect != requested:
            continue
        matches, _ = _action_matches_effect(
            action, effect=requested, ledger=ledger
        )
        if not matches:
            protected += 1
            continue
        if float(row["binding"]) < minimum_role_binding:
            continue
        candidates.append(action)
    candidate_common = common | {
        "requested_source_effect": requested,
        "protected_or_incompatible_candidates": protected,
    }
    if not candidates:
        return candidate_common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "SLOT_EFFECT_UNAVAILABLE_TARGET_ABSTENTION",
        }
    selected = max(
        candidates,
        key=lambda action: (
            _policy_score(action, grounded[action], history),
            _realization_score(action, grounded[action], history),
            action,
        ),
    )
    selected_effect = target_effect(str(grounded[selected]["option"]))
    realization = _realization_score(selected, grounded[selected], history)
    selected_policy = _policy_score(selected, grounded[selected], history)
    fallback_policy = _policy_score(fallback, grounded[fallback], history)
    ratio = selected_policy / max(fallback_policy, 1e-12)
    scored = candidate_common | {
        "best_realization_score": realization,
        "selected_target_policy_score": selected_policy,
        "fallback_target_policy_score": fallback_policy,
        "target_policy_ratio": ratio,
    }
    if realization < minimum_realization_score:
        diagnostic = "TARGET_REALIZATION_SCORE_ABSTENTION"
    elif ratio < minimum_target_policy_ratio:
        diagnostic = "TARGET_POLICY_RELATIVE_ABSTENTION"
    else:
        diagnostic = None
    if diagnostic:
        return scored | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": diagnostic,
        }
    return scored | {
        "action": selected,
        "target_realized_effect": selected_effect,
        "source_selected_effect": selected_effect,
        "source_admitted": True,
        "changed_action": selected != fallback,
        "changed_effect": selected_effect != fallback_effect,
        "diagnostic": "SLOT_AWARE_SOURCE_EFFECT_TARGET_NEURAL_REALIZATION",
    }


__all__ = [
    "CONDITIONS",
    "RECEIPTS",
    "choose_slot_aware_action",
    "condition_required_property",
    "initialize_slot_ledger",
    "observe_target_transition",
    "parameterize_slot_source_ir",
    "parse_goal_spec",
    "reconcile_visible_target_objects",
    "slot_state",
    "validate_slot_source_ir",
]
