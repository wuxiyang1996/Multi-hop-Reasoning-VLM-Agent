"""Oracle-free target-only policy protocol for DiscoveryWorld."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .discoveryworld_env import DiscoveryWorldObservation


TARGET_ONLY_SYSTEM_PROMPT = """You are a target-native DiscoveryWorld scientific agent.
Complete the stated task by observing the world, manipulating native objects, making
measurements, testing hypotheses, and acting on the results. Use only the supplied
policy observation. You never receive the hidden evaluator scorecard, critical
questions, critical hypotheses, or any source-game advice.

Choose exactly one native action per turn. Objects can be manipulated only when they
are in inventory or accessibleEnvironmentObjects. Use TELEPORT_TO_LOCATION for named
locations and TELEPORT_TO_OBJECT with a visible UUID to reduce navigation errors.
PICKUP requires an exact UUID in accessible objects; DROP/EAT require an inventory
UUID; other object actions require exact inventory/accessibility UUIDs. If a desired
object is only nearby, move or teleport instead of attempting the object action.
Nearby direction/distance is an exact relation, but an object's coordinates are not
given. Never invent object coordinates from the agent location. TELEPORT_TO_OBJECT
lands beside an object; after teleporting, re-read its target_native_facts relation
before moving. For a requirement such as "west of X", X appearing one tile east means
the agent is already at the required location.
If an action failed, diagnose the precondition rather than repeating it unchanged.
Maintain compact factual memory, distinguish observations from hypotheses, and state
the visible effect expected from the next action.

Return one JSON object. Outside dialog use:
{"action":"...", "arg1":..., "arg2":..., "memory":"...",
 "running_hypotheses":["..."], "expected_effect":"...", "reason":"..."}
Include only arguments required by the selected action. During dialog return
chosen_dialog_option_int instead of action/arg fields, plus the same memory fields.
Do not wrap JSON in prose or Markdown."""


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("DiscoveryWorld decision must be a JSON object")
    return value


def native_action_from_decision(
    decision: Mapping[str, Any], observation: DiscoveryWorldObservation,
) -> dict[str, Any]:
    if observation.in_dialog:
        if "chosen_dialog_option_int" not in decision:
            raise ValueError("dialog decision omitted chosen_dialog_option_int")
        choice = decision["chosen_dialog_option_int"]
        if not isinstance(choice, int) or isinstance(choice, bool) or choice < 0:
            raise ValueError("chosen_dialog_option_int must be a nonnegative integer")
        return {"chosen_dialog_option_int": choice}
    name = str(decision.get("action") or "")
    specification = observation.known_actions.get(name)
    if not isinstance(specification, Mapping):
        raise ValueError(f"decision selected unknown action: {name!r}")
    required = tuple(str(key) for key in specification.get("args") or ())
    missing = [key for key in required if key not in decision]
    if missing:
        raise ValueError(f"decision omitted required action arguments: {missing}")
    action = {key: decision[key] for key in ("action", *required)}
    # The official API requires integer UUIDs but models occasionally serialize
    # them as decimal JSON strings. This is protocol normalization, not object
    # name resolution: non-numeric names remain invalid and trigger repair.
    for key in required:
        value = action[key]
        if isinstance(value, str) and value.strip().isdigit():
            action[key] = int(value.strip())
    validate_target_native_preconditions(action, observation)
    return action


def _object_uuids(observation: DiscoveryWorldObservation) -> tuple[set[int], set[int], set[int]]:
    ui = observation.ui
    inventory = {
        int(row["uuid"])
        for row in (ui.get("inventoryObjects") or ())
        if isinstance(row, Mapping) and isinstance(row.get("uuid"), int)
    }
    accessible = {
        int(row["uuid"])
        for row in (ui.get("accessibleEnvironmentObjects") or ())
        if isinstance(row, Mapping) and isinstance(row.get("uuid"), int)
    }
    visible = set(inventory) | set(accessible)
    nearby = ui.get("nearbyObjects") or {}
    directional = nearby.get("objects") if isinstance(nearby, Mapping) else {}
    if isinstance(directional, Mapping):
        for rows in directional.values():
            if isinstance(rows, (list, tuple)):
                visible.update(
                    int(row["uuid"])
                    for row in rows
                    if isinstance(row, Mapping) and isinstance(row.get("uuid"), int)
                )
    return inventory, accessible, visible


def validate_target_native_preconditions(
    action: Mapping[str, Any], observation: DiscoveryWorldObservation,
) -> None:
    """Reject clear protocol/precondition errors before they consume a world tick."""

    name = str(action.get("action") or "")
    inventory, accessible, visible = _object_uuids(observation)
    object_arguments = {
        "PICKUP": ("arg1",), "DROP": ("arg1",), "PUT": ("arg1", "arg2"),
        "OPEN": ("arg1",), "CLOSE": ("arg1",), "ACTIVATE": ("arg1",),
        "DEACTIVATE": ("arg1",), "TALK": ("arg1",), "EAT": ("arg1",),
        "READ": ("arg1",), "USE": ("arg1", "arg2"),
        "TELEPORT_TO_OBJECT": ("arg1",),
    }
    for key in object_arguments.get(name, ()):
        value = action.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name}.{key} must be an integer UUID, not an object name")
    if name == "PICKUP" and action["arg1"] not in accessible:
        raise ValueError("PICKUP arg1 is not in accessibleEnvironmentObjects")
    if name in {"DROP", "EAT"} and action["arg1"] not in inventory:
        raise ValueError(f"{name} arg1 is not in inventoryObjects")
    if name == "PUT":
        if action["arg1"] not in inventory:
            raise ValueError("PUT arg1 is not in inventoryObjects")
        if action["arg2"] not in accessible | inventory:
            raise ValueError("PUT arg2 is not accessible")
    if name in {"OPEN", "CLOSE", "ACTIVATE", "DEACTIVATE", "TALK", "READ"}:
        if action["arg1"] not in accessible:
            raise ValueError(f"{name} arg1 is not in accessibleEnvironmentObjects")
    if name == "USE":
        if action["arg1"] not in accessible | inventory:
            raise ValueError("USE arg1 is not accessible or in inventory")
        if action["arg2"] not in accessible | inventory:
            raise ValueError("USE arg2 is not accessible or in inventory")
    if name == "TELEPORT_TO_OBJECT" and action["arg1"] not in visible:
        raise ValueError("TELEPORT_TO_OBJECT arg1 is not currently visible")
    if name == "MOVE_DIRECTION":
        available = set((observation.ui.get("agentLocation") or {}).get(
            "directions_you_can_move", ())
        )
        if action.get("arg1") not in available:
            raise ValueError(f"MOVE_DIRECTION arg1 is not currently movable: {sorted(available)}")
    if name == "ROTATE_DIRECTION" and action.get("arg1") not in {
        "north", "east", "south", "west",
    }:
        raise ValueError("ROTATE_DIRECTION arg1 must be cardinal")
    if name == "TELEPORT_TO_LOCATION" and action.get("arg1") not in observation.teleport_locations:
        raise ValueError("TELEPORT_TO_LOCATION arg1 is not a listed location")
    if name == "DISCOVERY_FEED_GET_POST_BY_ID":
        value = action.get("arg1")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("DISCOVERY_FEED_GET_POST_BY_ID arg1 must be an integer")


def target_native_facts(observation: DiscoveryWorldObservation) -> dict[str, Any]:
    """Extract compact, deterministic spatial and object relations from the UI."""

    ui = observation.ui
    nearby = ui.get("nearbyObjects") or {}
    directional = nearby.get("objects") if isinstance(nearby, Mapping) else {}
    salient = []
    ignored = {"floor", "grass", "wall", "path", "air", "sand"}
    if isinstance(directional, Mapping):
        for relation, rows in directional.items():
            if not isinstance(rows, (list, tuple)):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("name") or "")
                if name.lower() in ignored:
                    continue
                salient.append({
                    "relation_from_agent": str(relation),
                    "distance": row.get("distance"),
                    "uuid": row.get("uuid"),
                    "name": name,
                })
    salient.sort(key=lambda row: (
        int(row["distance"]) if isinstance(row.get("distance"), int) else 10**9,
        str(row["relation_from_agent"]),
        int(row["uuid"]) if isinstance(row.get("uuid"), int) else 10**9,
    ))
    return {
        "agent_location": dict(ui.get("agentLocation") or {}),
        "inventory": [dict(row) for row in (ui.get("inventoryObjects") or ())],
        "accessible_objects": [
            dict(row) for row in (ui.get("accessibleEnvironmentObjects") or ())
        ],
        "salient_relative_objects": salient,
        "task_progress": [dict(row) for row in (ui.get("taskProgress") or ())],
        "teleport_location_names": sorted(observation.teleport_locations),
        "last_action_message": str(ui.get("lastActionMessage") or ""),
    }


def prompt_payload(
    observation: DiscoveryWorldObservation,
    *,
    memory: str,
    hypotheses: Sequence[str],
    recent_decisions: Sequence[Mapping[str, Any]],
    schema_error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "policy_observation": observation.policy_payload(),
        "target_native_facts": target_native_facts(observation),
        "persistent_memory": str(memory)[-6000:],
        "running_hypotheses": [str(value) for value in hypotheses][-24:],
        "recent_decisions": [dict(row) for row in recent_decisions[-3:]],
        "response_contract": (
            "one JSON native action plus memory, running_hypotheses, "
            "expected_effect, and reason"
        ),
    }
    if schema_error:
        payload["previous_response_rejected"] = str(schema_error)[:1000]
    return payload


def updated_memory(
    decision: Mapping[str, Any], previous_memory: str, previous_hypotheses: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    memory = str(decision.get("memory") or previous_memory).strip()
    raw_hypotheses = decision.get("running_hypotheses", previous_hypotheses)
    if isinstance(raw_hypotheses, str):
        hypotheses = (raw_hypotheses.strip(),) if raw_hypotheses.strip() else ()
    elif isinstance(raw_hypotheses, (list, tuple)):
        hypotheses = tuple(str(value).strip() for value in raw_hypotheses if str(value).strip())
    else:
        raise ValueError("running_hypotheses must be a string or list")
    return memory[-6000:], hypotheses[-24:]


__all__ = [
    "TARGET_ONLY_SYSTEM_PROMPT",
    "native_action_from_decision",
    "parse_json_object",
    "prompt_payload",
    "target_native_facts",
    "updated_memory",
    "validate_target_native_preconditions",
]
