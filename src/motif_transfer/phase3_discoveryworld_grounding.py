"""Outcome-blind target-native grounding qualification for DiscoveryWorld.

The Phase-2 target actor used Qwen JSON mode.  Most schema fallbacks were not
reasoning errors: the provider returned ``None`` or malformed numeric strings.
This module keeps the old policy observation and native validator intact, but
adds an exact, observation-derived action catalog and a strict retry contract.
It never reads an evaluator scorecard or source-game artifact.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .discoveryworld_env import DiscoveryWorldObservation
from .discoveryworld_policy import (
    TARGET_ONLY_SYSTEM_PROMPT,
    native_action_from_decision,
    parse_json_object,
    prompt_payload,
)


PHASE3_GROUNDER_SYSTEM_PROMPT = TARGET_ONLY_SYSTEM_PROMPT + """

The payload includes exact_action_catalog. Treat it as the authoritative action
schema for this turn. Copy enum values and integer UUIDs exactly. Do not choose
an action whose required argument is absent from its allowed values. Return a
JSON object even if uncertain; use a safe catalog action and explain uncertainty
in reason. Do not output null, a number, an array, Markdown, or hidden reasoning.
For USE, both objects must be co-present: each UUID must currently be accessible
or in inventory. If a portable tool must be used at spatially separated targets,
first PICKUP the accessible tool, then navigate to the target; never attempt USE
with a stale or merely visible UUID.
"""

PHASE3_REPAIR_SYSTEM_PROMPT = """You repair one rejected DiscoveryWorld action.
Use only the current exact_action_catalog and current target_native_facts. The
previous action is invalid now. Return one replacement JSON action and the same
memory fields requested by response_contract. Never reuse an argument absent
from the current catalog. If USE failed catalog validation, return a prerequisite
such as PICKUP of an accessible portable tool or TELEPORT_TO_OBJECT; do not swap
USE arguments or repeat another USE whose UUID is absent. Do not return prose,
Markdown, null, or an array."""

PHASE3_AFFORDANCE_SYSTEM_PROMPT = """You are an outcome-blind DiscoveryWorld
native-action affordance checker. Decide whether the proposed manipulation has
an obvious physical/interaction precondition error using only the current
observed object names/descriptions and action catalog. Floors, walls, tables,
statues, animals, and plants are not portable PICKUP targets, but a named tool,
meter, or instrument is a plausible portable target. For USE, arg1 must be a
plausible usable tool or instrument; an animal, plant, or other specimen is a
plausible arg2 measurement target. Reject only when arg1 is not a tool or the
pair is otherwise nonsensical. Do not substitute ACTIVATE/DEACTIVATE for USE on
a meter unless its description explicitly says it is activatable. Do not predict task
success or optimize the plan.
Return exactly one JSON object:
{\"has_obvious_error\":true|false,\"reason\":\"...\"}."""

_AFFORDANCE_CHECK_ACTIONS = frozenset({
    "PICKUP", "DROP", "PUT", "OPEN", "CLOSE", "ACTIVATE", "DEACTIVATE",
    "TALK", "EAT", "READ", "USE",
})


def _integer_uuids(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({
        int(row["uuid"]) for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("uuid"), int)
        and not isinstance(row.get("uuid"), bool)
    })


def exact_action_catalog(
    observation: DiscoveryWorldObservation,
) -> dict[str, Any]:
    """Build exact legal argument domains from the current policy observation."""

    if observation.in_dialog:
        options = observation.ui.get("dialogOptions") or observation.ui.get("dialog_options") or ()
        return {
            "mode": "DIALOG",
            "chosen_dialog_option_int": list(range(len(options))) if options else [0],
        }
    ui = observation.ui
    inventory_rows = [
        row for row in (ui.get("inventoryObjects") or ())
        if isinstance(row, Mapping)
    ]
    accessible_rows = [
        row for row in (ui.get("accessibleEnvironmentObjects") or ())
        if isinstance(row, Mapping)
    ]
    inventory = _integer_uuids(inventory_rows)
    accessible = _integer_uuids(accessible_rows)
    visible_rows = [*inventory_rows, *accessible_rows]
    nearby = ui.get("nearbyObjects") or {}
    directional = nearby.get("objects") if isinstance(nearby, Mapping) else {}
    if isinstance(directional, Mapping):
        for rows in directional.values():
            if isinstance(rows, (list, tuple)):
                visible_rows.extend(row for row in rows if isinstance(row, Mapping))
    visible = _integer_uuids(visible_rows)
    known = observation.known_actions
    domains: dict[str, Any] = {}
    for name in sorted(known):
        specification = known[name]
        if not isinstance(specification, Mapping):
            continue
        required = [str(value) for value in (specification.get("args") or ())]
        row: dict[str, Any] = {"required_args": required}
        if name == "MOVE_DIRECTION":
            row["arg1_allowed"] = sorted(
                (ui.get("agentLocation") or {}).get("directions_you_can_move", ())
            )
        elif name == "ROTATE_DIRECTION":
            row["arg1_allowed"] = ["north", "east", "south", "west"]
        elif name == "TELEPORT_TO_LOCATION":
            row["arg1_allowed"] = sorted(observation.teleport_locations)
        elif name == "TELEPORT_TO_OBJECT":
            row["arg1_allowed"] = visible
        elif name == "PICKUP":
            row["arg1_allowed"] = accessible
        elif name in {"DROP", "EAT"}:
            row["arg1_allowed"] = inventory
        elif name == "PUT":
            row["arg1_allowed"] = inventory
            row["arg2_allowed"] = sorted(set(accessible) | set(inventory))
        elif name in {"OPEN", "CLOSE", "ACTIVATE", "DEACTIVATE", "TALK", "READ"}:
            row["arg1_allowed"] = accessible
        elif name == "USE":
            allowed = sorted(set(accessible) | set(inventory))
            row["arg1_allowed"] = allowed
            row["arg2_allowed"] = allowed
        elif name == "DISCOVERY_FEED_GET_POST_BY_ID":
            row["arg1_type"] = "integer_post_id"
        domains[name] = row
    object_facts = {}
    inventory_set = set(inventory)
    accessible_set = set(accessible)
    for row in visible_rows:
        uuid = row.get("uuid")
        if not isinstance(uuid, int) or isinstance(uuid, bool):
            continue
        roles = []
        if uuid in inventory_set:
            roles.append("inventory")
        if uuid in accessible_set:
            roles.append("accessible")
        object_facts[uuid] = {
            "uuid": uuid,
            "name": str(row.get("name") or ""),
            "description": str(row.get("description") or ""),
            "observed_roles": roles or ["nearby"],
        }
    return {
        "mode": "NATIVE_ACTION",
        "actions": domains,
        "inventory_uuids": inventory,
        "accessible_uuids": accessible,
        "visible_uuids": visible,
        "object_facts": [object_facts[key] for key in sorted(object_facts)],
    }


def grounding_prompt_payload(
    observation: DiscoveryWorldObservation,
    *,
    memory: str,
    hypotheses: Sequence[str],
    recent_decisions: Sequence[Mapping[str, Any]],
    schema_error: str | None = None,
) -> dict[str, Any]:
    payload = prompt_payload(
        observation,
        memory=memory,
        hypotheses=hypotheses,
        recent_decisions=recent_decisions,
        schema_error=schema_error,
    )
    payload["exact_action_catalog"] = exact_action_catalog(observation)
    payload["grounding_qualification_contract"] = {
        "selection_reads_official_success": False,
        "selection_reads_evaluator_scorecard": False,
        "source_game_context_present": False,
        "must_copy_action_arguments_from_catalog": True,
    }
    return payload


def grounding_repair_payload(
    observation: DiscoveryWorldObservation,
    *,
    memory: str,
    hypotheses: Sequence[str],
    rejected_response: str,
    validation_error: str,
) -> dict[str, Any]:
    """Build a current-turn-only repair payload without stale action history."""

    full = prompt_payload(
        observation,
        memory=memory,
        hypotheses=hypotheses,
        recent_decisions=(),
        schema_error=validation_error,
    )
    return {
        "target_native_facts": full["target_native_facts"],
        "exact_action_catalog": exact_action_catalog(observation),
        "rejected_response": str(rejected_response)[:4000],
        "validation_error": str(validation_error)[:1000],
        "persistent_memory": str(memory)[-3000:],
        "running_hypotheses": [str(value) for value in hypotheses][-12:],
        "response_contract": full["response_contract"],
        "grounding_qualification_contract": {
            "selection_reads_official_success": False,
            "selection_reads_evaluator_scorecard": False,
            "source_game_context_present": False,
        },
    }


def _neural_affordance_check(
    *, backend, observation: DiscoveryWorldObservation,
    action: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    payload = {
        "proposed_action": dict(action),
        "exact_action_catalog": exact_action_catalog(observation),
        "contract": {
            "reads_official_success": False,
            "reads_evaluator_scorecard": False,
            "checks_only_native_affordance": True,
        },
    }
    raw = backend.complete(
        "decision", PHASE3_AFFORDANCE_SYSTEM_PROMPT, payload,
    )
    usage = json.loads(json.dumps(dict(backend.last_usage or {}), default=str))
    try:
        verdict = parse_json_object(raw)
        has_error = verdict.get("has_obvious_error")
        if not isinstance(has_error, bool):
            raise ValueError(
                "affordance verdict omitted boolean has_obvious_error"
            )
        accept = not has_error
        return accept, {
            "accepted": accept,
            "has_obvious_error": has_error,
            "reason": str(verdict.get("reason") or "")[:1000],
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "usage": usage,
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, {
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "usage": usage,
        }


def call_qualified_decision(
    *,
    backend,
    observation: DiscoveryWorldObservation,
    memory: str,
    hypotheses: tuple[str, ...],
    recent: list[dict[str, Any]],
    attempts: int,
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]], bool]:
    """Return one validated neural action or an explicitly counted fallback."""

    schema_error = None
    rejected_response = ""
    audit = []
    for attempt in range(attempts):
        if schema_error is None:
            payload = grounding_prompt_payload(
                observation,
                memory=memory,
                hypotheses=hypotheses,
                recent_decisions=recent,
            )
            system_prompt = PHASE3_GROUNDER_SYSTEM_PROMPT
        else:
            payload = grounding_repair_payload(
                observation,
                memory=memory,
                hypotheses=hypotheses,
                rejected_response=rejected_response,
                validation_error=schema_error,
            )
            system_prompt = PHASE3_REPAIR_SYSTEM_PROMPT
        raw = backend.complete(
            "decision", system_prompt, payload,
        )
        usage = json.loads(json.dumps(dict(backend.last_usage or {}), default=str))
        affordance_audit = None
        try:
            decision = parse_json_object(raw)
            action = native_action_from_decision(decision, observation)
            if str(action.get("action") or "") in _AFFORDANCE_CHECK_ACTIONS:
                affordance_accept, affordance_audit = _neural_affordance_check(
                    backend=backend,
                    observation=observation,
                    action=action,
                )
                if not affordance_accept:
                    reason = str(affordance_audit.get("reason") or affordance_audit.get("error") or "")
                    raise ValueError(
                        f"neural native-affordance check rejected action: {reason}"
                    )
            audit.append({
                "attempt": attempt + 1,
                "accepted": True,
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "usage": usage,
                "affordance_check": affordance_audit,
            })
            return decision, action, raw, audit, False
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1,
                "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "usage": usage,
                "affordance_check": affordance_audit,
            })
            rejected_response = raw
    # The fallback is a native no-argument observation action.  It is counted
    # as grounding failure and cannot make the qualification gate look better.
    fallback = {"action": "DISCOVERY_FEED_GET_UPDATES"}
    decision = {
        **fallback,
        "memory": memory,
        "running_hypotheses": list(hypotheses),
        "expected_effect": "Read public feed after neural grounding failure.",
        "reason": "PHASE3_GROUNDING_SCHEMA_FALLBACK",
    }
    return decision, fallback, "", audit, True


__all__ = [
    "PHASE3_AFFORDANCE_SYSTEM_PROMPT",
    "PHASE3_GROUNDER_SYSTEM_PROMPT",
    "PHASE3_REPAIR_SYSTEM_PROMPT",
    "call_qualified_decision",
    "exact_action_catalog",
    "grounding_prompt_payload",
    "grounding_repair_payload",
]
