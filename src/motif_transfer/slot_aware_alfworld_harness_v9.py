"""Execute source-game symbolic edges with target-native ALFWorld grounding.

V8 carried a hash-bound source IR but derived its requested effect from a
target-coded state machine.  V9 makes the source graph operational: target
state selects a source guard, and the matching source edge supplies the next
effect.  Target code still owns goal-role parsing, native-action grounding,
postcondition monitoring, and safety.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .parameterized_alfworld_harness import PROPERTY_CLASSES
from .slot_aware_alfworld_harness import (
    _action_matches_effect,
    _normalized,
    _policy_score,
    _property_satisfied,
    _realization_score,
    _would_reopen_completed_slot,
    slot_state,
    validate_slot_source_ir,
)
from .typed_alfworld_harness import target_effect


CONDITIONS = (
    "target_only",
    "authentic_slot_ir",
    "edge_permuted_ir",
    "property_permuted_router",
)

_GUARD_REQUIRES_PROPERTY = "TARGET_NATIVE_SLOT_REQUIRES_UNARY_PROPERTY"
_GUARD_READY_FOR_RELATION = "TARGET_NATIVE_SLOT_READY_FOR_RELATION"
_GUARD_PROPERTY_OBSERVED = "OBSERVED_REQUIRED_PROPERTY_POSTCONDITION"
_SUPPORTED_GUARDS = frozenset({
    _GUARD_REQUIRES_PROPERTY,
    _GUARD_READY_FOR_RELATION,
    _GUARD_PROPERTY_OBSERVED,
})


def _node_effect(value: str) -> str:
    name = str(value).split("(", 1)[0]
    if name == "ACHIEVE_UNARY_GOAL":
        return "MUTATE"
    if name in {"BIND", "RELATE"}:
        return name
    raise ValueError(f"unsupported source effect node: {value}")


def compile_source_effect_graph(
    source_ir: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    """Compile the bound source IR; controls transform this exact graph."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown V9 condition: {condition}")
    validate_slot_source_ir(source_ir)
    nodes = sorted({
        _node_effect(str(row["effect"]))
        for row in source_ir["nodes"]
    })
    edges = []
    for row in source_ir["edges"]:
        guard = str(row["guard"])
        if guard not in _SUPPORTED_GUARDS:
            raise ValueError(f"unsupported source guard: {guard}")
        source = _node_effect(str(row["from"]))
        destination = _node_effect(str(row["to"]))
        if condition == "edge_permuted_ir":
            source, destination = destination, source
        edges.append({
            "from": source,
            "to": destination,
            "guard": guard,
            "supporting_source_tasks": sorted(set(map(
                str, row.get("supporting_source_tasks", ())
            ))),
            "original_from": str(row["from"]),
            "original_to": str(row["to"]),
        })
    if not edges:
        raise ValueError("source effect graph has no executable edges")
    body = {
        "schema_version": "compiled-source-effect-graph-v9",
        "source_ir_sha256": str(source_ir["ir_sha256"]),
        "condition": condition,
        "transformation": (
            "REVERSE_EACH_BOUND_SOURCE_EDGE"
            if condition == "edge_permuted_ir"
            else "IDENTITY"
        ),
        "nodes": nodes,
        "edges": sorted(
            edges,
            key=lambda row: (
                row["from"], row["to"], row["guard"],
                row["original_from"], row["original_to"],
            ),
        ),
    }
    return body | {"graph_sha256": stable_hash(body)}


def _explicit_goal_property(goal: str) -> str:
    """Read an explicit ALFWorld goal operator without rollout outcomes."""
    value = _normalized(goal)
    if value.startswith(("look at ", "examine ")):
        return "LIGHT"
    if value.startswith("clean ") or re.match(r"^put (?:a |some |the )?clean ", value):
        return "CLEAN"
    if value.startswith("heat ") or re.match(r"^put (?:a |some |the )?hot ", value):
        return "HEAT"
    if value.startswith("cool ") or re.match(
        r"^put (?:a |some |the )?(?:cool|cold) ", value
    ):
        return "COOL"
    return "NONE"


def _permuted_property(property_name: str) -> str:
    return {
        "NONE": "CLEAN",
        "CLEAN": "HEAT",
        "HEAT": "COOL",
        "COOL": "LIGHT",
        "LIGHT": "NONE",
    }[property_name]


def condition_required_property(
    goal: str,
    property_probabilities: Mapping[str, float],
    condition: str,
) -> tuple[str, str, float]:
    """Return structural property plus target-neural diagnostic support.

    ALFWorld states the operator explicitly in its target-language goal.  The
    neural router is retained as target-native diagnostic evidence, but its
    confidence cannot erase an observed completed-slot safety invariant.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown V9 condition: {condition}")
    structural = _explicit_goal_property(goal)
    required = (
        _permuted_property(structural)
        if condition == "property_permuted_router"
        else structural
    )
    neural = max(
        PROPERTY_CLASSES,
        key=lambda name: (
            float(property_probabilities.get(name, 0.0)), name
        ),
    )
    return required, neural, float(
        property_probabilities.get(structural, 0.0)
    )


def _source_guard(ledger: Mapping[str, Any]) -> tuple[str, str] | None:
    state = slot_state(ledger)
    if state["remaining_slots"] <= 0:
        return None
    carried = state["carried_object"]
    if not carried:
        return "START", "BIND"
    spec = ledger["goal_spec"]
    if str(carried).split(" ", 1)[0] != spec["goal_object_type"]:
        return None
    if not _property_satisfied(ledger, str(carried)):
        return _GUARD_REQUIRES_PROPERTY, "BIND"
    if spec["required_property"] == "NONE":
        return _GUARD_READY_FOR_RELATION, "BIND"
    return _GUARD_PROPERTY_OBSERVED, "MUTATE"


def route_source_effect(
    ledger: Mapping[str, Any],
    compiled_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Route solely through a compiled source node or guarded source edge."""
    request = _source_guard(ledger)
    if request is None:
        return {
            "effect": None,
            "diagnostic": "NO_UNSATISFIED_SOURCE_GRAPH_STATE",
            "source_transition": None,
        }
    guard, source = request
    if guard == "START":
        if source not in set(map(str, compiled_graph["nodes"])):
            return {
                "effect": None,
                "diagnostic": "SOURCE_GRAPH_START_NODE_MISSING",
                "source_transition": None,
            }
        return {
            "effect": source,
            "diagnostic": "SOURCE_GRAPH_START_NODE",
            "source_transition": {
                "kind": "NODE",
                "node": source,
                "graph_sha256": compiled_graph["graph_sha256"],
            },
        }
    candidates = [
        row for row in compiled_graph["edges"]
        if row["from"] == source and row["guard"] == guard
    ]
    if len(candidates) != 1:
        return {
            "effect": None,
            "diagnostic": "SOURCE_GRAPH_GUARDED_EDGE_UNAVAILABLE",
            "source_transition": {
                "kind": "EDGE_LOOKUP",
                "from": source,
                "guard": guard,
                "matches": len(candidates),
                "graph_sha256": compiled_graph["graph_sha256"],
            },
        }
    edge = candidates[0]
    return {
        "effect": str(edge["to"]),
        "diagnostic": "SOURCE_GRAPH_GUARDED_EDGE",
        "source_transition": {
            "kind": "EDGE",
            "from": str(edge["from"]),
            "to": str(edge["to"]),
            "guard": str(edge["guard"]),
            "supporting_source_tasks": list(
                edge["supporting_source_tasks"]
            ),
            "graph_sha256": compiled_graph["graph_sha256"],
        },
    }


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
    """Choose a native action whose effect is supplied by the source graph."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown V9 condition: {condition}")
    if not grounded:
        raise ValueError("slot-aware Harness received no target-native actions")
    allowed_effects = set(map(str, allowed_source_effects))
    if not allowed_effects or not allowed_effects.issubset(
        {"BIND", "MUTATE", "RELATE"}
    ):
        raise ValueError("invalid slot-aware source-effect scope")
    compiled = compile_source_effect_graph(source_ir, condition=condition)
    spec = ledger["goal_spec"]
    required = str(spec["required_property"])
    neural_property = max(
        PROPERTY_CLASSES,
        key=lambda name: (
            float(property_probabilities.get(name, 0.0)), name
        ),
    )
    neural_support = float(property_probabilities.get(required, 0.0))
    structural_scope_active = bool(
        spec["kind"] == "RELATE"
        and required in set(map(str, active_required_properties))
    )
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
        condition != "target_only" and spec["kind"] == "RELATE"
    )
    fallback = safe_fallback if safety_enabled else raw_fallback
    fallback_effect = target_effect(str(grounded[fallback]["option"]))
    common = {
        "fallback_action": fallback,
        "raw_target_fallback_action": raw_fallback,
        "slot_safety_shielded": fallback != raw_fallback,
        "slot_safety_enabled": safety_enabled,
        "transfer_scope_active": structural_scope_active,
        "allowed_source_effects": sorted(allowed_effects),
        "fallback_effect": fallback_effect,
        "slot_state": slot_state(ledger),
        "required_property": required,
        "neural_property_prediction": neural_property,
        "neural_required_property_support": neural_support,
        "minimum_property_confidence_diagnostic_only": (
            minimum_property_confidence
        ),
        "compiled_source_graph_sha256": compiled["graph_sha256"],
        "compiled_source_graph_transformation": compiled["transformation"],
    }
    if condition == "target_only":
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "source_transition": None,
            "diagnostic": "TARGET_ONLY",
        }
    if not structural_scope_active:
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "source_transition": None,
            "diagnostic": "STRUCTURAL_TRANSFER_SCOPE_TARGET_ABSTENTION",
        }
    routed = route_source_effect(ledger, compiled)
    requested = routed["effect"]
    route_common = common | {
        "requested_source_effect": requested,
        "source_transition": routed["source_transition"],
        "source_route_diagnostic": routed["diagnostic"],
    }
    if requested is None:
        return route_common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": routed["diagnostic"],
        }
    if requested not in allowed_effects:
        return route_common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "SOURCE_GRAPH_EFFECT_OUTSIDE_FROZEN_SCOPE",
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
    candidate_common = route_common | {
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
            "diagnostic": "SOURCE_GRAPH_EFFECT_UNAVAILABLE_TARGET_ABSTENTION",
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
        "diagnostic": "SOURCE_GRAPH_EDGE_TARGET_NEURAL_REALIZATION",
    }


__all__ = [
    "CONDITIONS",
    "choose_slot_aware_action",
    "compile_source_effect_graph",
    "condition_required_property",
    "route_source_effect",
]
