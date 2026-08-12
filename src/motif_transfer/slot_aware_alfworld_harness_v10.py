"""Budgeted V10 controls separating safety, source nodes, and source edges."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .parameterized_alfworld_harness import PROPERTY_CLASSES
from .slot_aware_alfworld_harness_v9 import (
    CONDITIONS,
    choose_slot_aware_action as choose_v9_action,
    compile_source_effect_graph,
    condition_required_property as v9_condition_required_property,
    route_source_effect,
)


CONDITION_SEMANTICS = {
    "target_only": "RAW_TARGET_POLICY",
    "property_permuted_router": "TARGET_NATIVE_COMPLETED_SLOT_SAFETY_ONLY",
    "edge_permuted_ir": (
        "SOURCE_BIND_NODE_PLUS_SAFETY_WITH_REVERSED_SOURCE_EDGES"
    ),
    "authentic_slot_ir": "FULL_AUTHENTIC_SOURCE_GRAPH_PLUS_SAFETY",
}


def condition_required_property(
    goal: str,
    property_probabilities: Mapping[str, float],
    condition: str,
) -> tuple[str, str, float]:
    """Keep the safety-only control in the same structural target task."""
    routed_condition = (
        "authentic_slot_ir"
        if condition == "property_permuted_router"
        else condition
    )
    return v9_condition_required_property(
        goal, property_probabilities, routed_condition
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
    """Run V9 graph execution or the V10 safety-only causal control."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown V10 condition: {condition}")
    routed_condition = (
        "authentic_slot_ir"
        if condition == "property_permuted_router"
        else condition
    )
    decision = choose_v9_action(
        condition=routed_condition,
        grounded=grounded,
        history=history,
        ledger=ledger,
        source_ir=source_ir,
        property_probabilities=property_probabilities,
        minimum_property_confidence=minimum_property_confidence,
        minimum_role_binding=minimum_role_binding,
        minimum_realization_score=minimum_realization_score,
        minimum_target_policy_ratio=minimum_target_policy_ratio,
        allowed_source_effects=allowed_source_effects,
        active_required_properties=active_required_properties,
    )
    decision["condition_semantics"] = CONDITION_SEMANTICS[condition]
    if condition != "property_permuted_router":
        return decision
    fallback = str(decision["fallback_action"])
    return decision | {
        "action": fallback,
        "target_realized_effect": decision["fallback_effect"],
        "source_selected_effect": None,
        "source_admitted": False,
        "changed_action": False,
        "changed_effect": False,
        "requested_source_effect": None,
        "source_transition": None,
        "source_route_diagnostic": "SOURCE_GRAPH_NOT_EXECUTED",
        "compiled_source_graph_transformation": (
            "NOT_EXECUTED_SAFETY_ONLY_CONTROL"
        ),
        "diagnostic": "TARGET_NATIVE_COMPLETED_SLOT_SAFETY_ONLY",
    }


__all__ = [
    "CONDITIONS",
    "CONDITION_SEMANTICS",
    "choose_slot_aware_action",
    "compile_source_effect_graph",
    "condition_required_property",
    "route_source_effect",
]
