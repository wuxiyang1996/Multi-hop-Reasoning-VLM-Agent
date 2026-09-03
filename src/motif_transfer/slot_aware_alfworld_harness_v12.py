"""Selective V12 source-edge execution learned from consumed gate traces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .parameterized_alfworld_harness import PROPERTY_CLASSES
from .slot_aware_alfworld_harness_v10 import (
    CONDITIONS,
    CONDITION_SEMANTICS as V10_CONDITION_SEMANTICS,
    choose_slot_aware_action as choose_v10_action,
    compile_source_effect_graph,
    condition_required_property,
    route_source_effect,
)


MINIMUM_SOURCE_EDGE_STEP = 9
CONDITION_SEMANTICS = dict(V10_CONDITION_SEMANTICS) | {
    "authentic_slot_ir": (
        "SELECTIVE_FULL_SOURCE_GRAPH_PLUS_SAFETY_AFTER_MINIMUM_EDGE_STEP"
    )
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
    """Apply the cross-version minimum-step shield to source edges."""
    decision = choose_v10_action(
        condition=condition,
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
    transition = decision.get("source_transition")
    is_edge = bool(
        isinstance(transition, dict) and transition.get("kind") == "EDGE"
    )
    step = len(history)
    applicability = {
        "schema_version": "minimum-source-edge-step-applicability-v12",
        "feature": "TARGET_EPISODE_STEP_INDEX",
        "value": step,
        "minimum_source_edge_step": MINIMUM_SOURCE_EDGE_STEP,
        "source_edge_candidate": is_edge,
        "source_edge_admitted": bool(
            is_edge and step >= MINIMUM_SOURCE_EDGE_STEP
        ),
        "selection_authority": (
            "CONSUMED_V9_V10_V11_GROUPED_CROSS_VERSION_AUDIT"
        ),
    }
    decision["source_applicability"] = applicability
    if (
        condition != "authentic_slot_ir"
        or not is_edge
        or step >= MINIMUM_SOURCE_EDGE_STEP
    ):
        return decision
    fallback = str(decision["fallback_action"])
    return decision | {
        "action": fallback,
        "target_realized_effect": decision["fallback_effect"],
        "source_selected_effect": None,
        "source_admitted": False,
        "changed_action": False,
        "changed_effect": False,
        "candidate_source_transition": transition,
        "source_transition": None,
        "source_applicability": applicability,
        "diagnostic": "SOURCE_EDGE_EARLY_APPLICABILITY_ABSTENTION",
    }


__all__ = [
    "CONDITIONS",
    "CONDITION_SEMANTICS",
    "MINIMUM_SOURCE_EDGE_STEP",
    "choose_slot_aware_action",
    "compile_source_effect_graph",
    "condition_required_property",
    "route_source_effect",
]
