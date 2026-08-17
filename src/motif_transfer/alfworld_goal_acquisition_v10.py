"""Typed, handle-preserving ALFWorld grounding of source acquisition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .alfworld_goal_relation_macro import (
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    CONDITIONS,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
    _conflicts_with_bound_handle,
    _mentions_exact_handle,
    _would_reopen_completed_slot,
    observe_goal_relation_transition,
    reconcile_bound_relation_objects,
    relation_coverage,
    target_relation_state,
)
from .alfworld_goal_relation_macro_v5 import (
    choose_goal_relation_action as _choose_v5,
)
from .alfworld_goal_acquisition_v9 import TargetAcquisitionExecutionState
from .alfworld_search_automaton_v16 import target_policy_rank
from .source_goal_acquisition_induction import (
    validate_goal_acquisition_program,
)


_SOURCE_ACQUISITION_ARTIFACT: Mapping[str, Any] | None = None
_ACQUISITION_CONDITIONS = frozenset({AUTHENTIC, CEILING})


def configure_source_acquisition(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> None:
    validate_goal_acquisition_program(artifact)
    if not confirmation.get("source_gate_passed"):
        raise ValueError("source acquisition program lacks held-out confirmation")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("source acquisition confirmation/artifact mismatch")
    global _SOURCE_ACQUISITION_ARTIFACT
    _SOURCE_ACQUISITION_ARTIFACT = artifact


def _control_position_operator_id(artifact: Mapping[str, Any]) -> str:
    program_ids = set(map(
        str, artifact["program"]["acquisition_operator_type_ids"],
    ))
    matches = [
        str(row["operator_type_id"])
        for row in artifact["operator_types"]
        if str(row["operator_type_id"]) in program_ids
        and row.get("operation") == "UPDATE"
        and row.get("predicate_family") == "CONTROL_STATE"
        and int(row.get("arity", 0)) == 1
        and row.get("value_kind") == "POSITION"
    ]
    if len(matches) != 1:
        raise ValueError("source acquisition lacks one control-position operator")
    return matches[0]


def grounded_acquisition_actions(
    *, grounded: Mapping[str, Mapping[str, Any]], history: Sequence[str],
    ledger: Mapping[str, Any], execution_state: TargetAcquisitionExecutionState,
    acquisition_artifact: Mapping[str, Any], obligation: str,
) -> tuple[list[str], str]:
    """Use neural policy ranking and preserve the bound relation argument."""

    validate_goal_acquisition_program(acquisition_artifact)
    operator_id = _control_position_operator_id(acquisition_artifact)
    state = target_relation_state(ledger)
    execution_state.begin_acquisition_cycle(
        int(state["completed_count"]), obligation,
    )
    bound_handle = state.get("bound_target_receptacle")
    candidates = {
        action: row for action, row in grounded.items()
        if str(row.get("option")) == "SEARCH"
        and str(row.get("required_option")) == "SEARCH"
        and action not in execution_state.attempted_acquisition_actions
        and not _would_reopen_completed_slot(action, ledger)
        and not _conflicts_with_bound_handle(action, ledger)
        and (
            obligation != "RELATE"
            or not bound_handle
            or _mentions_exact_handle(action, str(bound_handle))
        )
    }
    if not candidates:
        return [], operator_id
    # Selection remains target-native and neural.  V9's structured=True path
    # mixed in a hand-composed completion score and caused avoidable drift.
    return target_policy_rank(
        candidates, history, discount_repeats=True, structured=False,
    ), operator_id


def choose_goal_relation_action(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    goal: str,
    history: Sequence[str],
    ledger: Mapping[str, Any],
    execution_state: TargetAcquisitionExecutionState,
    source_artifact: Mapping[str, Any],
    target_causal_effect_head: Mapping[str, Any],
    step: int,
    max_steps: int,
    minimum_binding: float,
    minimum_realization: float,
    minimum_binding_margin: float,
    minimum_causal_effect: float,
) -> dict[str, Any]:
    decision = _choose_v5(
        condition=condition,
        grounded=grounded,
        goal=goal,
        history=history,
        ledger=ledger,
        execution_state=execution_state,
        source_artifact=source_artifact,
        target_causal_effect_head=target_causal_effect_head,
        step=step,
        max_steps=max_steps,
        minimum_binding=minimum_binding,
        minimum_realization=minimum_realization,
        minimum_binding_margin=minimum_binding_margin,
        minimum_causal_effect=minimum_causal_effect,
    )
    if (
        condition not in _ACQUISITION_CONDITIONS
        or decision.get("diagnostic")
        != "SOURCE_ARTIFACT_ZERO_BINDINGS_ABSTENTION"
    ):
        return decision
    if _SOURCE_ACQUISITION_ARTIFACT is None:
        raise RuntimeError("source acquisition program was not configured")
    if not isinstance(execution_state, TargetAcquisitionExecutionState):
        raise TypeError("target acquisition execution state was not installed")
    state = target_relation_state(ledger)
    if int(state["completed_count"]) < 1 or int(state["remaining_slots"]) < 1:
        return decision
    carried = state.get("carried_object")
    goal_type = str(state["goal_object_type"])
    if carried and str(carried).split(" ", 1)[0] != goal_type:
        return decision | {
            "source_admitted": False,
            "diagnostic": "SOURCE_ACQUISITION_TYPED_ENTITY_MISMATCH_ABSTENTION",
        }
    obligation = "RELATE" if carried else "BIND"
    ranked, operator_id = grounded_acquisition_actions(
        grounded=grounded,
        history=history,
        ledger=ledger,
        execution_state=execution_state,
        acquisition_artifact=_SOURCE_ACQUISITION_ARTIFACT,
        obligation=obligation,
    )
    if not ranked:
        return decision | {
            "source_admitted": False,
            "diagnostic": "SOURCE_ACQUISITION_TARGET_GROUNDING_EXHAUSTED",
            "acquisition_operator_type_id": operator_id,
        }
    selected = ranked[0]
    execution_state.attempted_acquisition_actions.add(selected)
    return decision | {
        "action": selected,
        "source_admitted": condition == AUTHENTIC,
        "changed_action": selected != decision["raw_fallback_action"],
        "program_active": True,
        "program_status": "SOURCE_INDUCED_ACQUISITION_PRECONDITION",
        "target_native_obligation": obligation,
        "diagnostic": "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED",
        "acquisition_operator_type_id": operator_id,
        "target_acquisition_option": "SEARCH",
        "bound_relation_handle_preserved": bool(
            obligation == "RELATE" and state.get("bound_target_receptacle")
        ),
    }


__all__ = [
    "AUTHENTIC",
    "CARDINALITY_CONTROL",
    "CEILING",
    "CONDITIONS",
    "EFFECT_CONTROL",
    "GENERIC",
    "RAW",
    "TargetAcquisitionExecutionState",
    "choose_goal_relation_action",
    "configure_source_acquisition",
    "grounded_acquisition_actions",
    "observe_goal_relation_transition",
    "reconcile_bound_relation_objects",
    "relation_coverage",
    "target_relation_state",
]
