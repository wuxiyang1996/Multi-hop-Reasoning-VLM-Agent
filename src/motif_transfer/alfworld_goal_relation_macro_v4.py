"""Effect-gated ALFWorld realization of the source-induced relation macro.

The source program's self-loop is guarded by an observed increase in
``entity_goal_relation_coverage``.  V3 admitted the program before that first
typed effect and therefore allowed a cross-domain program to perturb the
target-native choice of the first object/receptacle binding.  This runtime
keeps the target policy in control until one target-native relation effect has
actually been observed; only the induced recurrent transition is transferred.
"""

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
    TargetRelationExecutionState,
    choose_goal_relation_action as _choose_v3,
    observe_goal_relation_transition,
    reconcile_bound_relation_objects,
    relation_coverage,
    target_relation_state,
)


_EFFECT_GATED = {AUTHENTIC, CARDINALITY_CONTROL, GENERIC, CEILING}


def choose_goal_relation_action(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    goal: str,
    history: Sequence[str],
    ledger: Mapping[str, Any],
    execution_state: TargetRelationExecutionState,
    source_artifact: Mapping[str, Any],
    target_causal_effect_head: Mapping[str, Any],
    step: int,
    max_steps: int,
    minimum_binding: float,
    minimum_realization: float,
    minimum_binding_margin: float,
    minimum_causal_effect: float,
) -> dict[str, Any]:
    """Admit a source transition only after its typed guard is observed."""

    completed = int(target_relation_state(ledger)["completed_count"])
    effective_condition = (
        RAW if condition in _EFFECT_GATED and completed == 0 else condition
    )
    decision = _choose_v3(
        condition=effective_condition,
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
    if effective_condition != condition:
        return decision | {
            "program_active": False,
            "program_status": "SOURCE_RECURRENCE_AWAITS_FIRST_OBSERVED_RELATION",
            "source_admitted": False,
            "diagnostic": "SOURCE_RECURRENCE_AWAITS_FIRST_OBSERVED_RELATION",
        }
    return decision


__all__ = [
    "AUTHENTIC",
    "CARDINALITY_CONTROL",
    "CEILING",
    "CONDITIONS",
    "EFFECT_CONTROL",
    "GENERIC",
    "RAW",
    "TargetRelationExecutionState",
    "choose_goal_relation_action",
    "observe_goal_relation_transition",
    "reconcile_bound_relation_objects",
    "relation_coverage",
    "target_relation_state",
]
