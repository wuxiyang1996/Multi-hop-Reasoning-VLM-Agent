"""Fail-closed ALFWorld realization of the source-induced relation macro."""

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
    _action_matches_effect,
    _mentions_exact_handle,
    _would_reopen_completed_slot,
    choose_goal_relation_action as _choose_v3,
    observe_goal_relation_transition,
    reconcile_bound_relation_objects,
    relation_coverage,
    target_relation_state,
)
from .alfworld_goal_relation_macro_v4 import (
    choose_goal_relation_action as _choose_effect_gated,
)


def _choose_raw(**kwargs: Any) -> dict[str, Any]:
    return _choose_v3(**(kwargs | {"condition": RAW}))


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
    """Execute only uniquely grounded transitions; otherwise abstain.

    Native admissibility establishes that an action is executable.  For the
    BIND prerequisite, the neural binding and completion heads select its typed
    arguments; the option-applicability head is deliberately not used a second
    time to veto an action already exposed by the target environment.
    """

    kwargs = {
        "condition": condition,
        "grounded": grounded,
        "goal": goal,
        "history": history,
        "ledger": ledger,
        "execution_state": execution_state,
        "source_artifact": source_artifact,
        "target_causal_effect_head": target_causal_effect_head,
        "step": step,
        "max_steps": max_steps,
        "minimum_binding": minimum_binding,
        "minimum_realization": minimum_realization,
        "minimum_binding_margin": minimum_binding_margin,
        "minimum_causal_effect": minimum_causal_effect,
    }
    state = target_relation_state(ledger)
    if (
        condition not in {AUTHENTIC, CEILING}
        or int(state["completed_count"]) == 0
        or int(state["remaining_slots"]) == 0
    ):
        return _choose_effect_gated(**kwargs)

    obligation = "RELATE" if state["carried_object"] else "BIND"
    candidates: list[str] = []
    for action, row in grounded.items():
        if _would_reopen_completed_slot(action, ledger):
            continue
        matches, _ = _action_matches_effect(
            action, effect=obligation, ledger=ledger,
        )
        if (
            matches
            and obligation == "RELATE"
            and ledger.get("bound_target_receptacle")
        ):
            matches = _mentions_exact_handle(
                action, str(ledger["bound_target_receptacle"]),
            )
        if matches and float(row["binding"]) >= minimum_binding:
            candidates.append(action)

    if len(candidates) != 1:
        decision = _choose_raw(**kwargs)
        diagnostic = (
            "SOURCE_ARTIFACT_ZERO_BINDINGS_ABSTENTION"
            if not candidates
            else "SOURCE_ARTIFACT_MULTIPLE_BINDINGS_ABSTENTION"
        )
        return decision | {
            "program_active": True,
            "program_status": "UNSATISFIED_RELATION_MACRO",
            "target_native_obligation": obligation,
            "source_admitted": False,
            "candidate_count": len(candidates),
            "diagnostic": diagnostic,
        }

    effective_grounded = grounded
    if obligation == "BIND":
        # The action is already in the target-native admissible action set.
        # Keep neural argument binding/completion, but avoid double-counting a
        # separately trained option-applicability head as executability.
        selected = candidates[0]
        effective_grounded = {
            action: dict(row) for action, row in grounded.items()
        }
        effective_grounded[selected]["applicability"] = 1.0
        kwargs = kwargs | {"grounded": effective_grounded}
    return _choose_effect_gated(**kwargs)


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
