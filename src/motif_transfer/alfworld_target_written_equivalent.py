"""Source-blind target-written equivalent of the ALFWorld transfer policy.

This is an identifiability control, not a learned source program.  It authors
the recurrence directly from the ALFWorld multiplicity interface:

* wait until one native goal relation has been observed;
* preserve that exact relation handle;
* acquire another correctly typed entity when no binding exists; and
* realize the remaining relation, abstaining when grounding is ambiguous.

The implementation accepts the runner's ``source_artifact`` argument only for
signature compatibility and never reads it.  Concrete actions are still
ranked exclusively by the frozen target-native neural grounder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .alfworld_goal_relation_macro import (
    TargetRelationExecutionState,
    _action_tokens,
    _conflicts_with_bound_handle,
    _mentions_exact_handle,
    _realization_score,
    _would_reopen_completed_slot,
    relation_coverage,
    target_relation_state,
)
from .alfworld_search_automaton_v16 import target_policy_rank, target_scope_id
from .contracts import stable_hash
from .real_source_relation_causal_v20 import (
    action_causal_features,
    linear_probability,
)
from .slot_aware_alfworld_harness import _action_matches_effect


TARGET_WRITTEN_EQUIVALENT = "target_written_isomorphic_multiplicity_controller"


@dataclass
class TargetWrittenExecutionState(TargetRelationExecutionState):
    """Target-owned novelty state with no source fields or identifiers."""

    acquisition_cycle: tuple[int, str] | None = None
    attempted_acquisition_actions: set[str] = field(default_factory=set)

    def begin_acquisition_cycle(self, completed: int, obligation: str) -> None:
        cycle = (int(completed), str(obligation))
        if cycle != self.acquisition_cycle:
            self.acquisition_cycle = cycle
            self.attempted_acquisition_actions.clear()


def _common(
    grounded: Mapping[str, Mapping[str, Any]], history: Sequence[str],
    ledger: Mapping[str, Any], *, active: bool, status: str,
) -> dict[str, Any]:
    ranking = target_policy_rank(
        grounded, history, discount_repeats=True, structured=False,
    )
    raw = ranking[0]
    safe = [
        action for action in ranking
        if not _would_reopen_completed_slot(action, ledger)
    ]
    fallback = (safe or ranking)[0]
    return {
        "fallback_action": fallback,
        "raw_fallback_action": raw,
        "slot_safety_changed_fallback": fallback != raw,
        "relation_coverage": relation_coverage(ledger),
        "slot_state": target_relation_state(ledger),
        "program_active": active,
        "program_status": status,
    }


def _active_relation_decision(
    *, grounded: Mapping[str, Mapping[str, Any]], goal: str,
    history: Sequence[str], ledger: Mapping[str, Any],
    execution_state: TargetWrittenExecutionState,
    target_causal_effect_head: Mapping[str, Any], step: int, max_steps: int,
    minimum_binding: float, minimum_realization: float,
    minimum_binding_margin: float, minimum_causal_effect: float,
    common: Mapping[str, Any], obligation: str,
) -> dict[str, Any]:
    candidates = []
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
        if not matches or float(row["binding"]) < minimum_binding:
            continue
        realization = _realization_score(action, row, history)
        causal = linear_probability(
            target_causal_effect_head,
            action_causal_features(
                action=action, grounded_scores=row, ledger=ledger,
                history=history, step=step, max_steps=max_steps,
            ),
        )
        candidates.append((
            causal if obligation == "RELATE" else realization,
            float(row["binding"]), causal, realization, action,
        ))
    candidates.sort(reverse=True)
    if candidates:
        _, binding, causal, realization, selected = candidates[0]
        second_binding = candidates[1][1] if len(candidates) > 1 else 0.0
        margin = binding - second_binding
        qualifies = (
            causal >= minimum_causal_effect
            if obligation == "RELATE" else
            realization >= minimum_realization
        ) and margin >= minimum_binding_margin
        if qualifies:
            return dict(common) | {
                "action": selected,
                "target_native_obligation": obligation,
                "source_admitted": False,
                "changed_action": selected != common["raw_fallback_action"],
                "best_realization_score": realization,
                "best_causal_effect_probability": causal,
                "best_binding": binding,
                "binding_margin": margin,
                "candidate_count": len(candidates),
                "diagnostic": "TARGET_WRITTEN_NATIVE_RELATION_REALIZATION",
            }

    state = target_relation_state(ledger)
    bound = ledger.get("bound_target_receptacle")
    carried = state.get("carried_object")
    if bound and carried and str(carried).split(" ", 1)[0] == ledger[
        "goal_spec"
    ]["goal_object_type"]:
        handle_actions = [
            action for action in grounded
            if _mentions_exact_handle(action, str(bound))
            and not _conflicts_with_bound_handle(action, ledger)
            and _action_tokens(action)[:1] in {
                ("go",), ("open",), ("examine",),
            }
        ]
        if handle_actions:
            selected = target_policy_rank(
                {action: grounded[action] for action in handle_actions},
                history, discount_repeats=True, structured=True,
            )[0]
            return dict(common) | {
                "action": selected,
                "target_native_obligation": obligation,
                "source_admitted": False,
                "changed_action": selected != common["raw_fallback_action"],
                "candidate_count": len(candidates),
                "bound_target_receptacle": str(bound),
                "diagnostic": "TARGET_WRITTEN_RELATION_HANDLE_GROUNDING",
            }

    scope = target_scope_id(
        goal=goal, native_actions=tuple(grounded), history=history,
    )
    execution_state.begin_scope(scope)
    structured = target_policy_rank(
        grounded, history, discount_repeats=True, structured=True,
    )
    exploration = [
        action for action in structured
        if action not in execution_state.attempted_actions
        and not _would_reopen_completed_slot(action, ledger)
        and not _conflicts_with_bound_handle(action, ledger)
        and str(grounded[action]["option"]) == "SEARCH"
    ]
    if not exploration:
        return dict(common) | {
            "action": common["fallback_action"],
            "target_native_obligation": obligation,
            "source_admitted": False,
            "changed_action": (
                common["fallback_action"] != common["raw_fallback_action"]
            ),
            "candidate_count": len(candidates),
            "diagnostic": "TARGET_WRITTEN_BINDING_EXHAUSTED_ABSTENTION",
        }
    selected = exploration[0]
    execution_state.attempted_actions.add(selected)
    return dict(common) | {
        "action": selected,
        "target_native_obligation": obligation,
        "source_admitted": False,
        "changed_action": selected != common["raw_fallback_action"],
        "candidate_count": len(candidates),
        "target_scope_sha256": stable_hash(scope),
        "diagnostic": "TARGET_WRITTEN_SEARCH_FOR_UNIQUE_RELATION_BINDING",
    }


def choose_target_written_action(
    *, condition: str, grounded: Mapping[str, Mapping[str, Any]], goal: str,
    history: Sequence[str], ledger: Mapping[str, Any],
    execution_state: TargetWrittenExecutionState,
    source_artifact: Mapping[str, Any],
    target_causal_effect_head: Mapping[str, Any], step: int, max_steps: int,
    minimum_binding: float, minimum_realization: float,
    minimum_binding_margin: float, minimum_causal_effect: float,
) -> dict[str, Any]:
    """Select target actions without inspecting ``source_artifact``."""
    if condition != TARGET_WRITTEN_EQUIVALENT:
        raise ValueError(f"unsupported target-written condition: {condition}")
    if not grounded:
        raise ValueError("target-written controller received no native actions")
    # Deliberately do not inspect, validate, iterate, hash, or copy this value.
    del source_artifact

    state = target_relation_state(ledger)
    completed = int(state["completed_count"])
    remaining = int(state["remaining_slots"])
    active = completed >= 1 and remaining >= 1
    status = (
        "TARGET_WRITTEN_MULTIPLICITY_RECURRENCE"
        if active else
        "TARGET_WRITTEN_AWAITS_FIRST_RELATION"
        if completed == 0 else
        "TARGET_WRITTEN_TERMINAL_RELATION_SATISFIED"
    )
    common = _common(
        grounded, history, ledger, active=active, status=status,
    )
    if not active:
        return common | {
            "action": common["fallback_action"],
            "target_native_obligation": None,
            "source_admitted": False,
            "changed_action": (
                common["fallback_action"] != common["raw_fallback_action"]
            ),
            "diagnostic": status,
        }

    obligation = "RELATE" if state["carried_object"] else "BIND"
    uniquely_bound = []
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
            uniquely_bound.append(action)

    if len(uniquely_bound) == 1:
        effective = grounded
        if obligation == "BIND":
            selected = uniquely_bound[0]
            effective = {action: dict(row) for action, row in grounded.items()}
            effective[selected]["applicability"] = 1.0
        return _active_relation_decision(
            grounded=effective, goal=goal, history=history, ledger=ledger,
            execution_state=execution_state,
            target_causal_effect_head=target_causal_effect_head,
            step=step, max_steps=max_steps,
            minimum_binding=minimum_binding,
            minimum_realization=minimum_realization,
            minimum_binding_margin=minimum_binding_margin,
            minimum_causal_effect=minimum_causal_effect,
            common=common, obligation=obligation,
        )
    if len(uniquely_bound) > 1:
        return common | {
            "action": common["fallback_action"],
            "target_native_obligation": obligation,
            "source_admitted": False,
            "changed_action": (
                common["fallback_action"] != common["raw_fallback_action"]
            ),
            "candidate_count": len(uniquely_bound),
            "diagnostic": "TARGET_WRITTEN_MULTIPLE_BINDINGS_ABSTENTION",
        }

    carried = state.get("carried_object")
    goal_type = str(state["goal_object_type"])
    if carried and str(carried).split(" ", 1)[0] != goal_type:
        return common | {
            "action": common["fallback_action"],
            "target_native_obligation": obligation,
            "source_admitted": False,
            "changed_action": (
                common["fallback_action"] != common["raw_fallback_action"]
            ),
            "candidate_count": 0,
            "diagnostic": "TARGET_WRITTEN_TYPED_ENTITY_MISMATCH_ABSTENTION",
        }

    execution_state.begin_acquisition_cycle(completed, obligation)
    bound = state.get("bound_target_receptacle")
    acquisition = {
        action: row for action, row in grounded.items()
        if str(row.get("option")) == "SEARCH"
        and str(row.get("required_option")) == "SEARCH"
        and action not in execution_state.attempted_acquisition_actions
        and not _would_reopen_completed_slot(action, ledger)
        and not _conflicts_with_bound_handle(action, ledger)
        and (
            obligation != "RELATE"
            or not bound
            or _mentions_exact_handle(action, str(bound))
        )
    }
    if not acquisition:
        return common | {
            "action": common["fallback_action"],
            "target_native_obligation": obligation,
            "source_admitted": False,
            "changed_action": (
                common["fallback_action"] != common["raw_fallback_action"]
            ),
            "candidate_count": 0,
            "diagnostic": "TARGET_WRITTEN_ACQUISITION_GROUNDING_EXHAUSTED",
        }
    selected = target_policy_rank(
        acquisition, history, discount_repeats=True, structured=False,
    )[0]
    execution_state.attempted_acquisition_actions.add(selected)
    return common | {
        "action": selected,
        "target_native_obligation": obligation,
        "source_admitted": False,
        "changed_action": selected != common["raw_fallback_action"],
        "candidate_count": 0,
        "target_acquisition_option": "SEARCH",
        "diagnostic": "TARGET_WRITTEN_ACQUISITION_OPERATOR_GROUNDED",
    }


__all__ = [
    "TARGET_WRITTEN_EQUIVALENT",
    "TargetWrittenExecutionState",
    "choose_target_written_action",
]
