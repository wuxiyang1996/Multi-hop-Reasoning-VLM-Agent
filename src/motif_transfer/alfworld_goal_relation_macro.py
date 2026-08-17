"""Target-native grounding for a source-induced recurrent relation macro."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from .alfworld_search_automaton_v16 import target_policy_rank, target_scope_id
from .contracts import stable_hash
from .real_source_relation_causal_v20 import (
    action_causal_features,
    linear_probability,
)
from .slot_aware_alfworld_harness import (
    _action_matches_effect,
    _would_reopen_completed_slot,
    observe_target_transition,
    reconcile_visible_target_objects,
    slot_state,
)
from .source_goal_relation_induction import validate_goal_relation_macro_program


RAW = "raw_target_only"
AUTHENTIC = "authentic_source_goal_relation_macro"
CARDINALITY_CONTROL = "source_cardinality_exactly_one_control"
EFFECT_CONTROL = "source_effect_binding_permuted_control"
GENERIC = "generic_single_relation_scaffold"
CEILING = "target_native_recurrent_relation_ceiling"
CONDITIONS = (RAW, AUTHENTIC, CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC, CEILING)

_MOVE = re.compile(
    r"^move (?P<object>[a-z]+) (?P<object_id>\d+) to "
    r"(?P<receptacle>[a-z]+) (?P<receptacle_id>\d+)$"
)


def _action_tokens(action: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+|\d+", str(action).lower()))


def _mentions_exact_handle(action: str, handle: str) -> bool:
    action_tokens = _action_tokens(action)
    handle_tokens = _action_tokens(handle)
    return any(
        action_tokens[index:index + len(handle_tokens)] == handle_tokens
        for index in range(len(action_tokens) - len(handle_tokens) + 1)
    )


def _conflicts_with_bound_handle(
    action: str, ledger: Mapping[str, Any],
) -> bool:
    """Reject a target relation action bound to a different native instance."""

    bound = ledger.get("bound_target_receptacle")
    if not bound:
        return False
    move = _MOVE.match(str(action).lower())
    if move is None:
        return False
    spec = ledger["goal_spec"]
    return bool(
        move.group("object") == spec["goal_object_type"]
        and move.group("receptacle") == spec["target_receptacle_type"]
        and not _mentions_exact_handle(action, str(bound))
    )


@dataclass
class TargetRelationExecutionState:
    """Target-owned action-attempt ledger; it contains no source tokens."""

    scope_sha256: str | None = None
    attempted_actions: set[str] = field(default_factory=set)

    def begin_scope(self, scope_sha256: str) -> None:
        if scope_sha256 != self.scope_sha256:
            self.scope_sha256 = str(scope_sha256)
            self.attempted_actions.clear()


def _realization_score(action: str, row: Mapping[str, Any], history: Sequence[str]) -> float:
    return (
        float(row["applicability"])
        * (0.20 + 0.80 * float(row["completion"]))
        * (0.25 + 0.75 * float(row["binding"]))
        / (1.0 + history.count(action))
    )


def target_relation_state(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Return coverage for one entity-conditioned target relation handle."""

    state = slot_state(ledger)
    bound = ledger.get("bound_target_receptacle")
    if bound:
        spec = ledger["goal_spec"]
        completed = sorted(
            object_key for object_key, location in ledger["observed_locations"].items()
            if str(location) == str(bound)
            and str(object_key).split(" ", 1)[0] == spec["goal_object_type"]
        )
        required = int(state["required_count"])
        state = state | {
            "completed_count": min(len(completed), required),
            "remaining_slots": max(required - len(completed), 0),
            "completed_objects": completed,
        }
    return state | {"bound_target_receptacle": bound}


def reconcile_bound_relation_objects(
    ledger: Mapping[str, Any], observation: str,
) -> dict[str, Any]:
    """Reconcile visible objects while preserving one relation argument."""

    updated = reconcile_visible_target_objects(ledger, observation)
    bound = updated.get("bound_target_receptacle")
    if bound:
        state = target_relation_state(updated)
        updated["completed_objects"] = list(state["completed_objects"])
    return updated


def relation_coverage(ledger: Mapping[str, Any]) -> float:
    state = target_relation_state(ledger)
    return float(state["completed_count"]) / max(1, int(state["required_count"]))


def _program_active(
    *, condition: str, ledger: Mapping[str, Any], artifact: Mapping[str, Any],
) -> tuple[bool, str]:
    if condition == RAW:
        return False, "TARGET_ONLY"
    if condition == EFFECT_CONTROL:
        return False, "PERMUTED_EFFECT_HAS_NO_TARGET_NATIVE_BINDING"
    state = target_relation_state(ledger)
    if int(state["remaining_slots"]) <= 0:
        return False, "INDUCED_TERMINAL_PREDICATE_SATISFIED"
    if condition in {CARDINALITY_CONTROL, GENERIC} and int(
        state["completed_count"]
    ) >= 1:
        return False, "EXACTLY_ONE_RELATION_PROGRAM_TERMINATED"
    if condition in {AUTHENTIC, CEILING}:
        cardinality = artifact["program"]["transitions"][0]["cardinality"]
        if cardinality != "ONE_OR_MORE":
            return False, "SOURCE_RECURRENCE_NOT_AVAILABLE"
    return True, "UNSATISFIED_RELATION_MACRO"


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
    """Ground one target action while the source relation obligation is active."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown relation macro condition: {condition}")
    if not grounded:
        raise ValueError("relation macro received no target-native candidates")
    validate_goal_relation_macro_program(source_artifact)
    fallback_rank = target_policy_rank(
        grounded, history, discount_repeats=True, structured=False,
    )
    raw_fallback = fallback_rank[0]
    safe = [
        action for action in fallback_rank
        if not _would_reopen_completed_slot(action, ledger)
    ]
    fallback = (safe or fallback_rank)[0]
    active, reason = _program_active(
        condition=condition, ledger=ledger, artifact=source_artifact,
    )
    common = {
        "fallback_action": fallback,
        "raw_fallback_action": raw_fallback,
        "slot_safety_changed_fallback": fallback != raw_fallback,
        "relation_coverage": relation_coverage(ledger),
        "slot_state": target_relation_state(ledger),
        "program_active": active,
        "program_status": reason,
    }
    if not active:
        return common | {
            "action": fallback,
            "target_native_obligation": None,
            "source_admitted": False,
            "changed_action": fallback != raw_fallback,
            "diagnostic": reason,
        }

    state = target_relation_state(ledger)
    obligation = "RELATE" if state["carried_object"] else "BIND"
    candidates = []
    for action, row in grounded.items():
        if _would_reopen_completed_slot(action, ledger):
            continue
        matches, _ = _action_matches_effect(
            action, effect=obligation, ledger=ledger,
        )
        if matches and obligation == "RELATE" and ledger.get(
            "bound_target_receptacle"
        ):
            move = _MOVE.match(str(action).lower())
            matches = bool(
                move and (
                    f"{move.group('receptacle')} {move.group('receptacle_id')}"
                    == str(ledger["bound_target_receptacle"])
                )
            )
        if not matches or float(row["binding"]) < minimum_binding:
            continue
        realization = _realization_score(action, row, history)
        causal_probability = linear_probability(
            target_causal_effect_head,
            action_causal_features(
                action=action,
                grounded_scores=row,
                ledger=ledger,
                history=history,
                step=step,
                max_steps=max_steps,
            ),
        )
        candidates.append((
            causal_probability if obligation == "RELATE" else realization,
            float(row["binding"]),
            causal_probability,
            realization,
            action,
        ))
    candidates.sort(reverse=True)
    if candidates:
        (
            best_score, best_binding, best_causal_probability,
            best_realization, selected,
        ) = candidates[0]
        second_binding = candidates[1][1] if len(candidates) > 1 else 0.0
        binding_margin = best_binding - second_binding
        if (
            (
                best_causal_probability >= minimum_causal_effect
                if obligation == "RELATE"
                else best_realization >= minimum_realization
            )
            and binding_margin >= minimum_binding_margin
        ):
            return common | {
                "action": selected,
                "target_native_obligation": obligation,
                "source_admitted": condition == AUTHENTIC,
                "changed_action": selected != raw_fallback,
                "best_realization_score": best_realization,
                "best_causal_effect_probability": best_causal_probability,
                "best_binding": best_binding,
                "binding_margin": binding_margin,
                "candidate_count": len(candidates),
                "diagnostic": (
                    "SOURCE_MACRO_TARGET_NATIVE_PREREQUISITE"
                    if obligation == "BIND"
                    else "SOURCE_MACRO_TARGET_NATIVE_RELATION_REALIZATION"
                ),
            }

    bound = ledger.get("bound_target_receptacle")
    carried = state.get("carried_object")
    if bound and carried and str(carried).split(" ", 1)[0] == ledger[
        "goal_spec"
    ]["goal_object_type"]:
        handle_actions = [
            action for action in grounded
            if _mentions_exact_handle(action, str(bound))
            and not _conflicts_with_bound_handle(action, ledger)
            and _action_tokens(action)[:1] in {("go",), ("open",), ("examine",)}
        ]
        if handle_actions:
            selected = target_policy_rank(
                {action: grounded[action] for action in handle_actions},
                history,
                discount_repeats=True,
                structured=True,
            )[0]
            return common | {
                "action": selected,
                "target_native_obligation": obligation,
                "source_admitted": condition == AUTHENTIC,
                "changed_action": selected != raw_fallback,
                "candidate_count": len(candidates),
                "bound_target_receptacle": str(bound),
                "diagnostic": "TARGET_NATIVE_RELATION_HANDLE_GROUNDING",
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
        return common | {
            "action": fallback,
            "target_native_obligation": obligation,
            "source_admitted": False,
            "changed_action": fallback != raw_fallback,
            "candidate_count": len(candidates),
            "diagnostic": "TARGET_NATIVE_BINDING_EXHAUSTED_SOURCE_ABSTENTION",
        }
    selected = exploration[0]
    execution_state.attempted_actions.add(selected)
    return common | {
        "action": selected,
        "target_native_obligation": obligation,
        "source_admitted": condition == AUTHENTIC,
        "changed_action": selected != raw_fallback,
        "candidate_count": len(candidates),
        "target_scope_sha256": stable_hash(scope),
        "diagnostic": "TARGET_NATIVE_SEARCH_FOR_UNIQUE_RELATION_BINDING",
    }


def observe_goal_relation_transition(
    ledger: Mapping[str, Any], *, action: str, after_observation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = relation_coverage(ledger)
    updated, receipt = observe_target_transition(
        ledger, action=action, after_observation=after_observation,
    )
    if receipt == "RELATE_SLOT_CLOSED":
        move = _MOVE.match(str(action).lower())
        if move is None:
            raise RuntimeError("relation receipt has no parseable target handle")
        observed_handle = (
            f"{move.group('receptacle')} {move.group('receptacle_id')}"
        )
        bound = updated.get("bound_target_receptacle")
        if bound is None:
            updated["bound_target_receptacle"] = observed_handle
        elif str(bound) != observed_handle:
            receipt = "RELATE_NO_PROGRESS"
    updated = reconcile_bound_relation_objects(updated, after_observation)
    after = relation_coverage(updated)
    return updated, {
        "target_effect_receipt": receipt,
        "relation_coverage_before": before,
        "relation_coverage_after": after,
        "observed_relation_delta": after - before,
        "source_transition_advanced": after > before,
        "source_terminal_observed": abs(after - 1.0) <= 1e-9,
    }


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
