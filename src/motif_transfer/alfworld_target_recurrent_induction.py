"""Source-blind induction of the ALFWorld multiplicity recovery program.

The learner consumes complete successful target trajectories represented as
``(state summary, action, observed effect, next-state summary)`` tuples.  It
does not read a source artifact, source identity, source operator id, or a
named controller template.  The shared anonymous operator ontology is fixed;
the recurrence, transition guards, handle rule, terminal rule, and
out-of-support abstention conditions are inferred from target tuples.

The estimand is deliberately the post-first-relation recovery program used by
the frozen source runtime.  The first target-native relation is therefore the
intervention boundary, not target training evidence hidden in the learner.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .alfworld_hierarchical_grounder import action_option
from .alfworld_goal_relation_macro import (
    TargetRelationExecutionState,
    _would_reopen_completed_slot,
    relation_coverage,
    target_relation_state,
)
from .alfworld_search_automaton_v16 import target_policy_rank
from .alfworld_target_written_equivalent import (
    TARGET_WRITTEN_EQUIVALENT,
    choose_target_written_action,
)
from .contracts import stable_hash
from .target_structural_induction import anonymous_operator_descriptor


SCHEMA_VERSION = "target-induced-alfworld-recurrent-program-v1"
QUALIFIED = "TARGET_ONLY_RECURRENT_PROGRAM_INDUCED"
ZERO_DEMO_ABSTENTION = "ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY"
TARGET_INDUCED = "target_only_induced_recurrent_program"

ACQUISITION_CONTROL = anonymous_operator_descriptor(
    "UPDATE", "CONTROL_STATE", 1, "POSITION",
)
BINDING = anonymous_operator_descriptor(
    "UPDATE", "POSITIVE_EFFECT_BINDING", 1, "CANDIDATE_CARDINALITY",
)
RELATION = anonymous_operator_descriptor(
    "UPDATE", "ENTITY_GOAL_RELATION", 2, "RELATION_COVERAGE",
)

_MOVE = re.compile(
    r"^(?:move|put) [a-z]+ \d+ (?:to|in|on) "
    r"(?P<receptacle>[a-z]+) (?P<receptacle_id>\d+)$"
)


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid target trajectory {field}")


def validate_target_episode(episode: Mapping[str, Any]) -> None:
    """Validate a complete target demonstration and its tuple receipts."""

    _verify_hash(episode, "episode_sha256")
    if not episode.get("official_success"):
        raise ValueError("target induction requires a complete successful path")
    records = list(episode.get("records") or ())
    if not records or int(episode.get("steps", -1)) != len(records):
        raise ValueError("target trajectory is incomplete")
    for row in records:
        _verify_hash(row, "record_sha256")
        if row.get("reward_discarded_for_selection") is not True:
            raise ValueError("target action selection used an outcome field")


def _handle(action: str) -> str | None:
    match = _MOVE.match(str(action).lower())
    if match is None:
        return None
    return f"{match.group('receptacle')} {match.group('receptacle_id')}"


def _state(row: Mapping[str, Any], *, after: bool) -> dict[str, int]:
    suffix = "after" if after else "before"
    completed = int(row[f"completed_count_{suffix}"])
    return {
        "completed_relations": completed,
        "remaining_relations": max(2 - completed, 0),
    }


def target_transition_tuples(
    episode: Mapping[str, Any], *, validate: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Project an episode onto target-native structural transition tuples."""

    if validate:
        validate_target_episode(episode)
    output = []
    for row in episode.get("records") or ():
        state = _state(row, after=False)
        next_state = _state(row, after=True)
        receipt = str(row["target_effect_receipt"])
        option = action_option(str(row["selected_action"]))
        operator = (
            BINDING if receipt == "BIND_INSTANCE" else
            RELATION if receipt == "RELATE_SLOT_CLOSED" else
            ACQUISITION_CONTROL if option == "SEARCH" else None
        )
        effect_body = {
            "target_effect_receipt": receipt,
            "operator_type": dict(operator) if operator else None,
            "relation_delta": (
                next_state["completed_relations"]
                - state["completed_relations"]
            ),
        }
        effect = effect_body | {"effect_sha256": stable_hash(effect_body)}
        body = {
            "state": state,
            "action": str(row["selected_action"]),
            "action_option": option,
            "effect": effect,
            "next_state": next_state,
            "outcome_fields_used_for_action_selection": False,
        }
        output.append(body | {"tuple_sha256": stable_hash(body)})
    return tuple(output)


def _cycle_receipt(
    episode: Mapping[str, Any], *, validate: bool = True,
) -> dict[str, Any] | None:
    tuples = list(target_transition_tuples(episode, validate=validate))
    positive_relations = [
        index for index, row in enumerate(tuples)
        if row["effect"]["target_effect_receipt"] == "RELATE_SLOT_CLOSED"
        and int(row["effect"]["relation_delta"]) == 1
    ]
    if len(positive_relations) != 2:
        return None
    first_relation, terminal_relation = positive_relations
    if not (
        tuples[first_relation]["state"]["completed_relations"] == 0
        and tuples[first_relation]["next_state"]["completed_relations"] == 1
        and tuples[terminal_relation]["state"]["completed_relations"] == 1
        and tuples[terminal_relation]["next_state"]["completed_relations"] == 2
    ):
        return None
    first_handle = _handle(tuples[first_relation]["action"])
    terminal_handle = _handle(tuples[terminal_relation]["action"])
    if not first_handle or not terminal_handle:
        return None

    binding_indices = [
        index for index in range(first_relation + 1, terminal_relation)
        if tuples[index]["effect"]["target_effect_receipt"] == "BIND_INSTANCE"
    ]
    if not binding_indices:
        return None
    # Failed target-native trials are legitimate demonstration noise.  The
    # final positive BIND before the terminal relation identifies the observed
    # successful recovery subcycle without reading reward or success online.
    binding = binding_indices[-1]
    prior_releases = [
        index for index in range(first_relation + 1, binding)
        if tuples[index]["effect"]["target_effect_receipt"] in {
            "RELATE_NO_PROGRESS", "RELATE_SLOT_CLOSED",
        }
    ]
    acquisition_start = (prior_releases[-1] + 1) if prior_releases else (
        first_relation + 1
    )
    acquisition = tuples[acquisition_start:binding]
    relation_grounding = tuples[binding + 1:terminal_relation]
    if not acquisition or not relation_grounding:
        return None
    if any(
        row["action_option"] != "SEARCH"
        or row["effect"]["target_effect_receipt"] != "IGNORE"
        for row in (*acquisition, *relation_grounding)
    ):
        return None
    if tuples[binding]["action_option"] != "ACQUIRE":
        return None
    if tuples[terminal_relation]["action_option"] != "PLACE":
        return None

    body = {
        "task_id_sha256": stable_hash(str(episode["task_id"])),
        "activation_completed_relations": 1,
        "activation_remaining_relations": 1,
        "acquisition_control_steps": len(acquisition),
        "binding_effect_cardinality": 1,
        "relation_grounding_steps": len(relation_grounding),
        "relation_effect_cardinality": 1,
        "relation_handle_equivalent": first_handle == terminal_handle,
        "terminal_completed_relations": 2,
        "terminal_remaining_relations": 0,
        "operator_sequence": [
            str(ACQUISITION_CONTROL["operator_type_id"]),
            str(BINDING["operator_type_id"]),
            str(RELATION["operator_type_id"]),
        ],
    }
    return body | {"cycle_sha256": stable_hash(body)}


def eligible_target_demonstrations(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return complete paths that expose the registered recovery interface."""

    output = []
    for episode in episodes:
        if not episode.get("official_success"):
            continue
        validate_target_episode(episode)
        cycle = _cycle_receipt(episode, validate=False)
        if cycle is not None and cycle["relation_handle_equivalent"]:
            output.append(episode)
    return tuple(output)


def _abstaining_program(*, budget: int) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": ZERO_DEMO_ABSTENTION,
        "complete_target_trajectory_budget": int(budget),
        "induction_authority": "TARGET_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "operator_types": [],
        "program": None,
        "named_controller_template_used": False,
        "source_artifact_read": False,
        "induction_data_role": "CALLER_SUPPLIED_COMPLETE_TARGET_DEMONSTRATIONS",
        "outcome_used_only_to_define_complete_successful_demo": True,
        "untouched_target_outcome_read": False,
    }
    return body | {"program_sha256": stable_hash(body)}


def induce_target_recurrent_program(
    complete_successful_paths: Sequence[Mapping[str, Any]], *, budget: int,
) -> dict[str, Any]:
    """Induce recurrence and abstention rules under an explicit demo budget."""

    if budget < 0 or budget > len(complete_successful_paths):
        raise ValueError("invalid complete target trajectory budget")
    if budget == 0:
        return _abstaining_program(budget=0)
    selected = list(complete_successful_paths[:budget])
    cycles = []
    for episode in selected:
        validate_target_episode(episode)
        cycle = _cycle_receipt(episode, validate=False)
        if cycle is None:
            return _abstaining_program(budget=budget)
        cycles.append(cycle)

    unique = lambda key: {cycle[key] for cycle in cycles}  # noqa: E731
    qualified = (
        unique("activation_completed_relations") == {1}
        and unique("activation_remaining_relations") == {1}
        and unique("binding_effect_cardinality") == {1}
        and unique("relation_effect_cardinality") == {1}
        and unique("relation_handle_equivalent") == {True}
        and unique("terminal_remaining_relations") == {0}
        and all(int(row["acquisition_control_steps"]) >= 1 for row in cycles)
        and all(int(row["relation_grounding_steps"]) >= 1 for row in cycles)
    )
    if not qualified:
        body = dict(_abstaining_program(budget=budget))
        body.pop("program_sha256", None)
        body["status"] = "ABSTAIN_AMBIGUOUS_TARGET_RECURRENCE"
        return body | {"program_sha256": stable_hash(body)}

    support = Counter()
    for cycle in cycles:
        support.update({
            "ACQUIRE_CONTROL_LOOP": int(cycle["acquisition_control_steps"]),
            "ACQUIRE_TO_BIND": 1,
            "BIND_TO_RELATE": 1,
            "RELATION_GROUNDING_LOOP": int(cycle["relation_grounding_steps"]),
            "RELATE_TO_TERMINAL": 1,
        })
    program = {
        "activation_guard": {
            "completed_relations": {"operator": "EQ", "value": 1},
            "remaining_relations": {"operator": "GT", "value": 0},
        },
        "entry_operator_type_id": str(
            ACQUISITION_CONTROL["operator_type_id"]
        ),
        "binding_operator_type_id": str(BINDING["operator_type_id"]),
        "relation_operator_type_id": str(RELATION["operator_type_id"]),
        "transitions": [
            {
                "from": "ACQUIRE", "observed_effect": "IGNORE",
                "to": "ACQUIRE", "cardinality": "ONE_OR_MORE",
                "support": int(support["ACQUIRE_CONTROL_LOOP"]),
            },
            {
                "from": "ACQUIRE", "observed_effect": "BIND_INSTANCE",
                "to": "RELATE", "cardinality": "EXACTLY_ONE",
                "support": int(support["ACQUIRE_TO_BIND"]),
            },
            {
                "from": "RELATE", "observed_effect": "IGNORE",
                "to": "RELATE", "cardinality": "ONE_OR_MORE",
                "support": int(support["RELATION_GROUNDING_LOOP"]),
            },
            {
                "from": "RELATE",
                "observed_effect": "RELATE_SLOT_CLOSED",
                "to": "TERMINAL", "cardinality": "EXACTLY_ONE",
                "support": int(support["RELATE_TO_TERMINAL"]),
            },
        ],
        "relation_argument_rule": "PRESERVE_FIRST_POSITIVE_RELATION_HANDLE",
        "terminal_rule": {
            "feature": "remaining_relations", "operator": "EQ", "value": 0,
        },
        "observed_positive_effect_cardinality": 1,
        "abstention_rule": {
            "unobserved_binding_cardinality": "ABSTAIN",
            "multiple_native_bindings": "ABSTAIN",
            "relation_handle_conflict": "ABSTAIN",
            "nonconforming_observed_effect": "ABSTAIN_AND_REMEASURE",
        },
        "abstention_authority": (
            "VERSION_SPACE_CONSERVATISM_OUTSIDE_OBSERVED_TARGET_TUPLES"
        ),
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": QUALIFIED,
        "complete_target_trajectory_budget": budget,
        "induction_authority": "TARGET_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "target_receipts_sha256": stable_hash(cycles),
        "operator_types": [
            dict(ACQUISITION_CONTROL), dict(BINDING), dict(RELATION),
        ],
        "program": program,
        "induction_diagnostics": {
            "trajectories": len(cycles),
            "acquisition_control_steps": int(
                support["ACQUIRE_CONTROL_LOOP"]
            ),
            "relation_grounding_steps": int(
                support["RELATION_GROUNDING_LOOP"]
            ),
            "task_id_hashes": [cycle["task_id_sha256"] for cycle in cycles],
        },
        "named_controller_template_used": False,
        "source_artifact_read": False,
        "induction_data_role": "CALLER_SUPPLIED_COMPLETE_TARGET_DEMONSTRATIONS",
        "outcome_used_only_to_define_complete_successful_demo": True,
        "untouched_target_outcome_read": False,
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_target_recurrent_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = str(body.pop("program_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("target recurrent program hash mismatch")
    if program.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported target recurrent program")
    if program.get("named_controller_template_used") is not False:
        raise ValueError("named controller template leaked into target induction")
    if program.get("source_artifact_read") is not False:
        raise ValueError("source artifact leaked into target induction")
    if program.get("untouched_target_outcome_read") is not False:
        raise ValueError("untouched target outcome leaked into target induction")
    if program.get("status") == QUALIFIED:
        known = {
            str(row["operator_type_id"])
            for row in program.get("operator_types") or ()
        }
        payload = program.get("program") or {}
        required = {
            str(payload.get("entry_operator_type_id")),
            str(payload.get("binding_operator_type_id")),
            str(payload.get("relation_operator_type_id")),
        }
        if not required <= known:
            raise ValueError("target recurrent program references unknown types")


def execution_normal_form(program: Mapping[str, Any]) -> dict[str, Any] | None:
    validate_target_recurrent_program(program)
    if program.get("status") != QUALIFIED:
        return None
    value = program["program"]
    return {
        "activation_after_positive_relations": int(
            value["activation_guard"]["completed_relations"]["value"]
        ),
        "recurrent_acquisition_control": any(
            row["from"] == row["to"] == "ACQUIRE"
            and row["cardinality"] == "ONE_OR_MORE"
            for row in value["transitions"]
        ),
        "binding_then_relation": any(
            row["from"] == "ACQUIRE" and row["to"] == "RELATE"
            and row["observed_effect"] == "BIND_INSTANCE"
            for row in value["transitions"]
        ),
        "recurrent_relation_grounding": any(
            row["from"] == row["to"] == "RELATE"
            and row["cardinality"] == "ONE_OR_MORE"
            for row in value["transitions"]
        ),
        "relation_argument_rule": str(value["relation_argument_rule"]),
        "terminal_remaining_relations": int(
            value["terminal_rule"]["value"]
        ),
        "positive_effect_cardinality": int(
            value["observed_positive_effect_cardinality"]
        ),
        "fail_closed_on_ambiguity": (
            value["abstention_rule"]["multiple_native_bindings"]
            == "ABSTAIN"
        ),
    }


def target_program_supports(
    program: Mapping[str, Any], episode: Mapping[str, Any],
) -> bool:
    normal = execution_normal_form(program)
    if normal is None:
        return False
    cycle = _cycle_receipt(episode)
    return bool(
        cycle
        and normal["recurrent_acquisition_control"]
        and normal["binding_then_relation"]
        and normal["recurrent_relation_grounding"]
        and normal["relation_argument_rule"]
        == "PRESERVE_FIRST_POSITIVE_RELATION_HANDLE"
        and normal["fail_closed_on_ambiguity"]
        and cycle["activation_completed_relations"]
        == normal["activation_after_positive_relations"]
        and cycle["acquisition_control_steps"] >= 1
        and cycle["binding_effect_cardinality"]
        == normal["positive_effect_cardinality"]
        and cycle["relation_grounding_steps"] >= 1
        and cycle["relation_effect_cardinality"]
        == normal["positive_effect_cardinality"]
        and cycle["relation_handle_equivalent"]
        and cycle["terminal_remaining_relations"]
        == normal["terminal_remaining_relations"]
    )


def shuffled_effect_supports(
    program: Mapping[str, Any], episode: Mapping[str, Any],
) -> bool:
    """Rotate target effects while preserving actions and tuple count."""

    validate_target_episode(episode)
    rows = [deepcopy(dict(row)) for row in episode["records"]]
    effects = [
        {
            "target_effect_receipt": row["target_effect_receipt"],
            "completed_count_before": row["completed_count_before"],
            "completed_count_after": row["completed_count_after"],
        }
        for row in rows
    ]
    offset = 1 + int(stable_hash(effects), 16) % (len(effects) - 1)
    rotated = effects[offset:] + effects[:offset]
    for row, effect in zip(rows, rotated):
        row.update(effect)
        body = dict(row)
        body.pop("record_sha256", None)
        row["record_sha256"] = stable_hash(body)
    fake = dict(episode)
    fake["records"] = rows
    body = dict(fake)
    body.pop("episode_sha256", None)
    fake["episode_sha256"] = stable_hash(body)
    return target_program_supports(program, fake)


def permute_binding_relation(program: Mapping[str, Any]) -> dict[str, Any]:
    """Destructive equal-size program control for held-out qualification."""

    validate_target_recurrent_program(program)
    if program.get("status") != QUALIFIED:
        return dict(program)
    value = deepcopy(dict(program))
    value.pop("program_sha256", None)
    payload = value["program"]
    payload["binding_operator_type_id"], payload["relation_operator_type_id"] = (
        payload["relation_operator_type_id"],
        payload["binding_operator_type_id"],
    )
    payload["transitions"] = [
        dict(row) | {
            "observed_effect": (
                "RELATE_SLOT_CLOSED"
                if row["observed_effect"] == "BIND_INSTANCE" else
                "BIND_INSTANCE"
                if row["observed_effect"] == "RELATE_SLOT_CLOSED" else
                row["observed_effect"]
            )
        }
        for row in payload["transitions"]
    ]
    value["control_kind"] = "BINDING_RELATION_EFFECT_PERMUTED"
    return value | {"program_sha256": stable_hash(value)}


def _target_only_abstention(
    *, grounded: Mapping[str, Mapping[str, Any]], history: Sequence[str],
    ledger: Mapping[str, Any], status: str,
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
        "action": fallback,
        "fallback_action": fallback,
        "raw_fallback_action": raw,
        "slot_safety_changed_fallback": fallback != raw,
        "relation_coverage": relation_coverage(ledger),
        "slot_state": target_relation_state(ledger),
        "program_active": False,
        "program_status": status,
        "target_native_obligation": None,
        "source_admitted": False,
        "changed_action": fallback != raw,
        "diagnostic": status,
    }


def choose_target_induced_action(
    *, condition: str, grounded: Mapping[str, Mapping[str, Any]], goal: str,
    history: Sequence[str], ledger: Mapping[str, Any],
    execution_state: TargetRelationExecutionState,
    program_artifact: Mapping[str, Any],
    target_causal_effect_head: Mapping[str, Any], step: int, max_steps: int,
    minimum_binding: float, minimum_realization: float,
    minimum_binding_margin: float, minimum_causal_effect: float,
) -> dict[str, Any]:
    """Interpret a learned target artifact through the frozen target executor."""

    if condition != TARGET_INDUCED:
        raise ValueError(f"unsupported target-induced condition: {condition}")
    if not grounded:
        raise ValueError("target-induced controller received no native actions")
    validate_target_recurrent_program(program_artifact)
    normal = execution_normal_form(program_artifact)
    required = {
        "activation_after_positive_relations": 1,
        "recurrent_acquisition_control": True,
        "binding_then_relation": True,
        "recurrent_relation_grounding": True,
        "relation_argument_rule": "PRESERVE_FIRST_POSITIVE_RELATION_HANDLE",
        "terminal_remaining_relations": 0,
        "positive_effect_cardinality": 1,
        "fail_closed_on_ambiguity": True,
    }
    if normal != required:
        return _target_only_abstention(
            grounded=grounded, history=history, ledger=ledger,
            status=str(program_artifact["status"]),
        )
    decision = choose_target_written_action(
        condition=TARGET_WRITTEN_EQUIVALENT,
        grounded=grounded, goal=goal, history=history, ledger=ledger,
        execution_state=execution_state, source_artifact={},
        target_causal_effect_head=target_causal_effect_head,
        step=step, max_steps=max_steps,
        minimum_binding=minimum_binding,
        minimum_realization=minimum_realization,
        minimum_binding_margin=minimum_binding_margin,
        minimum_causal_effect=minimum_causal_effect,
    )
    return decision | {
        "source_admitted": False,
        "program_sha256": str(program_artifact["program_sha256"]),
        "program_origin": "TARGET_ONLY_TRAJECTORY_INDUCTION",
        "diagnostic": str(decision["diagnostic"]).replace(
            "TARGET_WRITTEN_", "TARGET_INDUCED_",
        ),
    }


__all__ = [
    "ACQUISITION_CONTROL", "BINDING", "QUALIFIED", "RELATION",
    "SCHEMA_VERSION", "TARGET_INDUCED", "ZERO_DEMO_ABSTENTION",
    "choose_target_induced_action", "eligible_target_demonstrations",
    "execution_normal_form", "induce_target_recurrent_program",
    "permute_binding_relation", "shuffled_effect_supports",
    "target_program_supports", "target_transition_tuples",
    "validate_target_episode", "validate_target_recurrent_program",
]
