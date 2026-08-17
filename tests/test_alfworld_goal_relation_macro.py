from __future__ import annotations

from motif_transfer.alfworld_goal_relation_macro import (
    AUTHENTIC,
    CARDINALITY_CONTROL,
    TargetRelationExecutionState,
    choose_goal_relation_action,
    observe_goal_relation_transition,
    target_relation_state,
)
from motif_transfer.alfworld_goal_relation_macro_v4 import (
    choose_goal_relation_action as choose_v4_action,
)
from motif_transfer.alfworld_goal_relation_macro_v5 import (
    choose_goal_relation_action as choose_v5_action,
)
from motif_transfer.alfworld_goal_acquisition_v9 import (
    TargetAcquisitionExecutionState,
    choose_goal_relation_action as choose_v9_action,
    configure_source_acquisition,
)
from motif_transfer.alfworld_goal_acquisition_v10 import (
    choose_goal_relation_action as choose_v10_action,
    configure_source_acquisition as configure_v10_acquisition,
)
from motif_transfer.source_goal_acquisition_induction import BINDING_OPERATOR
from motif_transfer.contracts import stable_hash
from motif_transfer.slot_aware_alfworld_harness import initialize_slot_ledger
from motif_transfer.target_structural_induction import anonymous_operator_descriptor


def _artifact() -> dict:
    operator = anonymous_operator_descriptor(
        "UPDATE", "ENTITY_GOAL_RELATION", 2, "RELATION_COVERAGE",
    )
    body = {
        "schema_version": "source-induced-goal-relation-macro-v3",
        "artifact_version": "SOURCE_INDUCED_GOAL_RELATION_MACRO_V3",
        "status": "SOURCE_GOAL_RELATION_MACRO_AWAITING_FRESH_CONFIRMATION",
        "source_receipts_sha256": "a" * 64,
        "induction_authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "operator_types": [operator],
        "program": {
            "entry_operator_type_id": operator["operator_type_id"],
            "transitions": [{
                "from_operator_type_id": operator["operator_type_id"],
                "observed_effect_guard": {
                    "feature": "entity_goal_relation_coverage",
                    "change_sign": "INCREASE",
                },
                "to_operator_type_id": operator["operator_type_id"],
                "cardinality": "ONE_OR_MORE",
            }],
            "terminal_predicates": [{
                "predicate_family": "ENTITY_GOAL_RELATION",
                "arity": 2,
                "value_kind": "RELATION_COVERAGE",
                "feature": "entity_goal_relation_coverage",
                "operator": "EQ",
                "value": 1.0,
            }],
            "terminal_rule": "INDUCED_PREDICATE_CONJUNCTION_AFTER_TYPED_EFFECT",
            "abstention_rule": {
                "zero_target_bindings": "ABSTAIN",
                "multiple_target_bindings": "ABSTAIN",
                "nonpositive_observed_relation_delta": "ABSTAIN",
                "terminal_predicate_unobservable": "ABSTAIN",
            },
        },
        "induction_diagnostics": {},
        "named_controller_template_used": False,
        "forbidden_named_controller_templates": [
            "EXPLORE", "BACKTRACK", "COMMIT",
        ],
        "target_binding": "TARGET_NATIVE_NEURAL_GOAL_RELATION_GROUNDER",
        "target_data_read": False,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _row(option: str, policy: float = 0.8) -> dict:
    return {
        "option": option,
        "policy": policy,
        "applicability": 0.95,
        "completion": 0.95,
        "binding": 0.99,
        "required_option": option,
    }


def _effect_head() -> dict:
    return {
        "feature_names": ["verb_move"],
        "means": [0.0],
        "scales": [1.0],
        "weights": [12.0],
        "intercept": -6.0,
    }


def _acquisition_artifact() -> dict:
    control = anonymous_operator_descriptor(
        "UPDATE", "CONTROL_STATE", 1, "POSITION",
    )
    relation = _artifact()["operator_types"][0]
    body = {
        "schema_version": "source-induced-goal-acquisition-program-v1",
        "artifact_version": "SOURCE_INDUCED_GOAL_ACQUISITION_PROGRAM_V1",
        "status": "SOURCE_GOAL_ACQUISITION_AWAITING_FRESH_CONFIRMATION",
        "source_receipts_sha256": "b" * 64,
        "relation_artifact_sha256": _artifact()["artifact_sha256"],
        "induction_authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "operator_types": [control, BINDING_OPERATOR, relation],
        "program": {
            "measured_precondition": {
                "feature": "positive_relation_intervention_cardinality",
                "operator": "EQ", "value": 0,
            },
            "acquisition_operator_type_ids": [control["operator_type_id"]],
            "binding_operator_type_id": BINDING_OPERATOR["operator_type_id"],
            "relation_operator_type_id": relation["operator_type_id"],
            "relation_binding_cardinality": {"operator": "EQ", "value": 1},
            "transition_graph": [],
            "binding_to_relation": {},
            "abstention_rule": {},
        },
        "induction_diagnostics": {},
        "named_controller_template_used": False,
        "forbidden_named_controller_templates": [
            "EXPLORE", "BACKTRACK", "COMMIT",
        ],
        "target_binding": "TARGET_NATIVE_NEURAL_STRUCTURAL_GROUNDER",
        "target_data_read": False,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def test_recurrent_program_preserves_one_target_relation_handle() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["carried_object"] = "apple 1"
    ledger, effect = observe_goal_relation_transition(
        ledger,
        action="move apple 1 to drawer 2",
        after_observation="You move the apple 1 to the drawer 2.",
    )
    assert effect["source_transition_advanced"] is True
    assert ledger["bound_target_receptacle"] == "drawer 2"
    ledger["carried_object"] = "apple 2"
    grounded = {
        "move apple 2 to drawer 1": _row("PLACE", 0.99),
        "move apple 2 to drawer 2": _row("PLACE", 0.80),
        "look": _row("SEARCH", 0.70),
    }
    decision = choose_goal_relation_action(
        condition=AUTHENTIC,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=TargetRelationExecutionState(),
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=4,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )
    assert decision["action"] == "move apple 2 to drawer 2"
    assert decision["diagnostic"] == (
        "SOURCE_MACRO_TARGET_NATIVE_RELATION_REALIZATION"
    )


def test_exactly_one_control_stops_after_first_relation() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    grounded = {"look": _row("SEARCH")}
    decision = choose_goal_relation_action(
        condition=CARDINALITY_CONTROL,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=TargetRelationExecutionState(),
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=4,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )
    assert decision["program_active"] is False
    assert decision["diagnostic"] == "EXACTLY_ONE_RELATION_PROGRAM_TERMINATED"
    assert target_relation_state(ledger)["remaining_slots"] == 1


def test_v4_waits_for_observed_relation_effect_before_transfer() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    grounded = {
        "go to drawer 1": _row("SEARCH", 0.7),
        "take apple 1 from table 1": _row("PICKUP", 0.99),
    }
    decision = choose_v4_action(
        condition=AUTHENTIC,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=TargetRelationExecutionState(),
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=0,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )
    assert decision["program_active"] is False
    assert decision["source_admitted"] is False
    assert decision["diagnostic"] == (
        "SOURCE_RECURRENCE_AWAITS_FIRST_OBSERVED_RELATION"
    )


def _choose_v5(ledger: dict, grounded: dict) -> dict:
    return choose_v5_action(
        condition=AUTHENTIC,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=TargetRelationExecutionState(),
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=10,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )


def test_v5_enforces_source_induced_zero_binding_abstention() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    decision = _choose_v5(ledger, {"go to table 1": _row("SEARCH")})
    assert decision["action"] == "go to table 1"
    assert decision["source_admitted"] is False
    assert decision["diagnostic"] == (
        "SOURCE_ARTIFACT_ZERO_BINDINGS_ABSTENTION"
    )


def test_v5_native_admissibility_does_not_double_veto_unique_binding() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    row = _row("ACQUIRE")
    row["applicability"] = 0.0001
    decision = _choose_v5(ledger, {
        "take apple 2 from table 1": row,
        "go to table 2": _row("SEARCH"),
    })
    assert decision["action"] == "take apple 2 from table 1"
    assert decision["source_admitted"] is True


def test_v9_grounds_source_acquisition_to_neural_target_search() -> None:
    acquisition = _acquisition_artifact()
    configure_source_acquisition(acquisition, {
        "source_gate_passed": True,
        "artifact_sha256": acquisition["artifact_sha256"],
    })
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    grounded = {
        "go to table 1": _row("SEARCH", 0.90),
        "go to shelf 1": _row("SEARCH", 0.70),
    }
    state = TargetAcquisitionExecutionState()
    kwargs = dict(
        condition=AUTHENTIC,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=state,
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=10,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )
    first = choose_v9_action(**kwargs)
    second = choose_v9_action(**kwargs)
    assert first["action"] == "go to table 1"
    assert second["action"] == "go to shelf 1"
    assert first["source_admitted"] is True
    assert first["diagnostic"] == (
        "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED"
    )


def test_v9_abstains_when_source_operator_has_no_target_native_binding() -> None:
    acquisition = _acquisition_artifact()
    configure_source_acquisition(acquisition, {
        "source_gate_passed": True,
        "artifact_sha256": acquisition["artifact_sha256"],
    })
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    row = _row("SEARCH")
    row["required_option"] = "ACQUIRE"
    decision = choose_v9_action(
        condition=AUTHENTIC,
        grounded={"go to table 1": row},
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=TargetAcquisitionExecutionState(),
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=10,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )
    assert decision["source_admitted"] is False
    assert decision["diagnostic"] == (
        "SOURCE_ACQUISITION_TARGET_GROUNDING_EXHAUSTED"
    )


def _choose_v10(ledger: dict, grounded: dict) -> dict:
    acquisition = _acquisition_artifact()
    configure_v10_acquisition(acquisition, {
        "source_gate_passed": True,
        "artifact_sha256": acquisition["artifact_sha256"],
    })
    return choose_v10_action(
        condition=AUTHENTIC,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=TargetAcquisitionExecutionState(),
        source_artifact=_artifact(),
        target_causal_effect_head=_effect_head(),
        step=10,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )


def test_v10_relation_acquisition_preserves_bound_handle() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    ledger["carried_object"] = "apple 2"
    decision = _choose_v10(ledger, {
        "go to drawer 2": _row("SEARCH", 0.99),
        "go to drawer 1": _row("SEARCH", 0.70),
    })
    assert decision["action"] == "go to drawer 1"
    assert decision["source_admitted"] is True


def test_v10_abstains_on_typed_distractor_binding() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 1"
    ledger["observed_locations"] = {"apple 1": "drawer 1"}
    ledger["completed_objects"] = ["apple 1"]
    ledger["carried_object"] = "cloth 2"
    decision = _choose_v10(ledger, {
        "go to drawer 1": _row("SEARCH", 0.90),
    })
    assert decision["source_admitted"] is False
    assert decision["diagnostic"] == (
        "SOURCE_ACQUISITION_TYPED_ENTITY_MISMATCH_ABSTENTION"
    )
