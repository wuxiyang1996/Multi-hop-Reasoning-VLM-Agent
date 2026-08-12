from __future__ import annotations

from motif_transfer.contracts import stable_hash
from motif_transfer.slot_aware_alfworld_harness import (
    initialize_slot_ledger,
    observe_target_transition,
    parameterize_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v9 import (
    choose_slot_aware_action,
    compile_source_effect_graph,
    condition_required_property,
    route_source_effect,
)


PARENT_IR = {
    "schema_version": "typed-effect-ir-v3",
    "nodes": ["POSITION", "BIND", "MUTATE", "RELATE"],
    "edges": [
        {
            "from": "BIND",
            "to": "MUTATE",
            "guard": "CARRIER_BOUND",
            "supporting_source_tasks": ["doorkey", "unlock"],
        },
        {
            "from": "BIND",
            "to": "RELATE",
            "guard": "CARRIER_BOUND",
            "supporting_source_tasks": ["put_near", "putnext_3d"],
        },
    ],
    "prohibited_runtime_fields": ["source_action_ordinal", "environment_id"],
    "target_grounding": "TARGET_NATIVE_NEURAL_PROBES_ONLY",
    "ir_sha256": "parent-v9-test",
    "induction_split": "development",
    "validation_splits": ["qualification", "heldout"],
    "source_lineage": ["intervention-receipt"],
}


def _ir() -> dict:
    return parameterize_slot_source_ir(PARENT_IR)


def _row(
    option: str,
    policy: float,
    *,
    binding: float = 0.99,
    completion: float = 0.9,
) -> dict:
    return {
        "option": option,
        "applicability": 0.99,
        "binding": binding,
        "completion": completion,
        "policy": policy,
    }


def _bound_relation_ledger() -> dict:
    ledger = initialize_slot_ledger(
        "find two pencil and put them in desk.",
        required_property="NONE",
    )
    ledger, receipt = observe_target_transition(
        ledger,
        action="take pencil 2 from garbagecan 1",
        after_observation=(
            "You pick up the pencil 2 from the garbagecan 1."
        ),
    )
    assert receipt == "BIND_INSTANCE"
    return ledger


def _choose(
    ledger: dict,
    grounded: dict,
    *,
    condition: str = "authentic_slot_ir",
    source_ir: dict | None = None,
    probabilities: dict | None = None,
) -> dict:
    return choose_slot_aware_action(
        condition=condition,
        grounded=grounded,
        history=[],
        ledger=ledger,
        source_ir=source_ir or _ir(),
        property_probabilities=probabilities or {
            "NONE": 0.95,
            "CLEAN": 0.02,
            "HEAT": 0.01,
            "COOL": 0.01,
            "LIGHT": 0.01,
        },
        minimum_property_confidence=0.8,
        minimum_role_binding=0.5,
        minimum_realization_score=0.05,
        minimum_target_policy_ratio=0.05,
        allowed_source_effects=("BIND", "RELATE"),
        active_required_properties=("NONE",),
    )


def test_authentic_route_executes_bound_source_edge() -> None:
    graph = compile_source_effect_graph(
        _ir(), condition="authentic_slot_ir"
    )
    routed = route_source_effect(_bound_relation_ledger(), graph)
    assert routed["effect"] == "RELATE"
    assert routed["source_transition"] == {
        "kind": "EDGE",
        "from": "BIND",
        "to": "RELATE",
        "guard": "TARGET_NATIVE_SLOT_READY_FOR_RELATION",
        "supporting_source_tasks": ["put_near", "putnext_3d"],
        "graph_sha256": graph["graph_sha256"],
    }


def test_reversing_bound_source_edge_removes_relation_route() -> None:
    graph = compile_source_effect_graph(
        _ir(), condition="edge_permuted_ir"
    )
    routed = route_source_effect(_bound_relation_ledger(), graph)
    assert routed["effect"] is None
    assert routed["diagnostic"] == "SOURCE_GRAPH_GUARDED_EDGE_UNAVAILABLE"
    assert routed["source_transition"]["matches"] == 0


def test_source_edge_changes_native_decision_and_graph_control_does_not() -> None:
    ledger = _bound_relation_ledger()
    grounded = {
        "go to sidetable 1": _row("SEARCH", 0.95, binding=0.01),
        "move pencil 2 to desk 1": _row("PLACE", 0.8),
    }
    authentic = _choose(ledger, grounded)
    control = _choose(
        ledger, grounded, condition="edge_permuted_ir"
    )
    assert authentic["action"] == "move pencil 2 to desk 1"
    assert authentic["changed_effect"]
    assert authentic["source_transition"]["guard"] == (
        "TARGET_NATIVE_SLOT_READY_FOR_RELATION"
    )
    assert control["action"] == "go to sidetable 1"
    assert not control["source_admitted"]


def test_mutating_source_destination_changes_requested_effect() -> None:
    source_ir = _ir()
    edge = next(
        row for row in source_ir["edges"]
        if row["guard"] == "TARGET_NATIVE_SLOT_READY_FOR_RELATION"
    )
    edge["to"] = (
        "ACHIEVE_UNARY_GOAL(goal_object_instance, required_property, "
        "unsatisfied_goal_slot)"
    )
    body = dict(source_ir)
    body.pop("ir_sha256")
    source_ir["ir_sha256"] = stable_hash(body)
    decision = _choose(
        _bound_relation_ledger(),
        {
            "go to sidetable 1": _row("SEARCH", 0.95, binding=0.01),
            "move pencil 2 to desk 1": _row("PLACE", 0.8),
        },
        source_ir=source_ir,
    )
    assert decision["requested_source_effect"] == "MUTATE"
    assert decision["diagnostic"] == (
        "SOURCE_GRAPH_EFFECT_OUTSIDE_FROZEN_SCOPE"
    )
    assert decision["action"] == "go to sidetable 1"


def test_explicit_goal_operator_is_not_erased_by_neural_uncertainty() -> None:
    required, neural, support = condition_required_property(
        "find two dishsponge and put them in toilet.",
        {
            "NONE": 0.40,
            "CLEAN": 0.45,
            "HEAT": 0.05,
            "COOL": 0.05,
            "LIGHT": 0.05,
        },
        "authentic_slot_ir",
    )
    assert required == "NONE"
    assert neural == "CLEAN"
    assert support == 0.40


def test_completed_slot_safety_is_independent_of_router_confidence() -> None:
    ledger = _bound_relation_ledger()
    ledger, receipt = observe_target_transition(
        ledger,
        action="move pencil 2 to desk 1",
        after_observation="You move the pencil 2 to the desk 1.",
    )
    assert receipt == "RELATE_SLOT_CLOSED"
    grounded = {
        "take pencil 2 from desk 1": _row("ACQUIRE", 0.99),
        "go to dresser 1": _row("SEARCH", 0.7, binding=0.01),
    }
    uncertain = {
        "NONE": 0.40,
        "CLEAN": 0.35,
        "HEAT": 0.10,
        "COOL": 0.10,
        "LIGHT": 0.05,
    }
    authentic = _choose(
        ledger, grounded, probabilities=uncertain
    )
    target_only = _choose(
        ledger,
        grounded,
        condition="target_only",
        probabilities=uncertain,
    )
    assert authentic["action"] == "go to dresser 1"
    assert authentic["slot_safety_shielded"]
    assert authentic["neural_required_property_support"] == 0.40
    assert target_only["action"] == "take pencil 2 from desk 1"


def test_runtime_structure_blocks_light_goal_scope_leak() -> None:
    required, _, _ = condition_required_property(
        "examine the watch with the desklamp.",
        {"LIGHT": 0.95, "NONE": 0.05},
        "property_permuted_router",
    )
    assert required == "NONE"
    ledger = initialize_slot_ledger(
        "examine the watch with the desklamp.",
        required_property=required,
    )
    decision = _choose(
        ledger,
        {
            "go to shelf 1": _row("SEARCH", 0.9, binding=0.01),
            "take watch 1 from shelf 2": _row("ACQUIRE", 0.8),
        },
        condition="property_permuted_router",
        probabilities={"LIGHT": 0.95, "NONE": 0.05},
    )
    assert not decision["transfer_scope_active"]
    assert not decision["source_admitted"]
    assert decision["diagnostic"] == (
        "STRUCTURAL_TRANSFER_SCOPE_TARGET_ABSTENTION"
    )
