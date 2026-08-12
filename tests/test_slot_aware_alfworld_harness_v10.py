from __future__ import annotations

from motif_transfer.slot_aware_alfworld_harness import (
    initialize_slot_ledger,
    observe_target_transition,
    parameterize_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v10 import (
    CONDITION_SEMANTICS,
    choose_slot_aware_action,
    condition_required_property,
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
    "prohibited_runtime_fields": [
        "source_action_ordinal", "environment_id"
    ],
    "target_grounding": "TARGET_NATIVE_NEURAL_PROBES_ONLY",
    "ir_sha256": "v10-parent-test",
    "induction_split": "development",
    "validation_splits": ["qualification", "heldout"],
    "source_lineage": ["intervention-receipt"],
}


def _ir() -> dict:
    return parameterize_slot_source_ir(PARENT_IR)


def _row(option: str, policy: float, binding: float = 0.99) -> dict:
    return {
        "option": option,
        "applicability": 0.99,
        "binding": binding,
        "completion": 0.9,
        "policy": policy,
    }


def _choose(condition: str, ledger: dict, grounded: dict) -> dict:
    return choose_slot_aware_action(
        condition=condition,
        grounded=grounded,
        history=[],
        ledger=ledger,
        source_ir=_ir(),
        property_probabilities={
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


def test_safety_only_control_keeps_structure_but_never_executes_source() -> None:
    required, _, _ = condition_required_property(
        "find two pencil and put them in desk.",
        {"NONE": 0.95},
        "property_permuted_router",
    )
    assert required == "NONE"
    ledger = initialize_slot_ledger(
        "find two pencil and put them in desk.",
        required_property=required,
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="take pencil 2 from dresser 1",
        after_observation=(
            "You pick up the pencil 2 from the dresser 1."
        ),
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="move pencil 2 to desk 1",
        after_observation="You move the pencil 2 to the desk 1.",
    )
    decision = _choose(
        "property_permuted_router",
        ledger,
        {
            "take pencil 2 from desk 1": _row("ACQUIRE", 0.99),
            "go to dresser 1": _row("SEARCH", 0.8, binding=0.01),
        },
    )
    assert decision["action"] == "go to dresser 1"
    assert decision["slot_safety_shielded"]
    assert not decision["source_admitted"]
    assert decision["source_transition"] is None
    assert decision["condition_semantics"] == (
        "TARGET_NATIVE_COMPLETED_SLOT_SAFETY_ONLY"
    )


def test_full_graph_relation_edge_is_absent_from_node_only_control() -> None:
    ledger = initialize_slot_ledger(
        "find two potato and put them in fridge.",
        required_property="NONE",
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="take potato 3 from countertop 1",
        after_observation=(
            "You pick up the potato 3 from the countertop 1."
        ),
    )
    grounded = {
        "close fridge 1": _row("SEARCH", 0.95, binding=0.01),
        "move potato 3 to fridge 1": _row("PLACE", 0.8),
    }
    authentic = _choose("authentic_slot_ir", ledger, grounded)
    node_only = _choose("edge_permuted_ir", ledger, grounded)
    assert authentic["action"] == "move potato 3 to fridge 1"
    assert authentic["source_transition"]["kind"] == "EDGE"
    assert authentic["changed_effect"]
    assert node_only["action"] == "close fridge 1"
    assert not node_only["source_admitted"]
    assert node_only["condition_semantics"] == CONDITION_SEMANTICS[
        "edge_permuted_ir"
    ]
