from __future__ import annotations

from motif_transfer.slot_aware_alfworld_harness import (
    initialize_slot_ledger,
    observe_target_transition,
    parameterize_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v12 import (
    MINIMUM_SOURCE_EDGE_STEP,
    choose_slot_aware_action,
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
    "ir_sha256": "v12-parent-test",
    "induction_split": "development",
    "validation_splits": ["qualification", "heldout"],
    "source_lineage": ["intervention-receipt"],
}


def _row(option: str, policy: float, binding: float = 0.99) -> dict:
    return {
        "option": option,
        "applicability": 0.99,
        "binding": binding,
        "completion": 0.9,
        "policy": policy,
    }


def _ledger() -> dict:
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
    return ledger


def _choose(step: int) -> dict:
    return choose_slot_aware_action(
        condition="authentic_slot_ir",
        grounded={
            "close fridge 1": _row("SEARCH", 0.95, binding=0.01),
            "move potato 3 to fridge 1": _row("PLACE", 0.8),
        },
        history=[f"look {index}" for index in range(step)],
        ledger=_ledger(),
        source_ir=parameterize_slot_source_ir(PARENT_IR),
        property_probabilities={"NONE": 0.99},
        minimum_property_confidence=0.8,
        minimum_role_binding=0.5,
        minimum_realization_score=0.05,
        minimum_target_policy_ratio=0.05,
        allowed_source_effects=("BIND", "RELATE"),
        active_required_properties=("NONE",),
    )


def test_early_source_edge_abstains_to_target_action() -> None:
    decision = _choose(MINIMUM_SOURCE_EDGE_STEP - 1)
    assert decision["action"] == "close fridge 1"
    assert not decision["source_admitted"]
    assert decision["source_transition"] is None
    assert decision["candidate_source_transition"]["kind"] == "EDGE"
    assert decision["diagnostic"] == (
        "SOURCE_EDGE_EARLY_APPLICABILITY_ABSTENTION"
    )
    assert not decision["source_applicability"]["source_edge_admitted"]


def test_source_edge_is_admitted_at_frozen_minimum_step() -> None:
    decision = _choose(MINIMUM_SOURCE_EDGE_STEP)
    assert decision["action"] == "move potato 3 to fridge 1"
    assert decision["source_admitted"]
    assert decision["source_transition"]["kind"] == "EDGE"
    assert decision["source_applicability"]["source_edge_admitted"]
    assert decision["changed_effect"]
