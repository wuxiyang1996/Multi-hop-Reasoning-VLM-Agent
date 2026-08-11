from __future__ import annotations

from motif_transfer.parameterized_alfworld_harness import (
    action_property,
    choose_parameterized_action,
    parameterize_source_ir,
    parameterized_cycle_state,
    property_label_from_actions,
    target_effect_receipt,
    validate_parameterized_source_ir,
)


PARENT_IR = {
    "schema_version": "typed-effect-ir-v3",
    "nodes": ["POSITION", "BIND", "MUTATE", "RELATE"],
    "edges": [
        {
            "from": "BIND",
            "to": "MUTATE",
            "guard": "CARRIER_BOUND",
            "supporting_source_tasks": ["door", "unlock"],
        },
        {
            "from": "BIND",
            "to": "RELATE",
            "guard": "CARRIER_BOUND",
            "supporting_source_tasks": ["put", "put3d"],
        },
    ],
    "prohibited_runtime_fields": ["source_action_ordinal", "environment_id"],
    "target_grounding": "TARGET_NATIVE_NEURAL_PROBES_ONLY",
    "ir_sha256": "parent",
    "induction_split": "development",
    "validation_splits": ["qualification", "heldout"],
    "source_lineage": ["receipt"],
}


def _row(
    option: str, policy: float, *, binding: float = 0.99, completion: float = 0.9
) -> dict:
    return {
        "option": option,
        "applicability": 0.99,
        "binding": binding,
        "completion": completion,
        "policy": policy,
    }


def _parameterized_ir() -> dict:
    parent = dict(PARENT_IR)
    # Parent validation checks the parent hash only through its other contract;
    # parameterization preserves it as immutable lineage.
    return parameterize_source_ir(parent)


def _choose(
    *,
    grounded: dict,
    receipts: list[str],
    probabilities: dict[str, float],
    condition: str = "authentic_parameterized_ir",
) -> dict:
    return choose_parameterized_action(
        condition=condition,
        grounded=grounded,
        history=[],
        effect_receipts=receipts,
        source_ir=_parameterized_ir(),
        property_probabilities=probabilities,
        minimum_property_confidence=0.8,
        minimum_role_binding=0.5,
        minimum_realization_score=0.05,
        minimum_target_policy_ratio=0.1,
    )


def test_parameterized_ir_preserves_source_lineage_and_adds_roles() -> None:
    source_ir = _parameterized_ir()
    validate_parameterized_source_ir(source_ir)
    assert source_ir["parent_ir_sha256"] == "parent"
    assert {row["guard"] for row in source_ir["edges"]} == {
        "TARGET_NEURAL_UNARY_GOAL_UNSATISFIED",
        "TARGET_NEURAL_NO_UNARY_GOAL_OR_SATISFIED",
        "TARGET_NEURAL_UNARY_GOAL_SATISFIED",
    }
    assert "source_task_id" in source_ir["prohibited_runtime_fields"]


def test_role_receipts_ignore_wrong_object_binding() -> None:
    wrong = target_effect_receipt(
        action="take box 1 from cabinet 1",
        grounding=_row("ACQUIRE", 0.9, binding=0.01),
        required_property="NONE",
        minimum_role_binding=0.5,
    )
    right = target_effect_receipt(
        action="take mug 1 from cabinet 1",
        grounding=_row("ACQUIRE", 0.9),
        required_property="COOL",
        minimum_role_binding=0.5,
    )
    assert wrong == "IGNORE"
    assert right == "BIND_ROLE"
    assert parameterized_cycle_state([wrong, right])["goal_object_bound"]


def test_property_subtype_receipt_requires_router_property() -> None:
    assert action_property("cool mug 1 with fridge 1") == "COOL"
    assert property_label_from_actions(["look", "cool mug 1 with fridge 1"]) == "COOL"
    row = _row("TRANSFORM", 0.8)
    assert target_effect_receipt(
        action="clean mug 1 with sinkbasin 1",
        grounding=row,
        required_property="COOL",
        minimum_role_binding=0.5,
    ) == "IGNORE"
    assert target_effect_receipt(
        action="cool mug 1 with fridge 1",
        grounding=row,
        required_property="COOL",
        minimum_role_binding=0.5,
    ) == "MUTATE_REQUIRED_PROPERTY"


def test_authentic_router_orders_bound_cool_before_relation() -> None:
    grounded = {
        "go to fridge 1": _row("SEARCH", 0.95, binding=0.01),
        "clean mug 1 with sinkbasin 1": _row("TRANSFORM", 0.9),
        "cool mug 1 with fridge 1": _row("TRANSFORM", 0.8),
        "move mug 1 to cabinet 1": _row("PLACE", 0.99),
    }
    decision = _choose(
        grounded=grounded,
        receipts=["BIND_ROLE"],
        probabilities={"COOL": 0.98, "NONE": 0.02},
    )
    assert decision["action"] == "cool mug 1 with fridge 1"
    assert decision["requested_source_effect"] == "MUTATE"
    assert decision["required_property"] == "COOL"


def test_direct_relation_and_property_control_are_distinct() -> None:
    grounded = {
        "go to drawer 1": _row("SEARCH", 0.7, binding=0.01),
        "clean mug 1 with sinkbasin 1": _row("TRANSFORM", 0.6),
        "move mug 1 to drawer 1": _row("PLACE", 0.9),
    }
    probabilities = {"NONE": 0.99, "CLEAN": 0.01}
    authentic = _choose(
        grounded=grounded,
        receipts=["BIND_ROLE"],
        probabilities=probabilities,
    )
    control = _choose(
        grounded=grounded,
        receipts=["BIND_ROLE"],
        probabilities=probabilities,
        condition="property_permuted_router",
    )
    assert authentic["requested_source_effect"] == "RELATE"
    assert authentic["action"] == "move mug 1 to drawer 1"
    assert control["requested_source_effect"] == "MUTATE"
    assert control["action"] == "clean mug 1 with sinkbasin 1"


def test_missing_parameterized_effect_abstains_to_exact_target_policy() -> None:
    grounded = {
        "go to fridge 1": _row("SEARCH", 0.9, binding=0.01),
        "clean mug 1 with sinkbasin 1": _row("TRANSFORM", 0.8),
    }
    decision = _choose(
        grounded=grounded,
        receipts=["BIND_ROLE"],
        probabilities={"COOL": 0.99, "NONE": 0.01},
    )
    assert decision["action"] == "go to fridge 1"
    assert not decision["source_admitted"]
    assert decision["diagnostic"] == (
        "PARAMETERIZED_EFFECT_UNAVAILABLE_TARGET_ABSTENTION"
    )
