from __future__ import annotations

from motif_transfer.slot_aware_alfworld_harness import (
    choose_slot_aware_action,
    initialize_slot_ledger,
    observe_target_transition,
    parameterize_slot_source_ir,
    parse_goal_spec,
    slot_state,
    validate_slot_source_ir,
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


def _ir() -> dict:
    return parameterize_slot_source_ir(PARENT_IR)


def _choose(ledger: dict, grounded: dict, *, condition: str = "authentic_slot_ir") -> dict:
    required = ledger["goal_spec"]["required_property"]
    probabilities = {name: 0.001 for name in ("NONE", "CLEAN", "HEAT", "COOL", "LIGHT")}
    probabilities[required] = 0.996
    return choose_slot_aware_action(
        condition=condition,
        grounded=grounded,
        history=[],
        ledger=ledger,
        source_ir=_ir(),
        property_probabilities=probabilities,
        minimum_property_confidence=0.8,
        minimum_role_binding=0.5,
        minimum_realization_score=0.05,
        minimum_target_policy_ratio=0.05,
    )


def test_goal_parser_covers_all_alfworld_families() -> None:
    assert parse_goal_spec(
        "find two remotecontrol and put them in armchair.",
        required_property="NONE",
    ) == {
        "kind": "RELATE",
        "goal_object_type": "remotecontrol",
        "target_receptacle_type": "armchair",
        "required_count": 2,
        "required_property": "NONE",
    }
    assert parse_goal_spec(
        "put a clean cup in shelf.", required_property="CLEAN"
    )["goal_object_type"] == "cup"
    assert parse_goal_spec(
        "heat some potato and put it in diningtable.",
        required_property="HEAT",
    )["target_receptacle_type"] == "diningtable"
    assert parse_goal_spec(
        "look at cd under the desklamp.", required_property="LIGHT"
    )["kind"] == "LIGHT"
    assert parse_goal_spec(
        "examine the bowl with the desklamp.", required_property="LIGHT"
    )["goal_object_type"] == "bowl"


def test_slot_ir_preserves_source_lineage_and_adds_slot_roles() -> None:
    source_ir = _ir()
    validate_slot_source_ir(source_ir)
    assert source_ir["parent_ir_sha256"] == "parent"
    assert "completed_goal_slots" in source_ir["monitor_state"]
    assert "official_success_for_action_selection" in source_ir[
        "prohibited_runtime_fields"
    ]


def test_completed_two_object_slot_cannot_be_rebound() -> None:
    ledger = initialize_slot_ledger(
        "find two remotecontrol and put them in armchair.",
        required_property="NONE",
    )
    ledger, receipt = observe_target_transition(
        ledger,
        action="take remotecontrol 2 from dresser 1",
        after_observation=(
            "You pick up the remotecontrol 2 from the dresser 1."
        ),
    )
    assert receipt == "BIND_INSTANCE"
    ledger, receipt = observe_target_transition(
        ledger,
        action="move remotecontrol 2 to armchair 1",
        after_observation=(
            "You move the remotecontrol 2 to the armchair 1."
        ),
    )
    assert receipt == "RELATE_SLOT_CLOSED"
    assert slot_state(ledger)["remaining_slots"] == 1
    grounded = {
        "go to sidetable 1": _row("SEARCH", 0.9, binding=0.01),
        "take remotecontrol 2 from armchair 1": _row("ACQUIRE", 0.99),
        "take remotecontrol 1 from coffeetable 1": _row("ACQUIRE", 0.8),
    }
    decision = _choose(ledger, grounded)
    assert decision["action"] == "take remotecontrol 1 from coffeetable 1"
    assert decision["slot_state"]["completed_objects"] == ["remotecontrol 2"]
    assert decision["protected_or_incompatible_candidates"] == 1


def test_no_property_goal_protects_visible_target_relation() -> None:
    ledger = initialize_slot_ledger(
        "put some toiletpaper on toilet.",
        required_property="NONE",
        initial_observation=(
            "You arrive at toilet 1. On the toilet 1, you see a "
            "toiletpaper 1."
        ),
    )
    assert slot_state(ledger)["remaining_slots"] == 0
    grounded = {
        "take toiletpaper 1 from toilet 1": _row("ACQUIRE", 0.99),
        "look": _row("SEARCH", 0.5, binding=0.01),
    }
    decision = _choose(ledger, grounded)
    assert decision["action"] == "look"
    assert not decision["source_admitted"]
    assert decision["slot_safety_shielded"]
    assert decision["diagnostic"] == "NO_UNSATISFIED_SLOT_TARGET_ABSTENTION"


def test_safety_shield_does_not_abstain_to_a_completed_object() -> None:
    ledger = initialize_slot_ledger(
        "find two spraybottle and put them in garbagecan.",
        required_property="NONE",
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="take spraybottle 2 from sidetable 1",
        after_observation=(
            "You pick up the spraybottle 2 from the sidetable 1."
        ),
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="move spraybottle 2 to garbagecan 1",
        after_observation=(
            "You move the spraybottle 2 to the garbagecan 1."
        ),
    )
    grounded = {
        "take spraybottle 2 from garbagecan 1": _row("ACQUIRE", 0.99),
        "go to sidetable 1": _row("SEARCH", 0.8, binding=0.01),
    }
    authentic = _choose(ledger, grounded)
    target_only = _choose(ledger, grounded, condition="target_only")
    assert authentic["action"] == "go to sidetable 1"
    assert authentic["slot_safety_shielded"]
    assert not authentic["source_admitted"]
    assert target_only["action"] == "take spraybottle 2 from garbagecan 1"


def test_low_confidence_property_abstains_with_raw_target_policy() -> None:
    ledger = initialize_slot_ledger(
        "put a clean knife in countertop.",
        required_property="NONE",
        initial_observation=(
            "On the countertop 1, you see a knife 1."
        ),
    )
    grounded = {
        "take knife 1 from countertop 1": _row("ACQUIRE", 0.99),
        "look": _row("SEARCH", 0.5, binding=0.01),
    }
    decision = choose_slot_aware_action(
        condition="authentic_slot_ir",
        grounded=grounded,
        history=[],
        ledger=ledger,
        source_ir=_ir(),
        property_probabilities={
            "NONE": 0.4,
            "CLEAN": 0.35,
            "HEAT": 0.1,
            "COOL": 0.1,
            "LIGHT": 0.05,
        },
        minimum_property_confidence=0.8,
        minimum_role_binding=0.5,
        minimum_realization_score=0.05,
        minimum_target_policy_ratio=0.05,
    )
    assert decision["diagnostic"] == "TARGET_PROPERTY_ROUTER_ABSTAINED"
    assert decision["action"] == "take knife 1 from countertop 1"
    assert not decision["slot_safety_enabled"]
    assert not decision["slot_safety_shielded"]


def test_relational_scope_abstains_on_unary_property_tasks() -> None:
    ledger = initialize_slot_ledger(
        "put a clean cup in shelf.", required_property="CLEAN"
    )
    grounded = {
        "take cup 1 from countertop 1": _row("ACQUIRE", 0.8),
        "go to sinkbasin 1": _row("SEARCH", 0.9, binding=0.01),
    }
    decision = choose_slot_aware_action(
        condition="authentic_slot_ir",
        grounded=grounded,
        history=[],
        ledger=ledger,
        source_ir=_ir(),
        property_probabilities={"CLEAN": 0.99, "NONE": 0.01},
        minimum_property_confidence=0.8,
        minimum_role_binding=0.5,
        minimum_realization_score=0.05,
        minimum_target_policy_ratio=0.05,
        allowed_source_effects=("BIND", "RELATE"),
        active_required_properties=("NONE",),
    )
    assert decision["action"] == "go to sinkbasin 1"
    assert decision["diagnostic"] == "TRANSFER_SCOPE_TARGET_ABSTENTION"
    assert not decision["transfer_scope_active"]


def test_dirty_object_at_target_is_not_a_completed_slot() -> None:
    ledger = initialize_slot_ledger(
        "put a clean cup in shelf.", required_property="CLEAN"
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="take cup 1 from countertop 1",
        after_observation="You pick up the cup 1 from the countertop 1.",
    )
    ledger, receipt = observe_target_transition(
        ledger,
        action="move cup 1 to shelf 1",
        after_observation="You move the cup 1 to the shelf 1.",
    )
    assert receipt == "RELATE_NO_PROGRESS"
    assert slot_state(ledger)["remaining_slots"] == 1
    decision = _choose(
        ledger,
        {
            "go to sinkbasin 1": _row("SEARCH", 0.9, binding=0.01),
            "take cup 1 from shelf 1": _row("ACQUIRE", 0.2),
        },
    )
    assert decision["action"] == "take cup 1 from shelf 1"
    assert decision["source_admitted"]


def test_mutation_and_relation_require_observed_postconditions() -> None:
    ledger = initialize_slot_ledger(
        "put a clean plate in shelf.", required_property="CLEAN"
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="take plate 1 from cabinet 1",
        after_observation="You pick up the plate 1 from the cabinet 1.",
    )
    unchanged, receipt = observe_target_transition(
        ledger,
        action="clean plate 1 with sinkbasin 1",
        after_observation="Nothing happens.",
    )
    assert receipt == "IGNORE"
    assert slot_state(unchanged)["required_property"] == "CLEAN"
    assert unchanged["observed_properties"] == {}
    cleaned, receipt = observe_target_transition(
        unchanged,
        action="clean plate 1 with sinkbasin 1",
        after_observation="You clean the plate 1 using the sinkbasin 1.",
    )
    assert receipt == "MUTATE_REQUIRED_PROPERTY"
    decision = _choose(
        cleaned,
        {
            "go to shelf 1": _row("SEARCH", 0.9, binding=0.01),
            "move plate 1 to shelf 1": _row("PLACE", 0.8),
            "move plate 1 to cabinet 1": _row("PLACE", 0.99),
        },
    )
    assert decision["requested_source_effect"] == "RELATE"
    assert decision["action"] == "move plate 1 to shelf 1"


def test_light_goal_closes_slot_only_after_observed_lamp_use() -> None:
    ledger = initialize_slot_ledger(
        "look at cd under the desklamp.", required_property="LIGHT"
    )
    ledger, _ = observe_target_transition(
        ledger,
        action="take cd 1 from sidetable 1",
        after_observation="You pick up the cd 1 from the sidetable 1.",
    )
    decision = _choose(
        ledger,
        {
            "go to desk 1": _row("SEARCH", 0.95, binding=0.01),
            "use desklamp 1": _row("TRANSFORM", 0.8),
        },
    )
    assert decision["requested_source_effect"] == "MUTATE"
    assert decision["action"] == "use desklamp 1"
    ledger, receipt = observe_target_transition(
        ledger,
        action="use desklamp 1",
        after_observation="You turn on the desklamp 1.",
    )
    assert receipt == "LIGHT_SLOT_CLOSED"
    assert slot_state(ledger)["remaining_slots"] == 0
