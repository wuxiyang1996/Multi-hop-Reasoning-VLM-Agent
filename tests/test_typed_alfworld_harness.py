from __future__ import annotations

from motif_transfer.typed_alfworld_harness import (
    choose_typed_action,
    target_cycle_state,
    validate_typed_effect_ir,
)


IR = {
    "nodes": ["POSITION", "BIND", "MUTATE", "RELATE"],
    "edges": [
        {"from": "BIND", "to": "MUTATE", "guard": "CARRIER_BOUND"},
        {"from": "BIND", "to": "RELATE", "guard": "CARRIER_BOUND"},
    ],
    "prohibited_runtime_fields": ["source_action_ordinal", "environment_id"],
    "target_grounding": "TARGET_NATIVE_NEURAL_PROBES_ONLY",
}


def _row(option: str, score: float) -> dict:
    return {
        "option": option,
        "applicability": score,
        "binding": score,
        "completion": score,
        "policy": score,
        "required_option": "SEARCH",
    }


GROUNDED = {
    "go to counter": _row("SEARCH", 0.9),
    "take mug": _row("ACQUIRE", 0.8),
    "clean mug": _row("TRANSFORM", 0.7),
    "put mug in desk": _row("PLACE", 0.6),
}


def test_ir_validation_and_target_cycle() -> None:
    validate_typed_effect_ir(IR)
    assert target_cycle_state([], []) == {
        "carrier_bound": False, "mutated_since_bind": False,
    }
    assert target_cycle_state(["a", "b"], ["ACQUIRE", "TRANSFORM"]) == {
        "carrier_bound": True, "mutated_since_bind": True,
    }
    assert target_cycle_state(
        ["a", "b", "c"], ["ACQUIRE", "TRANSFORM", "PLACE"]
    )["carrier_bound"] is False


def test_authentic_graph_orders_bind_mutate_relate() -> None:
    bind = choose_typed_action(
        condition="authentic_typed_ir", grounded=GROUNDED, history=[],
        grounded_history_options=[], source_ir=IR,
    )
    assert bind["target_realized_effect"] == "BIND"
    mutate = choose_typed_action(
        condition="authentic_typed_ir", grounded=GROUNDED, history=["take mug"],
        grounded_history_options=["ACQUIRE"], source_ir=IR,
    )
    assert mutate["target_realized_effect"] == "MUTATE"
    relate = choose_typed_action(
        condition="authentic_typed_ir", grounded=GROUNDED,
        history=["take mug", "clean mug"],
        grounded_history_options=["ACQUIRE", "TRANSFORM"], source_ir=IR,
    )
    assert relate["target_realized_effect"] == "RELATE"


def test_controls_change_graph_semantics_without_source_actions() -> None:
    permuted = choose_typed_action(
        condition="edge_permuted_ir", grounded=GROUNDED, history=[],
        grounded_history_options=[], source_ir=IR,
    )
    assert permuted["target_realized_effect"] == "MUTATE"
    wrong_guard = choose_typed_action(
        condition="wrong_guard_ir", grounded=GROUNDED, history=[],
        grounded_history_options=[], source_ir=IR,
    )
    assert wrong_guard["target_realized_effect"] == "MUTATE"
    assert "source_action_ordinal" not in str(permuted["action"])


def test_target_only_uses_neural_policy_fallback() -> None:
    decision = choose_typed_action(
        condition="target_only", grounded=GROUNDED, history=[],
        grounded_history_options=[], source_ir=IR,
    )
    assert decision["action"] == "go to counter"
    assert not decision["source_admitted"]


def test_target_policy_realizes_concrete_action_within_source_effect() -> None:
    grounded = dict(GROUNDED) | {
        "take plate": _row("ACQUIRE", 0.95) | {"policy": 0.2},
    }
    grounded["take mug"] = dict(grounded["take mug"]) | {"policy": 0.7}
    decision = choose_typed_action(
        condition="authentic_typed_ir",
        grounded=grounded,
        history=[],
        grounded_history_options=[],
        source_ir=IR,
        concrete_action_ranking="target_policy_within_effect",
    )
    assert decision["target_realized_effect"] == "BIND"
    assert decision["action"] == "take mug"


def test_relative_target_policy_gate_abstains_from_low_confidence_override() -> None:
    decision = choose_typed_action(
        condition="authentic_typed_ir",
        grounded=GROUNDED,
        history=[],
        grounded_history_options=[],
        source_ir=IR,
        concrete_action_ranking="target_policy_within_effect",
        minimum_target_policy_ratio=0.95,
    )
    assert decision["action"] == "go to counter"
    assert not decision["source_admitted"]
    assert decision["diagnostic"] == "TARGET_POLICY_RELATIVE_ABSTENTION"
    assert decision["target_policy_ratio"] < 0.95
