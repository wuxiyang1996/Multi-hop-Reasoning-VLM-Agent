from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.alfworld_target_recurrent_induction import (
    QUALIFIED,
    TARGET_INDUCED,
    ZERO_DEMO_ABSTENTION,
    choose_target_induced_action,
    eligible_target_demonstrations,
    execution_normal_form,
    induce_target_recurrent_program,
    permute_binding_relation,
    shuffled_effect_supports,
    target_program_supports,
    validate_target_recurrent_program,
)
from motif_transfer.alfworld_target_written_equivalent import (
    TargetWrittenExecutionState,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.slot_aware_alfworld_harness import initialize_slot_ledger


REPO = Path(__file__).resolve().parents[1]


def _record(
    step: int, action: str, receipt: str, before: int, after: int,
) -> dict:
    body = {
        "step": step,
        "selected_action": action,
        "target_effect_receipt": receipt,
        "completed_count_before": before,
        "completed_count_after": after,
        "reward_discarded_for_selection": True,
    }
    return body | {"record_sha256": stable_hash(body)}


def _episode(*, second_handle: str = "drawer 1") -> dict:
    records = [
        _record(0, "take apple 1 from table 1", "BIND_INSTANCE", 0, 0),
        _record(1, "go to drawer 1", "IGNORE", 0, 0),
        _record(2, "move apple 1 to drawer 1", "RELATE_SLOT_CLOSED", 0, 1),
        _record(3, "go to table 2", "IGNORE", 1, 1),
        _record(4, "take apple 2 from table 2", "BIND_INSTANCE", 1, 1),
        _record(5, f"go to {second_handle}", "IGNORE", 1, 1),
        _record(
            6, f"move apple 2 to {second_handle}",
            "RELATE_SLOT_CLOSED", 1, 2,
        ),
    ]
    body = {
        "task_id": "pick_two_obj_and_place-Apple-None-Drawer-X/game.tw-pddl",
        "official_success": True,
        "steps": len(records),
        "records": records,
    }
    return body | {"episode_sha256": stable_hash(body)}


def test_zero_demo_abstains_and_one_demo_induces_recurrent_program() -> None:
    zero = induce_target_recurrent_program((_episode(),), budget=0)
    assert zero["status"] == ZERO_DEMO_ABSTENTION
    assert execution_normal_form(zero) is None

    one = induce_target_recurrent_program((_episode(),), budget=1)
    assert one["status"] == QUALIFIED
    assert one["source_artifact_read"] is False
    assert one["named_controller_template_used"] is False
    assert target_program_supports(one, _episode())
    assert execution_normal_form(one) == {
        "activation_after_positive_relations": 1,
        "recurrent_acquisition_control": True,
        "binding_then_relation": True,
        "recurrent_relation_grounding": True,
        "relation_argument_rule": "PRESERVE_FIRST_POSITIVE_RELATION_HANDLE",
        "terminal_remaining_relations": 0,
        "positive_effect_cardinality": 1,
        "fail_closed_on_ambiguity": True,
    }


def test_handle_conflict_and_effect_controls_are_rejected() -> None:
    program = induce_target_recurrent_program((_episode(),), budget=1)
    assert eligible_target_demonstrations((_episode(second_handle="drawer 2"),)) == ()
    assert not shuffled_effect_supports(program, _episode())
    assert not target_program_supports(permute_binding_relation(program), _episode())


def test_committed_v16_one_demo_heldout_qualification_is_portable() -> None:
    report = json.loads((
        REPO / "docs/results/alfworld_target_acquisition_value_v16_qualification.json"
    ).read_text(encoding="utf-8"))
    program = json.loads((
        REPO / "docs/results/alfworld_target_only_k1_program_v16.json"
    ).read_text(encoding="utf-8"))
    validate_target_recurrent_program(program)
    curve = report["target_only_induction_curve"]

    assert report["development"]["eligible_trajectories"] == 9
    assert report["qualification"]["eligible_trajectories"] == 11
    assert curve[0]["complete_target_trajectory_budget"] == 0
    assert curve[0]["qualification_support"] == 0
    assert curve[1]["complete_target_trajectory_budget"] == 1
    assert curve[1]["program_sha256"] == program["program_sha256"]
    assert curve[1]["qualification_support"] == 11
    assert curve[1]["qualification_shuffled_effect_support"] == 0
    assert curve[1]["qualification_binding_relation_permuted_support"] == 0


def _grounded_row(option: str, policy: float) -> dict:
    return {
        "option": option,
        "policy": policy,
        "applicability": 0.95,
        "completion": 0.95,
        "binding": 0.99,
        "required_option": option,
    }


def _choose(program: dict) -> dict:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 2"
    ledger["observed_locations"] = {"apple 1": "drawer 2"}
    ledger["completed_objects"] = ["apple 1"]
    return choose_target_induced_action(
        condition=TARGET_INDUCED,
        grounded={
            "go to counter 1": _grounded_row("SEARCH", 0.80),
            "take mug 1 from shelf 1": _grounded_row("ACQUIRE", 1.0),
        },
        goal="put two apple in drawer.", history=(), ledger=ledger,
        execution_state=TargetWrittenExecutionState(),
        program_artifact=program,
        target_causal_effect_head={
            "feature_names": ["verb_move"], "means": [0.0],
            "scales": [1.0], "weights": [12.0], "intercept": -6.0,
        },
        step=4, max_steps=30, minimum_binding=0.5,
        minimum_realization=0.1, minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )


def test_learned_artifact_controls_execution_and_controls_abstain() -> None:
    qualified = induce_target_recurrent_program((_episode(),), budget=1)
    zero = induce_target_recurrent_program((_episode(),), budget=0)
    assert _choose(qualified)["action"] == "go to counter 1"
    assert _choose(qualified)["program_origin"] == (
        "TARGET_ONLY_TRAJECTORY_INDUCTION"
    )
    assert _choose(zero)["action"] == "take mug 1 from shelf 1"
    assert _choose(permute_binding_relation(qualified))["action"] == (
        "take mug 1 from shelf 1"
    )
