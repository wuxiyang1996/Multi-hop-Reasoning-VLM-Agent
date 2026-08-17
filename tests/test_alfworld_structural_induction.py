from __future__ import annotations

from motif_transfer.alfworld_structural_induction import (
    ADD_ID,
    REMOVE_ID,
    UPDATE_ID,
    binding_labels,
    induce_target_sequence_program,
    infer_demonstrated_bindings,
    observed_action_history_sequence,
    observed_transition_operator_ids,
    repeated_source_support,
    target_candidate_features,
)


def _transition(action: str, after: str, goal: str = "put two mug in cabinet."):
    return {
        "expert_action": action,
        "after_observation": after,
        "goal": goal,
        "native_actions": [action],
    }


def test_outcome_labels_require_target_native_effect_text() -> None:
    assert observed_transition_operator_ids(
        _transition("take mug 1 from desk 1", "You pick up the mug 1 from the desk 1.")
    ) == (ADD_ID,)
    assert observed_transition_operator_ids(
        _transition("move mug 1 to cabinet 1", "You move the mug 1 to the cabinet 1.")
    ) == (REMOVE_ID,)
    assert observed_transition_operator_ids(
        _transition("open cabinet 1", "You open the cabinet 1.")
    ) == (UPDATE_ID,)
    assert not observed_transition_operator_ids(
        _transition("take mug 1 from desk 1", "Nothing happens.")
    )


def test_bindings_are_induced_from_observed_add_remove_events() -> None:
    episode = {"transitions": [
        _transition("take mug 1 from desk 1", "You pick up the mug 1 from the desk 1."),
        _transition("move mug 1 to cabinet 1", "You move the mug 1 to the cabinet 1."),
        _transition("take mug 2 from shelf 1", "You pick up the mug 2 from the shelf 1."),
        _transition("move mug 2 to cabinet 1", "You move the mug 2 to the cabinet 1."),
    ]}
    bindings = infer_demonstrated_bindings(episode)
    assert bindings == {"entity": ("mug",), "destination": ("cabinet",)}
    assert binding_labels("take mug 3 from table 1", bindings) == (1, 0)
    assert binding_labels("go to cabinet 1", bindings) == (0, 1)


def test_target_count_and_source_subgraph_are_separate() -> None:
    target = (ADD_ID, REMOVE_ID, ADD_ID, REMOVE_ID)
    program = induce_target_sequence_program(
        [target, target], development_receipts_sha256="development",
    )
    assert program["induced_sequence"] == list(target)
    assert program["source_program_copied_as_target_body"] is False
    assert repeated_source_support((ADD_ID, REMOVE_ID), target) == {
        "applicable": True,
        "repeat_count": 2,
        "explained_operator_count": 4,
        "target_operator_count": 4,
    }
    assert not repeated_source_support((ADD_ID, UPDATE_ID), target)["applicable"]


def test_target_ledger_requires_distinct_bindings() -> None:
    goal = "put two mug in cabinet."
    history = (
        "take mug 1 from desk 1",
        "move mug 1 to cabinet 1",
        "take mug 1 from cabinet 1",
        "move mug 1 to cabinet 1",
        "take mug 2 from shelf 1",
        "move mug 2 to cabinet 1",
    )
    assert observed_action_history_sequence(goal, history) == (
        ADD_ID, REMOVE_ID, ADD_ID, REMOVE_ID, ADD_ID, REMOVE_ID,
    )


def test_binding_features_are_invariant_to_entity_alpha_renaming() -> None:
    left = target_candidate_features(
        goal="put two mug in cabinet.",
        observation="On the desk, you see a mug 1.",
        action="take mug 1 from desk 1", step=2, action_history=(), feature_bins=128,
    )
    right = target_candidate_features(
        goal="put two apple in shelf.",
        observation="On the table, you see an apple 1.",
        action="take apple 1 from table 1", step=2, action_history=(), feature_bins=128,
    )
    assert (left == right).all()
