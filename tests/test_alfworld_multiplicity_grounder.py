from __future__ import annotations

from motif_transfer.alfworld_multiplicity_grounder import (
    candidate_effect,
    infer_required_option,
    workflow_status,
)


GOAL = "put two cd in safe"


def test_distinct_object_bindings_not_place_event_count_define_progress() -> None:
    history = (
        "take cd 2 from desk 1",
        "move cd 2 to safe 1",
        "take cd 2 from safe 1",
        "move cd 2 to safe 1",
    )
    status = workflow_status(GOAL, history)
    assert status.placed_count == 1
    assert status.placed_object_ids == ("cd:2",)
    assert status.remaining_count == 1


def test_taking_completed_object_from_goal_receptacle_is_negative_effect() -> None:
    history = ("take cd 2 from desk 1", "move cd 2 to safe 1")
    effect = candidate_effect(GOAL, history, "take cd 2 from safe 1")
    assert effect["reverses_completed_binding"] is True
    assert effect["distinct_placed_before"] == 1
    assert effect["distinct_placed_after"] == 0
    assert effect["symbolic_progress_delta"] < 0


def test_second_distinct_object_advances_and_completes_multiplicity_goal() -> None:
    history = (
        "take cd 2 from desk 1",
        "move cd 2 to safe 1",
        "take cd 3 from drawer 1",
    )
    assert infer_required_option(
        goal=GOAL,
        native_actions=("move cd 3 to safe 1", "go to desk 1"),
        action_history=history,
    ) == "PLACE"
    complete = workflow_status(GOAL, (*history, "move cd 3 to safe 1"))
    assert complete.placed_count == 2
    assert complete.remaining_count == 0
    assert complete.progress_fraction == 1.0


def test_placed_identity_is_not_selected_as_the_remaining_acquire() -> None:
    history = ("take cd 2 from desk 1", "move cd 2 to safe 1")
    assert infer_required_option(
        goal=GOAL,
        native_actions=("take cd 2 from safe 1", "go to drawer 1"),
        action_history=history,
    ) == "SEARCH"
    assert infer_required_option(
        goal=GOAL,
        native_actions=("take cd 2 from safe 1", "take cd 3 from drawer 1"),
        action_history=history,
    ) == "ACQUIRE"
