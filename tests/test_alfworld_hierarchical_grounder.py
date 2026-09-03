import numpy as np

from motif_transfer.alfworld_hierarchical_grounder import (
    action_option,
    completion_label,
    grounder_features,
    infer_required_option,
    parse_goal,
    workflow_status,
)


def test_parse_goal_variants():
    assert parse_goal("put a clean cloth in cart.").target_object == "cloth"
    assert parse_goal("heat some egg and put it in fridge.").transform == "heat"
    assert parse_goal("find two creditcard and put them in armchair.").count == 2
    look = parse_goal("look at cd under the desklamp.")
    assert (look.target_object, look.destination, look.transform) == (
        "cd", "desklamp", "light",
    )


def test_option_and_phase_monitor_tracks_receipted_progress():
    goal = "put a clean cloth in cart."
    actions = ["go to table 1", "take cloth 1 from table 1"]
    assert infer_required_option(
        goal=goal,
        native_actions=["take cloth 1 from table 1", "take mug 1 from table 1"],
        action_history=actions[:1],
    ) == "ACQUIRE"
    assert infer_required_option(
        goal=goal,
        native_actions=["clean cloth 1 with sinkbasin 1"],
        action_history=actions,
    ) == "TRANSFORM"
    history = [*actions, "clean cloth 1 with sinkbasin 1"]
    assert workflow_status(goal, history).transformed
    assert action_option("move cloth 1 to cart 1") == "PLACE"


def test_search_completion_is_target_becoming_actionable():
    label = completion_label(
        goal="put a mug in cabinet.",
        before_native_actions=["go to table 1"],
        action_history=[],
        action="go to table 1",
        after_native_actions=["take mug 1 from table 1"],
        official_success_after=False,
    )
    assert label == 1


def test_grounder_features_are_fixed_and_deterministic():
    kwargs = dict(
        goal="put a mug in cabinet.",
        observation="On the table, you see a mug 1.",
        action="take mug 1 from table 1",
        required_option="ACQUIRE",
        step=3,
        action_history=["go to table 1"],
        feature_bins=32,
    )
    left = grounder_features(**kwargs)
    right = grounder_features(**kwargs)
    assert left.shape == (5 + 5 + 18 + 32,)
    assert np.array_equal(left, right)
