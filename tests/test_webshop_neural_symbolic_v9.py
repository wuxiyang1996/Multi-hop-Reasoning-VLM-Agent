import numpy as np

from motif_transfer.webshop_neural_symbolic_v9 import (
    OUTCOME_NAMES,
    OutcomeRow,
    _source_features,
    choose_transfer_action,
    fit_target_outcome_mlp,
    target_features,
    visible_goal_constraint_status,
)


def test_visible_goal_constraint_status_distinguishes_false_and_true() -> None:
    goal = "storage ottoman size 60x40x40cm"
    false_tree = "[30] radio '60x40x40cm', checked='false'"
    true_tree = "[30] radio '60x40x40cm', checked='true'"
    assert visible_goal_constraint_status(false_tree, goal) == (False, True)
    assert visible_goal_constraint_status(true_tree, goal) == (True, False)


def test_unsatisfied_constraint_overrides_partial_commit_reward_as_confidence() -> None:
    test, commit = _source_features(
        predicted_test_progress=0.9,
        predicted_commit_reward=0.85,
        predicted_test_change=0.95,
        visible_satisfied=False,
        visible_unsatisfied=True,
        remaining_fraction=0.5,
        repeated_test=False,
    )
    assert test[4] == 0.0
    assert commit[4] == 0.0
    assert test[1] == 0.9


def test_one_satisfied_and_one_unsatisfied_constraint_is_not_ready() -> None:
    test, commit = _source_features(
        predicted_test_progress=0.8,
        predicted_commit_reward=0.95,
        predicted_test_change=0.9,
        visible_satisfied=True,
        visible_unsatisfied=True,
        remaining_fraction=0.5,
        repeated_test=False,
    )
    assert test[4] == 0.0
    assert commit[4] == 0.0
    assert test[1] == 0.8


def test_target_outcome_mlp_learns_paired_progress_and_commit_reward() -> None:
    paired = {
        "verb": "click", "is_commit": False, "is_constraint": True,
        "is_goal_constraint": True, "paired_constraint_bid": "30",
        "is_navigation": False, "is_noop": False, "goal_overlap": 0.2,
    }
    commit = {
        "verb": "click", "is_commit": True, "is_constraint": False,
        "is_goal_constraint": False, "paired_constraint_bid": None,
        "is_navigation": False, "is_noop": False, "goal_overlap": 0.0,
    }
    rows = []
    for _ in range(12):
        rows.append(OutcomeRow(
            target_features(
                paired, visible_satisfied=False, visible_unsatisfied=True,
                prior_no_effect=True, step_index=5, maximum_steps=12,
            ),
            (1.0, 0.0, 0.0, 1.0),
        ))
        rows.append(OutcomeRow(
            target_features(
                commit, visible_satisfied=True, visible_unsatisfied=False,
                prior_no_effect=False, step_index=6, maximum_steps=12,
            ),
            (1.0, 1.0, 1.0, 0.0),
        ))
    model = fit_target_outcome_mlp(rows, seed=4, epochs=500)
    predictions = model.predict([row.features for row in rows[:2]])
    assert predictions[0, OUTCOME_NAMES.index("prerequisite_progress")] > 0.8
    assert predictions[1, OUTCOME_NAMES.index("reward")] > 0.8


class _FixedModel:
    def __init__(self, test_value: float, commit_value: float) -> None:
        self.values = np.asarray([test_value, commit_value])

    def predict(self, features):
        assert len(features) == 2
        return self.values, np.zeros(2)


def test_source_condition_controls_test_commit_switch() -> None:
    candidates = ("click('31')", "click('59')", "go_back()")
    semantics = (
        {"is_commit": False, "is_noop": False},
        {"is_commit": True, "is_noop": False},
        {"is_commit": False, "is_noop": False},
    )
    predictions = np.asarray([
        [0.95, 0.05, 0.05, 0.95],
        [0.95, 0.95, 0.95, 0.05],
        [0.80, 0.05, 0.05, 0.05],
    ])
    models = {
        "authentic_source_plus_target": _FixedModel(0.2, 0.9),
        "shuffled_source_plus_target": _FixedModel(0.9, 0.2),
    }
    authentic = choose_transfer_action(
        condition="authentic_source_plus_target",
        predictions=predictions,
        semantics=semantics,
        source_models=models,
        visible_satisfied=True,
        visible_unsatisfied=False,
        prior_no_effect=False,
        remaining_fraction=0.5,
        previous_action="click('31')",
        candidates=candidates,
        uncertainty_scale=0.5,
        decision_margin=0.0025,
    )
    shuffled = choose_transfer_action(
        condition="shuffled_source_plus_target",
        predictions=predictions,
        semantics=semantics,
        source_models=models,
        visible_satisfied=True,
        visible_unsatisfied=False,
        prior_no_effect=False,
        remaining_fraction=0.5,
        previous_action="click('31')",
        candidates=candidates,
        uncertainty_scale=0.5,
        decision_margin=0.0025,
    )
    assert authentic.abstract_kind == "COMMIT"
    assert authentic.selected_index == 1
    assert shuffled.abstract_kind == "TEST"
    assert shuffled.selected_index == 0
