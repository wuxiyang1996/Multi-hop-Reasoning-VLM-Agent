import numpy as np

from motif_transfer.alfworld_neural_grounder import (
    action_role,
    choose_grounded_action,
    grounder_features,
    mlp_score,
    target_symbolic_features,
)
from motif_transfer.controlled_exploration_transfer import RidgeValueModel, ValueEnsemble


def test_action_roles_are_target_native() -> None:
    assert action_role("go to cabinet 1") == "TEST"
    assert action_role("open cabinet 1") == "TEST"
    assert action_role("take mug 1 from cabinet 1") == "COMMIT"
    assert action_role("help") == "EXCLUDE"


def test_grounder_features_are_fixed_and_deterministic() -> None:
    kwargs = {
        "goal": "put mug in cabinet",
        "observation": "A mug 1 is on countertop 1.",
        "action": "take mug 1 from countertop 1",
        "step": 2,
        "action_history": ["go to countertop 1"],
        "feature_bins": 32,
    }
    left = grounder_features(**kwargs)
    right = grounder_features(**kwargs)
    assert left.shape == (len(16 * [0]) + 10 + 32,)
    assert np.array_equal(left, right)


def test_exported_mlp_inference_and_symbolic_features() -> None:
    artifact = {
        "feature_bins": 16,
        "hidden_activation": "tanh",
        "layers": [
            {"weights": [[1.0]] * 3, "bias": [0.0]},
            {"weights": [[1.0]], "bias": [0.0]},
        ],
    }
    assert 0.5 < mlp_score(np.ones(3), artifact) < 1.0
    features = target_symbolic_features(
        actions=["go to cabinet 1", "take mug 1 from table 1"],
        scores={"go to cabinet 1": 0.6, "take mug 1 from table 1": 0.4},
        step=2,
        max_steps=20,
        action_history=[],
    )
    assert features["go to cabinet 1"][0] == 1.0
    assert features["take mug 1 from table 1"][0] == 0.0


def test_source_controller_must_compare_test_and_commit() -> None:
    coefficients = [0.0] * 10
    coefficients[0] = 2.0
    model = ValueEnsemble((RidgeValueModel(
        feature_mean=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        coefficients=tuple(coefficients),
    ),))
    actions = ["go to cabinet 1", "take mug 1 from table 1"]
    symbolic = {
        actions[0]: (1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0),
        actions[1]: (0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.0),
    }
    decision = choose_grounded_action(
        actions=actions,
        grounder_scores={actions[0]: 0.2, actions[1]: 0.8},
        symbolic_features=symbolic,
        source_model=model,
        uncertainty_scale=0.5,
        decision_margin=0.0,
    )
    assert decision["action"] == actions[0]
    assert decision["changed_role"] is True
