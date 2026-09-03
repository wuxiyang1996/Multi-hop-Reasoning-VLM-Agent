from __future__ import annotations

import numpy as np

from motif_transfer.oracle_free_target_grounder import (
    DENSE_FEATURE_NAMES,
    policy_features,
)


def test_oracle_free_features_have_only_declared_dense_prefix() -> None:
    features = policy_features(
        goal="put two clean apple in the cabinet",
        observation="You see an apple and a cabinet.",
        action="take apple 1 from counter 2",
        step=7,
        action_history=("go to counter 2",),
        feature_bins=32,
    )
    assert features.shape == (len(DENSE_FEATURE_NAMES) + 32,)
    assert np.allclose(features[:5], [0, 1, 0, 0, 0])
    assert features[5] > 0
    assert features[6] > 0


def test_features_do_not_accept_required_option_or_workflow_status() -> None:
    first = policy_features(
        goal="put an apple in the cabinet",
        observation="You are at a counter.",
        action="open cabinet 1",
        step=3,
        action_history=("go to counter 1",),
        feature_bins=32,
    )
    second = policy_features(
        goal="put an apple in the cabinet",
        observation="You are at a counter.",
        action="open cabinet 1",
        step=3,
        action_history=("go to counter 1",),
        feature_bins=32,
    )
    assert np.array_equal(first, second)
