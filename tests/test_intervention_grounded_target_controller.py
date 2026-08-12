from __future__ import annotations

import pytest

from motif_transfer.hierarchical_skill_transfer import FEATURE_NAMES
from motif_transfer.intervention_grounded_target_controller import (
    FORBIDDEN_FEATURE_INDICES,
    target_option_features,
    within_state_target_effect_permutation,
)


def test_target_option_features_only_bind_neural_effect_and_observables() -> None:
    values = target_option_features(
        option="ACQUIRE",
        neural_effect_probability=0.7,
        representative_action="take apple 1 from counter 2",
        action_history=(
            "take apple 1 from counter 2",
            "take apple 1 from counter 2",
        ),
        step=9,
        max_steps=60,
    )
    assert len(values) == len(FEATURE_NAMES)
    assert values[:5] == (0.0, 1.0, 0.0, 0.0, 0.0)
    assert values[11] == 0.7
    assert values[12] == 0.7
    assert values[14] == 51 / 60
    assert values[16] == 0.5
    assert values[17] == pytest.approx(0.3)
    assert all(values[index] == 0.0 for index in FORBIDDEN_FEATURE_INDICES)


def test_unknown_source_costs_cancel_by_identical_zero_binding() -> None:
    search = target_option_features(
        option="SEARCH", neural_effect_probability=0.4,
        representative_action="open cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    place = target_option_features(
        option="PLACE", neural_effect_probability=0.4,
        representative_action="put apple 1 in cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    for index in (13, 15, 18, 19, 20, 21, 22):
        assert search[index] == place[index] == 0.0


def test_target_effect_control_rotates_receipts_not_option_identity() -> None:
    search = target_option_features(
        option="SEARCH", neural_effect_probability=0.2,
        representative_action="open cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    place = target_option_features(
        option="PLACE", neural_effect_probability=0.8,
        representative_action="put apple 1 in cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    grounded = {"option_features": {"PLACE": place, "SEARCH": search}}
    permuted = within_state_target_effect_permutation(grounded)
    assert permuted["option_features"]["PLACE"][:5] == place[:5]
    assert permuted["option_features"]["SEARCH"][:5] == search[:5]
    assert permuted["option_features"]["PLACE"][11] == search[11]
    assert permuted["option_features"]["SEARCH"][11] == place[11]
