from __future__ import annotations

import pytest

from motif_transfer.hierarchical_skill_transfer import FEATURE_NAMES
from motif_transfer.intervention_grounded_target_controller import (
    FORBIDDEN_FEATURE_INDICES,
    ground_target_options,
    source_shadow_decision,
    target_option_features,
    within_state_target_effect_permutation,
)
from motif_transfer.neurosymbolic_transfer_contract import (
    CAUSAL_EFFECT_SCORE_SEMANTICS,
    IMITATION_SCORE_SEMANTICS,
)


def test_target_option_features_only_bind_neural_effect_and_observables() -> None:
    values = target_option_features(
        option="ACQUIRE",
        causal_effect_probability=0.7,
        score_semantics=CAUSAL_EFFECT_SCORE_SEMANTICS,
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
        option="SEARCH", causal_effect_probability=0.4,
        score_semantics=CAUSAL_EFFECT_SCORE_SEMANTICS,
        representative_action="open cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    place = target_option_features(
        option="PLACE", causal_effect_probability=0.4,
        score_semantics=CAUSAL_EFFECT_SCORE_SEMANTICS,
        representative_action="put apple 1 in cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    for index in (13, 15, 18, 19, 20, 21, 22):
        assert search[index] == place[index] == 0.0


def test_target_effect_control_rotates_receipts_not_option_identity() -> None:
    search = target_option_features(
        option="SEARCH", causal_effect_probability=0.2,
        score_semantics=CAUSAL_EFFECT_SCORE_SEMANTICS,
        representative_action="open cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    place = target_option_features(
        option="PLACE", causal_effect_probability=0.8,
        score_semantics=CAUSAL_EFFECT_SCORE_SEMANTICS,
        representative_action="put apple 1 in cabinet 1", action_history=(),
        step=0, max_steps=60,
    )
    grounded = {"option_features": {"PLACE": place, "SEARCH": search}}
    permuted = within_state_target_effect_permutation(grounded)
    assert permuted["option_features"]["PLACE"][:5] == place[:5]
    assert permuted["option_features"]["SEARCH"][:5] == search[:5]
    assert permuted["option_features"]["PLACE"][11] == search[11]
    assert permuted["option_features"]["SEARCH"][11] == place[11]


def test_imitation_score_cannot_populate_causal_effect_slot() -> None:
    with pytest.raises(ValueError, match="non-causal score"):
        target_option_features(
            option="SEARCH",
            causal_effect_probability=0.9,
            score_semantics=IMITATION_SCORE_SEMANTICS,
            representative_action="open cabinet 1",
            action_history=(),
            step=0,
            max_steps=60,
        )


def test_legacy_imitation_grounder_abstains_and_preserves_baseline() -> None:
    grounded = ground_target_options(
        goal="put an apple in a cabinet",
        observation="You are in a kitchen.",
        native_actions=("open cabinet 1",),
        step=0,
        max_steps=60,
        action_history=(),
        target_grounder={
            "training_supervision": "expert_action_identity_only",
            "required_option_or_workflow_features_used": False,
            "feature_bins": 16,
            "policy_head": {
                "hidden_activation": "tanh",
                "layers": [{
                    "weights": [[0.0]] * 26,
                    "bias": [0.0],
                }],
            },
        },
    )
    assert grounded["fallback_action"] == "open cabinet 1"
    assert grounded["transfer_eligible"] is False
    assert grounded["option_features"] == {}
    assert "TARGET_SCORE_IS_NOT_CAUSAL_SUCCESSOR_EFFECT" in grounded[
        "transfer_abstention_reasons"
    ]


def test_source_admission_requires_joint_support_receipt() -> None:
    decision = source_shadow_decision(
        {
            "fallback_option": "SEARCH",
            "fallback_action": "look",
            "option_features": {"SEARCH": (0.0,)},
            "transfer_eligible": True,
            "transfer_abstention_reasons": [],
        },
        model=None,  # type: ignore[arg-type]
        conformal_error=0.0,
    )
    assert decision["action"] == "look"
    assert decision["source_admitted"] is False
    assert decision["transfer_abstention_reasons"] == [
        "SOURCE_TARGET_SUPPORT_RECEIPT_MISSING"
    ]
