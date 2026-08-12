from __future__ import annotations

import pytest

from motif_transfer.neurosymbolic_transfer_contract import (
    CAUSAL_EFFECT_SCORE_SEMANTICS,
    IMITATION_SCORE_SEMANTICS,
    require_causal_target_grounder,
    require_source_target_support,
    target_score_contract,
    transfer_contract_audit,
)


def test_legacy_expert_policy_is_inferred_as_imitation_not_effect() -> None:
    artifact = {"training_supervision": "expert_action_identity_only"}
    contract = target_score_contract(artifact)
    assert contract["score_semantics"] == IMITATION_SCORE_SEMANTICS
    assert contract["legacy_contract_inferred"] is True
    with pytest.raises(ValueError, match="TARGET_SCORE_IS_NOT_CAUSAL"):
        require_causal_target_grounder(artifact)


def test_causal_target_and_joint_support_both_pass() -> None:
    artifact = {
        "score_contract": {
            "score_semantics": CAUSAL_EFFECT_SCORE_SEMANTICS,
            "causal_successor_effect_certified": True,
            "counterfactual_action_supervision": True,
            "probability_calibration": "heldout_target_isotonic",
            "entity_conditioned_action_binding": True,
            "successor_event_prediction": True,
        }
    }
    support = {
        "status": "SOURCE_TARGET_SUPPORT_GATE_PASSED",
        "score_semantics": CAUSAL_EFFECT_SCORE_SEMANTICS,
        "exchangeability_or_covariate_coverage_certified": True,
        "source_calibration_states": 100,
        "target_adaptation_states": 50,
    }
    assert require_causal_target_grounder(artifact)[
        "score_semantics"
    ] == CAUSAL_EFFECT_SCORE_SEMANTICS
    require_source_target_support(support)
    assert transfer_contract_audit(
        target_grounder=artifact,
        source_target_support_receipt=support,
    )["status"] == "NEUROSYMBOLIC_TRANSFER_CONTRACT_PASSED"


def test_source_conformal_bound_fails_closed_without_joint_support() -> None:
    with pytest.raises(ValueError, match="SUPPORT_RECEIPT_MISSING"):
        require_source_target_support(None)
