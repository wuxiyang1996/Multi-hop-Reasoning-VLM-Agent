from __future__ import annotations

from motif_transfer.online_transfer_utility import (
    ApplicabilityReceipt,
    OnlineTransferUtilityGate,
    PairedOutcome,
)
from scripts.audit_online_transfer_utility_v1 import route_receipt


VALID = ApplicabilityReceipt(True, True, True, True, True)


def test_cold_start_and_small_positive_sample_abstain() -> None:
    gate = OnlineTransferUtilityGate()
    assert gate.decision(VALID).decision == "ABSTAIN"
    gate.update_many([PairedOutcome(True, False)] * 2)
    assert gate.decision(VALID).decision == "ABSTAIN"


def test_repeated_paired_wins_authorize_route() -> None:
    gate = OnlineTransferUtilityGate()
    gate.update_many([PairedOutcome(True, False)] * 7)
    receipt = gate.decision(VALID)
    assert receipt.decision == "SELECT_SKILL"
    assert receipt.posterior_lower_win_probability > 0.5
    assert receipt.state.wins == 7


def test_losses_and_ties_are_accounted_separately() -> None:
    gate = OnlineTransferUtilityGate()
    gate.update_many(
        [PairedOutcome(True, False)] * 9
        + [PairedOutcome(False, True)]
        + [PairedOutcome(True, True)] * 14
    )
    receipt = gate.decision(VALID)
    assert receipt.decision == "SELECT_SKILL"
    assert receipt.state.exposures == 24
    assert receipt.state.discordant == 10
    assert receipt.observed_disagreement_rate == 10 / 24


def test_current_outcome_cannot_affect_prior_decision() -> None:
    gate = OnlineTransferUtilityGate()
    before = gate.decision(VALID)
    assert before.decision == "ABSTAIN"
    gate.update(PairedOutcome(True, False))
    after = gate.decision(VALID)
    assert after.state.wins == 1


def test_any_failed_target_native_predicate_abstains() -> None:
    gate = OnlineTransferUtilityGate()
    gate.update_many([PairedOutcome(True, False)] * 9)
    invalid = ApplicabilityReceipt(True, True, True, False, True)
    receipt = gate.decision(invalid)
    assert receipt.decision == "ABSTAIN"
    assert receipt.reason == "STRUCTURAL_APPLICABILITY_FAILED"


def test_audit_route_receipt_is_conservative_at_small_n() -> None:
    assert route_receipt(
        wins=3, losses=0, ties=3, evidence_status="positive",
    )["post_replication"]["decision"] == "ABSTAIN"
    assert route_receipt(
        wins=7, losses=0, ties=25, evidence_status="positive",
    )["post_replication"]["decision"] == "SELECT_SKILL"
