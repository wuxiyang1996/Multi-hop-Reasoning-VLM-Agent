from dataclasses import replace

from motif_transfer.contracts import (
    ConditionOutcome,
    DecisionProposal,
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
    Observation,
    TransitionReceipt,
)
from motif_transfer.harness import DeterministicHarness


def receipt(index=0):
    before = Observation({"n": index}, ("go",))
    after = Observation({"n": index + 1}, ("go",))
    return TransitionReceipt.create(before, DecisionProposal(str(index), "go"), after, 0)


def motif(r1, r2, description="first prose"):
    return MotifCandidate(
        "m",
        ("episode",),
        (MotifNode("a", (r1.receipt_id,)), MotifNode("b", (r2.receipt_id,))),
        (MotifEdge("a", "b", (r1.receipt_id,), description),),
        untrusted_description=description,
    )


def test_receipt_tampering_is_detected():
    row = receipt()
    assert row.validate()
    assert not replace(row, reward=99).validate()


def test_prose_is_excluded_from_structural_fingerprint():
    r1, r2 = receipt(0), receipt(1)
    harness = DeterministicHarness()
    receipts = {row.receipt_id: row for row in (r1, r2)}
    first = harness.audit_motif(motif(r1, r2, "collect treasure"), receipts)
    second = harness.audit_motif(motif(r1, r2, "verify a web page"), receipts)
    assert first.accepted and second.accepted
    assert first.structural_fingerprint == second.structural_fingerprint


def outcome(condition, success):
    return ConditionOutcome(condition, "state", "prefix", "policy", "budget", success, float(success))


def test_matched_authentic_separation_supports_transfer():
    rows = [
        outcome("authentic", True),
        outcome("target_only", False),
        outcome("generic_protocol", False),
        outcome("shuffled_topology", False),
        outcome("other_source", False),
    ]
    assert DeterministicHarness().evaluate_matched(rows).status == Lifecycle.POSITIVE_TRANSFER


def test_missing_control_is_inconclusive():
    rows = [outcome("authentic", True), outcome("target_only", False)]
    assert DeterministicHarness().evaluate_matched(rows).status == Lifecycle.INCONCLUSIVE
