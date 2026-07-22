from dataclasses import replace

from motif_transfer.contracts import (
    ConditionOutcome,
    DecisionStepSignature,
    DecisionProposal,
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
    Observation,
    ReplayForkReceipt,
    SourceStepSignature,
    TransitionReceipt,
)
from motif_transfer.harness import DeterministicHarness


def receipt(index=0):
    before = Observation({"n": index}, ("go",))
    after = Observation({"n": index + 1}, ("go",))
    return TransitionReceipt.create(before, DecisionProposal(str(index), "go"), after, 0)


def replay(r1):
    return ReplayForkReceipt.create(
        source_transition_id=r1.receipt_id,
        prefix_hash="prefix",
        fork_state_hash=r1.before_hash,
        admissible_actions_hash=r1.native_actions_hash,
        alternative_action="go",
        alternative_after_hash="alternative",
    )


def motif(r1, r2, fork, description="first prose"):
    return MotifCandidate(
        "m",
        ("episode",),
        (
            MotifNode(
                "a",
                (r1.receipt_id,),
                (DecisionStepSignature(2, 0, "SUPPORTED", "CONTINUE"),),
            ),
            MotifNode(
                "b",
                (r2.receipt_id,),
                (DecisionStepSignature(2, 1, "REFUTED", "REPLAN"),),
            ),
        ),
        (MotifEdge("a", "b", (fork.receipt_id,), description),),
        untrusted_description=description,
    )


def test_receipt_tampering_is_detected():
    row = receipt()
    assert row.validate()
    assert not replace(row, reward=99).validate()


def test_prose_is_excluded_from_structural_fingerprint():
    r1, r2 = receipt(0), receipt(1)
    fork = replay(r1)
    harness = DeterministicHarness()
    receipts = {row.receipt_id: row for row in (r1, r2, fork)}
    first = harness.audit_motif(motif(r1, r2, fork, "collect treasure"), receipts)
    second = harness.audit_motif(motif(r1, r2, fork, "verify a web page"), receipts)
    assert first.accepted and second.accepted
    assert first.structural_fingerprint == second.structural_fingerprint


def test_receipt_identity_is_excluded_from_structural_fingerprint():
    first_r1, first_r2 = receipt(0), receipt(1)
    second_r1, second_r2 = receipt(10), receipt(11)
    first_fork, second_fork = replay(first_r1), replay(second_r1)
    harness = DeterministicHarness()
    first = harness.audit_motif(
        motif(first_r1, first_r2, first_fork),
        {row.receipt_id: row for row in (first_r1, first_r2, first_fork)},
    )
    second = harness.audit_motif(
        motif(second_r1, second_r2, second_fork),
        {row.receipt_id: row for row in (second_r1, second_r2, second_fork)},
    )
    assert first.accepted and second.accepted
    assert first.structural_fingerprint == second.structural_fingerprint


def test_skill_condition_label_is_excluded_from_structural_fingerprint():
    r1, r2 = receipt(0), receipt(1)
    fork = replay(r1)

    def source_candidate(conditioned):
        return MotifCandidate(
            "source",
            (r1.receipt_id, r2.receipt_id),
            (
                MotifNode("a", (r1.receipt_id,), (
                    SourceStepSignature(conditioned, "AGENT", "ZERO", False),
                )),
                MotifNode("b", (r2.receipt_id,), (
                    SourceStepSignature(conditioned, "POLICY_POSTPROCESSOR", "ZERO", False),
                )),
            ),
            (MotifEdge("a", "b", (fork.receipt_id,)),),
        )

    receipts = {row.receipt_id: row for row in (r1, r2, fork)}
    harness = DeterministicHarness()
    skill_on = harness.audit_motif(source_candidate(True), receipts)
    skill_off = harness.audit_motif(source_candidate(False), receipts)
    assert skill_on.accepted and skill_off.accepted
    assert skill_on.structural_fingerprint == skill_off.structural_fingerprint


def test_anonymous_skill_class_change_is_observable_control_variation():
    r1, r2 = receipt(0), receipt(1)
    fork = replay(r1)
    candidate = MotifCandidate(
        "source",
        (r1.receipt_id, r2.receipt_id),
        (
            MotifNode("a", (r1.receipt_id,), (
                SourceStepSignature(True, "AGENT", "ZERO", False, 0),
            )),
            MotifNode("b", (r2.receipt_id,), (
                SourceStepSignature(True, "AGENT", "ZERO", False, 1),
            )),
        ),
        (MotifEdge("a", "b", (fork.receipt_id,)),),
    )
    receipts = {row.receipt_id: row for row in (r1, r2, fork)}
    audit = DeterministicHarness().audit_motif(candidate, receipts)
    assert audit.accepted


def test_source_fingerprint_is_alpha_renamed_and_run_length_invariant():
    r1, r2, r3 = receipt(0), receipt(1), receipt(2)
    fork = replay(r1)

    def candidate(first_class, second_class, repeat):
        second_receipts = (r2.receipt_id, r3.receipt_id) if repeat else (r2.receipt_id,)
        second_signatures = tuple(
            SourceStepSignature(True, "AGENT", "ZERO", False, second_class)
            for _ in second_receipts
        )
        return MotifCandidate(
            "source",
            (r1.receipt_id, r2.receipt_id, r3.receipt_id),
            (
                MotifNode("a", (r1.receipt_id,), (
                    SourceStepSignature(True, "AGENT", "ZERO", False, first_class),
                )),
                MotifNode("b", second_receipts, second_signatures),
            ),
            (MotifEdge("a", "b", (fork.receipt_id,)),),
        )

    receipts = {row.receipt_id: row for row in (r1, r2, r3, fork)}
    harness = DeterministicHarness()
    first = harness.audit_motif(candidate(1, 7, True), receipts)
    second = harness.audit_motif(candidate(20, 30, False), receipts)
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


def test_uniform_chain_is_rejected():
    r1, r2 = receipt(0), receipt(1)
    fork = replay(r1)
    candidate = MotifCandidate(
        "uniform",
        ("episode",),
        (MotifNode("a", (r1.receipt_id,)), MotifNode("b", (r2.receipt_id,))),
        (MotifEdge("a", "b", (fork.receipt_id,)),),
    )
    audit = DeterministicHarness().audit_motif(
        candidate, {r1.receipt_id: r1, r2.receipt_id: r2, fork.receipt_id: fork}
    )
    assert not audit.accepted
    assert audit.reason == "no observable control variation"


def test_edge_replay_must_fork_from_its_source_node():
    r1, r2 = receipt(0), receipt(1)
    fork_from_r1 = replay(r1)
    candidate = MotifCandidate(
        "mislinked",
        (r1.receipt_id, r2.receipt_id),
        (
            MotifNode("a", (r2.receipt_id,), (DecisionStepSignature(2, 0, "SUPPORTED", "CONTINUE"),)),
            MotifNode("b", (r1.receipt_id,), (DecisionStepSignature(2, 1, "REFUTED", "REPLAN"),)),
        ),
        (MotifEdge("a", "b", (fork_from_r1.receipt_id,)),),
    )
    audit = DeterministicHarness().audit_motif(
        candidate,
        {r1.receipt_id: r1, r2.receipt_id: r2, fork_from_r1.receipt_id: fork_from_r1},
    )
    assert not audit.accepted
    assert audit.reason == "replay fork is not grounded in edge source node"


def test_edge_must_not_cherry_pick_source_node_forks():
    r1, r2 = receipt(0), receipt(1)
    first_fork = replay(r1)
    second_fork = ReplayForkReceipt.create(
        source_transition_id=r1.receipt_id,
        prefix_hash="prefix",
        fork_state_hash=r1.before_hash,
        admissible_actions_hash=r1.native_actions_hash,
        alternative_action="different",
        alternative_after_hash="other-alternative",
    )
    candidate = motif(r1, r2, first_fork)
    audit = DeterministicHarness().audit_motif(
        candidate,
        {row.receipt_id: row for row in (r1, r2, first_fork, second_fork)},
    )
    assert not audit.accepted
    assert audit.reason == "edge must carry every observed fork from its source node"


def test_multiple_pairs_are_aggregated():
    rows = []
    for pair_id in ("p1", "p2"):
        rows.extend(
            [
                replace(outcome("authentic", True), pair_id=pair_id),
                replace(outcome("target_only", False), pair_id=pair_id),
                replace(outcome("generic_protocol", False), pair_id=pair_id),
                replace(outcome("shuffled_topology", False), pair_id=pair_id),
                replace(outcome("other_source", False), pair_id=pair_id),
            ]
        )
    report = DeterministicHarness().evaluate_matched(rows)
    assert report.status == Lifecycle.POSITIVE_TRANSFER
    assert report.metrics["authentic"] == 1.0
