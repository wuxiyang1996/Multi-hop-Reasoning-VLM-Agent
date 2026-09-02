import dataclasses

import pytest

from motif_transfer.agqa_layer_b_contracts import AGQASemanticSlotReceipt, SemanticSlotNode
from motif_transfer.agqa_layer_b_epistemic import (
    AtomicVisualClaim, AtomicVisualClaimDecision, AtomicVisualClaimReceipt,
    extract_atomic_claims, source_open_world_commit,
    source_root_open_world_commit,
)
from motif_transfer.contracts import stable_hash


H = "a" * 64


def _semantic(surface="require exactly one grounded branch"):
    slots = (
        SemanticSlotNode("S0", "LITERAL", "opening a door"),
        SemanticSlotNode("S1", "LITERAL", "closing a door"),
        SemanticSlotNode("S2", "RELATION", "branch one", ("S0",)),
        SemanticSlotNode("S3", "RELATION", "branch two", ("S1",)),
        SemanticSlotNode("S4", "LOGICAL_CONSTRAINT", surface, ("S2", "S3")),
    )
    return AGQASemanticSlotReceipt.create(
        task_id="t", question_sha256=H, answer_kind="BOOLEAN", root_slot_id="S4",
        slots=slots, parser_sha256=H, parser_training_authority="DISJOINT_TARGET_TRAIN",
    )


def _receipt(statuses):
    claims = extract_atomic_claims(_semantic())
    decisions = tuple(AtomicVisualClaimDecision(
        claim.claim_id, status, .9, (0,), (H,), "visible transition",
    ) for claim, status in zip(claims, statuses))
    return AtomicVisualClaimReceipt.create(
        task_id="t", semantic_receipt_sha256=H,
        raw_event_graph_receipt_sha256=H, claims=claims, decisions=decisions,
        verifier_backend_sha256=H, frame_budget=2,
    )


def test_extracts_two_operator_free_logical_branches():
    claims = extract_atomic_claims(_semantic())
    assert [claim.semantic_root_slot_id for claim in claims] == ["S2", "S3"]
    assert "opening a door" in claims[0].proposition


def test_refutation_requires_pixels_and_receipt_is_content_addressed():
    claim = AtomicVisualClaim("C0", "S0", "door is closed")
    decision = AtomicVisualClaimDecision("C0", "REFUTED", .9, (), (), "not seen")
    with pytest.raises(ValueError, match="pixel evidence"):
        AtomicVisualClaimReceipt.create(
            task_id="t", semantic_receipt_sha256=H,
            raw_event_graph_receipt_sha256=H, claims=(claim,), decisions=(decision,),
            verifier_backend_sha256=H, frame_budget=2,
        )


def test_xor_requires_explicit_support_and_refutation():
    safe, _ = source_open_world_commit(
        required_operators=("EXISTS", "XOR"), symbolic_status="COMMITTED",
        symbolic_prediction="yes", evidence=_receipt(("SUPPORTED", "REFUTED")),
    )
    assert safe
    unsafe, reason = source_open_world_commit(
        required_operators=("EXISTS", "XOR"), symbolic_status="COMMITTED",
        symbolic_prediction="yes", evidence=_receipt(("SUPPORTED", "UNKNOWN")),
    )
    assert not unsafe and reason == "EXCLUSIVE_GUARDS_UNKNOWN"


def test_xor_no_requires_both_branches_decided_and_equal():
    assert source_open_world_commit(
        required_operators=("XOR",), symbolic_status="COMMITTED",
        symbolic_prediction="no", evidence=_receipt(("REFUTED", "REFUTED")),
    )[0]


def test_xor_choice_requires_one_supported_and_one_refuted():
    assert source_open_world_commit(
        required_operators=("XOR", "CHOOSE"), symbolic_status="COMMITTED",
        symbolic_prediction="chair", evidence=_receipt(("SUPPORTED", "REFUTED")),
    )[0]
    assert not source_open_world_commit(
        required_operators=("XOR",), symbolic_status="COMMITTED",
        symbolic_prediction="no", evidence=_receipt(("REFUTED", "UNKNOWN")),
    )[0]


def test_unknown_presence_never_becomes_false():
    receipt = _receipt(("UNKNOWN", "UNKNOWN"))
    safe, reason = source_open_world_commit(
        required_operators=("EXISTS",), symbolic_status="COMMITTED",
        symbolic_prediction="no", evidence=receipt,
    )
    assert not safe and reason == "CARDINALITY_GUARD_UNKNOWN"


def test_tampering_breaks_receipt_hash():
    receipt = _receipt(("SUPPORTED", "REFUTED"))
    with pytest.raises(ValueError, match="hash mismatch"):
        dataclasses.replace(receipt, frame_budget=3).validate()


def test_root_aware_gate_does_not_misclassify_equality_as_presence():
    semantic = _semantic("test semantic equality")
    safe, reason = source_root_open_world_commit(
        semantic=semantic, symbolic_status="COMMITTED", symbolic_prediction="yes",
        evidence=_receipt(("REFUTED", "UNKNOWN")),
    )
    assert safe and reason == "NO_OPEN_WORLD_BOOLEAN_GUARD_REQUIRED"
