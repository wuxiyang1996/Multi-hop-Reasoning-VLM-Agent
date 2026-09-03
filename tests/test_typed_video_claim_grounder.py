import pytest

from motif_transfer.typed_video_claim_grounder import (
    CHECK_KINDS,
    execute_binary_vector_guard,
    execute_mcq_guard,
    parse_typed_claim_receipt,
    rotate_bindings,
)


def _payload(status="SUPPORTED"):
    checks = []
    for kind in CHECK_KINDS:
        step_status = status
        if kind == "DYNAMICS_INFERENCE":
            step_status = "NOT_APPLICABLE"
        if kind in {"PRECONDITION", "POSTCONDITION", "DIRECTIONAL_OR_CAUSAL_LINK"}:
            step_status = status
        checks.append({
            "kind": kind,
            "status": step_status,
            "confidence": 0.9,
            "basis": "NOT_APPLICABLE" if step_status == "NOT_APPLICABLE" else "OBSERVED",
            "evidence_frames": [] if step_status == "NOT_APPLICABLE" else [2, 4],
            "fact": "visible fact",
        })
    return {
        "claim_status": status,
        "confidence": 0.9,
        "checks": checks,
        "uncertainties": [],
        "reason": "typed verification",
    }


def test_parser_enforces_entailment_and_no_binding_leakage():
    value = parse_typed_claim_receipt(_payload(), frame_count=8)
    assert value.claim_status == "SUPPORTED"
    with pytest.raises(ValueError, match="leaked"):
        parse_typed_claim_receipt(_payload() | {"slot": "A"}, frame_count=8)
    bad = _payload() | {"claim_status": "REFUTED"}
    with pytest.raises(ValueError, match="CLAIM_ENTAILMENT"):
        parse_typed_claim_receipt(bad, frame_count=8)


def test_parser_normalizes_provider_f_prefixed_frame_ids():
    payload = _payload()
    payload["checks"][0]["evidence_frames"] = ["F2", "F4"]
    receipt = parse_typed_claim_receipt(payload, frame_count=8)
    assert receipt.checks[0].evidence_frames == (2, 4)


def test_mcq_guard_changes_only_with_unique_support_and_all_others_refuted():
    supported = parse_typed_claim_receipt(_payload(), frame_count=8)
    refuted = parse_typed_claim_receipt(_payload("REFUTED"), frame_count=8)
    required = ("ENTITY_BINDING", "PRECONDITION", "POSTCONDITION", "DIRECTIONAL_OR_CAUSAL_LINK", "CLAIM_ENTAILMENT")
    bound = [
        {"slot": "A", "receipt": refuted},
        {"slot": "B", "receipt": supported},
        {"slot": "C", "receipt": refuted},
    ]
    assert execute_mcq_guard("A", bound, required_checks=required)["answer"] == "B"
    unknown = parse_typed_claim_receipt(_payload("UNKNOWN"), frame_count=8)
    blocked = [bound[0], bound[1], {"slot": "C", "receipt": unknown}]
    assert execute_mcq_guard("A", blocked, required_checks=required)["answer"] == "A"


def test_binary_vector_decides_claims_and_preserves_unknown_bits():
    supported = parse_typed_claim_receipt(_payload(), frame_count=8)
    refuted = parse_typed_claim_receipt(_payload("REFUTED"), frame_count=8)
    unknown = parse_typed_claim_receipt(_payload("UNKNOWN"), frame_count=8)
    bound = [
        {"slot": "0", "receipt": supported},
        {"slot": "1", "receipt": refuted},
        {"slot": "2", "receipt": unknown},
    ]
    result = execute_binary_vector_guard(
        "001", bound,
        required_checks=("ENTITY_BINDING", "PRECONDITION", "POSTCONDITION", "DIRECTIONAL_OR_CAUSAL_LINK", "CLAIM_ENTAILMENT"),
    )
    assert result["answer"] == "101"
    assert result["decided_indices"] == [0, 1]


def test_binding_rotation_preserves_slots_and_rotates_receipts():
    supported = parse_typed_claim_receipt(_payload(), frame_count=8)
    refuted = parse_typed_claim_receipt(_payload("REFUTED"), frame_count=8)
    bound = [{"slot": "A", "receipt": supported}, {"slot": "B", "receipt": refuted}]
    rotated = rotate_bindings(bound)
    assert [row["slot"] for row in rotated] == ["A", "B"]
    assert [row["receipt"].claim_status for row in rotated] == ["REFUTED", "SUPPORTED"]
