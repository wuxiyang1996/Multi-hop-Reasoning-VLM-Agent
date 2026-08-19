from motif_transfer.agqa_operand_normalization_v2 import (
    normalize_operand_receipt_syntax,
    parse_normalized_operand_receipt_v2,
)


def _payload():
    return {
        "operand_role": "B",
        "requested_operand": "tidying a blanket",
        "observations": [{
            "occurrence_id": "O0",
            "label": "tidying a blanket",
            "subject": "person",
            "predicate": "tidying",
            "object": "blanket",
            "observability": "OBSERVED",
            "start_frame": 21,
            "end_frame": 3,
            "evidence_frames": [21, 23, 0, 1],
            "confidence": 0.75,
            "uncertainties": [],
        }],
        "coverage": "SUFFICIENT",
        "uncertainties": [],
    }


def test_sorts_existing_evidence_then_closes_interval_envelope():
    normalized, markers = normalize_operand_receipt_syntax(
        _payload(), frame_count=24,
    )
    row = normalized["observations"][0]
    assert row["evidence_frames"] == [0, 1, 21, 23]
    assert (row["start_frame"], row["end_frame"]) == (0, 23)
    assert markers == (
        "O0:DETERMINISTIC_EVIDENCE_FRAME_ORDER",
        "O0:DETERMINISTIC_INTERVAL_EVIDENCE_ENVELOPE",
    )
    receipt = parse_normalized_operand_receipt_v2(
        _payload(), expected_role="B",
        expected_operand="tidying a blanket", frame_count=24,
    )
    assert receipt.observations[0].evidence_frames == (0, 1, 21, 23)


def test_does_not_invent_missing_evidence_or_observation():
    payload = _payload()
    payload["observations"][0].update({
        "observability": "UNOBSERVED",
        "start_frame": None,
        "end_frame": None,
        "evidence_frames": [],
    })
    normalized, markers = normalize_operand_receipt_syntax(
        payload, frame_count=24,
    )
    assert normalized["observations"][0]["evidence_frames"] == []
    assert normalized["observations"][0]["start_frame"] is None
    assert markers == ()
