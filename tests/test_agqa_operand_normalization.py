from motif_transfer.agqa_operand_normalization import (
    normalize_observation_interval_envelopes,
    parse_normalized_operand_receipt,
)


def _payload():
    return {
        "operand_role": "A",
        "requested_operand": "a person in an unknown object",
        "observations": [{
            "occurrence_id": "O0",
            "label": "a person in an unknown object",
            "subject": "person",
            "predicate": "in",
            "object": "clothes",
            "observability": "OBSERVED",
            "start_frame": 11,
            "end_frame": 15,
            "evidence_frames": [8, 9, 10, 11],
            "confidence": 0.8,
            "uncertainties": [],
        }],
        "coverage": "SUFFICIENT",
        "uncertainties": [],
    }


def test_normalizer_envelopes_existing_evidence_without_creating_evidence():
    normalized, markers = normalize_observation_interval_envelopes(
        _payload(), frame_count=24,
    )
    row = normalized["observations"][0]
    assert row["start_frame"] == 8
    assert row["end_frame"] == 15
    assert row["evidence_frames"] == [8, 9, 10, 11]
    assert markers == ("O0:DETERMINISTIC_INTERVAL_EVIDENCE_ENVELOPE",)
    parsed = parse_normalized_operand_receipt(
        _payload(), expected_role="A",
        expected_operand="a person in an unknown object", frame_count=24,
    )
    assert parsed.observations[0].start_frame == 8


def test_normalizer_does_not_repair_unobserved_or_out_of_range_evidence():
    payload = _payload()
    payload["observations"][0]["observability"] = "UNOBSERVED"
    normalized, markers = normalize_observation_interval_envelopes(
        payload, frame_count=24,
    )
    assert normalized["observations"][0]["start_frame"] == 11
    assert not markers

    payload = _payload()
    payload["observations"][0]["evidence_frames"] = [8, 30]
    normalized, markers = normalize_observation_interval_envelopes(
        payload, frame_count=24,
    )
    assert normalized["observations"][0]["start_frame"] == 11
    assert not markers
