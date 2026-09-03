import pytest

from motif_transfer.video_grounding_qualification import (
    ledger_localized_indices,
    localized_indices,
    parse_event_grounding_receipt,
    parse_event_ledger_receipt,
    shifted_indices,
    uniform_indices,
)


def _payload():
    return {
        "subject": "person",
        "predicate": "closes",
        "object": "book",
        "observability": "OBSERVED",
        "start_frame": 9,
        "end_frame": 13,
        "evidence_frames": [9, 11, 13],
        "before_state": "book open",
        "after_state": "book closed",
        "confidence": 0.8,
        "uncertainties": [],
        "reason": "visible state transition",
    }


def test_receipt_rejects_answer_bearing_keys_recursively():
    payload = _payload() | {"debug": {"selected_option": "B"}}
    with pytest.raises(ValueError, match="answer-bearing"):
        parse_event_grounding_receipt(payload, frame_count=24)


def test_receipt_requires_chronological_bounded_evidence():
    payload = _payload() | {"evidence_frames": [13, 9]}
    with pytest.raises(ValueError, match="chronological"):
        parse_event_grounding_receipt(payload, frame_count=24)

    payload = _payload() | {"evidence_frames": [8, 11]}
    with pytest.raises(ValueError, match="outside the grounded interval"):
        parse_event_grounding_receipt(payload, frame_count=24)


def test_localized_and_shifted_views_preserve_exact_budget():
    receipt = parse_event_grounding_receipt(_payload(), frame_count=24)
    localized = localized_indices(receipt, frame_count=24, budget=8)
    shifted = shifted_indices(localized, frame_count=24)
    assert len(localized) == len(set(localized)) == 8
    assert len(shifted) == len(set(shifted)) == 8
    assert set(receipt.evidence_frames) <= set(localized)
    assert set(localized) != set(shifted)


def test_unobserved_receipt_falls_back_to_uniform_view():
    payload = _payload() | {
        "observability": "UNOBSERVED",
        "start_frame": None,
        "end_frame": None,
        "evidence_frames": [],
    }
    receipt = parse_event_grounding_receipt(payload, frame_count=24)
    assert localized_indices(receipt, frame_count=24, budget=8) == uniform_indices(24, 8)


def test_candidate_blind_ledger_preserves_multiple_events_and_budget():
    second = _payload() | {
        "subject": "person",
        "predicate": "puts down",
        "object": "book",
        "start_frame": 18,
        "end_frame": 20,
        "evidence_frames": [18, 20],
    }
    ledger = parse_event_ledger_receipt({
        "events": [
            {"event_id": "E0", **_payload()},
            {"event_id": "E1", **second},
        ],
        "coverage": "PARTIAL",
        "uncertainties": ["off-camera interval"],
    }, frame_count=24)
    indices = ledger_localized_indices(ledger, frame_count=24, budget=8)
    assert len(indices) == len(set(indices)) == 8
    assert {9, 13, 18, 20} <= set(indices)


def test_candidate_blind_ledger_rejects_dense_frame_dump():
    dense = _payload() | {"evidence_frames": [9, 10, 11, 12, 13]}
    with pytest.raises(ValueError, match="sparse evidence"):
        parse_event_ledger_receipt({
            "events": [{"event_id": "E0", **dense}],
            "coverage": "PARTIAL",
            "uncertainties": [],
        }, frame_count=24)


def test_candidate_blind_ledger_rejects_answer_fields_and_unobserved_events():
    with pytest.raises(ValueError, match="answer-bearing"):
        parse_event_ledger_receipt({
            "events": [{"event_id": "E0", **_payload(), "answer": "B"}],
            "coverage": "SUFFICIENT",
            "uncertainties": [],
        }, frame_count=24)
    unobserved = _payload() | {
        "observability": "UNOBSERVED", "start_frame": None,
        "end_frame": None, "evidence_frames": [],
    }
    with pytest.raises(ValueError, match="OBSERVED/PARTIAL"):
        parse_event_ledger_receipt({
            "events": [{"event_id": "E0", **unobserved}],
            "coverage": "INSUFFICIENT",
            "uncertainties": [],
        }, frame_count=24)
