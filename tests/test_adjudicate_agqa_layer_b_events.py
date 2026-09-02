import pytest

from scripts.adjudicate_agqa_layer_b_events import _validate


def test_adjudication_requires_exact_candidate_set():
    payload = {"adjudications": [{
        "event_id": "E0", "status": "SUPPORTED", "start_frame": 1,
        "end_frame": 4, "evidence_frames": [1, 4], "confidence": 0.8,
        "reason": "visible",
    }]}
    result = _validate(payload, candidates=("E0",), frame_count=8)
    assert result["E0"]["status"] == "SUPPORTED"
    with pytest.raises(ValueError, match="one row per event"):
        _validate(payload, candidates=("E0", "E1"), frame_count=8)


def test_unknown_may_not_be_mistaken_for_supported_without_evidence():
    unknown = {"adjudications": [{
        "event_id": "E0", "status": "UNKNOWN", "start_frame": 0,
        "end_frame": 0, "evidence_frames": [], "confidence": 0.2,
        "reason": "occluded",
    }]}
    assert _validate(unknown, candidates=("E0",), frame_count=8)["E0"]["status"] == "UNKNOWN"
    unknown["adjudications"][0]["status"] = "SUPPORTED"
    with pytest.raises(ValueError, match="in-interval evidence"):
        _validate(unknown, candidates=("E0",), frame_count=8)


def test_adjudication_cannot_invent_event_id():
    payload = {"adjudications": [{
        "event_id": "E9", "status": "REFUTED", "start_frame": 0,
        "end_frame": 0, "evidence_frames": [], "confidence": 0.9,
        "reason": "contradicted",
    }]}
    with pytest.raises(ValueError, match="exactly match"):
        _validate(payload, candidates=("E0",), frame_count=8)
