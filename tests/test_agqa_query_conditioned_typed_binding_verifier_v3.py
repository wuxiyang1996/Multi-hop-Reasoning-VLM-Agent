from scripts.verify_agqa_query_conditioned_typed_binding_v3 import (
    _selected_frame_ids,
    _validate,
)


def test_selected_frames_prioritize_proposal_and_cover_scope():
    result = _selected_frame_ids(
        {"scope": [0, 63], "evidence_frame_ids": [2, 31, 61]},
        person_visible={2, 10, 31, 50, 61},
        candidate_visible={2, 31, 40, 61}, maximum=8,
    )
    assert {2, 31, 61}.issubset(result)
    assert len(result) <= 8
    assert result == sorted(set(result))


def test_verifier_keeps_only_track_bound_supported_pixels():
    payload = {
        "status": "SUPPORTED", "confidence": 0.9,
        "evidence_frame_ids": [2, 4],
    }
    assert _validate(
        payload, frame_ids={2, 4, 6}, supportable_frame_ids={2, 4},
    ) == payload
    assert _validate(
        payload, frame_ids={2, 4, 6}, supportable_frame_ids={2},
    ) == {
        "status": "SUPPORTED", "confidence": 0.9,
        "evidence_frame_ids": [2],
    }


def test_verifier_fails_closed_when_no_supported_citation_remains():
    assert _validate(
        {
            "status": "SUPPORTED", "confidence": 0.9,
            "evidence_frame_ids": [4],
        }, frame_ids={2, 4, 6}, supportable_frame_ids={2},
    ) == {
        "status": "UNKNOWN", "confidence": 0.0,
        "evidence_frame_ids": [],
    }


def test_unknown_is_information_decreasing():
    assert _validate(
        {
            "status": "UNKNOWN", "confidence": 0.4,
            "evidence_frame_ids": [2],
        }, frame_ids={2}, supportable_frame_ids={2},
    ) == {
        "status": "UNKNOWN", "confidence": 0.4,
        "evidence_frame_ids": [],
    }
