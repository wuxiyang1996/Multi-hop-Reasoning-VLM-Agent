from __future__ import annotations

import pytest

from scripts.verify_agqa_question_blind_event_candidates_v1 import (
    _response_format, _validate,
)


def test_binary_verifier_schema_cannot_name_or_select_an_object() -> None:
    schema = _response_format([2, 4, 6])["json_schema"]["schema"]
    assert set(schema["properties"]) == {
        "status", "confidence", "evidence_frame_ids",
    }
    assert schema["properties"]["status"]["enum"] == [
        "SUPPORTED", "REFUTED", "UNKNOWN",
    ]


def test_binary_verifier_requires_pixel_evidence_for_support() -> None:
    assert _validate({
        "status": "SUPPORTED", "confidence": 0.8,
        "evidence_frame_ids": [2, 6],
    }, [2, 4, 6]) == {
        "status": "SUPPORTED", "confidence": 0.8,
        "evidence_frame_ids": [2, 6],
    }
    with pytest.raises(ValueError, match="needs pixel evidence"):
        _validate({
            "status": "SUPPORTED", "confidence": 0.8,
            "evidence_frame_ids": [],
        }, [2, 4, 6])


def test_binary_verifier_fails_closed_on_unpresented_evidence() -> None:
    with pytest.raises(ValueError, match="unpresented"):
        _validate({
            "status": "REFUTED", "confidence": 0.9,
            "evidence_frame_ids": [5],
        }, [2, 4, 6])


def test_binary_verifier_fails_closed_without_same_frame_track_evidence() -> None:
    assert _validate({
        "status": "SUPPORTED", "confidence": 0.9,
        "evidence_frame_ids": [2, 6],
    }, [2, 4, 6], {2}) == {
        "status": "UNKNOWN", "confidence": 0.0,
        "evidence_frame_ids": [],
    }


def test_binary_verifier_discards_inspection_frames_from_unknown() -> None:
    assert _validate({
        "status": "UNKNOWN", "confidence": 0.7,
        "evidence_frame_ids": [4],
    }, [2, 4, 6], {4}) == {
        "status": "UNKNOWN", "confidence": 0.7,
        "evidence_frame_ids": [],
    }
