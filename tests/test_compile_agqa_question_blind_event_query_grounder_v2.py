import pytest

from scripts.compile_agqa_question_blind_event_query_grounder_v2 import (
    _combined_provider_success,
    _expected_statuses,
    _frozen_model_contract,
    _validate_anchor_row,
)


def test_development_and_fresh_statuses_cannot_be_confused() -> None:
    consumed = _expected_statuses(True)
    fresh = _expected_statuses(False)
    assert all("CONSUMED_DEVELOPMENT" in value for value in consumed)
    assert all("CONSUMED_DEVELOPMENT" not in value for value in fresh)


def test_anchor_row_is_bound_to_exact_parser_obligations_and_pixel_evidence() -> None:
    plan = {"action_obligations": [{"phrase": "opening a refrigerator"}]}
    anchor = {
        "video_id": "V0",
        "anchor_specs": [{"anchor_id": "A0", "phrase": "opening a refrigerator"}],
        "anchor_localizations": [{
            "anchor_id": "A0", "status": "SUPPORTED", "confidence": 0.9,
            "evidence_frame_ids": [12, 8],
        }],
        "anchor_intervals": [[8, 12]],
    }
    _validate_anchor_row(anchor, plan, video_id="V0")
    anchor["anchor_intervals"] = [[7, 12]]
    with pytest.raises(ValueError, match="pixel evidence"):
        _validate_anchor_row(anchor, plan, video_id="V0")


def test_unknown_anchor_must_fail_closed_without_evidence() -> None:
    plan = {"action_obligations": [{"phrase": "taking a cup"}]}
    anchor = {
        "video_id": "V0",
        "anchor_specs": [{"anchor_id": "A0", "phrase": "taking a cup"}],
        "anchor_localizations": [{
            "anchor_id": "A0", "status": "UNKNOWN", "confidence": 0.0,
            "evidence_frame_ids": [4],
        }],
        "anchor_intervals": [],
    }
    with pytest.raises(ValueError, match="fail-closed"):
        _validate_anchor_row(anchor, plan, video_id="V0")


def test_frozen_model_contract_pins_provider_seed_and_disables_fallback() -> None:
    assert _frozen_model_contract({
        "model": "qwen/example", "provider": "parasail", "seed": 0,
        "provider_allow_fallbacks": False,
    }) == {
        "id": "qwen/example", "omit_temperature": False, "seed": 0,
        "provider": {"only": ["parasail"], "allow_fallbacks": False},
    }


def test_provider_success_counts_event_and_actual_anchor_calls() -> None:
    events = [{"clips": [
        {"provider_error": None},
        {"provider_error": "NO_VISIBLE_PERSON_OBJECT_PAIR"},
        {"provider_error": "TimeoutError:x"},
    ]}]
    anchors = [
        {"call_receipt": None, "provider_error": None},
        {"call_receipt": {}, "provider_error": None},
        {"call_receipt": {}, "provider_error": "TimeoutError:y"},
    ]
    assert _combined_provider_success(events, anchors) == (3, 5, 0.6)
