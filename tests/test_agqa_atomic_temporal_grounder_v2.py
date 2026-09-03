from __future__ import annotations

import pytest

from scripts.pilot_agqa_atomic_temporal_grounder_v2 import (
    _execute_temporal,
    _native_to_sampled,
    _provider_failure,
    _uniform,
    _validate_rows,
)


def _candidates() -> list[dict]:
    return [
        {"candidate_id": "C0", "canonical_label": "cup", "detector_confidence": 0.9},
        {"candidate_id": "C1", "canonical_label": "food", "detector_confidence": 0.8},
    ]


def test_uniform_preserves_endpoints_without_duplicates() -> None:
    assert _uniform(range(10), 4) == [0, 3, 6, 9]


def test_native_indices_map_to_real_sgdet_sample_positions() -> None:
    raw = {"sampled_original_frame_indices": [0, 10, 20, 30]}
    assert _native_to_sampled(raw, [1, 19, 29]) == [0, 2, 3]


def test_atomic_validator_requires_every_fixed_candidate_once() -> None:
    with pytest.raises(ValueError, match="every fixed identifier"):
        _validate_rows({"events": [{
            "candidate_id": "C0", "status": "SUPPORTED", "confidence": 0.9,
            "evidence_frame_ids": [2],
        }]}, "events", "candidate_id", ["C0", "C1"], [2, 4])


def test_atomic_validator_fails_closed_on_unknown_with_evidence() -> None:
    with pytest.raises(ValueError, match="cannot cite"):
        _validate_rows({"events": [{
            "candidate_id": "C0", "status": "UNKNOWN", "confidence": 0.2,
            "evidence_frame_ids": [2],
        }]}, "events", "candidate_id", ["C0"], [2, 4])


def test_before_executor_chooses_nearest_atomic_event_without_vlm_temporal_input() -> None:
    decision = _execute_temporal(
        "BEFORE",
        [{"anchor_id": "A0", "status": "SUPPORTED", "confidence": 0.9,
          "evidence_frame_ids": [20, 21]}],
        [
            {"candidate_id": "C0", "status": "SUPPORTED", "confidence": 0.99,
             "evidence_frame_ids": [1, 2]},
            {"candidate_id": "C1", "status": "SUPPORTED", "confidence": 0.90,
             "evidence_frame_ids": [17, 19]},
        ],
        _candidates(),
    )
    assert decision["selected_label"] == "food"


def test_while_executor_requires_event_inside_anchor_interval() -> None:
    decision = _execute_temporal(
        "WHILE",
        [{"anchor_id": "A0", "status": "SUPPORTED", "confidence": 0.9,
          "evidence_frame_ids": [10, 14]}],
        [
            {"candidate_id": "C0", "status": "SUPPORTED", "confidence": 0.99,
             "evidence_frame_ids": [20]},
            {"candidate_id": "C1", "status": "SUPPORTED", "confidence": 0.90,
             "evidence_frame_ids": [11, 13]},
        ],
        _candidates(),
    )
    assert decision["selected_label"] == "food"


def test_provider_failure_preserves_billed_attempt_receipts() -> None:
    error = RuntimeError("bad contract")
    error.usage = {"reported_cost_usd": 0.2, "provider_attempts": 2}
    usage, message = _provider_failure(error)
    assert usage["reported_cost_usd"] == 0.2
    assert usage["provider_attempts"] == 2
    assert message == "RuntimeError:bad contract"
