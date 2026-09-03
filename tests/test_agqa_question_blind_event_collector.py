from __future__ import annotations

import pytest

from scripts.collect_agqa_question_blind_typed_event_inventory_v1 import (
    _artifact_status, _clip_frame_ids, _request_cache_contract, _response_format,
)


def test_partition_complete_covers_every_frozen_frame_exactly_once() -> None:
    clips = _clip_frame_ids(64, 8, 8, strategy="partition_complete")
    assert clips == [list(range(index, index + 8)) for index in range(0, 64, 8)]
    assert sorted(frame for clip in clips for frame in clip) == list(range(64))


def test_partition_complete_fails_when_budget_is_not_complete() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        _clip_frame_ids(64, 4, 8, strategy="partition_complete")


def test_provider_schema_can_only_emit_public_typed_track_bindings() -> None:
    value = _response_format(["T0", "T2", "T7"], ["T0"], [8, 9, 10], 10)
    schema = value["json_schema"]["schema"]
    assert set(schema["properties"]) == {"events"}
    events = schema["properties"]["events"]
    assert events["maxItems"] == 10
    event = events["items"]
    assert event["additionalProperties"] is False
    assert event["properties"]["subject_track_id"]["enum"] == ["T0"]
    assert event["properties"]["object_track_id"]["enum"] == ["T2", "T7"]
    assert event["properties"]["evidence_frame_ids"]["items"]["enum"] == [8, 9, 10]
    assert "answer" not in event["properties"]
    assert "question" not in event["properties"]


def test_request_cache_contract_covers_generation_and_schema_options() -> None:
    model = {"id": "provider/model", "omit_temperature": True}
    schema_a = _response_format(["T0", "T1"], ["T0"], [1, 2], 4)
    schema_b = _response_format(["T0", "T2"], ["T0"], [1, 2], 4)
    base = _request_cache_contract(
        model=model, max_tokens=512, response_format=schema_a, maximum_attempts=2,
    )
    assert base["temperature_mode"] == "omitted"
    assert base["seed"] is None
    assert base["provider"] == {"require_parameters": True}
    assert base["reasoning"] is None
    assert base != _request_cache_contract(
        model=model, max_tokens=1024, response_format=schema_a, maximum_attempts=2,
    )
    assert base != _request_cache_contract(
        model=model, max_tokens=512, response_format=schema_b, maximum_attempts=2,
    )
    assert base != _request_cache_contract(
        model={**model, "reasoning": {"enabled": False}},
        max_tokens=512, response_format=schema_a, maximum_attempts=2,
    )
    assert base != _request_cache_contract(
        model={
            **model, "seed": 0,
            "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        },
        max_tokens=512, response_format=schema_a, maximum_attempts=2,
    )


def test_consumed_development_pilot_cannot_claim_frozen_transfer_evidence() -> None:
    assert _artifact_status(True) == "CONSUMED_DEVELOPMENT_PILOT_NOT_TRANSFER_EVIDENCE"
    assert "FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME" in _artifact_status(False)
