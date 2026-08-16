import json

import pytest

from motif_transfer.webshop_candidate_failclosed_v3 import (
    FALLBACK_SCHEMA,
    failclosed_decision_candidates,
)


def test_valid_candidates_pass_through_without_fallback():
    def base(**kwargs):
        attempts = kwargs["attempts_out"]
        attempts.append({"attempt": 0, "valid_candidates": 1})
        return ("click('12')",), '{"candidates":[{"action":"click(\\"12\\")"}]}', attempts

    candidates, _, attempts = failclosed_decision_candidates(base)
    assert candidates == ("click('12')",)
    assert not any("deterministic_fallback" in row for row in attempts)


@pytest.mark.parametrize("error", [
    ValueError("Decision response contains no valid target-native action"),
    json.JSONDecodeError("invalid", "x", 0),
])
def test_exhausted_schema_validation_gets_audited_safe_noop(error):
    def base(**kwargs):
        attempts = kwargs["attempts_out"]
        attempts.append({"attempt": 0, "validation_error": type(error).__name__})
        raise error

    candidates, raw, attempts = failclosed_decision_candidates(base)
    assert candidates == ("noop()",)
    assert json.loads(raw)["candidates"] == [{"action": "noop()"}]
    fallback = attempts[-1]["deterministic_fallback"]
    assert fallback["schema_version"] == FALLBACK_SCHEMA
    assert fallback["task_or_goal_information_used"] is False
    assert fallback["source_information_used"] is False
    assert fallback["condition_information_used"] is False
    assert fallback["provider_call"] is False


def test_transport_errors_are_not_hidden_by_policy_fallback():
    def base(**kwargs):
        raise RuntimeError("HTTP 503")

    with pytest.raises(RuntimeError, match="HTTP 503"):
        failclosed_decision_candidates(base)
