from __future__ import annotations

import pytest

from harness.frozen_transfer_policy import (
    action_prompt,
    parse_exact_numbered_response,
)
from scripts.eval_principled_alfworld import _choose_action
from scripts.eval_principled_alfworld import _official_won
from scripts.aggregate_principled_alfworld import _require_error_free


@pytest.mark.parametrize("value", ["ACTION: 2", " action: 2\n"])
def test_exact_action_response(value: str) -> None:
    assert parse_exact_numbered_response(value, kind="action", n=3) == 1


@pytest.mark.parametrize(
    "value",
    [
        "I choose ACTION: 2",
        "REASONING: maybe\nACTION: 2",
        "ACTION: go north",
        "ACTION: 0",
        "ACTION: 4",
        "2",
    ],
)
def test_hallucinated_or_extracted_action_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_exact_numbered_response(value, kind="action", n=3)


def test_prompt_preserves_exact_action_strings() -> None:
    prompt = action_prompt(
        domain="alfworld",
        goal="put the mug away",
        observation="at the table",
        actions=["look", "take mug 1 from table 1"],
    )
    assert "1. look" in prompt
    assert "2. take mug 1 from table 1" in prompt
    assert "Return exactly `ACTION: N`" in prompt


class _BrokenClient:
    def complete(self, **_: object) -> tuple[str, dict[str, object]]:
        raise ConnectionError("endpoint unavailable")


def test_endpoint_failure_is_not_counted_as_model_abstention() -> None:
    with pytest.raises(RuntimeError, match="ACTION_ENDPOINT_FAILURE:ConnectionError"):
        _choose_action(
            condition="base",
            client=_BrokenClient(),  # type: ignore[arg-type]
            base_model="base",
            action_adapter="action",
            skill_adapter="skill",
            guard=None,  # type: ignore[arg-type]
            goal="put the mug away",
            observation="at the table",
            admissible=["look"],
            recent_actions=[],
        )


def test_only_official_won_counts_as_success() -> None:
    assert _official_won({"won": [True]}) is True
    assert _official_won({"won": [False], "raw_env_reward": 1.0}) is False
    assert _official_won({}) is False


def test_aggregation_rejects_any_endpoint_error() -> None:
    rows = [{"error": None}, {"error": "POLICY_ERROR:ACTION_ENDPOINT_FAILURE"}]
    with pytest.raises(RuntimeError, match="evaluation errors in base/eval"):
        _require_error_free(rows, condition="base", split="eval")
