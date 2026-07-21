from __future__ import annotations

import pytest

from harness.frozen_transfer_policy import (
    action_prompt,
    native_target_action_prompt,
    parse_exact_numbered_response,
    parse_native_target_plan_reply,
)
from scripts.eval_principled_alfworld import _choose_action
from scripts.eval_principled_alfworld import _official_won
from scripts.aggregate_principled_alfworld import _require_error_free
from scripts.propose_alfworld_bindings_35b import _request_proposal


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


def test_native_target_prompt_uses_all_target_history_without_source() -> None:
    history = [
        {
            "step": 0,
            "action": "go to drawer 1",
            "observation_after": "You arrive at drawer 1.",
            "native_reward": 0.0,
            "official_success": False,
        },
        {
            "step": 1,
            "action": "open drawer 1",
            "observation_after": "Drawer 1 is open and contains a mug.",
            "native_reward": 0.0,
            "official_success": False,
        },
    ]
    prompt = native_target_action_prompt(
        domain="alfworld",
        goal="put the mug in the cabinet",
        observation="Drawer 1 is open and contains a mug.",
        actions=["look", "take mug 1 from drawer 1"],
        interaction_history=history,
    )
    assert prompt.index("go to drawer 1") < prompt.index("open drawer 1")
    assert "Drawer 1 is open and contains a mug." in prompt
    assert "1. look" in prompt
    assert "2. take mug 1 from drawer 1" in prompt
    assert "No source-game conditioning is provided." in prompt
    assert "state_summary,next_subgoal,action_number" in prompt

    conditioned = native_target_action_prompt(
        domain="alfworld", goal="put the mug away", observation="at table",
        actions=["look"], interaction_history=history,
        source_conditioning=[{"receipt_sha256": "abc", "node_id": "N0"}],
    )
    assert "Untrusted source-side evidence receipts" in conditioned
    assert '"receipt_sha256": "abc"' in conditioned


def test_native_target_plan_is_closed_and_action_is_in_range() -> None:
    plan = parse_native_target_plan_reply(
        '{"state_summary":"drawer checked","next_subgoal":"search next location",'
        '"action_number":2}',
        n=3,
    )
    assert plan.action_index == 1
    assert plan.next_subgoal == "search next location"

    with pytest.raises(ValueError, match="keys"):
        parse_native_target_plan_reply(
            '{"state_summary":"x","next_subgoal":"y","action_number":1,'
            '"confidence":0.9}',
            n=3,
        )
    with pytest.raises(ValueError, match="out_of_range"):
        parse_native_target_plan_reply(
            '{"state_summary":"x","next_subgoal":"y","action_number":4}',
            n=3,
        )


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


def test_35b_endpoint_failure_is_not_counted_as_hallucination() -> None:
    reply, usage, proposal, error, endpoint_failure = _request_proposal(
        _BrokenClient(),  # type: ignore[arg-type]
        model="35b",
        prompt="proposal",
        program=object(),
    )
    assert (reply, usage, proposal) == ("", {}, None)
    assert error == "ENDPOINT_FAILURE:ConnectionError:endpoint unavailable"
    assert endpoint_failure is True
