from __future__ import annotations

import json

import pytest

from motif_transfer.alfworld_llm_decision import (
    DECISION_ROLE,
    InadmissibleActionError,
    decide_alfworld_action,
)
from motif_transfer.cross_domain_memory_runtime import MEMORY_PAYLOAD_KEY

COMMANDS = ("go to cabinet 1", "take mug 1 from cabinet 1", "look")


class ScriptedBackend:
    identity = {"model": "alfworld-decision-test"}
    last_usage = {"total_tokens": 3}

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, role, system, payload):
        self.requests.append((role, system, json.loads(json.dumps(payload))))
        return self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]


def _decide(backend, **kwargs):
    return decide_alfworld_action(
        backend,
        observation_text="You are in the middle of a room.",
        task_goal="put a clean mug in the coffeemachine",
        admissible_commands=COMMANDS,
        **kwargs,
    )


def test_valid_index_selects_the_environment_command():
    backend = ScriptedBackend([json.dumps({"action_index": 1, "reason": "grab it"})])
    decision = _decide(backend)
    assert decision.action == "take mug 1 from cabinet 1"
    assert decision.action_index == 1
    assert decision.attempts == 1
    assert backend.requests[0][0] == DECISION_ROLE


def test_out_of_range_index_is_retried_then_refused():
    backend = ScriptedBackend([json.dumps({"action_index": 99, "reason": "x"})])
    with pytest.raises(InadmissibleActionError, match="admissible-command schema"):
        _decide(backend, schema_retries=2)
    assert len(backend.requests) == 2
    assert "previous_error" in backend.requests[1][2]


def test_invented_command_cannot_be_returned():
    """The model can only emit an index, so a free-text action has no path in."""
    backend = ScriptedBackend([json.dumps({"action": "fly to the moon"})])
    with pytest.raises(InadmissibleActionError):
        _decide(backend, schema_retries=1)


def test_empty_admissible_set_is_an_error_not_a_guess():
    backend = ScriptedBackend([json.dumps({"action_index": 0})])
    with pytest.raises(InadmissibleActionError, match="no admissible commands"):
        decide_alfworld_action(
            backend, observation_text="o", task_goal="g", admissible_commands=(),
        )


def test_target_score_is_never_shown_to_the_decision_agent():
    backend = ScriptedBackend([json.dumps({"action_index": 0, "reason": "look"})])
    _decide(backend, history=[{"step": 0, "action": "look", "score": 0.5, "won": False}])
    sent = backend.requests[0][2]
    assert sent["history"] == [{"step": 0, "action": "look"}]
    assert "score" not in json.dumps(sent)


def test_payload_shape_matches_the_memory_adapter():
    """The memory wrapper reads task_goal/observation/admissible_commands."""
    from motif_transfer.cross_domain_memory_runtime import _ADAPTERS
    from motif_transfer.cross_domain_memory_baselines import TargetDomain

    backend = ScriptedBackend([json.dumps({"action_index": 0, "reason": "look"})])
    _decide(backend)
    payload = backend.requests[0][2]
    view = _ADAPTERS[TargetDomain.ALFWORLD](payload)
    assert view["task"] == "put a clean mug in the coffeemachine"
    assert view["native_actions"] == list(COMMANDS)
    assert view["observation"]["observation"].startswith("You are in the middle")
    assert MEMORY_PAYLOAD_KEY not in payload
