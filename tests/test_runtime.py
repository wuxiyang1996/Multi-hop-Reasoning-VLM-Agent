import pytest

from motif_transfer.contracts import Advisory, AdvisoryVerdict, DecisionProposal, Observation
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.harness import HarnessReject
from motif_transfer.runtime import TwoAgentRuntime


class Env:
    def __init__(self):
        self.actions = []

    def reset(self):
        return Observation({"n": 0}, ("native",))

    def step(self, action):
        self.actions.append(action)
        return Observation({"n": 1}, (), True, True, 1), 1


class MotifAgent:
    def __init__(self, verdict=AdvisoryVerdict.ADMIT):
        self.verdict = verdict

    def review(self, proposal, observation, binding, history):
        return Advisory(self.verdict, "advice has no action field")


def test_only_decision_agent_action_is_executed():
    env = Env()
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent()).run(env, "goal")
    assert env.actions == ["native"]
    assert len(result.receipts) == 1


def test_abstain_executes_nothing():
    env = Env()
    TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent(AdvisoryVerdict.ABSTAIN)).run(env, "goal")
    assert env.actions == []


class IllegalDecisionAgent:
    def propose(self, observation, goal, history, advisory):
        return DecisionProposal("bad", "invented")


def test_non_native_action_fails_closed():
    with pytest.raises(HarnessReject):
        TwoAgentRuntime(IllegalDecisionAgent(), MotifAgent()).run(Env(), "goal")
