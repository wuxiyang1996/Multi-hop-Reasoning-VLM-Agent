import pytest

from motif_transfer.contracts import (
    Advisory,
    AdvisoryVerdict,
    ContinuationDecision,
    DecisionProposal,
    DecisionProposalSet,
    EvidenceVerdict,
    Observation,
    PostTransitionAssessment,
)
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
        return Advisory(
            self.verdict,
            "advice has no action field",
            current_role="verification",
            information_need="observe the transition",
            expected_transition="state changes",
            failure_route="replan",
            termination_test="official success",
        )


def test_only_decision_agent_action_is_executed():
    env = Env()
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent()).run(env, "goal")
    assert env.actions == ["native"]
    assert len(result.receipts) == 1
    assert len(result.cycles) == 1
    assert result.cycles[0].validate()
    assert len(result.records) == 1
    assert result.records[0].validate()
    assert "action" not in Advisory.__dataclass_fields__


def test_abstain_executes_nothing():
    env = Env()
    TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent(AdvisoryVerdict.ABSTAIN)).run(env, "goal")
    assert env.actions == []


class IllegalDecisionAgent:
    def propose_set(self, observation, goal, history, advisory):
        proposal = DecisionProposal("bad", "invented")
        return DecisionProposalSet("bad-set", (proposal,), "bad")

    def assess_transition(self, before, proposal_set, after, reward, history):
        return PostTransitionAssessment(EvidenceVerdict.INCONCLUSIVE, ContinuationDecision.ABSTAIN)


def test_non_native_action_fails_closed():
    with pytest.raises(HarnessReject):
        TwoAgentRuntime(IllegalDecisionAgent(), MotifAgent()).run(Env(), "goal")
