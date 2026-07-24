import pytest

from motif_transfer.contracts import (
    Advisory,
    AdvisoryVerdict,
    BindingEvidence,
    BindingHypothesis,
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


def test_source_abstain_falls_back_and_executes_target_action():
    env = Env()
    binding = BindingHypothesis("b", "m", "claim", "prediction", ("demo",), "v")
    result = TwoAgentRuntime(
        FirstNativeDecisionAgent(), MotifAgent(AdvisoryVerdict.ABSTAIN)
    ).run(env, "goal", binding=binding)
    assert env.actions == ["native"]
    assert result.source_fallback_step == 0


class RefutingMotifAgent(MotifAgent):
    def verify_transition(self, binding, before, proposal, after, transition, history):
        return BindingEvidence(
            binding.binding_id,
            transition.receipt_id,
            binding.verifier_id,
            EvidenceVerdict.REFUTED,
        )


def test_post_transition_refutation_updates_version_space_and_fallback():
    env = Env()
    binding = BindingHypothesis("b", "m", "claim", "prediction", ("demo",), "v")
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), RefutingMotifAgent()).run(
        env, "goal", binding=binding
    )
    assert result.binding_evidence[0].verdict == EvidenceVerdict.REFUTED
    assert result.source_fallback_step == 1


class WrongReceiptMotifAgent(MotifAgent):
    def verify_transition(self, binding, before, proposal, after, transition, history):
        return BindingEvidence(
            binding.binding_id, "different-receipt", binding.verifier_id,
            EvidenceVerdict.SUPPORTED,
        )


def test_invalid_source_evidence_disables_source_without_aborting_target_episode():
    binding = BindingHypothesis("b", "m", "claim", "prediction", ("demo",), "v")
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), WrongReceiptMotifAgent()).run(
        Env(), "goal", binding=binding
    )
    assert result.final_observation.official_success is True
    assert result.source_fallback_step == 1
    assert "different live transition" in result.source_failures[0]


class BrokenReviewMotifAgent(MotifAgent):
    def review(self, proposal, observation, binding, history):
        raise ValueError("invalid source citation")


def test_invalid_source_review_executes_original_target_proposal():
    binding = BindingHypothesis("b", "m", "claim", "prediction", ("demo",), "v")
    env = Env()
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), BrokenReviewMotifAgent()).run(
        env, "goal", binding=binding,
    )
    assert env.actions == ["native"]
    assert result.source_fallback_step == 0
    assert result.source_failures == ("REVIEW:ValueError:invalid source citation",)


class IllegalDecisionAgent:
    def propose_set(self, observation, goal, history, advisory):
        proposal = DecisionProposal("bad", "invented")
        return DecisionProposalSet("bad-set", (proposal,), "bad")

    def assess_transition(self, before, proposal_set, after, reward, history):
        return PostTransitionAssessment(EvidenceVerdict.INCONCLUSIVE, ContinuationDecision.ABSTAIN)


def test_non_native_action_fails_closed():
    with pytest.raises(HarnessReject):
        TwoAgentRuntime(IllegalDecisionAgent(), MotifAgent()).run(Env(), "goal")
