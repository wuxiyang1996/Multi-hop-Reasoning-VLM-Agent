from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import (
    Advisory,
    ContinuationDecision,
    DecisionProposal,
    DecisionProposalSet,
    EvidenceVerdict,
    Observation,
    PostTransitionAssessment,
    TransitionReceipt,
)


class DecisionAgent(Protocol):
    """The only agent authorized to choose a target-native environment action."""

    def propose_set(
        self,
        observation: Observation,
        goal: str,
        history: Sequence[TransitionReceipt],
        advisory: Advisory | None,
    ) -> DecisionProposalSet: ...

    def assess_transition(
        self,
        before: Observation,
        proposal_set: DecisionProposalSet,
        after: Observation,
        reward: float,
        history: Sequence[TransitionReceipt],
    ) -> PostTransitionAssessment: ...


class FirstNativeDecisionAgent:
    """Deterministic smoke fixture; not an experimental policy."""

    def propose_set(self, observation, goal, history, advisory):
        if not observation.native_actions:
            raise ValueError("environment supplied no native actions")
        proposal = DecisionProposal(
            proposal_id=f"decision-{len(history)}",
            action=observation.native_actions[0],
            prediction="execute one native transition",
            rationale="smoke fixture",
        )
        return DecisionProposalSet(f"set-{len(history)}", (proposal,), proposal.proposal_id)

    def assess_transition(self, before, proposal_set, after, reward, history):
        return PostTransitionAssessment(
            EvidenceVerdict.SUPPORTED,
            ContinuationDecision.TERMINATE if after.terminal else ContinuationDecision.CONTINUE,
            "smoke fixture observes the environment transition",
        )
