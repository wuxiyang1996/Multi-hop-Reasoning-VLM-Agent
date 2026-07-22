from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import Advisory, DecisionProposal, Observation, TransitionReceipt


class DecisionAgent(Protocol):
    """The only agent authorized to choose a target-native environment action."""

    def propose(
        self,
        observation: Observation,
        goal: str,
        history: Sequence[TransitionReceipt],
        advisory: Advisory | None,
    ) -> DecisionProposal: ...


class FirstNativeDecisionAgent:
    """Deterministic smoke fixture; not an experimental policy."""

    def propose(self, observation, goal, history, advisory):
        if not observation.native_actions:
            raise ValueError("environment supplied no native actions")
        return DecisionProposal(
            proposal_id=f"decision-{len(history)}",
            action=observation.native_actions[0],
            prediction="execute one native transition",
            rationale="smoke fixture",
        )
