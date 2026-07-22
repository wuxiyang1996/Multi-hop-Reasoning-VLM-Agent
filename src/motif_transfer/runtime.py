from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import Advisory, BindingHypothesis, Observation, TransitionReceipt
from .decision_agent import DecisionAgent
from .harness import DeterministicHarness
from .motif_harness_agent import MotifHarnessAgent


class Environment(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: str) -> tuple[Observation, float]: ...


@dataclass(frozen=True)
class EpisodeResult:
    receipts: tuple[TransitionReceipt, ...]
    final_observation: Observation


class TwoAgentRuntime:
    def __init__(self, decision_agent: DecisionAgent, motif_agent: MotifHarnessAgent, harness=None):
        self.decision_agent = decision_agent
        self.motif_agent = motif_agent
        self.harness = harness or DeterministicHarness()

    def run(
        self,
        environment: Environment,
        goal: str,
        binding: BindingHypothesis | None = None,
        max_steps: int = 100,
    ) -> EpisodeResult:
        observation = environment.reset()
        receipts: list[TransitionReceipt] = []
        advisory: Advisory | None = None
        for _ in range(max_steps):
            if observation.terminal:
                break
            proposal = self.decision_agent.propose(observation, goal, receipts, advisory)
            self.harness.validate_proposal(observation, proposal)
            # Review can request replanning/abstention but its schema cannot carry an action.
            advisory = self.motif_agent.review(proposal, observation, binding, receipts)
            if advisory.verdict.value == "ABSTAIN":
                break
            if advisory.verdict.value == "REPLAN":
                proposal = self.decision_agent.propose(observation, goal, receipts, advisory)
                self.harness.validate_proposal(observation, proposal)
            after, reward = environment.step(proposal.action)
            receipt = TransitionReceipt.create(observation, proposal, after, reward)
            self.harness.validate_receipt(receipt)
            receipts.append(receipt)
            observation = after
        return EpisodeResult(tuple(receipts), observation)
