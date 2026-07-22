from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import (
    Advisory,
    AdvisoryVerdict,
    BindingHypothesis,
    ContinuationDecision,
    DecisionCycleRecord,
    DecisionCycleReceipt,
    Observation,
    TransitionReceipt,
)
from .decision_agent import DecisionAgent
from .harness import DeterministicHarness
from .motif_harness_agent import MotifHarnessAgent


class Environment(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: str) -> tuple[Observation, float]: ...


@dataclass(frozen=True)
class EpisodeResult:
    receipts: tuple[TransitionReceipt, ...]
    cycles: tuple[DecisionCycleReceipt, ...]
    records: tuple[DecisionCycleRecord, ...]
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
        cycles: list[DecisionCycleReceipt] = []
        records: list[DecisionCycleRecord] = []
        advisory: Advisory | None = None
        for _ in range(max_steps):
            try:
                if observation.terminal:
                    break
                proposal_set = self.decision_agent.propose_set(observation, goal, receipts, advisory)
                self.harness.validate_proposal_set(observation, proposal_set)
                proposal = proposal_set.selected
                # Review can request replanning/abstention but its schema cannot carry an action.
                advisory = self.motif_agent.review(proposal, observation, binding, receipts)
                if advisory.verdict == AdvisoryVerdict.ABSTAIN:
                    break
                if advisory.verdict == AdvisoryVerdict.REPLAN:
                    proposal_set = self.decision_agent.propose_set(observation, goal, receipts, advisory)
                    self.harness.validate_proposal_set(observation, proposal_set)
                    proposal = proposal_set.selected
                after, reward = environment.step(proposal.action)
                receipt = TransitionReceipt.create(observation, proposal, after, reward)
                self.harness.validate_receipt(receipt)
                assessment = self.decision_agent.assess_transition(
                    observation, proposal_set, after, reward, receipts
                )
                cycle = DecisionCycleReceipt.create(proposal_set, receipt, assessment)
                self.harness.validate_cycle(cycle)
                record = DecisionCycleRecord(
                    observation, proposal_set, advisory, after, reward, receipt, assessment, cycle,
                )
                if not record.validate():
                    raise RuntimeError("decision-cycle record failed self-validation")
                receipts.append(receipt)
                cycles.append(cycle)
                records.append(record)
                observation = after
                if assessment.continuation in {
                    ContinuationDecision.ABSTAIN,
                    ContinuationDecision.TERMINATE,
                }:
                    break
                if assessment.continuation == ContinuationDecision.REPLAN:
                    advisory = Advisory(
                        AdvisoryVerdict.REPLAN,
                        assessment.reason,
                        (receipt.receipt_id,),
                        failure_route="decision agent requested replanning after live evidence",
                    )
            except Exception as exc:
                exc.partial_episode_result = EpisodeResult(
                    tuple(receipts), tuple(cycles), tuple(records), observation
                )
                raise
        return EpisodeResult(tuple(receipts), tuple(cycles), tuple(records), observation)
