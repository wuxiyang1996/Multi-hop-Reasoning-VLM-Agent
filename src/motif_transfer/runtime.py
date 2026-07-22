from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import (
    Advisory,
    AdvisoryVerdict,
    BindingEvidence,
    BindingHypothesis,
    ContinuationDecision,
    DecisionCycleRecord,
    DecisionCycleReceipt,
    Observation,
    TransitionReceipt,
)
from .binding import BindingVersionSpace
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
    binding_evidence: tuple[BindingEvidence, ...] = ()
    source_fallback_step: int | None = None
    source_replans: int = 0


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
        bindings: Sequence[BindingHypothesis] = (),
        max_steps: int = 100,
        max_source_replans: int = 1,
    ) -> EpisodeResult:
        if binding is not None and bindings:
            raise ValueError("pass binding or bindings, not both")
        hypotheses = tuple(bindings) or ((binding,) if binding is not None else ())
        version_space = BindingVersionSpace(hypotheses) if hypotheses else None
        observation = environment.reset()
        receipts: list[TransitionReceipt] = []
        cycles: list[DecisionCycleReceipt] = []
        records: list[DecisionCycleRecord] = []
        advisory: Advisory | None = None
        binding_evidence: list[BindingEvidence] = []
        source_replans = 0
        source_fallback_step = None
        source_enabled = version_space is not None
        for step_index in range(max_steps):
            try:
                if observation.terminal:
                    break
                proposal_set = self.decision_agent.propose_set(observation, goal, receipts, advisory)
                self.harness.validate_proposal_set(observation, proposal_set)
                proposal = proposal_set.selected
                active_binding = None
                if source_enabled and version_space is not None:
                    viable = version_space.viable()
                    if viable:
                        active_binding = sorted(viable, key=lambda row: row.binding_id)[0]
                    else:
                        source_enabled = False
                        source_fallback_step = step_index
                if active_binding is not None:
                    # Review can inspect an already selected action but cannot output one.
                    advisory = self.motif_agent.review(
                        proposal, observation, active_binding, receipts
                    )
                else:
                    advisory = Advisory(
                        AdvisoryVerdict.ADMIT,
                        "target-only fallback; source intervention disabled",
                    )
                if advisory.verdict == AdvisoryVerdict.ABSTAIN:
                    source_enabled = False
                    source_fallback_step = step_index
                    active_binding = None
                    advisory = Advisory(
                        AdvisoryVerdict.ADMIT,
                        "source agent abstained; execute target-only proposal",
                    )
                elif advisory.verdict == AdvisoryVerdict.REPLAN:
                    source_replans += 1
                    proposal_set = self.decision_agent.propose_set(observation, goal, receipts, advisory)
                    self.harness.validate_proposal_set(observation, proposal_set)
                    proposal = proposal_set.selected
                    if source_replans >= max_source_replans:
                        source_enabled = False
                        source_fallback_step = step_index + 1
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
                verifier = getattr(self.motif_agent, "verify_transition", None)
                if active_binding is not None and callable(verifier):
                    evidence = verifier(
                        active_binding,
                        record.before,
                        record.proposal_set.selected,
                        record.after,
                        record.transition,
                        tuple(receipts),
                    )
                    if evidence.receipt_id != record.transition.receipt_id:
                        raise ValueError("binding evidence references a different live transition")
                    version_space.record(evidence)
                    binding_evidence.append(evidence)
                    if not version_space.viable():
                        source_enabled = False
                        source_fallback_step = step_index + 1
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
                    tuple(receipts), tuple(cycles), tuple(records), observation,
                    tuple(binding_evidence), source_fallback_step, source_replans,
                )
                raise
        return EpisodeResult(
            tuple(receipts), tuple(cycles), tuple(records), observation,
            tuple(binding_evidence), source_fallback_step, source_replans,
        )
