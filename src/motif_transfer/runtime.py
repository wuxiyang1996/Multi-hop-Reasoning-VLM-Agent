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
    source_failures: tuple[str, ...] = ()


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
        source_failures: list[str] = []
        source_fallback_step = None
        source_enabled = version_space is not None
        for step_index in range(max_steps):
            try:
                if observation.terminal:
                    break
                if not source_enabled and step_index > 0:
                    advisory = Advisory(
                        AdvisoryVerdict.ADMIT,
                        "target-only fallback; source intervention disabled",
                    )
                proposal_set = self.decision_agent.propose_set(observation, goal, receipts, advisory)
                self.harness.validate_proposal_set(observation, proposal_set)
                proposal = proposal_set.selected
                active_bindings: tuple[BindingHypothesis, ...] = ()
                if source_enabled and version_space is not None:
                    viable = version_space.viable()
                    if viable:
                        active_bindings = tuple(sorted(viable, key=lambda row: row.binding_id))
                    else:
                        source_enabled = False
                        source_fallback_step = step_index
                if active_bindings:
                    # Review can inspect an already selected action but cannot output one.
                    aggregate_review = getattr(self.motif_agent, "review_bindings", None)
                    try:
                        advisory = (
                            aggregate_review(proposal, observation, active_bindings, receipts)
                            if callable(aggregate_review)
                            else self.motif_agent.review(
                                proposal, observation, active_bindings[0], receipts
                            )
                        )
                    except Exception as exc:
                        source_failures.append(f"REVIEW:{type(exc).__name__}:{exc}")
                        source_enabled = False
                        source_fallback_step = step_index
                        active_bindings = ()
                        advisory = Advisory(
                            AdvisoryVerdict.ADMIT,
                            "target-only fallback; source intervention disabled",
                        )
                else:
                    advisory = Advisory(
                        AdvisoryVerdict.ADMIT,
                        "target-only fallback; source intervention disabled",
                    )
                if advisory.verdict == AdvisoryVerdict.ABSTAIN:
                    source_enabled = False
                    source_fallback_step = step_index
                    active_bindings = ()
                    advisory = Advisory(
                        AdvisoryVerdict.ADMIT,
                        "target-only fallback; source intervention disabled",
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
                aggregate_verifier = getattr(self.motif_agent, "verify_bindings", None)
                if active_bindings and (callable(aggregate_verifier) or callable(verifier)):
                    try:
                        evidence_rows = (
                            aggregate_verifier(
                                active_bindings, record.before, record.proposal_set.selected,
                                record.after, record.transition, tuple(receipts),
                            )
                            if callable(aggregate_verifier)
                            else tuple(verifier(
                                active_binding, record.before, record.proposal_set.selected,
                                record.after, record.transition, tuple(receipts),
                            ) for active_binding in active_bindings)
                        )
                        for evidence in evidence_rows:
                            if evidence.receipt_id != record.transition.receipt_id:
                                raise ValueError("binding evidence references a different live transition")
                            version_space.record(evidence)
                            binding_evidence.append(evidence)
                        if not version_space.viable():
                            source_enabled = False
                            source_fallback_step = step_index + 1
                    except Exception as exc:
                        source_failures.append(f"VERIFY:{type(exc).__name__}:{exc}")
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
                    tuple(source_failures),
                )
                raise
        return EpisodeResult(
            tuple(receipts), tuple(cycles), tuple(records), observation,
            tuple(binding_evidence), source_fallback_step, source_replans,
            tuple(source_failures),
        )
