"""`SkillCrafterService` — the crafter's outward-facing API.

Spec: PLAN-SKILL-CRAFTER §4 + PLAN-COMPONENTS-IMPLEMENTATION §4.

This is the **only** module the orchestrator calls into for crafter
work. It bundles `Composer`, `Generalizer`, `Hypothesizer`, the failure
diagnoser, and the failure memory, and ensures *every* output:

  1. is persisted to the artifact store as a typed proposal,
  2. is materialized as a DRAFT `SkillRecord` via the lifecycle manager,
  3. carries provenance (`parent_skill_ids`, `proposal_id`,
     `seed_failure_ids`).

The service does NOT decide what to do with the proposal next — that's
the gate's job. It does NOT mutate any active store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from common.enums import SkillType
from common.ids import new_skill_id
from common.models import BACKBONE_TEACHER_MODEL
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_record import SkillRecord
from crafter.composer import Composer
from crafter.failure_diagnoser import FailureDiagnoser
from crafter.failure_memory import FailureMemory, FailurePattern
from crafter.generalizer import Generalizer
from crafter.hypothesizer import Hypothesizer
from orchestrator.artifact_store import ArtifactStore
from skill_bank.lifecycle import SkillLifecycleManager


@dataclass
class CrafterCycleResult:
    n_failures_ingested: int
    n_patterns_examined: int
    proposals: List[BankMutationProposal]


class SkillCrafterService:
    def __init__(
        self,
        *,
        lifecycle: SkillLifecycleManager,
        artifact_store: ArtifactStore,
        composer: Optional[Composer] = None,
        generalizer: Optional[Generalizer] = None,
        hypothesizer: Optional[Hypothesizer] = None,
        diagnoser: Optional[FailureDiagnoser] = None,
        failure_memory: Optional[FailureMemory] = None,
        teacher_model: Optional[str] = None,
        hot_pattern_threshold: int = 3,
    ) -> None:
        self._lifecycle = lifecycle
        self._artifacts = artifact_store
        self._composer = composer or Composer()
        self._generalizer = generalizer or Generalizer()
        self._hypothesizer = hypothesizer or Hypothesizer()
        self._diagnoser = diagnoser or FailureDiagnoser()
        self._failures = failure_memory or FailureMemory()
        # Default teacher = project-wide backbone (currently GPT-4o); see
        # `common/models.py`. The 32B / 72B Qwen tracks are deferred and
        # may be re-enabled by passing `teacher_model="Qwen/Qwen2.5-72B"`.
        self._teacher = teacher_model or BACKBONE_TEACHER_MODEL
        self._threshold = hot_pattern_threshold

    # -- explicit invocations --------------------------------------------

    def propose_composition(
        self,
        components: Iterable[SkillRecord],
        *,
        name: str,
        rationale: str,
        target_domains: Optional[List[str]] = None,
    ) -> BankMutationProposal:
        proposal = self._composer.compose(
            components=components,
            name=name,
            rationale=rationale,
            target_domains=target_domains,
            teacher_model=self._teacher,
        )
        self._persist(proposal, skill_type=SkillType.MIXED, name=name)
        return proposal

    def propose_generalization(
        self,
        base: SkillRecord,
        *,
        new_domains: Iterable[str],
        rationale: str,
    ) -> BankMutationProposal:
        proposal = self._generalizer.generalize(
            base=base,
            new_domains=new_domains,
            rationale=rationale,
            teacher_model=self._teacher,
        )
        self._persist(proposal, skill_type=base.skill_type, name=proposal.name)
        return proposal

    def propose_retirement(self, skill_id: str, *, reason: str) -> BankMutationProposal:
        proposal = RetireProposal(
            target_skill_id=skill_id,
            rationale=reason,
            reason=reason,
            proposed_at=time.time(),
        )
        self._artifacts.put_proposal(proposal)
        self._artifacts.append_audit(
            {"kind": "proposal", "type": "RetireProposal", "proposal_id": proposal.proposal_id, "target_skill_id": skill_id}
        )
        return proposal

    # -- failure-driven cycle --------------------------------------------

    def ingest_failures(self, traces: Iterable[FailureTrace]) -> int:
        n = 0
        for t in traces:
            self._failures.add(t)
            self._artifacts.put_failure(t)
            n += 1
        return n

    def cycle(
        self,
        *,
        new_failures: Optional[Iterable[FailureTrace]] = None,
    ) -> CrafterCycleResult:
        n_in = self.ingest_failures(new_failures or [])
        proposals: List[BankMutationProposal] = []
        hot = self._failures.hot_patterns(min_count=self._threshold)
        for pattern in hot:
            diagnosis = self._diagnoser.diagnose(
                # Pull a representative trace for diagnosis.
                self._failures.trace(pattern.failure_ids[-1])  # type: ignore[arg-type]
            )
            hypothesis = self._hypothesizer.propose(
                pattern=pattern,
                diagnosis=diagnosis,
                teacher_model=self._teacher,
            )
            if hypothesis is None:
                continue
            self._persist(hypothesis, skill_type=SkillType.MIXED, name=hypothesis.name)
            proposals.append(hypothesis)
        return CrafterCycleResult(
            n_failures_ingested=n_in,
            n_patterns_examined=len(hot),
            proposals=proposals,
        )

    # -- internals --------------------------------------------------------

    def _persist(
        self,
        proposal: BankMutationProposal,
        *,
        skill_type: SkillType,
        name: str,
    ) -> None:
        self._artifacts.put_proposal(proposal)
        # Materialize a DRAFT skill record so the gate has something to evaluate.
        skill = self._proposal_to_draft(proposal, skill_type=skill_type, name=name)
        if skill is not None:
            self._lifecycle.ingest_draft(skill)
        self._artifacts.append_audit(
            {
                "kind": "proposal",
                "type": type(proposal).__name__,
                "proposal_id": proposal.proposal_id,
                "draft_skill_id": skill.skill_id if skill else None,
            }
        )

    def _proposal_to_draft(
        self,
        proposal: BankMutationProposal,
        *,
        skill_type: SkillType,
        name: str,
    ) -> Optional[SkillRecord]:
        if isinstance(proposal, ComposeProposal):
            protocol = proposal.composed_protocol
            contract = proposal.contract
            domains = proposal.target_domains
        elif isinstance(proposal, GeneralizeProposal):
            protocol = proposal.abstracted_protocol
            contract = proposal.contract
            domains = proposal.target_domains
        elif isinstance(proposal, HypothesisProposal):
            protocol = proposal.novel_protocol
            contract = proposal.contract
            domains = proposal.target_domains
        elif isinstance(proposal, PatchProposal):
            protocol = proposal.patched_protocol
            contract = proposal.patched_contract  # may be None; gate Stage 0 will catch
            domains = proposal.target_domains
        elif isinstance(proposal, RetireProposal):
            return None
        else:
            return None
        return SkillRecord.new(
            name=name or "unnamed",
            skill_type=skill_type,
            source_type=proposal.source_type,
            feasible_domains=list(domains),
            protocol=protocol,
            contract=contract or None,  # type: ignore[arg-type]
            proposal_id=proposal.proposal_id,
            parent_skill_ids=list(proposal.parent_skill_ids),
        )


__all__ = ["CrafterCycleResult", "SkillCrafterService"]
