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

from common.enums import RecoveryStrategy, SkillType
from common.ids import new_skill_id
from common.models import (
    BACKBONE_TEACHER_MODEL,
    is_frozen_qwen_teacher,
    phase_f_teacher_from_env,
    qwen3_vl_teacher,
)
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace
from data_structure.extensions.skill_record import SkillRecord
from crafter.composer import Composer
from crafter.failure_diagnoser import FailureDiagnoser
from crafter.failure_memory import FailureMemory, FailurePattern
from crafter.generalizer import Generalizer
from crafter.hypothesizer import Hypothesizer
from crafter.repairer import Repairer
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
        repairer: Optional[Repairer] = None,
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
        self._repairer = repairer or Repairer()
        self._diagnoser = diagnoser or FailureDiagnoser()
        self._failures = failure_memory or FailureMemory()
        # Default teacher = project-wide backbone (currently GPT-4o); see
        # `common/models.py`. The 8B / 32B / 72B Qwen tracks plus the
        # Phase-F frozen Qwen3-VL teachers are deferred and may be
        # re-enabled by passing `teacher_model="Qwen/Qwen3-VL-32B"` (or
        # the 235B-A22B variant) — see `common.models.qwen3_vl_teacher`.
        self._teacher = teacher_model or BACKBONE_TEACHER_MODEL
        self._threshold = hot_pattern_threshold

    # -- phase-F frozen teacher swap -------------------------------------

    @property
    def teacher_model(self) -> str:
        """The frozen-teacher backbone the crafter stamps on every proposal."""
        return self._teacher

    @property
    def is_phase_f_active(self) -> bool:
        """True iff the active teacher is one of the Phase-F frozen Qwen3-VL teachers."""
        return is_frozen_qwen_teacher(self._teacher)

    def set_teacher_model(self, model: str) -> None:
        """Swap the frozen-teacher backbone in place.

        Phase-F entry point — call with
        ``qwen3_vl_teacher("32b")`` or ``qwen3_vl_teacher("235b-a22b")``
        to flip the crafter's teacher without rebuilding the service.
        Existing component LLM hooks (set on `FailureDiagnoser`,
        `Hypothesizer`, `Repairer`) are preserved; this only changes
        which model name gets stamped on emitted proposals.
        """
        if not model:
            raise ValueError("teacher_model must be a non-empty string")
        self._teacher = model

    @classmethod
    def with_qwen3_vl_teacher(
        cls,
        *,
        lifecycle: SkillLifecycleManager,
        artifact_store: ArtifactStore,
        size: str = "32b",
        **kwargs,
    ) -> "SkillCrafterService":
        """Phase-F constructor — instantiate with a frozen Qwen3-VL teacher.

        Equivalent to passing
        ``teacher_model=qwen3_vl_teacher(size)`` explicitly; provided
        as a one-liner so deployment scripts don't need to import the
        ``common.models`` helpers themselves.
        """
        kwargs.setdefault("teacher_model", qwen3_vl_teacher(size))
        return cls(lifecycle=lifecycle, artifact_store=artifact_store, **kwargs)

    @classmethod
    def from_env(
        cls,
        *,
        lifecycle: SkillLifecycleManager,
        artifact_store: ArtifactStore,
        **kwargs,
    ) -> "SkillCrafterService":
        """Construct the service, honoring the Phase-F env switch.

        Reads ``VLM_AGENT_PHASE_F_TEACHER`` via
        :func:`common.models.phase_f_teacher_from_env`.  When set, the
        env value (e.g. ``qwen3-vl-32b``) overrides the default
        ``BACKBONE_TEACHER_MODEL``; otherwise behaviour matches the
        plain constructor.
        """
        phase_f = phase_f_teacher_from_env()
        if phase_f is not None:
            kwargs.setdefault("teacher_model", phase_f)
        return cls(lifecycle=lifecycle, artifact_store=artifact_store, **kwargs)

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

    def propose_repair(
        self,
        *,
        base_skill_id: Optional[str] = None,
        base: Optional[SkillRecord] = None,
        pattern_id: Optional[str] = None,
        pattern: Optional[FailurePattern] = None,
        diagnosis: Optional[FailureDiagnosis] = None,
        rationale: Optional[str] = None,
    ) -> Optional[BankMutationProposal]:
        """Phase-D entry point — emit a `PatchProposal` for a known skill.

        Resolves the base skill (via the lifecycle manager's read-through
        `get`), the failure pattern (via `FailureMemory`), and a
        diagnosis (via `FailureDiagnoser`) when not supplied, then asks
        the `Repairer` to build a `PatchProposal`. The proposal lands as
        a DRAFT `SkillRecord` whose `parent_skill_ids = [base.skill_id]`
        and whose `content_hash` differs from the base, so the gate
        revalidates from scratch (PLAN-UNIFIED-SKILL-GATE §3.2).

        Returns ``None`` when the diagnosis recommends retirement
        instead — in that case the caller (or the cycle loop) routes to
        `propose_retirement`.
        """
        base = base or self._resolve_base(base_skill_id)
        if base is None:
            raise ValueError(
                "propose_repair requires a base SkillRecord or a base_skill_id "
                "that resolves through the lifecycle manager."
            )

        pattern = pattern or self._resolve_pattern(pattern_id)
        if pattern is None:
            raise ValueError(
                "propose_repair requires a FailurePattern or a pattern_id "
                "that resolves through FailureMemory."
            )

        if diagnosis is None:
            diagnosis = self._diagnose_pattern(pattern)
            if diagnosis is None:
                # No representative trace available — cannot repair safely.
                return None

        proposal = self._repairer.repair(
            base=base,
            pattern=pattern,
            diagnosis=diagnosis,
            teacher_model=self._teacher,
            rationale=rationale,
        )
        if proposal is None:
            return None
        self._persist(proposal, skill_type=base.skill_type, name=f"{base.name}__patched")
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
            diagnosis = self._diagnose_pattern(pattern)
            if diagnosis is None:
                continue

            # Dispatch order (PLAN-SKILL-CRAFTER §6.5):
            #   1. If the failing `pattern.skill_id` resolves to an
            #      existing bank skill → propose a *patch* (Phase D).
            #   2. If the diagnosis recommends retirement → emit a
            #      `RetireProposal` (still gate-bound).
            #   3. Else fall back to the hypothesizer's novel-skill
            #      proposal (the original Phase C path).
            base = self._resolve_base(pattern.skill_id) if pattern.skill_id else None
            if base is not None:
                if diagnosis.recommended_strategy == RecoveryStrategy.SKILL_RETIREMENT:
                    proposals.append(
                        self.propose_retirement(
                            base.skill_id,
                            reason=diagnosis.root_cause or "persistent failure pattern",
                        )
                    )
                    continue
                patch = self.propose_repair(
                    base=base, pattern=pattern, diagnosis=diagnosis
                )
                if patch is not None:
                    proposals.append(patch)
                    continue

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

    def _resolve_base(self, skill_id: Optional[str]) -> Optional[SkillRecord]:
        if not skill_id:
            return None
        return self._lifecycle.get(skill_id)

    def _resolve_pattern(self, pattern_id: Optional[str]) -> Optional[FailurePattern]:
        if not pattern_id:
            return None
        return self._failures.pattern(pattern_id)

    def _diagnose_pattern(self, pattern: FailurePattern) -> Optional[FailureDiagnosis]:
        if not pattern.failure_ids:
            return None
        trace = self._failures.trace(pattern.failure_ids[-1])
        if trace is None:
            return None
        return self._diagnoser.diagnose(trace)


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
