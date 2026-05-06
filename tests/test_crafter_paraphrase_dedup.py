"""Regression tests for Fix-C — persist-time hypothesis paraphrase dedup.

Pins the contract introduced in 2026-05 in
``crafter.service.SkillCrafterService._already_have_similar_hypothesis``:

* a fresh ``HypothesisProposal`` whose name + protocol overlaps an
  existing skill in active / candidate / draft above
  ``hypothesis_paraphrase_jaccard`` is **dropped** (not persisted, not
  appended to the cycle's proposal list);
* a hypothesis with novel concept tokens passes through;
* the gate is a no-op for non-Hypothesis proposals (PatchProposal /
  RetireProposal flow through unchanged because they have their own
  coalesce / target-skill_id paths);
* ``hypothesis_paraphrase_jaccard=0.0`` fully disables the gate
  (preserves pre-fix behaviour for ablation studies).

The empirical receipt motivating this gate is in the v3 attribution
summary §"Diagnosis: LLM Hypothesizer mode collapse" — without it,
36 LLM hypothesis proposals collapsed to 10 ``evidence_gate_*``
synonyms covering 1 underlying concept and the actor uplift signal
flatlined.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.enums import (
    RecoveryStrategy,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from crafter import FailureMemory, SkillCrafterService
from crafter.failure_memory import FailurePattern
from crafter.hypothesizer import Hypothesizer
from data_structure.extensions.bank_mutation_proposal import (
    HypothesisProposal,
    PatchProposal,
)
from data_structure.extensions.failure_trace import FailureDiagnosis
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from orchestrator import ArtifactStore
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName


# --------------------------------------------------------------------- helpers


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(
            StoreName.CANDIDATE, os.path.join(root, "candidate"),
        ),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(
            StoreName.ARCHIVE, os.path.join(root, "archive"),
        ),
    )


def _seed_active_evidence_gate_skill(
    lifecycle: SkillLifecycleManager,
) -> SkillRecord:
    """Seed an ACTIVE skill that mirrors the v3 mode-collapse cluster."""
    skill = SkillRecord.new(
        name="evidence_gate_before_claim",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=["visual_reasoning"],
        protocol=[
            {"action": "GROUND", "payload": {"target": "evidence"}},
            {"action": "CHECK", "payload": {"target": "evidence_present"}},
            {"action": "COMMIT", "payload": {"target": "claim"}},
        ],
        contract=SkillContract(
            preconditions=["have_question"],
            expected_evidence_roles=["VERIFY", "GATHER"],
            success_criteria=["claim_evidence_grounded"],
            abort_criteria=["evidence_missing"],
        ),
    )
    lifecycle.ingest_draft(skill)
    lifecycle.transition(
        skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok",
    )
    lifecycle.transition(
        skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="seed",
    )
    return skill


def _hypothesis_paraphrase_of_evidence_gate() -> HypothesisProposal:
    """A fresh HypothesisProposal that the v3 LLM Hypothesizer would
    have minted as a paraphrase of ``evidence_gate_before_claim``."""
    return HypothesisProposal(
        name="evidence_gate_before_effect_claim",
        rationale="paraphrase of an existing evidence-gate skill",
        parent_skill_ids=[],
        seed_failure_ids=["fail-1", "fail-2"],
        target_domains=["visual_reasoning"],
        teacher_model="gpt-5.4",
        novel_protocol=[
            {"action": "GROUND", "payload": {"target": "evidence"}},
            {"action": "CHECK", "payload": {"target": "evidence_present"}},
            {"action": "COMMIT", "payload": {"target": "claim"}},
        ],
        contract=SkillContract(
            preconditions=["have_question"],
            expected_evidence_roles=["VERIFY", "GATHER"],
            success_criteria=["claim_evidence_grounded"],
            abort_criteria=["evidence_missing"],
        ),
        source_failure_pattern_ids=["pat-1"],
    )


def _hypothesis_genuinely_novel() -> HypothesisProposal:
    """A hypothesis whose tokens don't overlap an evidence-gate skill."""
    return HypothesisProposal(
        name="numeric_unit_consistency_check",
        rationale="ensure numeric answers carry consistent units",
        parent_skill_ids=[],
        seed_failure_ids=["fail-3"],
        target_domains=["visual_reasoning"],
        teacher_model="gpt-5.4",
        novel_protocol=[
            {"action": "RETRIEVE", "payload": {"target": "unit_table"}},
            {"action": "VERIFY", "payload": {"target": "units_match"}},
            {"action": "COMMIT", "payload": {"target": "numeric_answer"}},
        ],
        contract=SkillContract(
            preconditions=["have_numeric_question"],
            expected_evidence_roles=["GATHER"],
            success_criteria=["units_consistent"],
        ),
        source_failure_pattern_ids=["pat-2"],
    )


class _StaticHypothesizer(Hypothesizer):
    """Hypothesizer stub: returns a pre-baked proposal on every call,
    so the test controls the candidate the dedup gate sees."""

    def __init__(self, queue: List[Optional[HypothesisProposal]]) -> None:
        super().__init__()
        self._queue = list(queue)

    def propose(
        self, *, pattern, diagnosis, target_domains=None,
        teacher_model=None, existing_concepts=None,
    ):
        # ``existing_concepts`` (Fix-B) is accepted-but-ignored: this
        # stub just dequeues a pre-baked proposal so the test
        # controls the candidate the dedup gate sees.
        if not self._queue:
            return None
        return self._queue.pop(0)


def _failure_pattern_with_count(n: int) -> FailurePattern:
    return FailurePattern(
        pattern_id="pat-stub",
        skill_id="",                         # transfer-target style
        failure_class="INVARIANT_VIOLATION",
        failed_step_index=0,
        domains=["visual_reasoning"],
        failure_ids=[f"fail-{i}" for i in range(n)],
        sample_abort_reasons=["wrong answer"],
        semantic_bucket="wrong_answer/visual_toolbench/freeform",
    )


# --------------------------------------------------------------------- tests


class TestParaphraseDedup:
    def test_paraphrase_hypothesis_is_dropped(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        _seed_active_evidence_gate_skill(lifecycle)

        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            hypothesizer=_StaticHypothesizer([
                _hypothesis_paraphrase_of_evidence_gate(),
            ]),
            hot_pattern_threshold=1,
            hypothesize_min_recurrences=1,
            hypothesize_related_skill_jaccard=0.0,  # disable PRE-call gate
            hypothesis_paraphrase_jaccard=0.6,      # enable POST-call gate
        )

        # Manually inject a hot pattern (skip ingest path).
        pattern = _failure_pattern_with_count(2)
        crafter._failures._patterns[pattern.pattern_id] = pattern   # type: ignore[attr-defined]
        # Inject a sentinel diagnosis so _diagnose_pattern doesn't
        # reach into FailureMemory.trace().
        crafter._diagnoser.set_llm_diagnoser(                       # type: ignore[attr-defined]
            lambda trace: FailureDiagnosis(
                failure_id="fail-stub", locus="effect_check",
                root_cause="paraphrase test",
                recommended_strategy=RecoveryStrategy.HOP_INSERTION,
            )
        )
        # _run_failure_dispatch needs FailureMemory.trace(failure_id)
        # to return SOMETHING for the latest failure_id; stub it.
        from data_structure.extensions.failure_trace import FailureTrace as _FT
        for fid in pattern.failure_ids:
            crafter._failures._traces[fid] = _FT(                   # type: ignore[attr-defined]
                failure_id=fid, skill_id="",
                failure_class="INVARIANT_VIOLATION",
                failed_step_index=0,
                domain="visual_reasoning",
                abort_reason="x",
            )

        proposals, _, _ = crafter._run_failure_dispatch([pattern])  # type: ignore[attr-defined]

        assert proposals == [], (
            "Paraphrase HypothesisProposal whose tokens overlap an "
            "active skill above the Jaccard threshold must be "
            "dropped before persist."
        )
        # Audit trail records the drop with the matched skill name.
        import json as _json
        from pathlib import Path as _P
        audit_path = _P(artifacts.root) / "audit.jsonl"
        assert audit_path.is_file(), "audit log was never written"
        audits = [
            _json.loads(line) for line in
            audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kinds = [a.get("kind") for a in audits]
        assert "hypothesize_skipped_paraphrase_dedup" in kinds, (
            f"audit log should record the dedup; got: {kinds}"
        )
        match = next(
            a for a in audits
            if a.get("kind") == "hypothesize_skipped_paraphrase_dedup"
        )
        assert match["matched_existing_name"] == "evidence_gate_before_claim"
        assert match["jaccard"] >= 0.6

    def test_genuinely_novel_hypothesis_passes_through(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        _seed_active_evidence_gate_skill(lifecycle)

        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            hypothesizer=_StaticHypothesizer([_hypothesis_genuinely_novel()]),
            hot_pattern_threshold=1,
            hypothesize_min_recurrences=1,
            hypothesize_related_skill_jaccard=0.0,
            hypothesis_paraphrase_jaccard=0.6,
        )

        pattern = _failure_pattern_with_count(2)
        crafter._failures._patterns[pattern.pattern_id] = pattern   # type: ignore[attr-defined]
        crafter._diagnoser.set_llm_diagnoser(                       # type: ignore[attr-defined]
            lambda trace: FailureDiagnosis(
                failure_id="fail-stub", locus="effect_check",
                root_cause="novel test",
                recommended_strategy=RecoveryStrategy.HOP_INSERTION,
            )
        )
        from data_structure.extensions.failure_trace import FailureTrace as _FT
        for fid in pattern.failure_ids:
            crafter._failures._traces[fid] = _FT(                   # type: ignore[attr-defined]
                failure_id=fid, skill_id="",
                failure_class="INVARIANT_VIOLATION",
                failed_step_index=0,
                domain="visual_reasoning",
                abort_reason="x",
            )

        proposals, _, _ = crafter._run_failure_dispatch([pattern])  # type: ignore[attr-defined]
        assert len(proposals) == 1
        assert proposals[0].name == "numeric_unit_consistency_check"

    def test_jaccard_zero_disables_dedup(self, tmp_path) -> None:
        """Setting threshold to 0 reverts to pre-Fix-C behaviour:
        every hypothesis (including paraphrases) is persisted."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        _seed_active_evidence_gate_skill(lifecycle)

        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            hypothesizer=_StaticHypothesizer([
                _hypothesis_paraphrase_of_evidence_gate(),
            ]),
            hot_pattern_threshold=1,
            hypothesize_min_recurrences=1,
            hypothesize_related_skill_jaccard=0.0,
            hypothesis_paraphrase_jaccard=0.0,   # gate disabled
        )

        pattern = _failure_pattern_with_count(2)
        crafter._failures._patterns[pattern.pattern_id] = pattern   # type: ignore[attr-defined]
        crafter._diagnoser.set_llm_diagnoser(                       # type: ignore[attr-defined]
            lambda trace: FailureDiagnosis(
                failure_id="fail-stub", locus="effect_check",
                root_cause="ablation",
                recommended_strategy=RecoveryStrategy.HOP_INSERTION,
            )
        )
        from data_structure.extensions.failure_trace import FailureTrace as _FT
        for fid in pattern.failure_ids:
            crafter._failures._traces[fid] = _FT(                   # type: ignore[attr-defined]
                failure_id=fid, skill_id="",
                failure_class="INVARIANT_VIOLATION",
                failed_step_index=0,
                domain="visual_reasoning",
                abort_reason="x",
            )

        proposals, _, _ = crafter._run_failure_dispatch([pattern])  # type: ignore[attr-defined]
        assert len(proposals) == 1, (
            "with paraphrase_jaccard=0.0 the dedup gate must be "
            "disabled and the paraphrase must be persisted "
            "(pre-Fix-C behaviour)"
        )

    def test_helper_returns_none_for_non_hypothesis(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            hypothesis_paraphrase_jaccard=0.6,
        )
        # PatchProposal with overlapping tokens — gate must still skip
        # because this dedup is hypothesis-only.
        patch = PatchProposal(
            rationale="x", parent_skill_ids=["skill-x"],
            seed_failure_ids=[], target_domains=["gymv"],
            teacher_model="gpt-5.4", base_skill_id="skill-x",
            patched_protocol=[{"action": "VERIFY", "payload": {"target": "x"}}],
            patched_contract=SkillContract(
                expected_evidence_roles=["VERIFY"],
            ),
            recovery_strategy=RecoveryStrategy.HOP_INSERTION.value,
        )
        assert crafter._already_have_similar_hypothesis(patch) is None  # type: ignore[attr-defined]

    def test_signature_tokens_capture_concept(self) -> None:
        """The signature must concentrate concept tokens (not just
        the name) so two hypotheses with renamed surface forms but
        identical protocols are still flagged."""
        a = SkillCrafterService._hypothesis_signature_tokens(
            "evidence_gate_before_claim",
            [
                {"action": "GROUND", "payload": {"target": "evidence"}},
                {"action": "CHECK", "payload": {"target": "claim"}},
            ],
            SkillContract(
                expected_evidence_roles=["VERIFY"],
                success_criteria=["evidence_grounded"],
            ),
        )
        b = SkillCrafterService._hypothesis_signature_tokens(
            "claim_pre_evidence",  # renamed surface
            [
                {"action": "GROUND", "payload": {"target": "evidence"}},
                {"action": "CHECK", "payload": {"target": "claim"}},
            ],
            SkillContract(
                expected_evidence_roles=["VERIFY"],
                success_criteria=["evidence_grounded"],
            ),
        )
        # Concept tokens (ground / check / evidence / claim / verify /
        # evidence_grounded) appear on both sides; Jaccard should be
        # well above the default 0.6 threshold.
        jaccard = len(a & b) / len(a | b)
        assert jaccard >= 0.6, (
            f"signature should make renamed paraphrases collide; "
            f"got Jaccard {jaccard:.2f} on tokens a={a} b={b}"
        )
