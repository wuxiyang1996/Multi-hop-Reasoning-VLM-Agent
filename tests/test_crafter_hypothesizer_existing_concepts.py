"""Regression tests for Fix-B — existing-concepts injection.

Pins the contract introduced in 2026-05:

* ``crafter._llm_runtime._render_hypothesize_prompt(existing_concepts=...)``
  emits a constraint block before the failure-pattern payload that
  enumerates the existing concept names + hints; legacy callers that
  pass ``existing_concepts=None`` get the pre-Fix-B prompt verbatim.
* ``crafter.hypothesizer.Hypothesizer.propose(existing_concepts=...)``
  threads the kw down to the LLM hook and tolerates legacy hooks
  whose signature only accepts ``(pattern, diagnosis)`` via a
  ``TypeError`` retry.
* ``crafter.service.SkillCrafterService._collect_existing_concepts``
  builds the list out of every active / candidate / draft skill in
  the bank, honouring ``_EXISTING_CONCEPTS_RAW_CAP`` and producing
  ``"<name>: <success-criterion-or-evidence-roles>"`` descriptors.
"""

from __future__ import annotations

import os
import sys

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
from crafter import SkillCrafterService
from crafter._llm_runtime import (
    _HYPOTHESIZE_EXISTING_CONCEPTS_CAP,
    _render_hypothesize_prompt,
)
from crafter.failure_memory import FailurePattern
from crafter.hypothesizer import Hypothesizer
from data_structure.extensions.bank_mutation_proposal import HypothesisProposal
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


def _make_pattern() -> FailurePattern:
    return FailurePattern(
        pattern_id="pat-fixB",
        skill_id="",
        failure_class="INVARIANT_VIOLATION",
        failed_step_index=0,
        domains=["visual_reasoning"],
        failure_ids=["fail-1", "fail-2", "fail-3"],
        sample_abort_reasons=["wrong answer", "answer empty"],
        semantic_bucket="wrong_answer/visual_toolbench/freeform",
    )


def _make_diagnosis() -> FailureDiagnosis:
    return FailureDiagnosis(
        failure_id="fail-1", locus="effect_check",
        root_cause="committed an unverified claim",
        recommended_strategy=RecoveryStrategy.HOP_INSERTION,
    )


# --------------------------------------------------------------------- tests


# ─────────────────────────────────────────────────────────────────────
# 1. Prompt-renderer behaviour
# ─────────────────────────────────────────────────────────────────────


class TestPromptRenderer:
    def test_no_existing_concepts_keeps_legacy_prompt(self) -> None:
        """``existing_concepts=None`` (and ``[]``) must not add the
        'EXISTING SKILL BANK' block — pre-Fix-B output verbatim."""
        prompt_a = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(),
        )
        prompt_b = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(), existing_concepts=None,
        )
        prompt_c = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(), existing_concepts=[],
        )
        for p in (prompt_a, prompt_b, prompt_c):
            assert "EXISTING SKILL BANK" not in p

    def test_existing_concepts_block_appears_before_failure_pattern(self) -> None:
        """The constraint block must come BEFORE 'Failure pattern:'
        so the LLM treats it as a constraint, not afterthought."""
        prompt = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(),
            existing_concepts=[
                "evidence_gate_before_claim: claim_evidence_grounded",
                "numeric_unit_consistency_check: units_consistent",
            ],
        )
        i_block = prompt.find("EXISTING SKILL BANK")
        i_failure = prompt.find("Failure pattern:")
        assert i_block != -1
        assert i_failure != -1
        assert i_block < i_failure

    def test_existing_concepts_listed_as_bulleted_lines(self) -> None:
        prompt = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(),
            existing_concepts=[
                "skill_a: rationale_a",
                "skill_b: rationale_b",
            ],
        )
        assert "- skill_a: rationale_a" in prompt
        assert "- skill_b: rationale_b" in prompt

    def test_dedup_is_case_insensitive(self) -> None:
        """A caller passing ``["Foo", "FOO"]`` collapses to one line
        — defensive against the LLM Hypothesizer prompt bloating up
        on near-duplicate paraphrases minted across passes."""
        prompt = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(),
            existing_concepts=["Evidence_Gate", "evidence_gate", "EVIDENCE_GATE"],
        )
        assert prompt.count("- Evidence_Gate") == 1
        assert "- evidence_gate" not in prompt
        assert "- EVIDENCE_GATE" not in prompt

    def test_concept_list_is_capped(self) -> None:
        """``_HYPOTHESIZE_EXISTING_CONCEPTS_CAP`` lines is the budget;
        anything beyond is dropped."""
        many = [f"skill_{i}: criterion_{i}" for i in range(40)]
        prompt = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(), existing_concepts=many,
        )
        listed = sum(
            1 for line in prompt.splitlines()
            if line.startswith("- skill_")
        )
        assert listed == _HYPOTHESIZE_EXISTING_CONCEPTS_CAP
        # The first ones survive (FIFO), the last ones don't.
        assert "- skill_0: criterion_0" in prompt
        assert "- skill_39: criterion_39" not in prompt

    def test_empty_or_non_string_entries_are_skipped(self) -> None:
        prompt = _render_hypothesize_prompt(
            _make_pattern(), _make_diagnosis(),
            existing_concepts=[
                "ok_concept: x", "", "   ", None, 42, "ok_concept_2: y",
            ],
        )
        # Only 2 valid concepts + dedup → 2 bullets total.
        bullets = [
            line for line in prompt.splitlines() if line.startswith("- ")
        ]
        assert bullets == ["- ok_concept: x", "- ok_concept_2: y"]


# ─────────────────────────────────────────────────────────────────────
# 2. Hypothesizer wrapper — legacy + new hooks
# ─────────────────────────────────────────────────────────────────────


class TestHypothesizerWiring:
    def test_propose_threads_existing_concepts_to_new_hook(self) -> None:
        captured: dict = {}

        def _new_hook(pattern, diagnosis, *, existing_concepts=None):
            captured["pattern_id"] = pattern.pattern_id
            captured["existing_concepts"] = list(existing_concepts or [])
            return HypothesisProposal(
                name="x",
                rationale="y",
                novel_protocol=[
                    {"action": "VERIFY", "payload": {"target": "x"}},
                ],
                contract=SkillContract(),
            )

        hyp = Hypothesizer(_new_hook)
        prop = hyp.propose(
            pattern=_make_pattern(),
            diagnosis=_make_diagnosis(),
            existing_concepts=["evidence_gate_before_claim"],
        )
        assert prop is not None
        assert captured["existing_concepts"] == ["evidence_gate_before_claim"]

    def test_propose_falls_back_for_legacy_hook_signature(self) -> None:
        """A 2026-04 hook that only accepts ``(pattern, diagnosis)``
        must keep working — the wrapper retries without the kw on
        ``TypeError``."""
        calls: list = []

        def _legacy_hook(pattern, diagnosis):  # NO existing_concepts kw
            calls.append("ok")
            return HypothesisProposal(
                name="legacy", rationale="legacy",
                novel_protocol=[{"action": "VERIFY", "payload": {"target": "x"}}],
                contract=SkillContract(),
            )

        hyp = Hypothesizer(_legacy_hook)
        prop = hyp.propose(
            pattern=_make_pattern(),
            diagnosis=_make_diagnosis(),
            existing_concepts=["a", "b"],
        )
        assert prop is not None
        assert prop.name == "legacy"
        assert calls == ["ok"]

    def test_propose_rule_path_unchanged_when_llm_returns_none(self) -> None:
        """When the LLM hook is absent we fall through to the rule
        path; the new ``existing_concepts`` kw must not break that."""
        hyp = Hypothesizer()
        prop = hyp.propose(
            pattern=_make_pattern(),
            diagnosis=_make_diagnosis(),
            existing_concepts=["unused_concept"],
        )
        # HOP_INSERTION rule path is implemented and returns a stub.
        assert prop is not None


# ─────────────────────────────────────────────────────────────────────
# 3. Service-level concept collection
# ─────────────────────────────────────────────────────────────────────


def _seed_skill(
    lifecycle: SkillLifecycleManager,
    *,
    name: str,
    success_criteria=None,
    evidence_roles=None,
    status: SkillStatus = SkillStatus.ACTIVE,
) -> None:
    skill = SkillRecord.new(
        name=name,
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=["visual_reasoning"],
        protocol=[
            {"action": "VERIFY", "payload": {"target": "x"}},
        ],
        contract=SkillContract(
            preconditions=["have_question"],
            expected_evidence_roles=list(evidence_roles or ["VERIFY"]),
            success_criteria=list(success_criteria or []),
        ),
    )
    lifecycle.ingest_draft(skill)
    if status == SkillStatus.DRAFT:
        return
    lifecycle.transition(
        skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok",
    )
    if status == SkillStatus.CANDIDATE:
        return
    lifecycle.transition(
        skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="seed",
    )


class TestServiceConceptCollection:
    def test_empty_bank_yields_empty_list(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
        )
        assert crafter._collect_existing_concepts() == []          # type: ignore[attr-defined]

    def test_collects_active_candidate_and_draft_skills(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
        )

        _seed_skill(
            lifecycle, name="a_active",
            success_criteria=["a_done"], status=SkillStatus.ACTIVE,
        )
        _seed_skill(
            lifecycle, name="a_candidate",
            success_criteria=["b_done"], status=SkillStatus.CANDIDATE,
        )
        _seed_skill(
            lifecycle, name="a_draft",
            success_criteria=["c_done"], status=SkillStatus.DRAFT,
        )

        descriptors = crafter._collect_existing_concepts()         # type: ignore[attr-defined]
        names = sorted(d.split(":")[0].strip() for d in descriptors)
        assert names == ["a_active", "a_candidate", "a_draft"]
        for d in descriptors:
            assert ": " in d, (
                "every concept should ship a `<name>: <hint>` "
                "descriptor — got plain name only: " + d
            )

    def test_falls_back_to_evidence_roles_when_no_success_criteria(
        self, tmp_path,
    ) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
        )
        _seed_skill(
            lifecycle, name="no_criteria",
            success_criteria=None,
            evidence_roles=["VERIFY", "GATHER"],
        )
        descriptors = crafter._collect_existing_concepts()         # type: ignore[attr-defined]
        assert descriptors and descriptors[0] == (
            "no_criteria: evidence_roles=VERIFY/GATHER"
        )

    def test_caps_at_raw_collection_limit(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
        )
        for i in range(SkillCrafterService._EXISTING_CONCEPTS_RAW_CAP * 2):
            _seed_skill(
                lifecycle, name=f"sk_{i:02d}",
                success_criteria=[f"crit_{i}"],
            )
        descriptors = crafter._collect_existing_concepts()         # type: ignore[attr-defined]
        assert len(descriptors) == SkillCrafterService._EXISTING_CONCEPTS_RAW_CAP

    def test_dispatch_path_passes_concepts_into_llm(self, tmp_path) -> None:
        """End-to-end: when ``_run_failure_dispatch`` calls the
        Hypothesizer, the bank's concept list arrives at the
        LLM hook with the same content ``_collect_existing_concepts``
        returned."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "out"))
        _seed_skill(
            lifecycle, name="evidence_gate_before_claim",
            success_criteria=["claim_evidence_grounded"],
        )

        captured: dict = {}

        def _hook(pattern, diagnosis, *, existing_concepts=None):
            captured["concepts"] = list(existing_concepts or [])
            return HypothesisProposal(
                name="numeric_unit_consistency_check",
                rationale="novel",
                novel_protocol=[
                    {"action": "RETRIEVE", "payload": {"target": "units"}},
                ],
                contract=SkillContract(
                    expected_evidence_roles=["GATHER"],
                    success_criteria=["units_consistent"],
                ),
            )

        hypothesizer = Hypothesizer(_hook)
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            hypothesizer=hypothesizer,
            hot_pattern_threshold=1,
            hypothesize_min_recurrences=1,
            hypothesize_related_skill_jaccard=0.0,
            hypothesis_paraphrase_jaccard=0.0,  # don't drop the novel proposal
        )

        pattern = _make_pattern()
        crafter._failures._patterns[pattern.pattern_id] = pattern  # type: ignore[attr-defined]
        crafter._diagnoser.set_llm_diagnoser(                      # type: ignore[attr-defined]
            lambda trace: _make_diagnosis()
        )
        from data_structure.extensions.failure_trace import FailureTrace as _FT
        for fid in pattern.failure_ids:
            crafter._failures._traces[fid] = _FT(                  # type: ignore[attr-defined]
                failure_id=fid, skill_id="",
                failure_class="INVARIANT_VIOLATION",
                failed_step_index=0,
                domain="visual_reasoning",
                abort_reason="x",
            )

        crafter._run_failure_dispatch([pattern])                   # type: ignore[attr-defined]
        assert "concepts" in captured, "LLM hook was never called"
        assert any(
            "evidence_gate_before_claim" in c
            for c in captured["concepts"]
        ), (
            f"existing concepts did not reach the hook; got "
            f"{captured['concepts']!r}"
        )
