"""Fix-D regression tests: strategic_description / contract.description
end-to-end round-trip from the LLMHypothesizer output through
``crafter._proposal_to_draft`` and ``skill_bank.legacy_writeback`` to
the legacy bank envelope the actor's retrieval engine consumes.

What we cover
-------------
* **D1**: ``SkillContract.description`` is part of the dataclass and
  ``SkillContract.to_json()`` only emits it when non-empty (so existing
  snapshots' ``content_hash`` does not shift).
* **D1**: ``SkillRecord.strategic_description`` is part of the
  dataclass and serialised by ``to_json()``.
* **D2**: the LLMHypothesizer reads the new ``strategic_description``
  field from the LLM payload (and falls back to ``rationale`` when the
  LLM forgets) and threads it into ``HypothesisProposal``.
* **D3**: ``crafter.service.SkillCrafterService._proposal_to_draft``
  populates ``SkillRecord.strategic_description`` from the proposal.
* **D4**: ``skill_bank.legacy_writeback._project_to_legacy_envelope``
  surfaces the populated value into both
  ``skill.strategic_description`` and ``contract.description`` in the
  legacy envelope — the same shape ``labeling/skill_bank_qa/.../
  skill_bank.jsonl`` carries.

The fixture LLM payload mimics what ``gpt-5.4`` produces for a real
visual-reasoning failure pattern so the test catches both the
"happy-path" and the "LLM forgot the new field" branches.
"""

from __future__ import annotations

import json

from common.enums import (
    EVIDENCE_ROLES,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from crafter._llm_runtime import LLMHookConfig, LLMHypothesizer
from crafter.failure_memory import FailurePattern
from data_structure.extensions.bank_mutation_proposal import (
    HypothesisProposal,
    proposal_to_json,
)
from data_structure.extensions.failure_trace import FailureDiagnosis
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from skill_bank.legacy_writeback import _project_to_legacy_envelope


# ─────────────────────────────────────────────────────────────────────
# D1: dataclass shape
# ─────────────────────────────────────────────────────────────────────


def test_d1_contract_description_omitted_when_empty():
    """Empty ``description`` must NOT appear in ``to_json()`` so older
    snapshots that hashed without the field keep their content_hash
    stable. The gate's evaluation binding depends on this."""
    c = SkillContract()
    blob = c.to_json()
    assert "description" not in blob, (
        "SkillContract.to_json must skip 'description' when empty so "
        "pre-Fix-D snapshots keep their content_hash"
    )


def test_d1_contract_description_emitted_when_set():
    """A non-empty description must round-trip through ``to_json`` so
    the writeback projector can read it from the dict."""
    c = SkillContract(description="Invoke when claim asserts more than evidence supports.")
    blob = c.to_json()
    assert blob.get("description") == "Invoke when claim asserts more than evidence supports."


def test_d1_skill_record_strategic_description_default_empty():
    rec = SkillRecord(
        skill_id="sk-test-empty",
        name="any",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.CRAFTED,
        status=SkillStatus.DRAFT,
    )
    assert rec.strategic_description == ""
    blob = rec.to_json()
    assert blob["strategic_description"] == ""


def test_d1_skill_record_strategic_description_round_trip_through_json():
    """Use the ``new`` factory + the matching deserializer so we catch
    any field-list drift between the two."""
    from skill_bank.stores import _record_from_json

    contract = SkillContract(
        preconditions=["task asks for a quantity"],
        effects_add=["count_derived"],
        expected_evidence_roles=["GATHER"],
        description="Counts target instances under exclusion rules.",
    )
    rec = SkillRecord.new(
        name="Count target instances",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.CRAFTED,
        feasible_domains=["visual_reasoning", "browser"],
        contract=contract,
        protocol=[{"action": "GROUND", "payload": {}, "notes": "n"}],
        strategic_description=(
            "Invoke when a question asks how many objects, people, "
            "or marked items satisfy a visual or text-grounded criterion."
        ),
    )
    blob = rec.to_json()
    assert blob["strategic_description"].startswith("Invoke when a question")
    assert blob["contract"]["description"].startswith("Counts target instances")

    # Round-trip back through the deserializer.
    rec2 = _record_from_json(blob)
    assert rec2.strategic_description == rec.strategic_description
    assert rec2.contract.description == rec.contract.description


def test_d1_content_hash_stable_when_description_empty():
    """The Stage-0 evaluation gate binds reports to ``content_hash``;
    if the hash silently shifts after Fix-D, every old evaluation is
    invalidated. The protection is: ``description`` is not emitted by
    ``to_json`` when empty, so empty-description records hash exactly
    as they did pre-Fix-D."""
    pre = SkillRecord(
        skill_id="sk-stable",
        name="legacy",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        status=SkillStatus.DRAFT,
        feasible_domains=["gymv"],
        protocol=[{"action": "GROUND", "payload": {}}],
        contract=SkillContract(),
    )
    h_pre = pre.content_hash()

    # Equivalent record after Fix-D loads — same fields, no description.
    post = SkillRecord(
        skill_id="sk-stable",
        name="legacy",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        status=SkillStatus.DRAFT,
        feasible_domains=["gymv"],
        protocol=[{"action": "GROUND", "payload": {}}],
        contract=SkillContract(description=""),
        strategic_description="",
    )
    assert pre.content_hash() == post.content_hash() == h_pre, (
        "content_hash must be stable when description is empty so "
        "post-Fix-D loads of pre-Fix-D evaluations remain bound."
    )


# ─────────────────────────────────────────────────────────────────────
# D2: LLMHypothesizer parses strategic_description
# ─────────────────────────────────────────────────────────────────────


class _FakeAskModel:
    """Stub ``API_func.ask_model`` returning a canned JSON payload."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    def __call__(self, prompt, **kwargs) -> str:                     # noqa: ARG002
        self.calls += 1
        return json.dumps(self._payload)


def _patch_ask_model(monkeypatch, fake) -> None:
    """The runtime imports ``API_func.ask_model`` lazily inside
    ``_call_json``. We patch it on the module the import resolves to."""
    import API_func
    monkeypatch.setattr(API_func, "ask_model", fake)


def _make_pattern() -> FailurePattern:
    # ``count`` is a property over ``failure_ids``; we pass three IDs
    # so pattern.count == 3 (matches the prod hypothesizer-fallthrough
    # gate's ``count >= 1`` predicate).
    return FailurePattern(
        pattern_id="p-test-1",
        skill_id="sk-base",
        failure_class="evidence_gap",
        failed_step_index=2,
        domains=["visual_reasoning"],
        sample_abort_reasons=["claim exceeds evidence"],
        failure_ids=["f-1", "f-2", "f-3"],
        semantic_bucket="wrong_answer/visual_toolbench/freeform",
    )


def _make_diagnosis() -> FailureDiagnosis:
    from common.enums import RecoveryStrategy
    return FailureDiagnosis(
        failure_id="f-1",
        locus="commit",
        root_cause="Final answer asserted unsupported atoms.",
        recommended_strategy=RecoveryStrategy.HOP_INSERTION,
        confidence=0.8,
    )


def test_d2_llm_hypothesizer_threads_strategic_description(monkeypatch):
    """Happy path: LLM returns a payload with the new
    ``strategic_description`` field — it lands on the proposal AND on
    the contract's ``description`` mirror."""
    payload = {
        "name": "Gate evidence before commit",
        "strategic_description": (
            "Invoke when the model is about to assert a conclusion "
            "and at least one supporting atom has not been verified "
            "against the source. Forces an evidence-coverage check "
            "before commit."
        ),
        "novel_protocol": [
            {"action": "GROUND", "payload": {"target": "claim"}, "notes": "Pin claim."},
            {"action": "VERIFY", "payload": {"against": "evidence"}, "notes": "Match atoms."},
            {"action": "COMMIT", "payload": {"trim": "unsupported"}, "notes": "Drop unsupported."},
        ],
        "contract": {
            "preconditions": ["claim drafted", "evidence available"],
            "effects_add": ["claim_evidence_grounded"],
            "effects_del": ["unsupported_claim"],
            "expected_evidence_roles": ["VERIFY"],
            "success_criteria": ["every emitted atom has a citation"],
            "abort_criteria": ["evidence missing"],
        },
        "rationale": "Adds an evidence-coverage gate before commit.",
    }
    fake = _FakeAskModel(payload)
    _patch_ask_model(monkeypatch, fake)

    h = LLMHypothesizer(LLMHookConfig(model="gpt-5.4"))
    prop = h(_make_pattern(), _make_diagnosis())

    assert prop is not None
    assert prop.strategic_description.startswith("Invoke when the model")
    assert prop.contract.description.startswith("Invoke when the model")  # mirrored


def test_d2_llm_hypothesizer_falls_back_to_rationale(monkeypatch):
    """If the LLM forgets ``strategic_description``, the proposal must
    still carry a non-empty paragraph (we use ``rationale`` as fallback,
    since that field is always non-empty for live LLM hooks)."""
    payload = {
        "name": "Some skill",
        # NO strategic_description / NO contract.description
        "novel_protocol": [
            {"action": "GROUND", "payload": {}, "notes": "Pin."},
        ],
        "contract": {
            "preconditions": [],
            "effects_add": [],
            "effects_del": [],
            "expected_evidence_roles": ["GATHER"],
            "success_criteria": [],
            "abort_criteria": [],
        },
        "rationale": "fallback paragraph",
    }
    fake = _FakeAskModel(payload)
    _patch_ask_model(monkeypatch, fake)

    h = LLMHypothesizer(LLMHookConfig(model="gpt-5.4"))
    prop = h(_make_pattern(), _make_diagnosis())

    assert prop is not None
    assert prop.strategic_description == "fallback paragraph"
    assert prop.contract.description == "fallback paragraph"


def test_d2_llm_hypothesizer_prompt_advertises_new_field():
    """The prompt schema string must mention the new field so the LLM
    knows to emit it. This is a copy/paste guard against future
    unintended deletions."""
    from crafter._llm_runtime import _render_hypothesize_prompt
    prompt = _render_hypothesize_prompt(_make_pattern(), _make_diagnosis())
    assert "strategic_description" in prompt
    assert "skill.strategic_description" in prompt or "Skill Bank Agent" in prompt


def test_d2_proposal_to_json_includes_strategic_description():
    """The wire format the offline mirror writes to disk must carry the
    new field so promotion-side tooling sees it."""
    p = HypothesisProposal(
        name="N",
        strategic_description="when X happens, do Y",
        contract=SkillContract(description="when X happens, do Y"),
    )
    blob = proposal_to_json(p)
    assert blob.get("strategic_description") == "when X happens, do Y"
    assert blob["contract"].get("description") == "when X happens, do Y"


# ─────────────────────────────────────────────────────────────────────
# D3: _proposal_to_draft surfaces strategic_description
# ─────────────────────────────────────────────────────────────────────


def test_d3_proposal_to_draft_uses_proposal_strategic_description():
    """When the proposal carries strategic_description, the DRAFT
    SkillRecord must inherit it."""
    from crafter.service import SkillCrafterService

    p = HypothesisProposal(
        name="Some skill",
        strategic_description="Invoke when claim_extras_appear.",
        novel_protocol=[{"action": "GROUND", "payload": {}}],
        contract=SkillContract(description="Invoke when claim_extras_appear."),
        target_domains=["visual_reasoning"],
        rationale="audit text",
    )
    rec = SkillCrafterService._proposal_to_draft(
        SkillCrafterService.__new__(SkillCrafterService),
        p, skill_type=SkillType.MIXED, name=p.name,
    )
    assert rec is not None
    assert rec.strategic_description == "Invoke when claim_extras_appear."
    assert rec.contract.description == "Invoke when claim_extras_appear."


def test_d3_proposal_to_draft_falls_back_to_rationale_when_empty():
    """An older proposal without strategic_description still produces a
    non-empty SkillRecord.strategic_description via the rationale
    fallback. This makes Fix-D additive: pre-Fix-D callers that emit
    HypothesisProposal directly (no LLM hook) still benefit."""
    from crafter.service import SkillCrafterService

    p = HypothesisProposal(
        name="Some skill",
        novel_protocol=[{"action": "GROUND", "payload": {}}],
        contract=SkillContract(),  # no description
        target_domains=["visual_reasoning"],
        rationale="brief audit-style summary of what this fixes",
    )
    rec = SkillCrafterService._proposal_to_draft(
        SkillCrafterService.__new__(SkillCrafterService),
        p, skill_type=SkillType.MIXED, name=p.name,
    )
    assert rec is not None
    assert rec.strategic_description == "brief audit-style summary of what this fixes"


# ─────────────────────────────────────────────────────────────────────
# D4: writeback projector surfaces both fields into the legacy envelope
# ─────────────────────────────────────────────────────────────────────


def test_d4_writeback_envelope_matches_skill_bank_agent_shape():
    """End-to-end shape check: take a SkillRecord that carries
    Fix-D fields, project through writeback, and assert the resulting
    legacy envelope exposes both ``skill.strategic_description`` and
    ``contract.description`` populated with the same paragraph the
    legacy ``labeling/skill_bank_qa/.../skill_bank.jsonl`` Skill Bank
    Agent would produce."""
    rec = SkillRecord(
        skill_id="skill-test-d4",
        name="Gate evidence before commit",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.CRAFTED,
        status=SkillStatus.DRAFT,
        feasible_domains=["visual_reasoning"],
        protocol=[
            {"action": "GROUND", "payload": {"target": "claim"}, "notes": "Pin claim."},
            {"action": "VERIFY", "payload": {"against": "evidence"}, "notes": "Match atoms."},
        ],
        contract=SkillContract(
            preconditions=["claim drafted"],
            effects_add=["claim_evidence_grounded"],
            expected_evidence_roles=["VERIFY"],
            success_criteria=["every atom cited"],
            description="Invoke when the model is about to assert a conclusion.",
        ),
        strategic_description="Invoke when the model is about to assert a conclusion.",
    )
    envelope = _project_to_legacy_envelope(rec.to_json())
    assert envelope is not None

    sk = envelope["skill"]
    # 1. strategic_description must be populated (not empty)
    assert sk["strategic_description"].startswith("Invoke when the model")

    # 2. contract.description must mirror it
    assert sk["contract"]["description"].startswith("Invoke when the model")

    # 3. the rest of the Skill-Bank-Agent-shape fields must still be
    #    populated (the same fields a real ``labeling/skill_bank_qa/...
    #    /skill_bank.jsonl`` row carries).
    assert sk["name"] == "Gate evidence before commit"
    assert sk["evidence_role"] == "VERIFY"
    assert sk["contract"]["eff_add"] == ["claim_evidence_grounded"]
    assert sk["protocol"]["preconditions"] == ["claim drafted"]
    assert sk["protocol"]["steps"], "protocol.steps should not be empty"
    assert sk["applicable_domains"] == ["visual_reasoning"]


def test_d4_writeback_falls_back_to_contract_description():
    """If only ``contract.description`` is populated (older proposal,
    no skill-level strategic_description), the writeback must mirror
    the contract description into ``skill.strategic_description``."""
    rec = SkillRecord(
        skill_id="skill-test-fallback",
        name="X",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.CRAFTED,
        status=SkillStatus.DRAFT,
        feasible_domains=["gymv"],
        protocol=[{"action": "GROUND", "payload": {}}],
        contract=SkillContract(description="contract-only paragraph"),
        strategic_description="",  # empty
    )
    envelope = _project_to_legacy_envelope(rec.to_json())
    assert envelope is not None
    assert envelope["skill"]["strategic_description"] == "contract-only paragraph"
    assert envelope["skill"]["contract"]["description"] == "contract-only paragraph"


def test_d4_writeback_legacy_skills_keep_empty_strategic_description():
    """A pre-Fix-D record (neither field populated) must still
    project — and surface empty strategic_description, identical to
    the legacy Skill Bank's pre-Fix-D shape. No regression for
    cold-start records."""
    rec = SkillRecord(
        skill_id="skill-legacy",
        name="Legacy",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        status=SkillStatus.DRAFT,
        feasible_domains=["gymv"],
        protocol=[{"action": "GROUND", "payload": {}}],
        contract=SkillContract(),
    )
    envelope = _project_to_legacy_envelope(rec.to_json())
    assert envelope is not None
    assert envelope["skill"]["strategic_description"] == ""
    assert envelope["skill"]["contract"]["description"] == ""
