from __future__ import annotations

import hashlib
import json

import pytest

from harness.advisory_reasoning_harness import (
    MatchedHarnessCallLedger,
    admit_policy_proposal,
    close_policy_execution,
    parse_adaptation_binding_set,
    policy_execution_identity_receipt,
)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _reply(proposal_id: str, *, verdict: str = "ADMIT") -> str:
    return json.dumps({
        "policy_proposal_id": proposal_id,
        "binding_id": "b0",
        "verdict": verdict,
        "predicted_observable_delta": "the visible room state may change",
        "reason": "review of the policy proposal and frozen receipts",
    })


def _proposal_id() -> str:
    reply_hash = hashlib.sha256(json.dumps(
        "ACTION: 2", sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    payload = {
        "state": "before", "native_actions": ["look", "open door"],
        "policy_prompt_sha256": _h("policy-prompt"),
        "policy_reply_sha256": reply_hash, "policy_action": "open door",
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def test_advisory_agent_can_only_admit_exact_policy_action() -> None:
    admission = admit_policy_proposal(
        treatment="correct", target_state="before",
        native_actions=["look", "open door"],
        policy_prompt_sha256=_h("policy-prompt"), policy_reply="ACTION: 2",
        policy_action="open door", advisory_prompt_sha256=_h("advisory-prompt"),
        advisory_reply=_reply(_proposal_id()), allowed_binding_ids=["b0"],
    )
    assert admission.status == "POLICY_ACTION_ADMITTED"
    assert admission.approved_policy_action == admission.policy_action == "open door"
    receipt = close_policy_execution(
        admission, executed_action="open door", after_state="after",
        reward=0.0, done=False,
    )
    assert receipt.status == "POLICY_EXECUTION_VERIFIED"


def test_advisory_agent_cannot_substitute_an_action() -> None:
    admission = admit_policy_proposal(
        treatment="correct", target_state="before",
        native_actions=["look", "open door"],
        policy_prompt_sha256=_h("policy-prompt"), policy_reply="ACTION: 2",
        policy_action="open door", advisory_prompt_sha256=_h("advisory-prompt"),
        advisory_reply=_reply(_proposal_id()), allowed_binding_ids=["b0"],
    )
    receipt = close_policy_execution(
        admission, executed_action="look", after_state="after",
        reward=0.0, done=False,
    )
    assert receipt.status == "REJECTED"
    assert receipt.failure_code == "EXECUTED_ACTION_NOT_EXACT_POLICY_PROPOSAL"


def test_replan_has_no_executable_action() -> None:
    admission = admit_policy_proposal(
        treatment="correct", target_state="before",
        native_actions=["look", "open door"],
        policy_prompt_sha256=_h("policy-prompt"), policy_reply="ACTION: 2",
        policy_action="open door", advisory_prompt_sha256=_h("advisory-prompt"),
        advisory_reply=_reply(_proposal_id(), verdict="REPLAN"),
        allowed_binding_ids=["b0"],
    )
    assert admission.status == "POLICY_REPLAN_REQUESTED"
    assert admission.approved_policy_action is None


def test_target_only_cannot_cite_source_binding() -> None:
    admission = admit_policy_proposal(
        treatment="target_only", target_state="before",
        native_actions=["look", "open door"],
        policy_prompt_sha256=_h("policy-prompt"), policy_reply="ACTION: 2",
        policy_action="open door", advisory_prompt_sha256=_h("advisory-prompt"),
        advisory_reply=_reply(_proposal_id()), allowed_binding_ids=[],
    )
    assert admission.status == "REJECTED"
    assert "TARGET_ONLY_ADVISORY_CANNOT_CITE_BINDING" in admission.failure_code


def test_matched_call_ledger_has_fixed_slots_and_fails_closed() -> None:
    ledger = MatchedHarnessCallLedger(max_calls=3, audit_interval=5)
    assert ledger.scheduled_pre_action(0)
    assert not ledger.scheduled_pre_action(1)
    assert ledger.scheduled_pre_action(5)
    for phase in ("ADAPTATION", "PRE_ACTION", "PADDING"):
        ledger.record(
            phase=phase, step=None, effective=phase != "PADDING",
            prompt_sha256=_h(phase), generation_id=phase,
        )
    assert ledger.to_dict()["used_calls"] == 3
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        ledger.record(
            phase="PADDING", step=None, effective=False,
            prompt_sha256=_h("extra"), generation_id="extra",
        )


def test_adaptation_only_references_frozen_source_receipts() -> None:
    result = parse_adaptation_binding_set(json.dumps({
        "hypotheses": [{
            "binding_id": "b0", "source_ref": "source-0",
            "target_reasoning_claim": "try preserving ordered progress",
            "testable_target_prediction": "the next observation should change",
        }],
        "decision": "INITIALIZE",
    }), allowed_source_refs=["source-0"], target_only=False)
    assert result.hypotheses[0].source_ref == "source-0"


def test_target_only_adaptation_must_remain_empty() -> None:
    result = parse_adaptation_binding_set(json.dumps({
        "hypotheses": [], "decision": "NEED_MORE_EVIDENCE",
    }), allowed_source_refs=[], target_only=True)
    assert result.hypotheses == ()


def test_replanned_execution_still_has_to_be_exactly_the_policy_action() -> None:
    receipt = policy_execution_identity_receipt(
        policy_proposal_id_value="replan-p1", policy_action="look",
        executed_action="look", native_actions=["look", "open door"],
    )
    assert receipt.execution_matches_policy
    rejected = policy_execution_identity_receipt(
        policy_proposal_id_value="replan-p1", policy_action="look",
        executed_action="open door", native_actions=["look", "open door"],
    )
    assert not rejected.execution_matches_policy
