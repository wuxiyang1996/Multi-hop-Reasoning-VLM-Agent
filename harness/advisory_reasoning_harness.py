"""Advisory-only reasoning Harness for a separate target policy.

The Harness Agent may review a policy proposal, predict an observable delta,
and request re-planning.  It can never name or select a replacement action.
Only the target policy's exact native action can be admitted and executed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class AdaptationHypothesis:
    binding_id: str
    source_ref: str
    target_reasoning_claim: str
    testable_target_prediction: str


@dataclass(frozen=True)
class AdaptationBindingSet:
    hypotheses: tuple[AdaptationHypothesis, ...]
    decision: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "hypotheses": [asdict(item) for item in self.hypotheses],
            "decision": self.decision,
        }

    def content_hash(self) -> str:
        return _hash(self.to_dict())


def parse_adaptation_binding_set(
    raw: str, *, allowed_source_refs: Sequence[str], target_only: bool,
) -> AdaptationBindingSet:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("ADAPTATION_NOT_JSON") from exc
    if not isinstance(payload, dict) or not (
        raw.lstrip().startswith("{") and raw.rstrip().endswith("}")
    ):
        raise ValueError("ADAPTATION_NOT_EXACT_JSON_OBJECT")
    if set(payload) != {"hypotheses", "decision"}:
        raise ValueError("WRONG_ADAPTATION_KEYS")
    if payload["decision"] not in {"INITIALIZE", "NEED_MORE_EVIDENCE", "ABSTAIN"}:
        raise ValueError("UNKNOWN_ADAPTATION_DECISION")
    if not isinstance(payload["hypotheses"], list):
        raise ValueError("ADAPTATION_HYPOTHESES_NOT_LIST")
    if len(payload["hypotheses"]) > 3:
        raise ValueError("TOO_MANY_ADAPTATION_HYPOTHESES")
    refs = set(allowed_source_refs)
    hypotheses = []
    for row in payload["hypotheses"]:
        if not isinstance(row, dict) or set(row) != {
            "binding_id", "source_ref", "target_reasoning_claim",
            "testable_target_prediction",
        }:
            raise ValueError("WRONG_ADAPTATION_HYPOTHESIS_KEYS")
        if not all(isinstance(row[key], str) for key in row):
            raise ValueError("WRONG_ADAPTATION_HYPOTHESIS_TYPES")
        if row["source_ref"] not in refs:
            raise ValueError("UNKNOWN_ADAPTATION_SOURCE_REF")
        if not row["binding_id"] or not row["target_reasoning_claim"] or not row[
            "testable_target_prediction"
        ]:
            raise ValueError("EMPTY_ADAPTATION_HYPOTHESIS_FIELD")
        hypotheses.append(AdaptationHypothesis(**row))
    if len({item.binding_id for item in hypotheses}) != len(hypotheses):
        raise ValueError("DUPLICATE_ADAPTATION_BINDING_ID")
    if target_only and hypotheses:
        raise ValueError("TARGET_ONLY_CANNOT_INITIALIZE_SOURCE_BINDING")
    if payload["decision"] == "INITIALIZE" and not hypotheses:
        raise ValueError("INITIALIZE_REQUIRES_HYPOTHESIS")
    if payload["decision"] != "INITIALIZE" and hypotheses:
        raise ValueError("NON_INITIALIZE_REQUIRES_EMPTY_HYPOTHESES")
    result = AdaptationBindingSet(tuple(hypotheses), payload["decision"])
    return result


def policy_proposal_id(
    *, target_state: Any, native_actions: Sequence[str],
    policy_prompt_sha256: str, policy_reply: str, policy_action: str,
) -> str:
    return _hash({
        "state": target_state,
        "native_actions": [str(item) for item in native_actions],
        "policy_prompt_sha256": policy_prompt_sha256,
        "policy_reply_sha256": _hash(policy_reply),
        "policy_action": str(policy_action),
    })


@dataclass(frozen=True)
class AdvisoryReview:
    policy_proposal_id: str
    binding_id: str | None
    verdict: str
    predicted_observable_delta: str
    reason: str

    def validate(self, *, expected_policy_proposal_id: str,
                 allowed_binding_ids: Sequence[str]) -> None:
        if self.policy_proposal_id != expected_policy_proposal_id:
            raise ValueError("ADVISORY_POLICY_PROPOSAL_ID_MISMATCH")
        if self.verdict not in {"ADMIT", "REPLAN", "ABSTAIN"}:
            raise ValueError("UNKNOWN_ADVISORY_VERDICT")
        allowed = set(allowed_binding_ids)
        if not allowed and self.binding_id is not None:
            raise ValueError("TARGET_ONLY_ADVISORY_CANNOT_CITE_BINDING")
        if self.binding_id is not None and self.binding_id not in allowed:
            raise ValueError("UNKNOWN_ADVISORY_BINDING_ID")
        if not self.predicted_observable_delta or not self.reason:
            raise ValueError("EMPTY_ADVISORY_TEXT_FIELD")

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return _hash(self.to_dict())


def parse_advisory_review(
    raw: str, *, expected_policy_proposal_id: str,
    allowed_binding_ids: Sequence[str],
) -> AdvisoryReview:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("ADVISORY_NOT_JSON") from exc
    if not isinstance(payload, dict) or raw.strip() != json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) and not (raw.lstrip().startswith("{") and raw.rstrip().endswith("}")):
        raise ValueError("ADVISORY_NOT_EXACT_JSON_OBJECT")
    expected = {
        "policy_proposal_id", "binding_id", "verdict",
        "predicted_observable_delta", "reason",
    }
    if set(payload) != expected:
        raise ValueError("WRONG_ADVISORY_KEYS")
    if (
        not isinstance(payload["policy_proposal_id"], str)
        or payload["binding_id"] is not None
        and not isinstance(payload["binding_id"], str)
        or not isinstance(payload["verdict"], str)
        or not isinstance(payload["predicted_observable_delta"], str)
        or not isinstance(payload["reason"], str)
    ):
        raise ValueError("WRONG_ADVISORY_FIELD_TYPES")
    review = AdvisoryReview(**payload)
    review.validate(
        expected_policy_proposal_id=expected_policy_proposal_id,
        allowed_binding_ids=allowed_binding_ids,
    )
    return review


@dataclass(frozen=True)
class PolicyProposalAdmission:
    treatment: str
    target_state_sha256: str
    native_actions_sha256: str
    policy_prompt_sha256: str
    policy_reply_sha256: str
    policy_proposal_id: str
    policy_action: str
    policy_action_is_native: bool
    advisory_prompt_sha256: str
    advisory_reply_sha256: str
    advisory_review: Mapping[str, Any] | None
    advisory_review_sha256: str | None
    status: str
    approved_policy_action: str | None
    failure_code: str | None
    admission_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("admission_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.admission_sha256:
            raise ValueError("ADVISORY_ADMISSION_HASH_MISMATCH")


def admit_policy_proposal(
    *, treatment: str, target_state: Any, native_actions: Sequence[str],
    policy_prompt_sha256: str, policy_reply: str, policy_action: str,
    advisory_prompt_sha256: str, advisory_reply: str,
    allowed_binding_ids: Sequence[str],
) -> PolicyProposalAdmission:
    actions = [str(item) for item in native_actions]
    proposal_id = policy_proposal_id(
        target_state=target_state, native_actions=actions,
        policy_prompt_sha256=policy_prompt_sha256, policy_reply=policy_reply,
        policy_action=policy_action,
    )
    native = str(policy_action) in actions
    review = None
    failure = None
    status = "REJECTED"
    approved = None
    if not native:
        failure = "POLICY_ACTION_NOT_NATIVE"
    else:
        try:
            review = parse_advisory_review(
                advisory_reply,
                expected_policy_proposal_id=proposal_id,
                allowed_binding_ids=allowed_binding_ids,
            )
            if review.verdict == "ADMIT":
                status = "POLICY_ACTION_ADMITTED"
                approved = str(policy_action)
            elif review.verdict == "REPLAN":
                status = "POLICY_REPLAN_REQUESTED"
            else:
                status = "ADVISORY_ABSTAINED"
        except Exception as exc:
            failure = f"{type(exc).__name__}:{exc}"
    unsigned = {
        "treatment": treatment,
        "target_state_sha256": _hash(target_state),
        "native_actions_sha256": _hash(actions),
        "policy_prompt_sha256": str(policy_prompt_sha256),
        "policy_reply_sha256": _hash(policy_reply),
        "policy_proposal_id": proposal_id,
        "policy_action": str(policy_action),
        "policy_action_is_native": native,
        "advisory_prompt_sha256": str(advisory_prompt_sha256),
        "advisory_reply_sha256": _hash(advisory_reply),
        "advisory_review": review.to_dict() if review is not None else None,
        "advisory_review_sha256": review.content_hash() if review is not None else None,
        "status": status,
        "approved_policy_action": approved,
        "failure_code": failure,
    }
    receipt = PolicyProposalAdmission(**unsigned, admission_sha256=_hash(unsigned))
    receipt.validate_hash()
    return receipt


@dataclass(frozen=True)
class AdvisoryExecutionReceipt:
    admission_sha256: str
    executed_action: str
    executed_action_sha256: str
    after_state_sha256: str
    reward: float
    done: bool
    execution_matches_policy: bool
    status: str
    failure_code: str | None
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("ADVISORY_EXECUTION_HASH_MISMATCH")


def close_policy_execution(
    admission: PolicyProposalAdmission, *, executed_action: str,
    after_state: Any, reward: float, done: bool,
) -> AdvisoryExecutionReceipt:
    admission.validate_hash()
    matches = (
        admission.status == "POLICY_ACTION_ADMITTED"
        and admission.approved_policy_action == str(executed_action)
        and admission.policy_action == str(executed_action)
    )
    failure = None if matches else "EXECUTED_ACTION_NOT_EXACT_POLICY_PROPOSAL"
    unsigned = {
        "admission_sha256": admission.admission_sha256,
        "executed_action": str(executed_action),
        "executed_action_sha256": _hash(str(executed_action)),
        "after_state_sha256": _hash(after_state),
        "reward": float(reward),
        "done": bool(done),
        "execution_matches_policy": matches,
        "status": "POLICY_EXECUTION_VERIFIED" if matches else "REJECTED",
        "failure_code": failure,
    }
    receipt = AdvisoryExecutionReceipt(**unsigned, receipt_sha256=_hash(unsigned))
    receipt.validate_hash()
    return receipt


@dataclass(frozen=True)
class PolicyExecutionIdentityReceipt:
    policy_proposal_id: str
    policy_action: str
    executed_action: str
    policy_action_is_native: bool
    execution_matches_policy: bool
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("POLICY_EXECUTION_IDENTITY_HASH_MISMATCH")


def policy_execution_identity_receipt(
    *, policy_proposal_id_value: str, policy_action: str,
    executed_action: str, native_actions: Sequence[str],
) -> PolicyExecutionIdentityReceipt:
    native = str(policy_action) in [str(item) for item in native_actions]
    matches = native and str(policy_action) == str(executed_action)
    unsigned = {
        "policy_proposal_id": str(policy_proposal_id_value),
        "policy_action": str(policy_action),
        "executed_action": str(executed_action),
        "policy_action_is_native": native,
        "execution_matches_policy": matches,
    }
    receipt = PolicyExecutionIdentityReceipt(
        **unsigned, receipt_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return receipt


class MatchedHarnessCallLedger:
    """Content-independent call slots shared by every experimental treatment."""

    def __init__(self, *, max_calls: int, audit_interval: int) -> None:
        if max_calls < 1 or audit_interval < 1:
            raise ValueError("call budget and audit interval must be positive")
        self.max_calls = max_calls
        self.audit_interval = audit_interval
        self._rows: list[dict[str, Any]] = []

    def scheduled_pre_action(self, step: int) -> bool:
        return step % self.audit_interval == 0 and len(self._rows) < self.max_calls

    def record(self, *, phase: str, step: int | None, effective: bool,
               prompt_sha256: str, generation_id: str) -> None:
        if len(self._rows) >= self.max_calls:
            raise ValueError("HARNESS_AGENT_CALL_BUDGET_EXHAUSTED")
        if phase not in {"ADAPTATION", "PRE_ACTION", "POST_TRANSITION", "PADDING"}:
            raise ValueError("UNKNOWN_HARNESS_CALL_PHASE")
        self._rows.append({
            "slot": len(self._rows), "phase": phase, "step": step,
            "effective": bool(effective), "prompt_sha256": prompt_sha256,
            "generation_id_sha256": _hash(generation_id),
        })

    @property
    def remaining(self) -> int:
        return self.max_calls - len(self._rows)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "max_calls": self.max_calls,
            "audit_interval": self.audit_interval,
            "used_calls": len(self._rows),
            "effective_calls": sum(row["effective"] for row in self._rows),
            "padding_calls": sum(not row["effective"] for row in self._rows),
            "remaining_calls": self.remaining,
            "rows": list(self._rows),
            "ledger_sha256": _hash(self._rows),
        }


__all__ = [
    "AdaptationBindingSet", "AdaptationHypothesis", "AdvisoryExecutionReceipt",
    "AdvisoryReview", "MatchedHarnessCallLedger", "PolicyExecutionIdentityReceipt",
    "PolicyProposalAdmission",
    "admit_policy_proposal", "close_policy_execution",
    "parse_adaptation_binding_set", "parse_advisory_review",
    "policy_execution_identity_receipt", "policy_proposal_id",
]
