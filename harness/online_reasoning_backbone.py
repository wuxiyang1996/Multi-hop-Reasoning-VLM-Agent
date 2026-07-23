"""Fail-closed online target execution of an Agent reasoning cycle.

The Harness never maps source actions or predicates into the target. Source
receipts may condition the untrusted Agent prompt, while admission is based
only on target-native actions and content-addressed target evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from harness.agent_reasoning_cycle import (
    parse_agent_action_proposal_set,
    parse_agent_post_transition_verdict,
)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class FrozenBackboneConditioning:
    treatment: str
    source_artifact_sha256: str | None
    adaptation_demo_sha256: str
    prompt_template_sha256: str

    def validate(self) -> None:
        if self.treatment not in {
            "correct", "renamed", "randomized", "receipt_null", "target_only",
        }:
            raise ValueError("unknown reasoning-backbone treatment")
        if self.treatment == "target_only" and self.source_artifact_sha256 is not None:
            raise ValueError("target-only conditioning cannot cite a source artifact")
        if self.treatment != "target_only" and not self.source_artifact_sha256:
            raise ValueError("source treatment requires a source artifact receipt")
        if not self.adaptation_demo_sha256 or not self.prompt_template_sha256:
            raise ValueError("conditioning requires frozen adaptation/prompt receipts")

    def content_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


@dataclass(frozen=True)
class OnlineBackbonePlanAdmission:
    conditioning_sha256: str
    target_state_sha256: str
    native_actions_sha256: str
    agent_prompt_sha256: str
    raw_response_sha256: str
    status: str
    proposal_set: Mapping[str, Any] | None
    proposal_set_sha256: str | None
    selected_proposal_id: str | None
    admitted_native_action: str | None
    failure_code: str | None
    admission_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("admission_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.admission_sha256:
            raise ValueError("online backbone plan admission hash mismatch")


@dataclass(frozen=True)
class OnlineBackboneCycleReceipt:
    admission_sha256: str
    before_state_sha256: str
    after_state_sha256: str
    executed_action: str
    reward: float
    done: bool
    verifier_prompt_sha256: str
    raw_verdict_sha256: str
    status: str
    verdict: Mapping[str, Any] | None
    verdict_sha256: str | None
    runtime_directive: str
    failure_code: str | None
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("online backbone cycle receipt hash mismatch")


def admit_online_backbone_plan(
    raw_response: str, *, conditioning: FrozenBackboneConditioning,
    target_state: Any, native_actions: Sequence[str], agent_prompt_sha256: str,
) -> OnlineBackbonePlanAdmission:
    conditioning_hash = conditioning.content_hash()
    actions = [str(item) for item in native_actions]
    proposal = None
    status = "REJECTED"
    selected_id = None
    admitted_action = None
    failure = None
    try:
        proposal = parse_agent_action_proposal_set(
            raw_response, n_native_actions=len(actions),
        )
        if proposal.decision == "ABSTAIN":
            status = "ABSTAINED"
        else:
            selected = proposal.selected()
            selected_id = selected.proposal_id
            admitted_action = actions[selected.action_number - 1]
            status = "ADMITTED"
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
    unsigned = {
        "conditioning_sha256": conditioning_hash,
        "target_state_sha256": _hash(target_state),
        "native_actions_sha256": _hash(actions),
        "agent_prompt_sha256": str(agent_prompt_sha256),
        "raw_response_sha256": _hash(raw_response),
        "status": status,
        "proposal_set": proposal.to_dict() if proposal is not None else None,
        "proposal_set_sha256": proposal.content_hash() if proposal is not None else None,
        "selected_proposal_id": selected_id,
        "admitted_native_action": admitted_action,
        "failure_code": failure,
    }
    receipt = OnlineBackbonePlanAdmission(
        **unsigned, admission_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return receipt


def close_online_backbone_cycle(
    admission: OnlineBackbonePlanAdmission, *, executed_action: str,
    before_state: Any, after_state: Any, reward: float, done: bool,
    raw_verdict: str, verifier_prompt_sha256: str,
) -> OnlineBackboneCycleReceipt:
    admission.validate_hash()
    verdict = None
    failure = None
    if admission.status != "ADMITTED":
        failure = "PLAN_NOT_ADMITTED"
    elif str(executed_action) != admission.admitted_native_action:
        failure = "EXECUTED_ACTION_DIFFERS_FROM_ADMISSION"
    elif _hash(before_state) != admission.target_state_sha256:
        failure = "BEFORE_STATE_DIFFERS_FROM_ADMISSION"
    else:
        try:
            verdict = parse_agent_post_transition_verdict(
                raw_verdict,
                expected_proposal_id=admission.selected_proposal_id,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}:{exc}"
    status = "VERIFIED_AGENT_CYCLE" if verdict is not None else "REJECTED"
    directive = {
        "CONTINUE": "AGENT_CYCLE_CONTINUE",
        "REPLAN": "AGENT_CYCLE_REPLAN",
        "ABSTAIN": "FALLBACK_TARGET_ONLY",
    }.get(verdict.decision if verdict is not None else "", "FALLBACK_TARGET_ONLY")
    unsigned = {
        "admission_sha256": admission.admission_sha256,
        "before_state_sha256": _hash(before_state),
        "after_state_sha256": _hash(after_state),
        "executed_action": str(executed_action),
        "reward": float(reward), "done": bool(done),
        "verifier_prompt_sha256": str(verifier_prompt_sha256),
        "raw_verdict_sha256": _hash(raw_verdict),
        "status": status,
        "verdict": verdict.to_dict() if verdict is not None else None,
        "verdict_sha256": verdict.content_hash() if verdict is not None else None,
        "runtime_directive": directive,
        "failure_code": failure,
    }
    receipt = OnlineBackboneCycleReceipt(**unsigned, receipt_sha256=_hash(unsigned))
    receipt.validate_hash()
    return receipt


__all__ = [
    "FrozenBackboneConditioning", "OnlineBackbonePlanAdmission",
    "OnlineBackboneCycleReceipt", "admit_online_backbone_plan",
    "close_online_backbone_cycle",
]
