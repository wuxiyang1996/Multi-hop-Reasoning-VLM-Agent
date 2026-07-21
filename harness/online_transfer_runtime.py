"""Episode-local online source-use control with tamper-evident receipts.

This module records operational evidence only.  It deliberately does not label
an episode or transition as negative transfer: that causal claim requires a
matched target-only experiment that is unavailable to a live runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence


def _hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class OnlineTransferState(str, Enum):
    SOURCE_ACTIVE = "SOURCE_ACTIVE"
    REBIND_REQUIRED = "REBIND_REQUIRED"
    TARGET_ONLY = "TARGET_ONLY"


class OnlineTransferVerdict(str, Enum):
    CONTINUE = "CONTINUE"
    NOT_REFUTED_LOCALLY = "NOT_REFUTED_LOCALLY"
    NO_OBSERVABLE_DELTA = "NO_OBSERVABLE_DELTA"
    REBIND = "REBIND"
    REBIND_ACCEPTED = "REBIND_ACCEPTED"
    SOURCE_DISABLED = "SOURCE_DISABLED"
    TERMINAL_WITHOUT_SUCCESS = "TERMINAL_WITHOUT_SUCCESS"


@dataclass(frozen=True)
class NativeTransitionEvidence:
    """Hard target-native evidence produced after exactly one action."""

    step: int
    command: str
    before_observation_sha256: str
    after_observation_sha256: str
    before_actions_sha256: str
    after_actions_sha256: str
    reward: float
    official_success: bool
    command_was_admissible: bool
    executed_action_admissible_after: bool
    terminated: bool
    truncated: bool
    receipt_sha256: str

    @classmethod
    def build(
        cls,
        *,
        step: int,
        command: str,
        before_observation_sha256: str,
        after_observation_sha256: str,
        before_actions_sha256: str,
        after_actions_sha256: str,
        reward: float,
        official_success: bool,
        command_was_admissible: bool,
        executed_action_admissible_after: bool,
        terminated: bool,
        truncated: bool,
    ) -> "NativeTransitionEvidence":
        payload = {
            "step": int(step),
            "command": str(command),
            "before_observation_sha256": str(before_observation_sha256),
            "after_observation_sha256": str(after_observation_sha256),
            "before_actions_sha256": str(before_actions_sha256),
            "after_actions_sha256": str(after_actions_sha256),
            "reward": float(reward),
            "official_success": bool(official_success),
            "command_was_admissible": bool(command_was_admissible),
            "executed_action_admissible_after": bool(executed_action_admissible_after),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        return cls(**payload, receipt_sha256=_hash(payload))

    def validate_hash(self) -> None:
        payload = asdict(self)
        receipt_sha256 = payload.pop("receipt_sha256")
        if _hash(payload) != receipt_sha256:
            raise ValueError("native transition receipt hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_hash()
        return asdict(self)

    @property
    def has_observable_delta(self) -> bool:
        return (
            self.before_observation_sha256 != self.after_observation_sha256
            or self.before_actions_sha256 != self.after_actions_sha256
            or self.reward != 0.0
            or self.official_success
            or self.terminated
            or self.truncated
        )


@dataclass(frozen=True)
class OnlineTransferEvent:
    sequence: int
    step: int
    state_before: OnlineTransferState
    state_after: OnlineTransferState
    verdict: OnlineTransferVerdict
    reason: str
    transition_receipt_sha256: str | None
    binding_receipt_sha256: str | None
    rebind_requests: int
    consecutive_no_delta: int
    previous_event_sha256: str | None
    event_sha256: str

    def unsigned_payload(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "step": self.step,
            "state_before": self.state_before.value,
            "state_after": self.state_after.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "transition_receipt_sha256": self.transition_receipt_sha256,
            "binding_receipt_sha256": self.binding_receipt_sha256,
            "rebind_requests": self.rebind_requests,
            "consecutive_no_delta": self.consecutive_no_delta,
            "previous_event_sha256": self.previous_event_sha256,
        }

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.event_sha256:
            raise ValueError("online transfer event hash mismatch")


class OnlineTransferController:
    """Bounded online source use for one target episode.

    A rebind is never fabricated internally.  The controller can request one,
    but only an externally supplied content-addressed binding receipt can move
    it back to ``SOURCE_ACTIVE``.  If no online rebinder is available, callers
    must explicitly invoke :meth:`fallback_to_target_only`.
    """

    def __init__(
        self,
        *,
        max_rebind_requests: int = 1,
        max_consecutive_no_delta: int = 2,
    ) -> None:
        if max_rebind_requests < 0:
            raise ValueError("max_rebind_requests must be non-negative")
        if max_consecutive_no_delta < 1:
            raise ValueError("max_consecutive_no_delta must be positive")
        self.max_rebind_requests = int(max_rebind_requests)
        self.max_consecutive_no_delta = int(max_consecutive_no_delta)
        self.state = OnlineTransferState.SOURCE_ACTIVE
        self.rebind_requests = 0
        self.consecutive_no_delta = 0
        self._events: list[OnlineTransferEvent] = []

    @property
    def source_enabled(self) -> bool:
        return self.state == OnlineTransferState.SOURCE_ACTIVE

    @property
    def events(self) -> Sequence[OnlineTransferEvent]:
        return tuple(self._events)

    def _append(
        self,
        *,
        step: int,
        state_before: OnlineTransferState,
        verdict: OnlineTransferVerdict,
        reason: str,
        transition_receipt_sha256: str | None = None,
        binding_receipt_sha256: str | None = None,
    ) -> OnlineTransferEvent:
        unsigned = {
            "sequence": len(self._events),
            "step": int(step),
            "state_before": state_before.value,
            "state_after": self.state.value,
            "verdict": verdict.value,
            "reason": str(reason),
            "transition_receipt_sha256": transition_receipt_sha256,
            "binding_receipt_sha256": binding_receipt_sha256,
            "rebind_requests": self.rebind_requests,
            "consecutive_no_delta": self.consecutive_no_delta,
            "previous_event_sha256": (
                self._events[-1].event_sha256 if self._events else None
            ),
        }
        event = OnlineTransferEvent(
            sequence=unsigned["sequence"],
            step=unsigned["step"],
            state_before=state_before,
            state_after=self.state,
            verdict=verdict,
            reason=unsigned["reason"],
            transition_receipt_sha256=transition_receipt_sha256,
            binding_receipt_sha256=binding_receipt_sha256,
            rebind_requests=self.rebind_requests,
            consecutive_no_delta=self.consecutive_no_delta,
            previous_event_sha256=unsigned["previous_event_sha256"],
            event_sha256=_hash(unsigned),
        )
        self._events.append(event)
        return event

    def _request_rebind(
        self,
        *,
        step: int,
        reason: str,
        transition_receipt_sha256: str | None = None,
    ) -> OnlineTransferEvent:
        before = self.state
        if self.rebind_requests >= self.max_rebind_requests:
            self.state = OnlineTransferState.TARGET_ONLY
            return self._append(
                step=step,
                state_before=before,
                verdict=OnlineTransferVerdict.SOURCE_DISABLED,
                reason=f"REBIND_BUDGET_EXHAUSTED:{reason}",
                transition_receipt_sha256=transition_receipt_sha256,
            )
        self.rebind_requests += 1
        self.state = OnlineTransferState.REBIND_REQUIRED
        return self._append(
            step=step,
            state_before=before,
            verdict=OnlineTransferVerdict.REBIND,
            reason=reason,
            transition_receipt_sha256=transition_receipt_sha256,
        )

    def observe_source_abstention(self, *, step: int, reason: str) -> OnlineTransferEvent:
        if self.state != OnlineTransferState.SOURCE_ACTIVE:
            raise ValueError("source abstention requires SOURCE_ACTIVE state")
        return self._request_rebind(step=step, reason=f"SOURCE_ABSTAINED:{reason}")

    def observe_source_transition(
        self, evidence: NativeTransitionEvidence,
    ) -> OnlineTransferEvent:
        """Fail closed when a source action had no predeclared contract."""
        if self.state != OnlineTransferState.SOURCE_ACTIVE:
            raise ValueError("source transition requires SOURCE_ACTIVE state")
        evidence.validate_hash()
        if not evidence.command_was_admissible:
            return self._request_rebind(
                step=evidence.step,
                reason="COMMAND_WAS_NOT_ADMISSIBLE",
                transition_receipt_sha256=evidence.receipt_sha256,
            )
        return self._request_rebind(
            step=evidence.step,
            reason="MISSING_PREDECLARED_EVIDENCE_CONTRACT",
            transition_receipt_sha256=evidence.receipt_sha256,
        )

    def observe_contract_transition(
        self,
        evidence: NativeTransitionEvidence,
        *,
        evidence_contract_satisfied: bool,
        contract_kind: str,
    ) -> OnlineTransferEvent:
        """Evaluate one contract frozen before the target-native action."""
        if self.state != OnlineTransferState.SOURCE_ACTIVE:
            raise ValueError("contract transition requires SOURCE_ACTIVE state")
        evidence.validate_hash()
        before = self.state
        if not evidence.command_was_admissible:
            return self._request_rebind(
                step=evidence.step,
                reason="COMMAND_WAS_NOT_ADMISSIBLE",
                transition_receipt_sha256=evidence.receipt_sha256,
            )
        if (evidence.terminated or evidence.truncated) and not evidence.official_success:
            return self._append(
                step=evidence.step,
                state_before=before,
                verdict=OnlineTransferVerdict.TERMINAL_WITHOUT_SUCCESS,
                reason="EPISODE_ENDED_WITHOUT_OFFICIAL_SUCCESS",
                transition_receipt_sha256=evidence.receipt_sha256,
            )
        if evidence_contract_satisfied:
            self.consecutive_no_delta = 0
            return self._append(
                step=evidence.step,
                state_before=before,
                verdict=OnlineTransferVerdict.NOT_REFUTED_LOCALLY,
                reason=(
                    "PREDECLARED_CONTRACT_SATISFIED_NOT_TRANSFER_PROOF:"
                    + str(contract_kind)
                ),
                transition_receipt_sha256=evidence.receipt_sha256,
            )
        self.consecutive_no_delta += 1
        return self._request_rebind(
            step=evidence.step,
            reason="PREDECLARED_CONTRACT_REFUTED",
            transition_receipt_sha256=evidence.receipt_sha256,
        )

    def accept_rebind(
        self,
        *,
        step: int,
        binding_receipt_sha256: str,
        known_binding_receipt_sha256s: Sequence[str],
    ) -> OnlineTransferEvent:
        if self.state != OnlineTransferState.REBIND_REQUIRED:
            raise ValueError("rebind receipt requires REBIND_REQUIRED state")
        if not _valid_sha256(binding_receipt_sha256):
            raise ValueError("binding receipt must be a sha256 identity")
        if binding_receipt_sha256 not in set(known_binding_receipt_sha256s):
            raise ValueError("binding receipt is not in the admission registry")
        before = self.state
        self.state = OnlineTransferState.SOURCE_ACTIVE
        self.consecutive_no_delta = 0
        return self._append(
            step=step,
            state_before=before,
            verdict=OnlineTransferVerdict.REBIND_ACCEPTED,
            reason="EXTERNAL_BINDING_RECEIPT_ACCEPTED",
            binding_receipt_sha256=binding_receipt_sha256,
        )

    def observe_rebind_transition(
        self,
        evidence: NativeTransitionEvidence,
        *,
        evidence_contract_satisfied: bool,
    ) -> OnlineTransferEvent:
        """Apply a mechanically evaluated, predeclared evidence contract."""
        if self.state != OnlineTransferState.SOURCE_ACTIVE:
            raise ValueError("rebind transition requires SOURCE_ACTIVE state")
        evidence.validate_hash()
        return self.observe_contract_transition(
            evidence,
            evidence_contract_satisfied=evidence_contract_satisfied,
            contract_kind="ONLINE_REBIND",
        )

    def fallback_to_target_only(self, *, step: int, reason: str) -> OnlineTransferEvent:
        if self.state == OnlineTransferState.TARGET_ONLY:
            raise ValueError("target-only fallback already active")
        before = self.state
        self.state = OnlineTransferState.TARGET_ONLY
        return self._append(
            step=step,
            state_before=before,
            verdict=OnlineTransferVerdict.SOURCE_DISABLED,
            reason=reason,
        )

    def validate_chain(self) -> None:
        previous = None
        for index, event in enumerate(self._events):
            event.validate_hash()
            if event.sequence != index or event.previous_event_sha256 != previous:
                raise ValueError("broken online transfer event chain")
            previous = event.event_sha256

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_chain()
        events = []
        for item in self._events:
            row = asdict(item)
            row["state_before"] = item.state_before.value
            row["state_after"] = item.state_after.value
            row["verdict"] = item.verdict.value
            events.append(row)
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "claim_scope": "operational_online_evidence_not_negative_transfer",
            "state": self.state.value,
            "max_rebind_requests": self.max_rebind_requests,
            "max_consecutive_no_delta": self.max_consecutive_no_delta,
            "events": events,
        }
        payload["log_sha256"] = _hash(payload)
        return payload


def online_transfer_log_from_dict(
    payload: Mapping[str, Any],
) -> Sequence[OnlineTransferEvent]:
    """Load a serialized controller log and verify its outer and event hashes."""
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported online transfer log schema")
    unsigned_log = {
        "schema_version": 1,
        "claim_scope": str(payload.get("claim_scope") or ""),
        "state": str(payload.get("state") or ""),
        "max_rebind_requests": int(payload.get("max_rebind_requests", -1)),
        "max_consecutive_no_delta": int(payload.get("max_consecutive_no_delta", -1)),
        "events": list(payload.get("events") or ()),
    }
    if _hash(unsigned_log) != payload.get("log_sha256"):
        raise ValueError("online transfer log hash mismatch")
    if unsigned_log["claim_scope"] != "operational_online_evidence_not_negative_transfer":
        raise ValueError("unsupported online transfer claim scope")
    events = []
    previous = None
    for index, row in enumerate(unsigned_log["events"]):
        event = OnlineTransferEvent(
            sequence=int(row["sequence"]),
            step=int(row["step"]),
            state_before=OnlineTransferState(str(row["state_before"])),
            state_after=OnlineTransferState(str(row["state_after"])),
            verdict=OnlineTransferVerdict(str(row["verdict"])),
            reason=str(row["reason"]),
            transition_receipt_sha256=row.get("transition_receipt_sha256"),
            binding_receipt_sha256=row.get("binding_receipt_sha256"),
            rebind_requests=int(row["rebind_requests"]),
            consecutive_no_delta=int(row["consecutive_no_delta"]),
            previous_event_sha256=row.get("previous_event_sha256"),
            event_sha256=str(row["event_sha256"]),
        )
        event.validate_hash()
        if event.sequence != index or event.previous_event_sha256 != previous:
            raise ValueError("broken online transfer event chain")
        previous = event.event_sha256
        events.append(event)
    final_state = events[-1].state_after.value if events else OnlineTransferState.SOURCE_ACTIVE.value
    if unsigned_log["state"] != final_state:
        raise ValueError("online transfer final state mismatch")
    return tuple(events)


__all__ = [
    "NativeTransitionEvidence",
    "OnlineTransferController",
    "OnlineTransferEvent",
    "OnlineTransferState",
    "OnlineTransferVerdict",
    "online_transfer_log_from_dict",
]
