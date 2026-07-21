"""Tamper-evident event sourcing for source and target Agent rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence


class ReasoningEventKind(str, Enum):
    RESET = "RESET"
    OBSERVATION = "OBSERVATION"
    AGENT_PROPOSAL_SET = "AGENT_PROPOSAL_SET"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    PARSED_DECISION = "PARSED_DECISION"
    POLICY_TRANSFORM = "POLICY_TRANSFORM"
    NATIVE_ADMISSIBILITY = "NATIVE_ADMISSIBILITY"
    AGENT_DECISION = "AGENT_DECISION"
    ENVIRONMENT_STEP = "ENVIRONMENT_STEP"
    NATIVE_DELTA = "NATIVE_DELTA"
    ONLINE_TRANSFER_VERDICT = "ONLINE_TRANSFER_VERDICT"
    OFFICIAL_STOP = "OFFICIAL_STOP"


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReasoningEvent:
    episode_id: str
    sequence: int
    kind: ReasoningEventKind
    payload: Mapping[str, Any]
    previous_event_sha256: str | None
    event_sha256: str

    def unsigned_payload(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "payload": dict(self.payload),
            "previous_event_sha256": self.previous_event_sha256,
        }

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.event_sha256:
            raise ValueError(f"event hash mismatch at sequence {self.sequence}")


class ReasoningEventRecorder:
    """Append-only in-memory recorder; callers decide durable storage."""

    def __init__(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id is required")
        self.episode_id = episode_id
        self._events: list[ReasoningEvent] = []

    @property
    def events(self) -> Sequence[ReasoningEvent]:
        return tuple(self._events)

    def append(self, kind: ReasoningEventKind, payload: Mapping[str, Any]) -> ReasoningEvent:
        unsigned = {
            "episode_id": self.episode_id,
            "sequence": len(self._events),
            "kind": kind.value,
            "payload": dict(payload),
            "previous_event_sha256": (
                self._events[-1].event_sha256 if self._events else None
            ),
        }
        event = ReasoningEvent(
            episode_id=self.episode_id,
            sequence=unsigned["sequence"],
            kind=kind,
            payload=unsigned["payload"],
            previous_event_sha256=unsigned["previous_event_sha256"],
            event_sha256=_hash(unsigned),
        )
        self._events.append(event)
        return event

    def validate_chain(self) -> None:
        previous = None
        for index, event in enumerate(self._events):
            event.validate_hash()
            if event.episode_id != self.episode_id or event.sequence != index:
                raise ValueError("event identity/sequence mismatch")
            if event.previous_event_sha256 != previous:
                raise ValueError("broken event hash chain")
            previous = event.event_sha256

    def to_dict(self) -> Dict[str, Any]:
        self.validate_chain()
        events = []
        for item in self._events:
            row = asdict(item)
            row["kind"] = item.kind.value
            events.append(row)
        payload = {"schema_version": 1, "episode_id": self.episode_id, "events": events}
        payload["log_sha256"] = _hash(payload)
        return payload


def validate_reasoning_protocol(
    events: Sequence[ReasoningEvent], *, profile: str = "generic",
) -> Sequence[str]:
    """Check event completeness/order, never the quality of reasoning.

    ``source_agent`` additionally requires the three events that separate a
    raw model response from parsing and policy-side action transformation.
    Target runtimes do not have that policy stack and therefore use the core
    ``generic`` protocol.
    """
    failures: list[str] = []
    if not events or events[0].kind != ReasoningEventKind.RESET:
        failures.append("MISSING_INITIAL_RESET")
    if not events or events[-1].kind != ReasoningEventKind.OFFICIAL_STOP:
        failures.append("MISSING_FINAL_OFFICIAL_STOP")
    kinds = {item.kind for item in events}
    required_kinds = {
        ReasoningEventKind.RESET,
        ReasoningEventKind.OBSERVATION,
        ReasoningEventKind.AGENT_PROPOSAL_SET,
        ReasoningEventKind.NATIVE_ADMISSIBILITY,
        ReasoningEventKind.AGENT_DECISION,
        ReasoningEventKind.ENVIRONMENT_STEP,
        ReasoningEventKind.NATIVE_DELTA,
        ReasoningEventKind.OFFICIAL_STOP,
    }
    if profile == "source_agent":
        required_kinds.update({
            ReasoningEventKind.AGENT_RESPONSE,
            ReasoningEventKind.PARSED_DECISION,
            ReasoningEventKind.POLICY_TRANSFORM,
        })
    elif profile != "generic":
        failures.append(f"UNKNOWN_PROTOCOL_PROFILE:{profile}")
    for required in sorted(required_kinds, key=lambda item: item.value):
        if required not in kinds:
            failures.append(f"MISSING_EVENT_KIND:{required.value}")
    return tuple(failures)


def reasoning_event_log_from_dict(payload: Mapping[str, Any]) -> Sequence[ReasoningEvent]:
    """Load and verify a serialized event log without trusting its hashes."""
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported reasoning event log schema")
    unsigned_log = {
        "schema_version": 1,
        "episode_id": str(payload.get("episode_id") or ""),
        "events": list(payload.get("events") or ()),
    }
    if _hash(unsigned_log) != payload.get("log_sha256"):
        raise ValueError("reasoning event log hash mismatch")
    events = []
    for row in unsigned_log["events"]:
        event = ReasoningEvent(
            episode_id=str(row["episode_id"]),
            sequence=int(row["sequence"]),
            kind=ReasoningEventKind(str(row["kind"])),
            payload=dict(row["payload"]),
            previous_event_sha256=row.get("previous_event_sha256"),
            event_sha256=str(row["event_sha256"]),
        )
        events.append(event)
    previous = None
    for index, event in enumerate(events):
        event.validate_hash()
        if event.episode_id != unsigned_log["episode_id"] or event.sequence != index:
            raise ValueError("event identity/sequence mismatch")
        if event.previous_event_sha256 != previous:
            raise ValueError("broken event hash chain")
        previous = event.event_sha256
    return tuple(events)


__all__ = [
    "ReasoningEvent",
    "ReasoningEventKind",
    "ReasoningEventRecorder",
    "reasoning_event_log_from_dict",
    "validate_reasoning_protocol",
]
