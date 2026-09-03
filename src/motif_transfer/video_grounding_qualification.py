"""Fail-closed contracts for source-free video-grounding qualification.

The neural grounder may inspect a public question and chronological proxy
frames.  It may *not* inspect answer candidates or emit an answer or
option preference.  Its only product is a target-native event receipt that a
separate answer model can consume.  Keeping this boundary explicit lets the
qualification harness test whether localization and semantic receipts help,
rather than silently treating another full QA call as a grounding tool.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


OBSERVABILITY = ("OBSERVED", "PARTIAL", "UNOBSERVED")
FORBIDDEN_ANSWER_KEYS = {
    "answer",
    "answer_slot",
    "best_answer",
    "choice",
    "choice_id",
    "correct_option",
    "prediction",
    "selected_option",
}


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_ANSWER_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def _text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    output = value.strip()
    if not output and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    return output


def _optional_index(value: Any, *, field: str, frame_count: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    index = int(value)
    if index != value or not 0 <= index < frame_count:
        raise ValueError(f"{field} is outside the proxy frame range")
    return index


@dataclass(frozen=True)
class EventGroundingReceipt:
    subject: str
    predicate: str
    object: str
    observability: str
    start_frame: int | None
    end_frame: int | None
    evidence_frames: tuple[int, ...]
    before_state: str
    after_state: str
    confidence: float
    uncertainties: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "observability": self.observability,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "evidence_frames": list(self.evidence_frames),
            "before_state": self.before_state,
            "after_state": self.after_state,
            "confidence": self.confidence,
            "uncertainties": list(self.uncertainties),
            "reason": self.reason,
        }

    def evidence_only_dict(self) -> dict[str, Any]:
        """Return the receipt fields safe for a downstream answer model."""

        return self.as_dict()


@dataclass(frozen=True)
class EventLedgerReceipt:
    events: tuple[EventGroundingReceipt, ...]
    coverage: str
    uncertainties: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {"event_id": f"E{index}", **event.as_dict()}
                for index, event in enumerate(self.events)
            ],
            "coverage": self.coverage,
            "uncertainties": list(self.uncertainties),
        }


def parse_event_grounding_receipt(
    payload: Mapping[str, Any], *, frame_count: int,
) -> EventGroundingReceipt:
    """Validate a neural event receipt and reject answer-bearing payloads."""

    if frame_count < 2:
        raise ValueError("event grounding needs at least two proxy frames")
    forbidden = _forbidden_paths(payload)
    if forbidden:
        raise ValueError(
            "event receipt contains answer-bearing keys: " + ", ".join(forbidden)
        )
    observability = _text(payload.get("observability"), field="observability")
    if observability not in OBSERVABILITY:
        raise ValueError(f"observability must be one of {OBSERVABILITY}")
    start = _optional_index(
        payload.get("start_frame"), field="start_frame", frame_count=frame_count,
    )
    end = _optional_index(
        payload.get("end_frame"), field="end_frame", frame_count=frame_count,
    )
    if (start is None) != (end is None):
        raise ValueError("start_frame and end_frame must both be null or integers")
    if start is not None and end is not None and start > end:
        raise ValueError("start_frame must not exceed end_frame")
    raw_evidence = payload.get("evidence_frames")
    if not isinstance(raw_evidence, list):
        raise ValueError("evidence_frames must be a list")
    evidence: list[int] = []
    for raw in raw_evidence:
        index = _optional_index(raw, field="evidence_frames", frame_count=frame_count)
        assert index is not None
        if index not in evidence:
            evidence.append(index)
    if evidence != sorted(evidence):
        raise ValueError("evidence_frames must be chronological")
    if observability == "OBSERVED" and not evidence:
        raise ValueError("an OBSERVED receipt needs at least one evidence frame")
    if start is not None and any(index < start or index > end for index in evidence):
        raise ValueError("evidence frame falls outside the grounded interval")
    confidence = float(payload.get("confidence", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    raw_uncertainties = payload.get("uncertainties")
    if not isinstance(raw_uncertainties, list) or not all(
        isinstance(value, str) for value in raw_uncertainties
    ):
        raise ValueError("uncertainties must be a string list")
    return EventGroundingReceipt(
        subject=_text(payload.get("subject"), field="subject", allow_empty=True),
        predicate=_text(payload.get("predicate"), field="predicate"),
        object=_text(payload.get("object"), field="object", allow_empty=True),
        observability=observability,
        start_frame=start,
        end_frame=end,
        evidence_frames=tuple(evidence),
        before_state=_text(
            payload.get("before_state"), field="before_state", allow_empty=True,
        ),
        after_state=_text(
            payload.get("after_state"), field="after_state", allow_empty=True,
        ),
        confidence=confidence,
        uncertainties=tuple(value.strip() for value in raw_uncertainties),
        reason=_text(payload.get("reason"), field="reason", allow_empty=True),
    )


def parse_event_ledger_receipt(
    payload: Mapping[str, Any], *, frame_count: int,
) -> EventLedgerReceipt:
    """Validate a candidate-blind list of observed target-native events."""

    forbidden = _forbidden_paths(payload)
    if forbidden:
        raise ValueError(
            "event ledger contains answer-bearing keys: " + ", ".join(forbidden)
        )
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not 1 <= len(raw_events) <= 6:
        raise ValueError("event ledger needs between one and six events")
    events: list[EventGroundingReceipt] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError("event ledger rows must be objects")
        expected_id = f"E{index}"
        if str(raw.get("event_id")) != expected_id:
            raise ValueError("event ledger IDs must be consecutive E0,E1,...")
        event_payload = {key: value for key, value in raw.items() if key != "event_id"}
        event = parse_event_grounding_receipt(event_payload, frame_count=frame_count)
        if event.observability == "UNOBSERVED":
            raise ValueError("an event ledger may contain only OBSERVED/PARTIAL events")
        if not 1 <= len(event.evidence_frames) <= 3:
            raise ValueError("each ledger event needs one to three sparse evidence frames")
        events.append(event)
    coverage = _text(payload.get("coverage"), field="coverage")
    if coverage not in {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}:
        raise ValueError("invalid event-ledger coverage")
    raw_uncertainties = payload.get("uncertainties")
    if not isinstance(raw_uncertainties, list) or not all(
        isinstance(value, str) for value in raw_uncertainties
    ):
        raise ValueError("event-ledger uncertainties must be a string list")
    return EventLedgerReceipt(
        events=tuple(events),
        coverage=coverage,
        uncertainties=tuple(value.strip() for value in raw_uncertainties),
    )


def uniform_indices(frame_count: int, budget: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 1 <= budget <= frame_count:
        raise ValueError("budget must be in [1, frame_count]")
    if budget == 1:
        return [frame_count // 2]
    return [
        round(slot * (frame_count - 1) / (budget - 1))
        for slot in range(budget)
    ]


def localized_indices(
    receipt: EventGroundingReceipt, *, frame_count: int, budget: int,
) -> list[int]:
    """Select an exact-budget chronological view around grounded evidence."""

    fallback = uniform_indices(frame_count, budget)
    seeds = list(receipt.evidence_frames)
    if receipt.start_frame is not None:
        seeds.extend([receipt.start_frame, receipt.end_frame])
    seeds = sorted(set(int(value) for value in seeds))
    if not seeds:
        return fallback

    selected = set(seeds[:budget])
    radius = 1
    while len(selected) < budget and radius < frame_count:
        for seed in seeds:
            for candidate in (seed - radius, seed + radius):
                if 0 <= candidate < frame_count:
                    selected.add(candidate)
                    if len(selected) == budget:
                        break
            if len(selected) == budget:
                break
        radius += 1
    if len(selected) < budget:
        for candidate in fallback:
            selected.add(candidate)
            if len(selected) == budget:
                break
    return sorted(selected)[:budget]


def ledger_localized_indices(
    ledger: EventLedgerReceipt, *, frame_count: int, budget: int,
) -> list[int]:
    """Allocate an exact frame budget across every ledger event."""

    fallback = uniform_indices(frame_count, budget)
    seeds: list[int] = []
    for event in ledger.events:
        if event.start_frame is not None:
            seeds.extend([event.start_frame, event.end_frame])
        seeds.extend(event.evidence_frames)
    seeds = sorted(set(seeds))
    if not seeds:
        return fallback
    if len(seeds) > budget:
        positions = uniform_indices(len(seeds), budget)
        return [seeds[index] for index in positions]
    selected = set(seeds)
    radius = 1
    while len(selected) < budget and radius < frame_count:
        for seed in seeds:
            for candidate in (seed - radius, seed + radius):
                if 0 <= candidate < frame_count:
                    selected.add(candidate)
                    if len(selected) == budget:
                        break
            if len(selected) == budget:
                break
        radius += 1
    for candidate in fallback:
        if len(selected) == budget:
            break
        selected.add(candidate)
    return sorted(selected)[:budget]


def shifted_indices(
    indices: Sequence[int], *, frame_count: int,
) -> list[int]:
    """Create a deterministic half-cycle temporal destructive control."""

    if not indices:
        raise ValueError("cannot shift an empty evidence view")
    shift = max(1, frame_count // 2)
    shifted = sorted({(int(index) + shift) % frame_count for index in indices})
    if len(shifted) != len(indices):
        raise AssertionError("temporal shift changed the evidence budget")
    return shifted


def receipt_prompt_text(receipt: EventGroundingReceipt) -> str:
    return json.dumps(
        receipt.evidence_only_dict(), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


def ledger_prompt_text(receipt: EventLedgerReceipt) -> str:
    return json.dumps(
        receipt.as_dict(), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "EventGroundingReceipt",
    "EventLedgerReceipt",
    "FORBIDDEN_ANSWER_KEYS",
    "OBSERVABILITY",
    "localized_indices",
    "ledger_localized_indices",
    "ledger_prompt_text",
    "parse_event_grounding_receipt",
    "parse_event_ledger_receipt",
    "receipt_prompt_text",
    "shifted_indices",
    "uniform_indices",
]
