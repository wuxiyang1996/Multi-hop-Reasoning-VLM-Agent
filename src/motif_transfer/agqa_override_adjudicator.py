"""Independent target-native adjudication for risky AGQA symbolic overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AGQAOverrideAdjudication:
    decision: str
    confidence: float
    evidence_frames: tuple[int, ...]
    observed_events: tuple[str, ...]
    ambiguity: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_frames"] = list(self.evidence_frames)
        payload["observed_events"] = list(self.observed_events)
        return payload


def parse_override_adjudication(
    payload: Mapping[str, Any], *, allowed_decisions: Sequence[str],
    frame_count: int,
) -> AGQAOverrideAdjudication:
    forbidden = {
        "gold", "gold_answer", "correct", "correctness", "functional_program",
        "scene_graph", "source_identity", "typed_prediction", "direct_response",
    }
    leaked = forbidden & {str(key).casefold() for key in payload}
    if leaked:
        raise ValueError(f"AGQA adjudication leaked forbidden fields: {sorted(leaked)}")
    allowed = {str(value).strip().casefold() for value in allowed_decisions}
    decision = str(payload.get("decision") or "").strip().casefold()
    if decision not in allowed | {"unknown"}:
        raise ValueError("AGQA adjudication decision is outside the closed answer set")
    confidence = float(payload.get("confidence", -1.0))
    if not 0 <= confidence <= 1:
        raise ValueError("AGQA adjudication confidence must be in [0,1]")
    raw_frames = payload.get("evidence_frames")
    if not isinstance(raw_frames, list) or len(raw_frames) > 6:
        raise ValueError("AGQA adjudication may cite at most six evidence frames")
    frames: list[int] = []
    for raw in raw_frames:
        if isinstance(raw, bool):
            raise ValueError("AGQA adjudication frame IDs must be integers")
        index = int(raw)
        if index < 0 or index >= frame_count:
            raise ValueError("AGQA adjudication cited a frame outside the proxy video")
        if index not in frames:
            frames.append(index)
    if frames != sorted(frames):
        raise ValueError("AGQA adjudication evidence must be chronological")
    events = payload.get("observed_events")
    if (
        not isinstance(events, list)
        or len(events) > 6
        or not all(isinstance(value, str) for value in events)
    ):
        raise ValueError("AGQA adjudication observed_events must be a short string list")
    if decision != "unknown" and not frames:
        raise ValueError("a decisive AGQA adjudication requires cited frames")
    return AGQAOverrideAdjudication(
        decision=decision,
        confidence=confidence,
        evidence_frames=tuple(frames),
        observed_events=tuple(value.strip() for value in events),
        ambiguity=str(payload.get("ambiguity") or "").strip(),
        reason=str(payload.get("reason") or "").strip(),
    )


def adjudication_supports_typed_override(
    adjudication: AGQAOverrideAdjudication, *, typed_decision: str,
    minimum_confidence: float,
) -> bool:
    return (
        adjudication.decision != "unknown"
        and adjudication.decision.strip().casefold()
        == str(typed_decision).strip().casefold()
        and adjudication.confidence >= minimum_confidence
        and bool(adjudication.evidence_frames)
    )


__all__ = [
    "AGQAOverrideAdjudication", "adjudication_supports_typed_override",
    "parse_override_adjudication",
]
