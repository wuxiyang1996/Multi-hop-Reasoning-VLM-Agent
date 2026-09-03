"""Candidate-local second-pass adjudication for typed video claims."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .typed_video_claim_grounder import parse_typed_claim_receipt


STATUSES = ("SUPPORTED", "REFUTED", "UNKNOWN", "NOT_APPLICABLE")


def focus_indices(
    receipt: Mapping[str, Any], *, frame_count: int, radius: int = 1, limit: int = 20,
) -> tuple[int, ...]:
    """Expand transition evidence locally without using answer outcomes."""
    seeds = sorted({
        int(index)
        for check in list(receipt.get("checks") or ())[1:]
        for index in list(check.get("evidence_frames") or ())
    })
    if not seeds:
        seeds = [0, frame_count - 1]
    expanded = sorted({0, frame_count - 1} | {
        value
        for seed in seeds
        for value in range(max(0, seed - radius), min(frame_count, seed + radius + 1))
    })
    if len(expanded) <= limit:
        return tuple(expanded)
    # Deterministic temporal coverage with both endpoints retained.
    chosen = {
        expanded[round(index * (len(expanded) - 1) / (limit - 1))]
        for index in range(limit)
    }
    return tuple(sorted(chosen))


def _indices(value: Any, *, frame_count: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError("adjudication evidence_frames must contain at most three frames")
    output = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError("adjudication frame must be an integer")
        normalized = raw.strip()[1:] if isinstance(raw, str) and raw.strip().upper().startswith("F") else raw
        index = int(normalized)
        if index < 0 or index >= frame_count:
            raise ValueError("adjudication frame outside proxy video")
        if index not in output:
            output.append(index)
    if output != sorted(output):
        raise ValueError("adjudication evidence must be chronological")
    return tuple(output)


@dataclass(frozen=True)
class FocusedAdjudication:
    entity_binding: str
    precondition: str
    postcondition: str
    transition_direction: str
    claim_entailment: str
    evidence_frames: tuple[int, ...]
    alternative_explanation: str
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_binding": self.entity_binding,
            "precondition": self.precondition,
            "postcondition": self.postcondition,
            "transition_direction": self.transition_direction,
            "claim_entailment": self.claim_entailment,
            "evidence_frames": list(self.evidence_frames),
            "alternative_explanation": self.alternative_explanation,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def parse_focused_adjudication(
    payload: Mapping[str, Any], *, frame_count: int,
) -> FocusedAdjudication:
    forbidden = {"answer", "choice", "choice_id", "correct_option", "slot"}
    leaked = forbidden & {str(key).casefold() for key in payload}
    if leaked:
        raise ValueError(f"focused adjudication leaked binding fields: {sorted(leaked)}")
    values = {
        key: str(payload.get(key) or "")
        for key in (
            "entity_binding", "precondition", "postcondition",
            "transition_direction", "claim_entailment",
        )
    }
    if any(value not in STATUSES for value in values.values()):
        raise ValueError("invalid focused adjudication status")
    if values["claim_entailment"] == "NOT_APPLICABLE":
        raise ValueError("focused claim entailment cannot be NOT_APPLICABLE")
    if values["claim_entailment"] == "SUPPORTED" and any(
        values[key] != "SUPPORTED"
        for key in ("entity_binding", "precondition", "postcondition", "transition_direction")
    ):
        raise ValueError("supported adjudication requires every transition premise")
    confidence = float(payload.get("confidence", -1.0))
    if not 0 <= confidence <= 1:
        raise ValueError("focused confidence must be in [0,1]")
    evidence = _indices(payload.get("evidence_frames"), frame_count=frame_count)
    if values["claim_entailment"] != "UNKNOWN" and not evidence:
        raise ValueError("supported/refuted focused adjudication needs visible evidence")
    return FocusedAdjudication(
        **values,
        evidence_frames=evidence,
        alternative_explanation=str(payload.get("alternative_explanation") or "").strip(),
        confidence=confidence,
        reason=str(payload.get("reason") or "").strip(),
    )


def fuse_supported_receipt(
    receipt: Mapping[str, Any],
    adjudication: FocusedAdjudication,
    *,
    frame_count: int,
) -> dict[str, Any]:
    """Conjoin an initial support with its independent focused adjudication."""
    parsed = parse_typed_claim_receipt(receipt, frame_count=frame_count)
    if parsed.claim_status != "SUPPORTED":
        raise ValueError("only an initially supported receipt may be adjudicated")
    if adjudication.claim_entailment == "SUPPORTED":
        return parsed.as_dict()
    fused = deepcopy(parsed.as_dict())
    verdict = adjudication.claim_entailment
    fused["claim_status"] = verdict
    fused["confidence"] = min(float(fused["confidence"]), adjudication.confidence)
    entailment = fused["checks"][-1]
    entailment["status"] = verdict
    entailment["basis"] = "OBSERVED"
    entailment["evidence_frames"] = list(adjudication.evidence_frames)
    entailment["confidence"] = adjudication.confidence
    entailment["fact"] = adjudication.reason or adjudication.alternative_explanation
    fused["reason"] = adjudication.reason
    if verdict == "UNKNOWN" and not adjudication.evidence_frames:
        entailment["basis"] = "INFERRED"
    return parse_typed_claim_receipt(fused, frame_count=frame_count).as_dict()


__all__ = [
    "FocusedAdjudication",
    "STATUSES",
    "focus_indices",
    "fuse_supported_receipt",
    "parse_focused_adjudication",
]
