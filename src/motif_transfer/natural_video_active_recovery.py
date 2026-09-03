"""Fail-closed contracts for intervention-grounded natural-video recovery."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .natural_video_recovery import PROOF_KINDS, PROOF_STATUSES, normalize_probabilities


SOURCE_COMPATIBLE_FAMILIES = {
    "star": frozenset({"Interaction", "Sequence"}),
    "nextqa": frozenset({"Causal", "Temporal"}),
}
ACTIVE_PREDICATE_KINDS = (
    "ENTITY_GROUNDING",
    "EVENT_OCCURRENCE",
    "TEMPORAL_ORDER",
    "CAUSAL_LINK",
)


def source_compatible(benchmark: str, family: str) -> bool:
    """Whether a recorded clip can expose the committed action effect.

    Static descriptions and unexecuted future/feasibility questions do not have
    an observable post-action effect, so the Sokoban VERIFY operator must abstain.
    """

    return family in SOURCE_COMPATIBLE_FAMILIES.get(benchmark, ())


def parse_active_probe(
    payload: Mapping[str, Any],
    *,
    claim_ids: Sequence[str],
    duration_seconds: float,
    frames_per_probe: int,
    maximum_window_fraction: float,
    maximum_window_seconds: float,
) -> dict[str, Any]:
    if tuple(payload.get("claim_ids") or ()) != tuple(claim_ids):
        raise ValueError("active probe must preserve blinded claim order")
    if str(payload.get("tool") or "") != "sample_frames":
        raise ValueError("active natural-video probe must use wrapper sample_frames")
    predicate_kind = str(payload.get("predicate_kind") or "")
    if predicate_kind not in ACTIVE_PREDICATE_KINDS:
        raise ValueError("unsupported active predicate kind")
    start = float(payload.get("start_sec", -1))
    end = float(payload.get("end_sec", -1))
    if not all(math.isfinite(value) for value in (start, end)):
        raise ValueError("active probe window must be finite")
    if start < 0 or end <= start or end > duration_seconds + 1e-6:
        raise ValueError("active probe lies outside the native clip")
    maximum_span = min(
        maximum_window_seconds,
        maximum_window_fraction * duration_seconds,
    )
    if end - start > maximum_span + 1e-6:
        raise ValueError("active probe window exceeds the frozen sensing budget")
    expected_facts = payload.get("expected_facts")
    if not isinstance(expected_facts, Mapping) or set(expected_facts) != set(claim_ids):
        raise ValueError("active probe needs one expected fact per blinded claim")
    facts = {claim_id: str(expected_facts[claim_id]).strip() for claim_id in claim_ids}
    if any(not value for value in facts.values()):
        raise ValueError("active probe expected facts cannot be empty")
    return {
        "claim_ids": list(claim_ids),
        "tool": "sample_frames",
        "arguments": {
            "n": int(frames_per_probe),
            "start_sec": start,
            "end_sec": end,
        },
        "predicate_kind": predicate_kind,
        "expected_facts": facts,
        "why_discriminative": str(payload.get("why_discriminative") or "").strip(),
    }


def _parse_candidate_proof(
    value: Mapping[str, Any], *, expected_claim_id: str,
) -> dict[str, Any]:
    if str(value.get("claim_id") or "") != expected_claim_id:
        raise ValueError("arbiter candidate proofs must preserve blinded claim order")
    steps = list(value.get("proof_steps") or ())
    if [str(step.get("kind") or "") for step in steps] != list(PROOF_KINDS):
        raise ValueError("arbiter proof must preserve the five typed steps")
    parsed_steps = []
    for step in steps:
        status = str(step.get("status") or "")
        confidence = float(step.get("confidence", -1))
        if status not in PROOF_STATUSES or not 0 <= confidence <= 1:
            raise ValueError("invalid arbiter proof status/confidence")
        parsed_steps.append({
            "kind": str(step["kind"]),
            "status": status,
            "confidence": confidence,
            "visible_fact": str(step.get("visible_fact") or "").strip(),
        })
    return {"claim_id": expected_claim_id, "proof_steps": parsed_steps}


def parse_active_arbitration(
    payload: Mapping[str, Any],
    *,
    slots: Sequence[str],
    claim_ids: Sequence[str],
) -> dict[str, Any]:
    probabilities = normalize_probabilities(payload.get("probabilities") or {}, slots)
    answer = str(payload.get("answer") or "")
    ordered = sorted(slots, key=lambda slot: (-probabilities[slot], slots.index(slot)))
    if answer != ordered[0] or probabilities[ordered[0]] <= probabilities[ordered[1]]:
        raise ValueError("active arbiter answer must be the unique probability argmax")
    proofs = list(payload.get("candidate_proofs") or ())
    if len(proofs) != len(claim_ids):
        raise ValueError("active arbiter must evaluate both blinded claims")
    parsed = [
        _parse_candidate_proof(value, expected_claim_id=claim_id)
        for value, claim_id in zip(proofs, claim_ids)
    ]
    for key in ("observed_evidence", "unresolved_uncertainties"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"active arbiter {key} must be a string list")
    return {
        "answer": answer,
        "probabilities": probabilities,
        "candidate_proofs": parsed,
        "observed_evidence": list(payload["observed_evidence"]),
        "unresolved_uncertainties": list(payload["unresolved_uncertainties"]),
        "reason": str(payload.get("reason") or "").strip(),
    }


def authentic_recovery_decision(
    arbitration: Mapping[str, Any],
    *,
    claim_to_slot: Mapping[str, str],
    primary_slot: str,
    alternative_slot: str,
) -> bool:
    slot_to_claim = {slot: claim for claim, slot in claim_to_slot.items()}
    if set(slot_to_claim) != {primary_slot, alternative_slot}:
        raise ValueError("active arbitration claims do not bind the compared slots")
    by_claim = {
        str(row["claim_id"]): row for row in arbitration["candidate_proofs"]
    }
    primary_final = by_claim[slot_to_claim[primary_slot]]["proof_steps"][-1]["status"]
    alternative_final = by_claim[slot_to_claim[alternative_slot]]["proof_steps"][-1]["status"]
    return bool(
        arbitration["answer"] == alternative_slot
        and primary_final == "REFUTED"
        and alternative_final == "SUPPORTED"
    )


__all__ = [
    "ACTIVE_PREDICATE_KINDS",
    "SOURCE_COMPATIBLE_FAMILIES",
    "authentic_recovery_decision",
    "parse_active_arbitration",
    "parse_active_probe",
    "source_compatible",
]
