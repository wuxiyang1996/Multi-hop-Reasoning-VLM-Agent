"""Outcome-blind destructive controls for typed natural-video verification."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .natural_video_recovery import PROOF_KINDS


TOPOLOGY_DERANGEMENT = {
    kind: PROOF_KINDS[(index + 1) % len(PROOF_KINDS)]
    for index, kind in enumerate(PROOF_KINDS)
}


def _candidate_map(
    proof: Mapping[str, Any], *, shuffled_binding: bool,
) -> dict[str, Mapping[str, Any]]:
    candidates = list(proof["candidates"])
    slots = [str(candidate["slot"]) for candidate in candidates]
    if len(slots) < 2 or len(slots) != len(set(slots)):
        raise ValueError("typed proof candidates need distinct native slots")
    if not shuffled_binding:
        return {slot: candidate for slot, candidate in zip(slots, candidates)}
    # Rotate the proof objects while retaining the target slot names.  This is a
    # cardinality-preserving derangement for every supported choice count.
    rotated = candidates[1:] + candidates[:1]
    return {slot: candidate for slot, candidate in zip(slots, rotated)}


def _status(
    candidate: Mapping[str, Any], *, executor_kind: str, shuffled_topology: bool,
) -> str:
    source_kind = (
        TOPOLOGY_DERANGEMENT[executor_kind] if shuffled_topology else executor_kind
    )
    matches = [
        str(step["status"])
        for step in candidate["proof_steps"] if str(step["kind"]) == source_kind
    ]
    if len(matches) != 1:
        raise ValueError("typed proof must contain each required proof kind exactly once")
    return matches[0]


def recovery_decision(
    primary_answer: str,
    proof: Mapping[str, Any],
    *,
    shuffled_binding: bool = False,
    shuffled_topology: bool = False,
) -> bool:
    """Execute REFUTED(primary) & SUPPORTED(alternative) -> REPLAN."""

    proof_answer = str(proof["answer"])
    if proof_answer == primary_answer:
        return False
    candidates = _candidate_map(proof, shuffled_binding=shuffled_binding)
    if primary_answer not in candidates or proof_answer not in candidates:
        raise ValueError("primary/proof answer is outside the native candidate slots")
    primary_status = _status(
        candidates[primary_answer],
        executor_kind="ANSWER_ENTAILMENT",
        shuffled_topology=shuffled_topology,
    )
    proof_status = _status(
        candidates[proof_answer],
        executor_kind="ANSWER_ENTAILMENT",
        shuffled_topology=shuffled_topology,
    )
    return primary_status == "REFUTED" and proof_status == "SUPPORTED"


def execute_recovery(
    primary_answer: str,
    proof: Mapping[str, Any],
    *,
    shuffled_binding: bool = False,
    shuffled_topology: bool = False,
) -> str:
    return (
        str(proof["answer"])
        if recovery_decision(
            primary_answer,
            proof,
            shuffled_binding=shuffled_binding,
            shuffled_topology=shuffled_topology,
        )
        else primary_answer
    )


def validate_topology_derangement() -> None:
    if set(TOPOLOGY_DERANGEMENT) != set(PROOF_KINDS):
        raise ValueError("topology control lost a proof kind")
    if set(TOPOLOGY_DERANGEMENT.values()) != set(PROOF_KINDS):
        raise ValueError("topology control changed proof-kind cardinality")
    if any(kind == mapped for kind, mapped in TOPOLOGY_DERANGEMENT.items()):
        raise ValueError("topology control is not a derangement")


validate_topology_derangement()


__all__ = [
    "TOPOLOGY_DERANGEMENT",
    "execute_recovery",
    "recovery_decision",
    "validate_topology_derangement",
]
