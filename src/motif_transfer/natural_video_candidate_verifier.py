"""Candidate-level target-native verifier for natural-video transfer.

The source game contributes only the controller topology::

    COMMIT -> VERIFY_EXPECTED_EFFECT -> REPLAN_OR_ABSTAIN

The verifier is deliberately target-native: it learns how STAR/NExT-QA proof
nodes ground in natural video.  It never consumes gold labels at runtime and it
does not encode answer-slot identity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .natural_video_recovery import FAMILIES, PROOF_KINDS


SOURCE_CONTRACT = (
    "COMMIT",
    "VERIFY_EXPECTED_EFFECT",
    "EXPECTED_EFFECT_REFUTED",
    "REPLAN_OR_ABSTAIN",
)

STATUS_VALUE = {"SUPPORTED": 1.0, "REFUTED": -1.0, "UNKNOWN": 0.0}

BASE_FEATURE_NAMES = (
    "is_star",
    "is_nextqa",
    *(f"family_{family}" for family in FAMILIES),
    "choice_count_fraction",
    "direct_candidate_probability",
    "direct_max_probability",
    "direct_margin",
    "direct_entropy_fraction",
    "candidate_is_direct_commit",
    "candidate_minus_direct_max",
)

MARGINAL_PROOF_FEATURE_NAMES = (
    "proof_candidate_probability",
    "proof_max_probability",
    "proof_margin",
    "proof_entropy_fraction",
    "candidate_is_proof_commit",
    "candidate_support_probability",
    "candidate_sensor_reliability",
    "candidate_supported_step_fraction",
    "candidate_refuted_step_fraction",
    "candidate_unknown_step_fraction",
)

TOPOLOGY_FEATURE_NAMES = tuple(
    f"{kind}_{suffix}"
    for kind in PROOF_KINDS
    for suffix in ("status_value", "confidence", "signed_confidence")
)

MARGINAL_FEATURE_NAMES = BASE_FEATURE_NAMES + MARGINAL_PROOF_FEATURE_NAMES
FULL_FEATURE_NAMES = MARGINAL_FEATURE_NAMES + TOPOLOGY_FEATURE_NAMES


def _distribution_features(
    probabilities: Mapping[str, Any], slots: Sequence[str],
) -> tuple[float, float, float]:
    values = sorted((float(probabilities[slot]) for slot in slots), reverse=True)
    total = sum(values)
    if total <= 0:
        raise ValueError("candidate probabilities have no mass")
    values = [value / total for value in values]
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    return values[0], values[0] - values[1], entropy / math.log(len(values))


def _candidate_proof_features(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    steps = list(candidate["proof_steps"])
    if tuple(str(step["kind"]) for step in steps) != PROOF_KINDS:
        raise ValueError("candidate proof topology drift")
    statuses = [str(step["status"]) for step in steps]
    if any(status not in STATUS_VALUE for status in statuses):
        raise ValueError("unknown proof status")
    marginal = (
        float(candidate["support_probability"]),
        float(candidate["sensor_reliability"]),
        statuses.count("SUPPORTED") / len(steps),
        statuses.count("REFUTED") / len(steps),
        statuses.count("UNKNOWN") / len(steps),
    )
    topology = []
    for step in steps:
        value = STATUS_VALUE[str(step["status"])]
        confidence = float(step["confidence"])
        topology.extend((value, confidence, value * confidence))
    return tuple(map(float, marginal + tuple(topology)))


def build_candidate_features(
    *,
    benchmark: str,
    family: str,
    direct: Mapping[str, Any],
    proof: Mapping[str, Any],
    proof_binding: Sequence[int] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Build one slot-invariant feature vector per native answer candidate.

    ``proof_binding`` maps each answer candidate to a proof candidate.  A
    rotation supplies the answer-binding falsification control while exactly
    preserving the proof receipt and its marginals.
    """

    if benchmark not in {"star", "nextqa"} or family not in FAMILIES:
        raise ValueError("unsupported benchmark/family")
    slots = tuple(str(slot) for slot in direct["probabilities"])
    if tuple(str(slot) for slot in proof["probabilities"]) != slots:
        raise ValueError("direct/proof slots do not align")
    candidates = list(proof["candidates"])
    if tuple(str(row["slot"]) for row in candidates) != slots:
        raise ValueError("proof candidates do not align")
    binding = tuple(range(len(slots))) if proof_binding is None else tuple(proof_binding)
    if sorted(binding) != list(range(len(slots))):
        raise ValueError("proof binding must be a permutation")
    direct_dist = _distribution_features(direct["probabilities"], slots)
    proof_dist = _distribution_features(proof["probabilities"], slots)
    output = []
    for index, slot in enumerate(slots):
        proof_index = binding[index]
        proof_slot = slots[proof_index]
        candidate = candidates[proof_index]
        candidate_features = _candidate_proof_features(candidate)
        base = (
            float(benchmark == "star"),
            float(benchmark == "nextqa"),
            *(float(family == value) for value in FAMILIES),
            len(slots) / 5,
            float(direct["probabilities"][slot]),
            *direct_dist,
            float(str(direct["answer"]) == slot),
            float(direct["probabilities"][slot]) - direct_dist[0],
        )
        proof_head = (
            float(proof["probabilities"][proof_slot]),
            *proof_dist,
            float(str(proof["answer"]) == proof_slot),
        )
        row = tuple(map(float, base + proof_head + candidate_features))
        if len(row) != len(FULL_FEATURE_NAMES):
            raise AssertionError("candidate feature schema drift")
        output.append(row)
    return tuple(output)


def rotated_candidate_binding(choice_count: int) -> tuple[int, ...]:
    if choice_count < 2:
        raise ValueError("candidate rotation requires at least two choices")
    return tuple((index + 1) % choice_count for index in range(choice_count))


__all__ = [
    "BASE_FEATURE_NAMES",
    "FULL_FEATURE_NAMES",
    "MARGINAL_FEATURE_NAMES",
    "SOURCE_CONTRACT",
    "build_candidate_features",
    "rotated_candidate_binding",
]
