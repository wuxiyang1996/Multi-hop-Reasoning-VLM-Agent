"""Typed runtime contracts and features for natural-video recovery transfer."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


SLOTS = ("A", "B", "C", "D", "E")
FAMILIES = (
    "Interaction",
    "Sequence",
    "Prediction",
    "Feasibility",
    "Causal",
    "Temporal",
    "Descriptive",
)
PROOF_KINDS = (
    "ENTITY_GROUNDING",
    "EVENT_OCCURRENCE",
    "TEMPORAL_ORDER",
    "CAUSAL_LINK",
    "ANSWER_ENTAILMENT",
)
PROOF_STATUSES = ("SUPPORTED", "REFUTED", "UNKNOWN")


def normalize_probabilities(
    value: Mapping[str, Any], slots: Sequence[str],
) -> dict[str, float]:
    if set(value) != set(slots):
        raise ValueError("probabilities must contain exactly the native answer slots")
    output = {slot: float(value[slot]) for slot in slots}
    if any(not math.isfinite(item) or item < 0 for item in output.values()):
        raise ValueError("answer probabilities must be finite and nonnegative")
    total = sum(output.values())
    if total <= 0:
        raise ValueError("answer probabilities must have positive mass")
    return {slot: item / total for slot, item in output.items()}


def committed_answer(probabilities: Mapping[str, float], slots: Sequence[str]) -> str:
    return max(slots, key=lambda slot: (float(probabilities[slot]), -slots.index(slot)))


def parse_primary_receipt(
    payload: Mapping[str, Any], slots: Sequence[str],
) -> dict[str, Any]:
    probabilities = normalize_probabilities(payload.get("probabilities") or {}, slots)
    answer = str(payload.get("answer") or "")
    if answer != committed_answer(probabilities, slots):
        raise ValueError("primary answer must equal probability argmax")
    evidence = payload.get("observed_evidence")
    uncertainties = payload.get("unresolved_uncertainties")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError("primary observed_evidence must be a string list")
    if not isinstance(uncertainties, list) or not all(isinstance(item, str) for item in uncertainties):
        raise ValueError("primary unresolved_uncertainties must be a string list")
    return {
        "answer": answer,
        "probabilities": probabilities,
        "observed_evidence": list(evidence),
        "unresolved_uncertainties": list(uncertainties),
        "reason": str(payload.get("reason") or "").strip(),
    }


def parse_proof_receipt(
    payload: Mapping[str, Any], slots: Sequence[str],
) -> dict[str, Any]:
    probabilities = normalize_probabilities(payload.get("probabilities") or {}, slots)
    answer = str(payload.get("answer") or "")
    if answer != committed_answer(probabilities, slots):
        raise ValueError("proof answer must equal probability argmax")
    candidates = list(payload.get("candidates") or ())
    if [str(row.get("slot")) for row in candidates] != list(slots):
        raise ValueError("proof candidates must exactly preserve slot order")
    parsed = []
    for row in candidates:
        support = float(row.get("support_probability", -1))
        reliability = float(row.get("sensor_reliability", -1))
        if not 0 <= support <= 1 or not 0.5 <= reliability <= 1:
            raise ValueError("proof candidate probabilities/reliability are invalid")
        steps = list(row.get("proof_steps") or ())
        if [str(step.get("kind")) for step in steps] != list(PROOF_KINDS):
            raise ValueError("proof steps must exactly preserve the typed proof order")
        parsed_steps = []
        for step in steps:
            status = str(step.get("status") or "")
            confidence = float(step.get("confidence", -1))
            if status not in PROOF_STATUSES or not 0 <= confidence <= 1:
                raise ValueError("invalid typed proof status/confidence")
            parsed_steps.append({
                "kind": str(step["kind"]),
                "status": status,
                "confidence": confidence,
                "visible_fact": str(step.get("visible_fact") or "").strip(),
            })
        parsed.append({
            "slot": str(row["slot"]),
            "support_probability": support,
            "sensor_reliability": reliability,
            "proof_steps": parsed_steps,
        })
    uncertainties = payload.get("global_uncertainties")
    if not isinstance(uncertainties, list) or not all(isinstance(item, str) for item in uncertainties):
        raise ValueError("proof global_uncertainties must be a string list")
    return {
        "answer": answer,
        "probabilities": probabilities,
        "candidates": parsed,
        "global_uncertainties": list(uncertainties),
        "reason": str(payload.get("reason") or "").strip(),
    }


def parse_focused_verification(
    payload: Mapping[str, Any], slots: Sequence[str], expected_answer: str,
) -> dict[str, Any]:
    # The commitment is immutable controller state, not a neural prediction.
    # A model-generated echo must never be allowed to rewrite it.
    probabilities = normalize_probabilities(payload.get("probabilities") or {}, slots)
    recovery_answer = str(payload.get("recovery_answer") or "")
    if recovery_answer != committed_answer(probabilities, slots):
        raise ValueError("focused recovery answer must equal probability argmax")
    steps = list(payload.get("expected_answer_proof_steps") or ())
    if [str(step.get("kind")) for step in steps] != list(PROOF_KINDS):
        raise ValueError("focused proof steps must preserve typed proof order")
    parsed_steps = []
    for step in steps:
        status = str(step.get("status") or "")
        confidence = float(step.get("confidence", -1))
        if status not in PROOF_STATUSES or not 0 <= confidence <= 1:
            raise ValueError("invalid focused proof status/confidence")
        parsed_steps.append({
            "kind": str(step["kind"]),
            "status": status,
            "confidence": confidence,
            "visible_fact": str(step.get("visible_fact") or "").strip(),
        })
    verification_status = str(payload.get("verification_status") or "")
    expected_status = {
        "SUPPORTED": "OBSERVED",
        "REFUTED": "REFUTED",
        "UNKNOWN": "UNRESOLVED",
    }[parsed_steps[-1]["status"]]
    if verification_status != expected_status:
        raise ValueError("verification status must be determined by ANSWER_ENTAILMENT")
    for key in ("supporting_evidence", "counterevidence", "unresolved_uncertainties"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"focused {key} must be a string list")
    return {
        "expected_answer": expected_answer,
        "verification_status": verification_status,
        "recovery_answer": recovery_answer,
        "probabilities": probabilities,
        "expected_answer_proof_steps": parsed_steps,
        "supporting_evidence": list(payload["supporting_evidence"]),
        "counterevidence": list(payload["counterevidence"]),
        "unresolved_uncertainties": list(payload["unresolved_uncertainties"]),
        "reason": str(payload.get("reason") or "").strip(),
    }


BASE_FEATURE_NAMES = (
    "is_star",
    "is_nextqa",
    *(f"family_{value}" for value in FAMILIES),
    "choice_count_fraction",
    *(f"primary_probability_{slot}" for slot in SLOTS),
    *(f"proof_probability_{slot}" for slot in SLOTS),
    "primary_max_probability",
    "primary_margin",
    "primary_entropy_fraction",
    "proof_max_probability",
    "proof_margin",
    "proof_entropy_fraction",
    "answer_disagreement",
    "proof_probability_of_primary_answer",
    "primary_probability_of_proof_answer",
    "probability_l1_fraction",
    "probability_js_divergence",
    *(f"primary_answer_{slot}" for slot in SLOTS),
    *(f"proof_answer_{slot}" for slot in SLOTS),
)

PROOF_FEATURE_NAMES = (
    "primary_observed_evidence_count_fraction",
    "primary_uncertainty_count_fraction",
    "proof_global_uncertainty_count_fraction",
    "primary_candidate_support_probability",
    "primary_candidate_sensor_reliability",
    "proof_candidate_support_probability",
    "proof_candidate_sensor_reliability",
    "all_candidate_supported_step_fraction",
    "all_candidate_refuted_step_fraction",
    "all_candidate_unknown_step_fraction",
    *(
        f"primary_candidate_{kind}_{suffix}"
        for kind in PROOF_KINDS for suffix in ("status_value", "confidence")
    ),
    *(
        f"proof_candidate_{kind}_{suffix}"
        for kind in PROOF_KINDS for suffix in ("status_value", "confidence")
    ),
)
FEATURE_NAMES = BASE_FEATURE_NAMES + PROOF_FEATURE_NAMES


def _distribution_features(
    probabilities: Mapping[str, float], slots: Sequence[str],
) -> tuple[float, float, float]:
    ordered = sorted((float(probabilities[slot]) for slot in slots), reverse=True)
    entropy = -sum(value * math.log(value) for value in ordered if value > 0)
    return ordered[0], ordered[0] - ordered[1], entropy / math.log(len(slots))


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    middle = [(a + b) / 2 for a, b in zip(left, right)]

    def kl(values: Sequence[float], reference: Sequence[float]) -> float:
        return sum(
            value * math.log(value / target)
            for value, target in zip(values, reference) if value > 0 and target > 0
        )

    return (kl(left, middle) + kl(right, middle)) / (2 * math.log(2))


def build_features(
    *, benchmark: str, family: str, primary: Mapping[str, Any], proof: Mapping[str, Any],
) -> tuple[float, ...]:
    if benchmark not in {"star", "nextqa"} or family not in FAMILIES:
        raise ValueError("unsupported natural-video benchmark/family")
    slots = tuple(primary["probabilities"])
    if tuple(proof["probabilities"]) != slots or not 2 <= len(slots) <= len(SLOTS):
        raise ValueError("primary/proof native answer slots must align")
    primary_values = [float(primary["probabilities"][slot]) for slot in slots]
    proof_values = [float(proof["probabilities"][slot]) for slot in slots]
    primary_answer = str(primary["answer"])
    proof_answer = str(proof["answer"])
    padded_primary = [float(primary["probabilities"].get(slot, 0.0)) for slot in SLOTS]
    padded_proof = [float(proof["probabilities"].get(slot, 0.0)) for slot in SLOTS]
    primary_dist = _distribution_features(primary["probabilities"], slots)
    proof_dist = _distribution_features(proof["probabilities"], slots)
    base = (
        float(benchmark == "star"),
        float(benchmark == "nextqa"),
        *(float(family == value) for value in FAMILIES),
        len(slots) / len(SLOTS),
        *padded_primary,
        *padded_proof,
        *primary_dist,
        *proof_dist,
        float(primary_answer != proof_answer),
        float(proof["probabilities"][primary_answer]),
        float(primary["probabilities"][proof_answer]),
        sum(abs(a - b) for a, b in zip(primary_values, proof_values)) / 2,
        _js_divergence(primary_values, proof_values),
        *(float(primary_answer == slot) for slot in SLOTS),
        *(float(proof_answer == slot) for slot in SLOTS),
    )
    by_slot = {str(row["slot"]): row for row in proof["candidates"]}
    primary_candidate = by_slot[primary_answer]
    proof_candidate = by_slot[proof_answer]
    all_steps = [step for row in proof["candidates"] for step in row["proof_steps"]]
    status_value = {"SUPPORTED": 1.0, "REFUTED": -1.0, "UNKNOWN": 0.0}

    def step_features(candidate: Mapping[str, Any]) -> tuple[float, ...]:
        output = []
        for step in candidate["proof_steps"]:
            output.extend((status_value[str(step["status"])], float(step["confidence"])))
        return tuple(output)

    typed = (
        min(len(primary["observed_evidence"]), 10) / 10,
        min(len(primary["unresolved_uncertainties"]), 10) / 10,
        min(len(proof["global_uncertainties"]), 10) / 10,
        float(primary_candidate["support_probability"]),
        float(primary_candidate["sensor_reliability"]),
        float(proof_candidate["support_probability"]),
        float(proof_candidate["sensor_reliability"]),
        sum(step["status"] == "SUPPORTED" for step in all_steps) / len(all_steps),
        sum(step["status"] == "REFUTED" for step in all_steps) / len(all_steps),
        sum(step["status"] == "UNKNOWN" for step in all_steps) / len(all_steps),
        *step_features(primary_candidate),
        *step_features(proof_candidate),
    )
    features = tuple(map(float, base + typed))
    if len(base) != len(BASE_FEATURE_NAMES) or len(features) != len(FEATURE_NAMES):
        raise AssertionError("natural-video recovery feature contract drift")
    return features


__all__ = [
    "BASE_FEATURE_NAMES",
    "FAMILIES",
    "FEATURE_NAMES",
    "PROOF_FEATURE_NAMES",
    "PROOF_KINDS",
    "PROOF_STATUSES",
    "SLOTS",
    "build_features",
    "committed_answer",
    "normalize_probabilities",
    "parse_primary_receipt",
    "parse_proof_receipt",
    "parse_focused_verification",
]
