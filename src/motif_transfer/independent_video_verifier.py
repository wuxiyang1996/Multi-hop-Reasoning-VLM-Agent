"""Independent neural candidate receipts and a deterministic symbolic executor."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .natural_video_recovery import FAMILIES, PROOF_KINDS, PROOF_STATUSES


REQUIRED_KINDS = {
    "Interaction": ("ENTITY_GROUNDING", "EVENT_OCCURRENCE"),
    "Sequence": ("ENTITY_GROUNDING", "EVENT_OCCURRENCE", "TEMPORAL_ORDER"),
    "Prediction": ("ENTITY_GROUNDING", "EVENT_OCCURRENCE", "TEMPORAL_ORDER"),
    "Feasibility": ("ENTITY_GROUNDING", "EVENT_OCCURRENCE", "TEMPORAL_ORDER"),
    "Causal": (
        "ENTITY_GROUNDING", "EVENT_OCCURRENCE", "TEMPORAL_ORDER", "CAUSAL_LINK",
    ),
    "Temporal": ("ENTITY_GROUNDING", "EVENT_OCCURRENCE", "TEMPORAL_ORDER"),
    "Descriptive": ("ENTITY_GROUNDING", "EVENT_OCCURRENCE"),
}
STATUS_RANK = {"REFUTED": 0, "UNKNOWN": 1, "SUPPORTED": 2}
TOPOLOGY_DERANGEMENT = {
    kind: PROOF_KINDS[(index + 1) % len(PROOF_KINDS)]
    for index, kind in enumerate(PROOF_KINDS)
}


def parse_independent_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    support = float(payload.get("support_probability", -1))
    reliability = float(payload.get("sensor_reliability", -1))
    if not 0 <= support <= 1 or not 0 <= reliability <= 1:
        raise ValueError("independent candidate probabilities are invalid")
    steps = list(payload.get("proof_steps") or ())
    if [str(step.get("kind")) for step in steps] != list(PROOF_KINDS):
        raise ValueError("independent candidate must preserve the typed proof order")
    parsed_steps = []
    for step in steps:
        status = str(step.get("status") or "")
        confidence = float(step.get("confidence", -1))
        if status not in PROOF_STATUSES or not 0 <= confidence <= 1:
            raise ValueError("invalid independent typed step")
        parsed_steps.append({
            "kind": str(step["kind"]),
            "status": status,
            "confidence": confidence,
            "visible_fact": str(step.get("visible_fact") or "").strip(),
        })
    uncertainties = payload.get("unresolved_uncertainties")
    if not isinstance(uncertainties, list) or not all(
        isinstance(value, str) for value in uncertainties
    ):
        raise ValueError("independent candidate uncertainties must be a string list")
    return {
        "support_probability": support,
        "sensor_reliability": reliability,
        "proof_steps": parsed_steps,
        "unresolved_uncertainties": list(uncertainties),
        "reason": str(payload.get("reason") or "").strip(),
    }


def _step_map(
    candidate: Mapping[str, Any], *, shuffled_topology: bool,
) -> dict[str, Mapping[str, Any]]:
    raw = {str(step["kind"]): step for step in candidate["proof_steps"]}
    if set(raw) != set(PROOF_KINDS):
        raise ValueError("candidate proof-step kinds are incomplete")
    return {
        executor_kind: raw[
            TOPOLOGY_DERANGEMENT[executor_kind]
            if shuffled_topology else executor_kind
        ]
        for executor_kind in PROOF_KINDS
    }


def candidate_state(
    candidate: Mapping[str, Any],
    *,
    family: str,
    shuffled_topology: bool = False,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError("unsupported natural-video family")
    steps = _step_map(candidate, shuffled_topology=shuffled_topology)
    required = [steps[kind] for kind in REQUIRED_KINDS[family]]
    answer = steps["ANSWER_ENTAILMENT"]
    any_refuted = answer["status"] == "REFUTED" or any(
        step["status"] == "REFUTED" for step in required
    )
    all_supported = answer["status"] == "SUPPORTED" and all(
        step["status"] == "SUPPORTED" for step in required
    )
    tier = 0 if any_refuted else (2 if all_supported else 1)
    signed_confidence = sum(
        {"SUPPORTED": 1.0, "REFUTED": -1.0, "UNKNOWN": 0.0}[str(step["status"])]
        * float(step["confidence"])
        for step in [*required, answer]
    )
    return {
        "tier": tier,
        "answer_status": str(answer["status"]),
        "answer_status_rank": STATUS_RANK[str(answer["status"])],
        "supported_required_steps": sum(
            step["status"] == "SUPPORTED" for step in required
        ),
        "signed_required_confidence": signed_confidence,
        "support_reliability": (
            float(candidate["support_probability"])
            * float(candidate["sensor_reliability"])
        ),
        "refuted": any_refuted,
        "supported": answer["status"] == "SUPPORTED" and not any_refuted,
    }


def _bound_candidates(
    candidates: Sequence[Mapping[str, Any]], *, shuffled_binding: bool,
) -> list[Mapping[str, Any]]:
    values = list(candidates)
    slots = [str(candidate["slot"]) for candidate in values]
    if len(values) < 2 or len(slots) != len(set(slots)):
        raise ValueError("independent candidates need distinct native slots")
    if not shuffled_binding:
        return values
    receipts = values[1:] + values[:1]
    return [
        {**receipt, "slot": slot} for slot, receipt in zip(slots, receipts)
    ]


def execute_candidate_program(
    candidates: Sequence[Mapping[str, Any]],
    *,
    family: str,
    shuffled_binding: bool = False,
    shuffled_topology: bool = False,
) -> dict[str, Any]:
    bound = _bound_candidates(candidates, shuffled_binding=shuffled_binding)
    scored = []
    for index, candidate in enumerate(bound):
        state = candidate_state(
            candidate, family=family, shuffled_topology=shuffled_topology,
        )
        key = (
            int(state["tier"]),
            int(state["answer_status_rank"]),
            int(state["supported_required_steps"]),
            float(state["signed_required_confidence"]),
            float(state["support_reliability"]),
            -index,
        )
        scored.append({"slot": str(candidate["slot"]), "state": state, "key": key})
    selected = max(scored, key=lambda row: row["key"])
    return {"answer": selected["slot"], "candidates": scored}


def execute_source_guard(
    primary_answer: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    family: str,
    shuffled_binding: bool = False,
    shuffled_topology: bool = False,
) -> dict[str, Any]:
    execution = execute_candidate_program(
        candidates,
        family=family,
        shuffled_binding=shuffled_binding,
        shuffled_topology=shuffled_topology,
    )
    by_slot = {row["slot"]: row["state"] for row in execution["candidates"]}
    alternative = str(execution["answer"])
    if primary_answer not in by_slot:
        raise ValueError("primary answer is outside independent candidates")
    recover = bool(
        alternative != primary_answer
        and bool(by_slot[primary_answer]["refuted"])
        and bool(by_slot[alternative]["supported"])
    )
    return {
        "answer": alternative if recover else primary_answer,
        "recover": recover,
        "alternative": alternative,
        "execution": execution,
    }


__all__ = [
    "REQUIRED_KINDS",
    "TOPOLOGY_DERANGEMENT",
    "candidate_state",
    "execute_candidate_program",
    "execute_source_guard",
    "parse_independent_candidate",
]
