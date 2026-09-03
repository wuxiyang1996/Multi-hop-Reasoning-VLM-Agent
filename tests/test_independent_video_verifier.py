from __future__ import annotations

from motif_transfer.independent_video_verifier import (
    execute_candidate_program,
    execute_source_guard,
    parse_independent_candidate,
)
from motif_transfer.natural_video_recovery import PROOF_KINDS


def _candidate(slot: str, statuses: dict[str, str], probability: float) -> dict:
    return {
        "slot": slot,
        "support_probability": probability,
        "sensor_reliability": 0.9,
        "proof_steps": [
            {
                "kind": kind,
                "status": statuses.get(kind, "UNKNOWN"),
                "confidence": 0.9,
                "visible_fact": kind,
            }
            for kind in PROOF_KINDS
        ],
        "unresolved_uncertainties": [],
        "reason": "test",
    }


def test_parser_requires_candidate_factorized_order() -> None:
    payload = _candidate("A", {}, 0.4)
    payload.pop("slot")
    assert parse_independent_candidate(payload)["support_probability"] == 0.4


def test_parser_accepts_low_but_valid_sensor_reliability() -> None:
    payload = _candidate("A", {}, 0.4)
    payload.pop("slot")
    payload["sensor_reliability"] = 0.2
    assert parse_independent_candidate(payload)["sensor_reliability"] == 0.2


def test_executor_uses_required_program_before_probability() -> None:
    supported = {
        "ENTITY_GROUNDING": "SUPPORTED",
        "EVENT_OCCURRENCE": "SUPPORTED",
        "ANSWER_ENTAILMENT": "SUPPORTED",
    }
    candidates = [
        _candidate("A", supported, 0.6),
        _candidate("B", {**supported, "EVENT_OCCURRENCE": "REFUTED"}, 0.99),
    ]
    assert execute_candidate_program(candidates, family="Interaction")["answer"] == "A"


def test_source_guard_replans_only_refuted_to_supported() -> None:
    supported = {
        "ENTITY_GROUNDING": "SUPPORTED",
        "EVENT_OCCURRENCE": "SUPPORTED",
        "ANSWER_ENTAILMENT": "SUPPORTED",
    }
    candidates = [
        _candidate("A", {**supported, "ANSWER_ENTAILMENT": "REFUTED"}, 0.1),
        _candidate("B", supported, 0.8),
    ]
    result = execute_source_guard("A", candidates, family="Interaction")
    assert result["recover"]
    assert result["answer"] == "B"


def test_binding_control_rotates_receipts() -> None:
    supported = {
        "ENTITY_GROUNDING": "SUPPORTED",
        "EVENT_OCCURRENCE": "SUPPORTED",
        "ANSWER_ENTAILMENT": "SUPPORTED",
    }
    candidates = [
        _candidate("A", supported, 0.9),
        _candidate("B", {**supported, "ANSWER_ENTAILMENT": "REFUTED"}, 0.1),
    ]
    assert execute_candidate_program(candidates, family="Interaction")["answer"] == "A"
    assert execute_candidate_program(
        candidates, family="Interaction", shuffled_binding=True,
    )["answer"] == "B"
