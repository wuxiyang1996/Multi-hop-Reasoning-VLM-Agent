from __future__ import annotations

from motif_transfer.natural_video_recovery import PROOF_KINDS
from motif_transfer.natural_video_symbolic_controls import (
    TOPOLOGY_DERANGEMENT,
    execute_recovery,
    recovery_decision,
)


def _candidate(slot: str, answer_status: str, entity_status: str = "UNKNOWN") -> dict:
    statuses = {kind: "UNKNOWN" for kind in PROOF_KINDS}
    statuses["ANSWER_ENTAILMENT"] = answer_status
    statuses["ENTITY_GROUNDING"] = entity_status
    return {
        "slot": slot,
        "proof_steps": [
            {"kind": kind, "status": statuses[kind], "confidence": 0.9}
            for kind in PROOF_KINDS
        ],
    }


def test_authentic_guard_replans_on_refuted_supported_transition() -> None:
    proof = {
        "answer": "B",
        "candidates": [_candidate("A", "REFUTED"), _candidate("B", "SUPPORTED")],
    }
    assert recovery_decision("A", proof)
    assert execute_recovery("A", proof) == "B"


def test_binding_rotation_is_a_destructive_derangement() -> None:
    proof = {
        "answer": "B",
        "candidates": [_candidate("A", "REFUTED"), _candidate("B", "SUPPORTED")],
    }
    assert not recovery_decision("A", proof, shuffled_binding=True)


def test_topology_control_preserves_kinds_but_breaks_answer_entailment() -> None:
    assert set(TOPOLOGY_DERANGEMENT) == set(PROOF_KINDS)
    assert set(TOPOLOGY_DERANGEMENT.values()) == set(PROOF_KINDS)
    assert all(left != right for left, right in TOPOLOGY_DERANGEMENT.items())
    proof = {
        "answer": "B",
        "candidates": [
            _candidate("A", "REFUTED", entity_status="SUPPORTED"),
            _candidate("B", "SUPPORTED", entity_status="REFUTED"),
        ],
    }
    assert not recovery_decision("A", proof, shuffled_topology=True)
