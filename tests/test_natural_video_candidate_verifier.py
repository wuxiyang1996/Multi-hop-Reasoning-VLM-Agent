from __future__ import annotations

import pytest

from motif_transfer.natural_video_candidate_verifier import (
    BASE_FEATURE_NAMES,
    FULL_FEATURE_NAMES,
    MARGINAL_FEATURE_NAMES,
    build_candidate_features,
    rotated_candidate_binding,
)
from motif_transfer.natural_video_recovery import PROOF_KINDS


def _receipts():
    slots = ("A", "B", "C", "D")
    direct = {
        "answer": "A",
        "probabilities": {"A": .7, "B": .1, "C": .1, "D": .1},
    }
    proof = {
        "answer": "B",
        "probabilities": {"A": .1, "B": .7, "C": .1, "D": .1},
        "candidates": [
            {
                "slot": slot,
                "support_probability": .8 if slot == "B" else .2,
                "sensor_reliability": .9,
                "proof_steps": [
                    {
                        "kind": kind,
                        "status": "SUPPORTED" if slot == "B" else "REFUTED",
                        "confidence": .8,
                    }
                    for kind in PROOF_KINDS
                ],
            }
            for slot in slots
        ],
    }
    return direct, proof


def test_candidate_features_are_slot_invariant_and_have_declared_boundaries():
    direct, proof = _receipts()
    rows = build_candidate_features(
        benchmark="star", family="Sequence", direct=direct, proof=proof,
    )
    assert len(rows) == 4
    assert all(len(row) == len(FULL_FEATURE_NAMES) for row in rows)
    assert len(BASE_FEATURE_NAMES) < len(MARGINAL_FEATURE_NAMES) < len(FULL_FEATURE_NAMES)
    # No answer-slot one-hot or slot-name feature may leak answer priors.
    assert not any("slot" in name for name in FULL_FEATURE_NAMES)


def test_rotation_exactly_misbinds_candidate_proofs():
    direct, proof = _receipts()
    identity = build_candidate_features(
        benchmark="star", family="Sequence", direct=direct, proof=proof,
    )
    binding = rotated_candidate_binding(4)
    rotated = build_candidate_features(
        benchmark="star", family="Sequence", direct=direct, proof=proof,
        proof_binding=binding,
    )
    assert binding == (1, 2, 3, 0)
    assert identity[0][:len(BASE_FEATURE_NAMES)] == rotated[0][:len(BASE_FEATURE_NAMES)]
    assert identity[0][len(BASE_FEATURE_NAMES):] == rotated[3][len(BASE_FEATURE_NAMES):]


def test_invalid_binding_is_rejected():
    direct, proof = _receipts()
    with pytest.raises(ValueError):
        build_candidate_features(
            benchmark="star", family="Sequence", direct=direct, proof=proof,
            proof_binding=(0, 0, 1, 2),
        )
