from __future__ import annotations

import pytest

from motif_transfer.natural_video_recovery import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    PROOF_KINDS,
    build_features,
    parse_focused_verification,
    parse_primary_receipt,
    parse_proof_receipt,
)


def _receipts():
    primary = parse_primary_receipt({
        "answer": "A",
        "probabilities": {"A": 0.6, "B": 0.3, "C": 0.1, "D": 0.0},
        "observed_evidence": ["visible action"],
        "unresolved_uncertainties": ["occlusion"],
        "reason": "coarse view",
    }, ("A", "B", "C", "D"))
    candidates = []
    for slot, support in zip(("A", "B", "C", "D"), (0.2, 0.7, 0.05, 0.05)):
        candidates.append({
            "slot": slot,
            "support_probability": support,
            "sensor_reliability": 0.8,
            "proof_steps": [
                {
                    "kind": kind,
                    "status": "REFUTED" if slot == "A" else "SUPPORTED",
                    "confidence": 0.8,
                    "visible_fact": "fact",
                }
                for kind in PROOF_KINDS
            ],
        })
    proof = parse_proof_receipt({
        "answer": "B",
        "probabilities": {"A": 0.2, "B": 0.7, "C": 0.05, "D": 0.05},
        "candidates": candidates,
        "global_uncertainties": [],
        "reason": "dense proof",
    }, ("A", "B", "C", "D"))
    return primary, proof


def test_typed_natural_video_features_capture_refutation() -> None:
    primary, proof = _receipts()
    features = build_features(
        benchmark="star", family="Interaction", primary=primary, proof=proof,
    )
    assert len(features) == len(FEATURE_NAMES)
    assert len(BASE_FEATURE_NAMES) < len(features)
    assert features[len(BASE_FEATURE_NAMES) + 7] > 0


def test_primary_rejects_inconsistent_argmax() -> None:
    with pytest.raises(ValueError, match="argmax"):
        parse_primary_receipt({
            "answer": "B", "probabilities": {"A": 0.8, "B": 0.2},
            "observed_evidence": [], "unresolved_uncertainties": [],
        }, ("A", "B"))


def test_focused_verifier_status_is_tied_to_entailment_step() -> None:
    payload = {
        "expected_answer": "A",
        "verification_status": "REFUTED",
        "recovery_answer": "B",
        "probabilities": {"A": 0.2, "B": 0.8},
        "expected_answer_proof_steps": [
            {
                "kind": kind,
                "status": "REFUTED" if kind == "ANSWER_ENTAILMENT" else "UNKNOWN",
                "confidence": 0.8,
                "visible_fact": "fact",
            }
            for kind in PROOF_KINDS
        ],
        "supporting_evidence": [],
        "counterevidence": ["contradiction"],
        "unresolved_uncertainties": [],
        "reason": "expected answer contradicted",
    }
    parsed = parse_focused_verification(payload, ("A", "B"), "A")
    assert parsed["verification_status"] == "REFUTED"

    payload["verification_status"] = "OBSERVED"
    with pytest.raises(ValueError, match="determined"):
        parse_focused_verification(payload, ("A", "B"), "A")
