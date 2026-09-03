from __future__ import annotations

import pytest

from scripts.collect_natural_video_active_recovery_v18 import _decode_json_object

from motif_transfer.natural_video_active_recovery import (
    authentic_recovery_decision,
    parse_active_arbitration,
    parse_active_probe,
    source_compatible,
)
from motif_transfer.natural_video_recovery import PROOF_KINDS


def _proof(claim_id: str, final: str) -> dict:
    return {
        "claim_id": claim_id,
        "proof_steps": [
            {
                "kind": kind,
                "status": final if kind == "ANSWER_ENTAILMENT" else "SUPPORTED",
                "confidence": 0.8,
                "visible_fact": kind,
            }
            for kind in PROOF_KINDS
        ],
    }


def test_source_compatibility_is_semantic_not_family_tuned() -> None:
    assert source_compatible("star", "Interaction")
    assert source_compatible("star", "Sequence")
    assert not source_compatible("star", "Prediction")
    assert not source_compatible("star", "Feasibility")
    assert source_compatible("nextqa", "Causal")
    assert source_compatible("nextqa", "Temporal")
    assert not source_compatible("nextqa", "Descriptive")


def test_active_probe_is_bound_and_budgeted() -> None:
    payload = {
        "claim_ids": ["C0", "C1"],
        "tool": "sample_frames",
        "predicate_kind": "TEMPORAL_ORDER",
        "start_sec": 2.0,
        "end_sec": 6.0,
        "expected_facts": {"C0": "door then book", "C1": "book then door"},
        "why_discriminative": "tests order",
    }
    parsed = parse_active_probe(
        payload,
        claim_ids=("C0", "C1"),
        duration_seconds=20,
        frames_per_probe=24,
        maximum_window_fraction=0.5,
        maximum_window_seconds=12,
    )
    assert parsed["arguments"] == {"n": 24, "start_sec": 2.0, "end_sec": 6.0}
    with pytest.raises(ValueError, match="exceeds"):
        parse_active_probe(
            {**payload, "end_sec": 15.0},
            claim_ids=("C0", "C1"),
            duration_seconds=20,
            frames_per_probe=24,
            maximum_window_fraction=0.5,
            maximum_window_seconds=12,
        )


def test_active_recovery_requires_refute_support_and_answer_agreement() -> None:
    payload = {
        "answer": "B",
        "probabilities": {"A": 0.2, "B": 0.8},
        "candidate_proofs": [_proof("C0", "REFUTED"), _proof("C1", "SUPPORTED")],
        "observed_evidence": ["visible event"],
        "unresolved_uncertainties": [],
        "reason": "focused evidence",
    }
    parsed = parse_active_arbitration(
        payload, slots=("A", "B"), claim_ids=("C0", "C1"),
    )
    assert authentic_recovery_decision(
        parsed,
        claim_to_slot={"C0": "A", "C1": "B"},
        primary_slot="A",
        alternative_slot="B",
    )
    parsed["answer"] = "A"
    assert not authentic_recovery_decision(
        parsed,
        claim_to_slot={"C0": "A", "C1": "B"},
        primary_slot="A",
        alternative_slot="B",
    )


def test_provider_transport_normalization_only_removes_code_fence() -> None:
    payload, wrapper = _decode_json_object('```json\n{"answer":"A"}\n```')
    assert payload == {"answer": "A"}
    assert wrapper["kind"] == "markdown_code_fence"
    assert wrapper["prefix_chars"] == "7"
    assert wrapper["suffix_chars"] == "3"
    payload, wrapper = _decode_json_object('Here is JSON:\n{"answer":"A"}')
    assert payload == {"answer": "A"}
    assert wrapper["kind"] == "bounded_outer_text"
    assert wrapper["prefix_chars"] == "13"
    payload, wrapper = _decode_json_object('[{"answer":"A"}]')
    assert payload == {"answer": "A"}
    assert wrapper["kind"] == "singleton_list"
    with pytest.raises(ValueError, match="one bounded JSON object"):
        _decode_json_object('First {"answer":"A"}, then {"answer":"B"}')
