from scripts.build_agqa_action_genome_sgdet_query_plans import native_temporal_window
from scripts.compile_agqa_action_genome_query_grounder_v2 import (
    _canonical_probability, _shared_frame_view, _support_confidence,
)
import pytest
from scripts.evaluate_agqa_query_grounder_v2_qualification import _support_threshold


def test_native_temporal_windows_use_frozen_action_evidence() -> None:
    base = {
        "source_frame_count": 100,
        "action_obligations": [{"native_frame_index_view": list(range(20, 31))}],
    }
    assert native_temporal_window({**base, "temporal_operator": "BEFORE"}) == (0, 19)
    assert native_temporal_window({**base, "temporal_operator": "AFTER"}) == (31, 99)
    assert native_temporal_window({**base, "temporal_operator": "WHILE"}) == (20, 30)
    assert native_temporal_window({
        **base, "temporal_operator": "BETWEEN",
        "action_obligations": [
            {"native_frame_index_view": list(range(10, 21))},
            {"native_frame_index_view": list(range(40, 51))},
        ],
    }) == (21, 39)


def test_shared_frame_view_is_union_of_both_frozen_grounders() -> None:
    raw = {
        "sampled_original_frame_indices": [0, 10, 20],
        "selected_frame_sha256s": ["a" * 64, "b" * 64, "c" * 64],
    }
    slowfast = {"presented_frame_receipts": [
        {"native_frame_index": 5, "frame_sha256": "d" * 64},
        {"native_frame_index": 10, "frame_sha256": "e" * 64},
    ]}
    indices, hashes, remap = _shared_frame_view(raw, slowfast)
    assert indices == (0, 5, 10, 20)
    assert hashes == ("a" * 64, "d" * 64, "b" * 64, "c" * 64)
    assert remap == {0: 0, 1: 2, 2: 3}


def test_support_confidence_rejects_relative_winner_when_all_evidence_is_weak() -> None:
    weak = (
        {"candidate_label": "book", "score": .09},
        {"candidate_label": "table", "score": .002},
    )
    strong = (
        {"candidate_label": "book", "score": .90},
        {"candidate_label": "table", "score": .02},
    )
    assert _support_confidence("book", weak, (), .6) == .09
    assert _support_confidence("book", strong, (), .6) > .9 - 1e-9


def test_probability_boundary_normalizes_only_float32_numerical_slack() -> None:
    assert _canonical_probability(1.0000001192092896) == 1.0
    assert _canonical_probability(-1e-8) == 0.0
    with pytest.raises(ValueError, match="materially outside"):
        _canonical_probability(1.01)


def test_qualification_threshold_accepts_both_frozen_protocol_schemas() -> None:
    assert _support_threshold({
        "frozen_grounder": {"candidate_confidence": {"support_threshold": .7}}
    }) == .7
    assert _support_threshold({
        "frozen_grounder": {
            "candidate_confidence": "documented confidence definition",
            "candidate_support_threshold": .675,
        }
    }) == .675
