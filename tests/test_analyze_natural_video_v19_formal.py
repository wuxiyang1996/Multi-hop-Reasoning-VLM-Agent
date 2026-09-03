from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_natural_video_v19_formal.py"
SPEC = importlib.util.spec_from_file_location("analyze_natural_video_v19_formal", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(sample: str, family: str, recover: bool, compatible: bool = True) -> dict:
    return {
        "benchmark": "star",
        "sample_id": sample,
        "video_id": sample.split("_")[0],
        "family": family,
        "source_compatible": compatible,
        "authentic_recover": recover,
        "primary": {"answer": "0"},
        "proof": {"answer": "1"},
        "gold_answer": "1",
    }


def test_fixed_controls_preserve_recovery_counts_and_compatibility() -> None:
    rows = [
        _row("a_1", "Interaction", True),
        _row("a_2", "Interaction", False),
        _row("b_1", "Sequence", True),
        _row("b_2", "Sequence", False),
        _row("c_1", "Prediction", False, False),
    ]
    masks = MODULE.build_fixed_control_masks(rows, shuffled_seed=7, marginal_seed=11)
    for mask in masks.values():
        assert sum(mask) == 2
        assert not mask[-1]
    shuffled = masks["within_benchmark_family_shuffled_recovery_mask"]
    assert sum(shuffled[:2]) == 1
    assert sum(shuffled[2:4]) == 1


def test_cluster_metrics_resamples_video_clusters() -> None:
    rows = [
        _row("a_1", "Interaction", True),
        _row("a_2", "Interaction", True),
        _row("b_1", "Interaction", False),
    ]
    metrics = MODULE.cluster_metrics(
        rows, [True, True, False], [False, False, True], resamples=200, seed=3,
    )
    assert metrics["clusters"] == 2
    assert metrics["positive"] == 1
    assert metrics["negative"] == 1
    assert metrics["by_benchmark"]["star"]["questions"] == 3
