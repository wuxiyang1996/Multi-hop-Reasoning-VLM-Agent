from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _module():
    path = REPO / "scripts/analyze_minigrid_orientation_target_v31.py"
    spec = importlib.util.spec_from_file_location("analyze_v31", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_v31_summary_closes_prospective_target_gap():
    report = _module().analyze(
        REPO / "configs/minigrid_orientation_target_v31.json"
    )
    assert report["status"] == "PROSPECTIVE_V28_TARGET_RESERVE_VALIDATED"
    assert all(report["gates"].values())
    formal = report["stages"]["formal_reserve"]
    assert formal["source_induced_success"] == 48
    assert formal["neural_only_success"] == 16
    assert formal["source_vs_neural_only"]["wins"] == 32
    assert formal["source_vs_neural_only"]["losses"] == 0
    assert formal["target_written_isomorphic_success"] == 48
