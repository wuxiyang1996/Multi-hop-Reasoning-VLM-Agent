from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from motif_transfer.anonymous_video_harness import compile_anonymous_source_controller


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/audit_agqa_anonymous_harness_substitution_v1.py"
    spec = importlib.util.spec_from_file_location("agqa_anonymous_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_both_consumed_agqa_results_are_equivalent_under_anonymous_controller() -> None:
    module = _module()
    controller = compile_anonymous_source_controller(
        root=ROOT,
        lineage_directory=Path("runs/phase3_source_induction_v1_development/lineages"),
    )
    cases = [
        ("broad", "runs/agqa2_layer_b_raw_video_v1/qualification_v4/five_arm_epistemic_qualification_full512_v4.json", 232),
        ("temporal", "runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/five_arm_typed_temporal_full256_v1.json", 107),
    ]
    for kind, relative, expected_commits in cases:
        result = json.loads((ROOT / relative).read_text())
        audit = module.audit_one(controller, result, kind=kind)
        assert all(audit["gates"].values())
        assert audit["anonymous_commits"] == expected_commits
        assert not audit["mismatches"]
