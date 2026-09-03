from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/qualify_discoveryworld_normal_source_value_v26.py"


def _module():
    spec = importlib.util.spec_from_file_location("normal_v26_qualification", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v26_qualification_passes_without_opening_formal_reserve():
    report = json.loads((
        REPO / "docs/results/discoveryworld_normal_source_value_v26_qualification.json"
    ).read_text())
    grounder = json.loads((
        REPO / "docs/results/discoveryworld_normal_source_value_v26_grounder.json"
    ).read_text())
    report_body = dict(report)
    assert report_body.pop("report_sha256") == stable_hash(report_body)
    grounder_body = dict(grounder)
    assert grounder_body.pop("grounder_sha256") == stable_hash(grounder_body)
    assert report["status"] == "DISCOVERYWORLD_NORMAL_V26_QUALIFIED"
    assert report["all_qualification_gates_passed"] is True
    assert all(report["gates"].values())
    assert report["metrics"]["qualification_official_successes"] == 8
    assert report["metrics"]["grounding"]["exact_accuracy"] == 1.0
    assert report["metrics"]["complete_target_trajectories_replaced"] == 1
    assert report["target_only_induction_curve"][0]["status"].startswith("ABSTAIN")
    assert report["target_only_induction_curve"][1]["matches_source_phase_program"] is True
    assert report["lineage"]["formal_seeds_read"] == []
    assert report["gates"]["formal_reserve_still_sealed"] is True
    assert grounder["outcome_fields_used_at_inference"] is False


def test_v26_qualification_rebuilds_when_raw_receipts_are_available():
    raw = REPO / "runs/discoveryworld_normal_source_value_v26_development"
    if raw.is_dir():
        report, _ = _module().build_report()
        assert report["all_qualification_gates_passed"] is True
