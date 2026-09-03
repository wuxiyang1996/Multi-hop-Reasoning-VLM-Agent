from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit_phase9_source_program_heterogeneity_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase9_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase9_audit_selects_three_programs_and_passes_route_controls():
    report = _module().build_report()

    assert report["status"] == (
        "PHASE9_SOURCE_PROGRAM_HETEROGENEITY_AND_TARGET_UTILITY_VALIDATED"
    )
    assert report["source_catalog_size"] == 11
    assert report["selected_distinct_programs"] == 3
    assert len(report["route_audits"]) == 4
    assert all(report["gates"].values())
    assert report["descriptive_across_routes"]["wins"] == 38
    assert report["descriptive_across_routes"]["losses"] == 0
    assert report["descriptive_across_routes"][
        "pooled_iid_pvalue_reported"
    ] is False


def test_phase9_wrong_source_family_is_fail_closed():
    report = _module().build_report()

    for row in report["route_audits"]:
        assert row["wrong_family_selection"]["status"] == (
            "SOURCE_CONTRACT_SELECTION_ABSTAINED"
        )
        assert row["selection"]["source_identity_used_as_feature"] is False
        assert row["selection"]["target_action_emitted"] is False
        assert row["selection"]["target_outcome_read"] is False
