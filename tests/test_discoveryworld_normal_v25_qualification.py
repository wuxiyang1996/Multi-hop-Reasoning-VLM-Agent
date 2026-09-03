from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit_discoveryworld_normal_v25_qualification.py"


def _module():
    spec = importlib.util.spec_from_file_location("dw_normal_v25", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_formal_fails_closed_after_acquisition_repair():
    report = _module().build_report()

    assert report["status"] == "DISCOVERYWORLD_NORMAL_FORMAL_REMAINS_BLOCKED"
    assert report["formal_reserve_authorized"] is False
    assert report["development_evidence"]["commit_coverage"] == 2
    assert report["development_evidence"]["official_successes"] == 2
    assert report["development_evidence"]["target_native_myopic_successes"] == 1
    assert report["development_evidence"]["authentic_source_successes"] == 0
    assert "target_comparator_has_positive_success_headroom_at_fork" in report[
        "failed_qualification_gates"
    ]
    assert "target_relation_vocabulary_represents_required_adjacency" in report[
        "failed_qualification_gates"
    ]
