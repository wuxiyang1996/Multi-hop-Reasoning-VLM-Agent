from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit_discoveryworld_normal_source_transfer_v26.py"


def _module():
    spec = importlib.util.spec_from_file_location("normal_v26_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v26_independent_audit_passes_and_rejects_success_gain_claim():
    audit = json.loads((
        REPO / "docs/results/discoveryworld_normal_source_transfer_v26_audit.json"
    ).read_text())
    body = dict(audit)
    assert body.pop("audit_sha256") == stable_hash(body)
    assert audit["status"] == "DISCOVERYWORLD_NORMAL_V26_INDEPENDENT_AUDIT_PASSED"
    assert audit["all_audit_gates_passed"] is True
    assert all(audit["gates"].values())
    assert audit["metrics"]["authentic_source_successes"] == 24
    assert audit["metrics"]["neural_only_successes"] == 24
    assert audit["metrics"]["source_permuted_successes"] == 0
    assert audit["interpretation"]["program_transfer_validated"] is True
    assert audit["interpretation"]["incremental_success_rate_gain_validated"] is False
    assert audit["interpretation"]["source_provenance_identifiable_from_behavior_alone"] is False


def test_v26_audit_rebuilds_when_raw_formal_receipts_are_available():
    raw = REPO / "runs/discoveryworld_normal_source_transfer_v26_formal/formal_report.json"
    if raw.is_file():
        assert _module().build_audit()["all_audit_gates_passed"] is True
