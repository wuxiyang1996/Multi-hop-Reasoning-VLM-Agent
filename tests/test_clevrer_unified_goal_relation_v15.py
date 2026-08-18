from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def test_v15_prospective_report_passes_independent_audit(tmp_path: Path):
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/audit_clevrer_unified_goal_relation_v15.py"),
            "--output", str(output),
        ],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == (
        "CLEVRER_V15_PROSPECTIVE_RESERVE_VALIDATED_WITH_TARGET_BASE_AMBIGUITY"
    )
    assert all(summary["gates"].values())
    assert summary["primary"]["authentic"]["correct"] == 252
    assert summary["primary"]["neural_only"]["correct"] == 236
    assert summary["warnings"]["target_base_statistically_indistinguishable"]
    assert summary["cost"]["external_provider_cost_usd"] == 0.0
    assert result.returncode == 0


def test_v15_portable_gzip_artifact_passes_same_audit(tmp_path: Path):
    output = tmp_path / "portable-summary.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/audit_clevrer_unified_goal_relation_v15.py"),
            "--report",
            str(REPO / "artifacts/video_event_graph_v15/formal_report.json.gz"),
            "--output", str(output),
        ],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == (
        "CLEVRER_V15_PROSPECTIVE_RESERVE_VALIDATED_WITH_TARGET_BASE_AMBIGUITY"
    )
    assert summary["lineage"]["formal_report_sha256"] == (
        "bc788b73808be87b7f8b51428e6c208b0733ebfdb9f77dab3607805951414491"
    )
