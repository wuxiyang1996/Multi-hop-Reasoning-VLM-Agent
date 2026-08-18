import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from freeze_agqa2_active_grounding_v3_reserve import freeze_reserve  # noqa: E402


def test_reserve_reuses_exact_grounder_and_frozen_gates(tmp_path):
    summary_path = tmp_path / "summary.json"
    reserve_path = tmp_path / "reserve.json"
    summary, reserve = freeze_reserve(
        development_config_path=(
            REPO / "configs/agqa2_active_grounding_v3_development.json"
        ),
        development_report_path=(
            REPO / "runs/agqa2_active_grounding_v3_development/report.json"
        ),
        summary_path=summary_path,
        reserve_config_path=reserve_path,
    )
    development = json.loads((
        REPO / "configs/agqa2_active_grounding_v3_development.json"
    ).read_text())
    preregistration = json.loads((
        REPO / "configs/agqa2_active_grounding_v3_preregistration.json"
    ).read_text())
    for key in (
        "grounder", "model", "parser_model", "rescan_model",
        "nonrecurrent_model", "local_object_grounder", "media",
        "acquisition", "sources",
    ):
        assert reserve[key] == development[key]
    assert reserve["qualification_gates"] == preregistration["reserve_gates"]
    assert summary["grounder_qualified"] is True
    assert all(summary["qualification_gates"].values())


def test_reserve_freeze_rejects_unqualified_dependency(tmp_path):
    report = json.loads((
        REPO / "runs/agqa2_active_grounding_v3_development/report.json"
    ).read_text())
    report["grounder_qualified"] = False
    report.pop("report_sha256")
    from motif_transfer.contracts import stable_hash
    report["report_sha256"] = stable_hash(report)
    bad_report = tmp_path / "bad.json"
    bad_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not qualified"):
        freeze_reserve(
            development_config_path=(
                REPO / "configs/agqa2_active_grounding_v3_development.json"
            ),
            development_report_path=bad_report,
            summary_path=tmp_path / "summary.json",
            reserve_config_path=tmp_path / "reserve.json",
        )
