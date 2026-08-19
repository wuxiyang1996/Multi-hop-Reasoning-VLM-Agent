from pathlib import Path

import pytest

from scripts.audit_agqa2_robust_temporal_v33_development import run


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "runs/agqa2_active_grounding_v17_powered_reserve/report.json",
    ROOT / "runs/agqa2_temporal_selective_v19_reserve/report.json",
)


pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in INPUTS),
    reason="development replay requires gitignored historical AGQA reports",
)


def test_consumed_temporal_pair_reanalysis_qualifies_future_only(tmp_path):
    report = run(input_paths=INPUTS, output_path=tmp_path / "report.json")
    assert report["status"] == (
        "AGQA2_ROBUST_TEMPORAL_V33_DEVELOPMENT_QUALIFIED"
    )
    assert report["confirmatory_claim"] is False
    assert report["rows"] == 35
    assert report["source_authorizations"] == 12
    assert report["source_vs_target_native"]["wins"] == 4
    assert report["source_vs_target_native"]["losses"] == 0
    assert report["future_route_calibration"]["decision"] == "SELECT_SKILL"
    assert all(report["qualification_gates"].values())
    assert report["provider_calls"] == 0
