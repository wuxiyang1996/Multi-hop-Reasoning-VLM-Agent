from pathlib import Path

import pytest

from scripts.audit_agqa2_goal_relation_v29_development import run


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_REPORTS = (
    ROOT / "runs/agqa2_query_object_v25_reserve/report.json",
    ROOT / "runs/agqa2_query_object_v28_reserve/report.json",
)


pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in HISTORICAL_REPORTS),
    reason="development replay requires gitignored historical AGQA reports",
)


def test_consumed_reports_qualify_only_future_disjoint_route(tmp_path):
    result = run(
        input_paths=[
            ROOT / "runs/agqa2_query_object_v25_reserve/report.json",
            ROOT / "runs/agqa2_query_object_v28_reserve/report.json",
        ],
        artifact_path=(
            ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
        ),
        confirmation_path=(
            ROOT
            / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
        ),
        output_path=tmp_path / "report.json",
    )
    assert result["status"] == (
        "AGQA2_GOAL_RELATION_V29_DEVELOPMENT_QUALIFIED"
    )
    assert result["rows"] == 150
    assert result["source_vs_target_native"]["wins"] == 4
    assert result["source_vs_target_native"]["losses"] == 0
    assert result["future_route_calibration"]["utility_vs_target_native"][
        "decision"
    ] == "SELECT_SKILL"
    assert result["confirmatory_claim"] is False
    assert result["provider_calls"] == 0


def test_report_keeps_generic_scaffold_and_target_written_controls(tmp_path):
    result = run(
        input_paths=[
            ROOT / "runs/agqa2_query_object_v25_reserve/report.json",
            ROOT / "runs/agqa2_query_object_v28_reserve/report.json",
        ],
        artifact_path=(
            ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
        ),
        confirmation_path=(
            ROOT
            / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
        ),
        output_path=tmp_path / "report.json",
    )
    assert result["source_vs_generic_scaffold"]["left_correct"] == 52
    assert result["source_vs_generic_scaffold"]["right_correct"] == 54
    assert result["source_vs_target_written_equivalent"]["wins"] == 0
    assert result["source_vs_target_written_equivalent"]["losses"] == 0
