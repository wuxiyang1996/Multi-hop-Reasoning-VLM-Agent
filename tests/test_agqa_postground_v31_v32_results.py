import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _summary():
    return json.loads((
        ROOT
        / "docs/results/agqa2_postground_neurosymbolic_v29_v32_summary.json"
    ).read_text())


def test_v31_development_calibration_is_future_only_and_qualified():
    report = _summary()["historical_sequence"]["v31_development"]
    assert report["status"] == "AGQA2_POSTGROUND_V31_DEVELOPMENT_QUALIFIED"
    assert report["wins"] == 15
    assert report["losses"] == 1


def test_v32_fresh_formal_passes_all_frozen_gates():
    report = _summary()["formal_v32"]
    assert report["status"] == "AGQA2_POSTGROUND_V32_FORMAL_QUALIFIED"
    assert report["rows"] == 120
    assert report["source_authorizations"] == 56
    assert report["source_correct"] == 40
    assert report["target_native_correct"] == 34
    assert report["source_minus_target_correct"] == 6
    assert report["source_wins"] == 6
    assert report["source_losses"] == 0
    assert report["source_vs_target_exact_one_sided_pvalue"] == 0.015625
    assert report["all_gates_passed"] is True
    assert report["reported_provider_cost_usd"] < 1.30


def test_v32_source_matches_ceiling_but_shuffled_source_matches_target():
    summary = _summary()
    report = summary["formal_v32"]
    interpretation = summary["interpretation"]
    assert report["source_correct"] == report["generic_scaffold_correct"]
    assert report["source_correct"] == report["target_written_equivalent_correct"]
    assert interpretation["source_induced_program_matched_handwritten_ceiling"]
    assert not interpretation["source_effect_shuffled_improved_target"]
