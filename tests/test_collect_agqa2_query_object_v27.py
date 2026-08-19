from scripts.collect_agqa2_query_object_v27 import _relabel


def test_v27_relabel_does_not_change_paired_metrics():
    original = {
        "status": "AGQA2_QUERY_OBJECT_V26_SOURCE_SPECIFIC_QUALIFIED",
        "source_specific_metrics": {"source_vs_target_only_wins": 5},
        "report_sha256": "old",
    }
    result = _relabel(original)
    assert result["status"] == "AGQA2_QUERY_OBJECT_V27_SOURCE_SPECIFIC_QUALIFIED"
    assert result["source_specific_metrics"] == original["source_specific_metrics"]
    assert result["v26_paired_outcome_calculation_unchanged"] is True
