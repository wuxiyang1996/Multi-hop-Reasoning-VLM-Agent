from scripts.collect_agqa2_query_object_v28 import _relabel


def test_v28_relabel_preserves_metrics():
    source_metrics = {"source_vs_target_only_wins": 7}
    result = _relabel({
        "status": "AGQA2_QUERY_OBJECT_V26_SOURCE_SPECIFIC_QUALIFIED",
        "source_specific_metrics": source_metrics,
        "report_sha256": "old",
    })
    assert result["status"] == "AGQA2_QUERY_OBJECT_V28_SOURCE_SPECIFIC_QUALIFIED"
    assert result["source_specific_metrics"] == source_metrics
    assert result["decision_confidence_evidence_schema_unchanged"] is True
