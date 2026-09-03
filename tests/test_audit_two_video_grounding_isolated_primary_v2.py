from scripts.audit_two_video_grounding_isolated_primary_v2 import audit


def test_two_video_v2_audit_is_grounding_isolated_and_keeps_raw_failures():
    result = audit()
    assert result["status"] == "PASSED"
    assert all(result["gates"].values())
    assert result["primary"]["agqa2"]["grounding_induced_source_losses"] == 0
    assert result["primary"]["clevrer"]["grounding_induced_source_losses_vs_neural"] == 0
    assert result["secondary_raw_video"]["agqa2_qwen235_multiclass_v2"]["status"] == "POST_V74_EXPLORATORY_FAILED"
