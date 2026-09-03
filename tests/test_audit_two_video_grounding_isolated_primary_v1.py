from scripts.audit_two_video_grounding_isolated_primary_v1 import audit


def test_frozen_two_video_primary_is_grounding_isolated() -> None:
    result = audit()
    assert result["status"] == "PASSED"
    assert all(result["gates"].values())
    assert result["agqa2"]["grounding_induced_source_losses"] == 0
    assert result["clevrer"]["grounding_induced_source_losses_vs_neural"] == 0
    assert result["grounding_policy"]["vlm_grounder_used_in_primary_arm_delta"] is False
    assert result["grounding_policy"]["same_grounding_receipt_or_backend_for_all_matched_arms"] is True
