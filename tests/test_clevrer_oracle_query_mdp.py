import pytest

from motif_transfer.clevrer_oracle_query_mdp import (
    official_annotation_to_factual_prediction,
)


ANNOTATION = {
    "object_property": [
        {"object_id": 0, "color": "red", "material": "metal", "shape": "sphere"},
        {"object_id": 1, "color": "blue", "material": "rubber", "shape": "cube"},
    ],
    "motion_trajectory": [{
        "frame_id": 0,
        "objects": [
            {"object_id": 0, "location": [1, 2, 0], "inside_camera_view": True},
            {"object_id": 1, "location": [3, 4, 0], "inside_camera_view": False},
        ],
    }],
    "collision": [{"frame_id": 4, "object_ids": [0, 1]}],
}


def test_official_factual_annotation_converts_without_answer_or_program():
    result = official_annotation_to_factual_prediction(
        ANNOTATION, question_family="explanatory",
    )
    assert result["predictions"][0]["what_if"] == -1
    assert len(result["predictions"][0]["trajectory"][0]["objects"]) == 1
    assert result["predictions"][0]["collisions"][0]["objects"][1]["color"] == "blue"
    assert result["authority"]["answer_read"] is False
    assert result["authority"]["counterfactual_rollout_synthesized"] is False


@pytest.mark.parametrize("family", ["predictive", "counterfactual"])
def test_factual_adapter_refuses_missing_future_or_counterfactual_authority(family):
    with pytest.raises(ValueError, match="cannot supply"):
        official_annotation_to_factual_prediction(ANNOTATION, question_family=family)
