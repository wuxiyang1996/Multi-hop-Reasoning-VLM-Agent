from scripts.fuse_agqa_slowfast_multiresolution import (
    fuse_bindings,
    fuse_query_plans,
)


def _plan(view, score):
    return {
        "report_sha256": str(view),
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "rows": [{
            "task_id": "q1", "predicate": "holding", "temporal_operator": "BEFORE",
            "source_frame_count": 100, "status": "QUERY_PLAN_FROZEN",
            "inspection_indices": list(view), "native_frame_index_views": [list(view)],
            "action_obligations": [{
                "slot_id": "s1", "phrase": "holding", "argmax_window": 0,
                "max_score": score, "native_frame_index_view": list(view),
            }],
        }],
    }


def _binding(view, score):
    return {
        "report_sha256": str(view),
        "routing_rule": "typed", "scoring_rule": "max",
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "action_grounding_file_sha256": str(view),
        "rows": [{
            "task_id": "q1", "predicate": "holding", "video_id": "v1",
            "video_sha256": "video", "source_frame_count": 100,
            "native_frame_index_views": [list(view)],
            "presented_frame_receipts": [
                {"native_frame_index": x, "frame_sha256": f"h{x}"} for x in view
            ],
            "status": "BOUND", "top_candidate": "book",
            "candidates": [{"candidate_label": "book", "action_score": score}],
        }],
    }


def test_fusion_preserves_uniform_scores_and_uses_dense_localization():
    plans = fuse_query_plans(_plan((0, 50), .8), _plan((20, 21), .2))
    obligation = plans["rows"][0]["action_obligations"][0]
    assert obligation["max_score"] == .8
    assert obligation["native_frame_index_view"] == [0, 50]
    assert obligation["localization_native_frame_index_view"] == [20, 21]
    assert obligation["localization_sampling"] == "dense10x32"

    bindings = fuse_bindings(
        _binding((0, 50), .8), _binding((20, 21), .2),
        fused_plan_sha256="plan",
    )
    row = bindings["rows"][0]
    assert row["candidates"][0]["action_score"] == .8
    assert row["native_frame_index_views"] == [[20, 21]]
    assert [x["native_frame_index"] for x in row["presented_frame_receipts"]] == [0, 20, 21, 50]
    assert bindings["frame_presentation_budget"] == 4


def test_fusion_fails_on_authority_violation():
    uniform = _plan((0, 50), .8)
    uniform["target_outcome_read"] = True
    try:
        fuse_query_plans(uniform, _plan((20, 21), .2))
    except ValueError as error:
        assert "authority boundary" in str(error)
    else:
        raise AssertionError("authority violation was accepted")
