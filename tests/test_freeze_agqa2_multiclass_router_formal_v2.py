from scripts.freeze_agqa2_multiclass_router_formal_v2 import best_per_video


def test_best_per_video_uses_score_then_frozen_rank():
    rows = [
        {"video_id": "v1", "router_score": 0.9, "rank_sha256": "a", "task_id": "a"},
        {"video_id": "v1", "router_score": 0.95, "rank_sha256": "0", "task_id": "b"},
        {"video_id": "v2", "router_score": 0.8, "rank_sha256": "z", "task_id": "c"},
    ]
    selected = best_per_video(rows)
    assert {row["task_id"] for row in selected} == {"b", "c"}
    assert len({row["video_id"] for row in selected}) == 2
