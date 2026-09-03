from scripts.freeze_agqa_query_grounder_v2_powered_qualification import _select


def test_select_is_deterministic_and_balanced_per_video():
    candidates = {
        "V1": [
            {"task_id": "V1-A", "video_id": "V1"},
            {"task_id": "V1-B", "video_id": "V1"},
            {"task_id": "V1-C", "video_id": "V1"},
        ],
        "V2": [
            {"task_id": "V2-A", "video_id": "V2"},
            {"task_id": "V2-B", "video_id": "V2"},
        ],
        "TOO_SMALL": [{"task_id": "X", "video_id": "TOO_SMALL"}],
    }
    videos_a, rows_a = _select(
        candidates, videos=2, tasks_per_video=2, salt="frozen",
    )
    videos_b, rows_b = _select(
        candidates, videos=2, tasks_per_video=2, salt="frozen",
    )
    assert videos_a == videos_b
    assert rows_a == rows_b
    assert set(videos_a) == {"V1", "V2"}
    assert len(rows_a) == 4
    assert all(
        sum(row["video_id"] == video_id for row in rows_a) == 2
        for video_id in videos_a
    )
