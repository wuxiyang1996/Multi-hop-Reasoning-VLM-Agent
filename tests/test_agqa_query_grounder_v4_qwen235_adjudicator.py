from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    candidate_pool, response_format, selected_frame_ids,
)


def test_candidate_pool_is_label_unique_ranked_and_budgeted():
    row = {"candidate_ranking": [
        {"candidate_label": "cup", "track_id": "T1", "score": 1.0,
         "sources": ["sgdet"], "evidence_frames": [5]},
        {"candidate_label": "cup", "track_id": "T2", "score": 0.9,
         "sources": ["slowfast"], "evidence_frames": [6]},
        {"candidate_label": "book", "track_id": "T3", "score": 0.8,
         "sources": ["sgdet"], "evidence_frames": [7]},
        {"candidate_label": "door", "track_id": None, "score": 0.7,
         "sources": ["slowfast"], "evidence_frames": []},
    ]}
    assert candidate_pool(row, 2) == [
        {"candidate_id": "C0", "label": "cup", "track_id": "T1",
         "frozen_rank": 1, "frozen_fusion_score": 1.0,
         "sources": ["sgdet"], "base_evidence_frames": [5]},
        {"candidate_id": "C1", "label": "book", "track_id": "T3",
         "frozen_rank": 2, "frozen_fusion_score": 0.8,
         "sources": ["sgdet"], "base_evidence_frames": [7]},
    ]


def test_selected_frames_respect_scope_and_budget():
    candidates = [
        {"base_evidence_frames": [12]},
        {"base_evidence_frames": [17]},
        {"base_evidence_frames": [40]},
    ]
    frames = selected_frame_ids(
        lower=10, upper=20, candidates=candidates, frame_count=64, maximum=8,
    )
    assert frames == sorted(set(frames))
    assert {10, 15, 20, 12, 17} <= set(frames)
    assert 40 not in frames
    assert len(frames) <= 8


def test_response_schema_avoids_unsupported_unique_items_keyword():
    schema = response_format(["C0"], [1, 2])
    evidence = schema["json_schema"]["schema"]["properties"]["evidence_frame_ids"]
    assert "uniqueItems" not in evidence
    assert evidence["items"]["enum"] == [1, 2]
