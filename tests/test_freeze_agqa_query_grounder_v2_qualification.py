from scripts.freeze_agqa_query_grounder_v2_qualification import (
    _content_addressed_video_ids,
)


def test_frame_evidence_is_associated_with_its_enclosing_video_only():
    mixed = {
        "rows": [
            {
                "video_id": "SEEN1",
                "receipt": {"selected_frame_sha256s": ["abc"]},
            },
            {"video_id": "UNSEEN", "question": "public metadata only"},
        ],
        "video_receipts": [
            {"video_id": "HASHED_ONLY", "video_sha256": "def"},
        ],
    }
    assert _content_addressed_video_ids(mixed) == {"SEEN1"}


def test_presented_frame_receipt_inherits_parent_video_id():
    row = {
        "video_id": "SEEN2",
        "model_output": {
            "presented_frame_receipts": [
                {"native_frame_index": 3, "frame_sha256": "abc"},
            ],
        },
    }
    assert _content_addressed_video_ids(row) == {"SEEN2"}
