from __future__ import annotations

from scripts.freeze_clevrer_full_raw_video_v1 import _contains_forbidden, _public_projection


def test_public_projection_strips_answers_and_programs() -> None:
    raw = {
        "question_id": 3, "question": "Which event happens next?",
        "question_type": "predictive", "program": ["oracle"],
        "choices": [{"choice_id": 0, "choice": "A collision", "program": ["x"], "answer": "correct"}],
    }
    public = _public_projection(raw)
    assert public == {
        "question_id": 3, "question": "Which event happens next?",
        "question_type": "predictive", "question_subtype": "",
        "choices": [{"choice_id": 0, "choice": "A collision"}],
    }
    assert not _contains_forbidden(public)
