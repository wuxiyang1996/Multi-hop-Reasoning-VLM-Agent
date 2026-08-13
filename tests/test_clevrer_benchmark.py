from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_reasoning_wrapper.benchmarks.clevrer import (
    iter_clevrer_choice_samples,
    load_clevrer_scenes,
)


def _fixture(root: Path) -> Path:
    questions = root / "questions"
    questions.mkdir(parents=True)
    payload = [{
        "scene_index": 10000,
        "video_filename": "video_10000.mp4",
        "questions": [{
            "question_id": 7,
            "question": "Which event will happen next?",
            "question_type": "predictive",
            "program": ["unseen_events", "belong_to"],
            "choices": [
                {
                    "choice_id": 0,
                    "choice": "The cube collides with the sphere",
                    "program": ["filter_collision"],
                    "answer": "correct",
                },
                {
                    "choice_id": 1,
                    "choice": "The cylinder exits",
                    "program": ["filter_out"],
                    "answer": "wrong",
                },
            ],
        }],
    }]
    (questions / "validation.json").write_text(json.dumps(payload))
    return root


def test_choice_loader_preserves_binary_official_labels(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    samples = list(iter_clevrer_choice_samples(clevrer_root=root))

    assert [sample.sample_id for sample in samples] == [
        "video_10000.mp4.Q7.C0",
        "video_10000.mp4.Q7.C1",
    ]
    assert [sample.answer for sample in samples] == ["A", "B"]
    assert samples[0].answer_slots == ("A", "B")
    assert "functional" not in samples[0].format_question().lower()
    assert "filter_collision" not in samples[0].format_question()
    assert "question_program" not in samples[0].to_dict()
    assert samples[0].to_dict(include_oracle_programs=True)["choice_program"] == [
        "filter_collision"
    ]


def test_choice_loader_filters_by_id_and_requires_video(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert list(iter_clevrer_choice_samples(
        clevrer_root=root,
        sample_ids=["video_10000.mp4.Q7.C1"],
        require_video=True,
    )) == []

    video = root / "videos" / "validation" / "video_10000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    samples = list(iter_clevrer_choice_samples(
        clevrer_root=root,
        sample_ids=["video_10000.mp4.Q7.C1"],
        require_video=True,
    ))
    assert len(samples) == 1
    assert samples[0].video_path == video


def test_loader_rejects_descriptive_choice_mode(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    with pytest.raises(ValueError, match="causal question types"):
        list(iter_clevrer_choice_samples(
            clevrer_root=root, question_types=["descriptive"],
        ))


def test_load_scenes_accepts_val_alias(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert load_clevrer_scenes("val", clevrer_root=root)[0]["scene_index"] == 10000
