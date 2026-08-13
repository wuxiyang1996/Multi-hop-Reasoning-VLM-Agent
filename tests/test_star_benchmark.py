from __future__ import annotations

import json
from pathlib import Path

from visual_reasoning_wrapper.benchmarks.star import iter_star_samples


def _fixture(root: Path) -> Path:
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    payload = [{
        "question_id": "Prediction_T1_4",
        "question": "What will the person do next?",
        "video_id": "ABC12",
        "start": 2.5,
        "end": 8.0,
        "answer": "Open the door.",
        "question_program": [{"function": "Future_Actions", "value_input": []}],
        "choices": [
            {"choice_id": 0, "choice": "Sit down.", "choice_program": []},
            {"choice_id": 1, "choice": "Open the door.", "choice_program": []},
            {"choice_id": 2, "choice": "Wash a cup.", "choice_program": []},
            {"choice_id": 3, "choice": "Close a window.", "choice_program": []},
        ],
        "situations": {"000001": {"actions": ["a001"]}},
    }]
    (annotations / "STAR_val.json").write_text(json.dumps(payload))
    return root


def test_star_loader_hides_graph_and_program_by_default(tmp_path: Path) -> None:
    sample = next(iter_star_samples(star_root=_fixture(tmp_path)))
    assert sample.sample_id == "Prediction_T1_4"
    assert sample.question_type == "Prediction"
    assert sample.answer == "B"
    assert sample.answer_slots == ("A", "B", "C", "D")
    assert "Future_Actions" not in sample.format_question()
    assert "situations" not in sample.to_dict()
    oracle = sample.to_dict(include_oracle_graph=True)
    assert oracle["situations"]["000001"]["actions"] == ["a001"]


def test_star_loader_requires_real_video_when_requested(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert list(iter_star_samples(star_root=root, require_video=True)) == []
    video = root / "videos" / "charades" / "ABC12.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    sample = next(iter_star_samples(star_root=root, require_video=True))
    assert sample.video_path == video
