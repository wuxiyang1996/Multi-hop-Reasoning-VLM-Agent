from __future__ import annotations

import csv
import json
from pathlib import Path

from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples


def _fixture(root: Path) -> Path:
    annotation = root / "dataset" / "nextqa"
    annotation.mkdir(parents=True)
    (annotation / "map_vid_vidorID.json").write_text(json.dumps({"123": "0001/123"}))
    fields = [
        "video", "frame_count", "width", "height", "question", "answer",
        "qid", "type", "a0", "a1", "a2", "a3", "a4",
    ]
    with (annotation / "val.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "video": "123", "frame_count": "120", "width": "640", "height": "480",
            "question": "Why did the person open the door?", "answer": "2",
            "qid": "9", "type": "CW", "a0": "to sit", "a1": "to sleep",
            "a2": "to leave", "a3": "to eat", "a4": "to read",
        })
    return root


def test_nextqa_loader_preserves_native_five_way_answer(tmp_path: Path) -> None:
    sample = next(iter_nextqa_samples(nextqa_root=_fixture(tmp_path)))
    assert sample.sample_id == "123.Q9"
    assert sample.question_family == "Causal"
    assert sample.answer == "C"
    assert sample.answer_slots == ("A", "B", "C", "D", "E")
    assert sample.vidor_path == "0001/123"
    assert "C. to leave" in sample.format_question()


def test_nextqa_loader_resolves_nested_video_and_family_filter(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert list(iter_nextqa_samples(
        nextqa_root=root, question_families=("Temporal",),
    )) == []
    video = root / "videos" / "0001" / "123.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    sample = next(iter_nextqa_samples(nextqa_root=root, require_video=True))
    assert sample.video_path == video
