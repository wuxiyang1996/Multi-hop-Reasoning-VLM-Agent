from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from motif_transfer.clevrer_cv_tool_grounder import ground_clevrer_video


ROOT = Path(__file__).resolve().parents[1]


def test_raw_video_grounder_is_question_independent_and_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (160, 96))
    assert writer.isOpened()
    for index in range(32):
        frame = np.full((96, 160, 3), 120, np.uint8)
        cv2.circle(frame, (8 + 4 * index, 32), 8, (30, 30, 210), -1)
        cv2.rectangle(frame, (144 - 3 * index, 58), (156 - 3 * index, 72), (210, 40, 30), -1)
        writer.write(frame)
    writer.release()
    config = json.loads((ROOT / "configs/clevrer_cv_tool_grounder_v1.json").read_text())
    receipt = ground_clevrer_video(path, config)
    assert receipt.provider_calls == 0
    assert receipt.question_read is False
    assert receipt.official_annotation_read is False
    assert receipt.answer_read is False
    assert receipt.source_controller_read is False
    assert len(receipt.selected_frame_indices) == config["frame_budget"]
    assert len(receipt.tracks) >= 2
    assert len(receipt.receipt_sha256) == 64
