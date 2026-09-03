"""Content-addressed raw-video receipts for official frozen NS-DR outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import cv2
import numpy as np

from .contracts import stable_hash


@dataclass(frozen=True)
class CLEVRERNSDRReceipt:
    video_id: int
    video_sha256: str
    selected_frame_indices: tuple[int, ...]
    selected_frame_sha256s: tuple[str, ...]
    prediction_sha256: str
    prediction_payload_sha256: str
    grounder_config_sha256: str
    object_count: int
    prediction_world_count: int
    observed_world_present: bool
    counterfactual_worlds_complete: bool
    observed_collision_count: int
    provider_calls: int
    cached_off_the_shelf_model_output: bool
    question_read: bool
    processed_proposals_read: bool
    official_annotation_read: bool
    functional_program_read: bool
    answer_read: bool
    source_controller_read: bool
    receipt_sha256: str


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha(frame: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(frame))).hexdigest()


def _video_id(video_path: Path) -> int:
    match = re.fullmatch(r"video_(\d+)\.mp4", video_path.name)
    if match is None:
        raise ValueError("CLEVRER raw video filename must be video_N.mp4")
    return int(match.group(1))


def _sample_raw_frames(video_path: Path, budget: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release(); raise ValueError("CLEVRER video has no decodable frames")
    indices = tuple(int(x) for x in np.linspace(0, total - 1, min(total, budget), dtype=int))
    hashes = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release(); raise ValueError(f"cannot decode raw-video evidence frame {index}")
        hashes.append(_frame_sha(frame))
    capture.release()
    return indices, tuple(hashes)


def load_prediction_payload(path: Path) -> Mapping[str, Any]:
    """Load the prediction-only NS-DR schema and reject oracle-like fields."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"objects", "predictions"}:
        raise ValueError("NS-DR prediction schema contains unexpected fields")
    forbidden = {"answer", "annotation", "ground_truth", "program", "questions"}
    if forbidden & {str(key).casefold() for key in payload}:
        raise ValueError("NS-DR payload crossed the oracle boundary")
    objects = payload.get("objects")
    predictions = payload.get("predictions")
    if not isinstance(objects, list) or not isinstance(predictions, list):
        raise ValueError("NS-DR objects/predictions must be lists")
    expected_worlds = {-1, *range(len(objects))}
    actual_worlds = {int(row["what_if"]) for row in predictions}
    if actual_worlds != expected_worlds:
        raise ValueError("NS-DR observed/counterfactual world inventory is incomplete")
    return payload


def bind_cached_nsdr_prediction(
    *, video_path: Path, prediction_path: Path, config: Mapping[str, Any],
) -> CLEVRERNSDRReceipt:
    if config.get("status") != "FROZEN_BEFORE_CLEVRER_RAW_VIDEO_RESERVE_SELECTION":
        raise ValueError("NS-DR grounder config is not frozen")
    authority = str(config.get("authority"))
    if "NO_ANNOTATIONS" not in authority or "NO_PROCESSED_PROPOSALS" not in authority:
        raise ValueError("NS-DR grounder authority drift")
    video_id = _video_id(video_path)
    if prediction_path.name != f"sim_{video_id:05d}.json":
        raise ValueError("raw video and cached NS-DR prediction IDs differ")
    payload = load_prediction_payload(prediction_path)
    indices, frame_hashes = _sample_raw_frames(video_path, int(config["frame_budget"]))
    objects = payload["objects"]
    predictions = payload["predictions"]
    observed = next(row for row in predictions if int(row["what_if"]) == -1)
    body = {
        "video_id": video_id,
        "video_sha256": _file_sha(video_path),
        "selected_frame_indices": indices,
        "selected_frame_sha256s": frame_hashes,
        "prediction_sha256": _file_sha(prediction_path),
        "prediction_payload_sha256": stable_hash(payload),
        "grounder_config_sha256": stable_hash(config),
        "object_count": len(objects),
        "prediction_world_count": len(predictions),
        "observed_world_present": True,
        "counterfactual_worlds_complete": len(predictions) == len(objects) + 1,
        "observed_collision_count": len(observed.get("collisions") or ()),
        "provider_calls": 0,
        "cached_off_the_shelf_model_output": True,
        "question_read": False,
        "processed_proposals_read": False,
        "official_annotation_read": False,
        "functional_program_read": False,
        "answer_read": False,
        "source_controller_read": False,
    }
    return CLEVRERNSDRReceipt(**body, receipt_sha256=stable_hash(body))


__all__ = ["CLEVRERNSDRReceipt", "bind_cached_nsdr_prediction", "load_prediction_payload"]
