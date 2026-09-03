"""Question-independent frozen CV tool grounder for CLEVRER raw videos.

This module deliberately produces measurements rather than answers.  It uses
only decoded pixels and a frozen configuration.  Official object traces,
scene annotations, functional programs, question text, answer labels, and the
source controller are outside its input boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .contracts import stable_hash


@dataclass(frozen=True)
class CVObservation:
    sampled_frame: int
    source_frame: int
    center_x: float
    center_y: float
    area: int
    bbox: tuple[int, int, int, int]
    radius: float
    hue: float
    saturation: float
    value: float
    contour_circularity: float


@dataclass(frozen=True)
class CVTrack:
    track_id: str
    observations: tuple[CVObservation, ...]
    color: str
    shape: str
    material: str
    attribute_confidence: float


@dataclass(frozen=True)
class CVEvent:
    kind: str
    subject_track_id: str
    object_track_id: str | None
    sampled_frame: int
    source_frame: int
    confidence: float


@dataclass(frozen=True)
class CLEVRERCVReceipt:
    video_sha256: str
    grounder_config_sha256: str
    selected_frame_indices: tuple[int, ...]
    selected_frame_sha256s: tuple[str, ...]
    tracks: tuple[CVTrack, ...]
    events: tuple[CVEvent, ...]
    provider_calls: int
    question_read: bool
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


def _sample_indices(total: int, budget: int) -> tuple[int, ...]:
    if total <= 0 or budget <= 0:
        raise ValueError("video and frame budget must be positive")
    return tuple(int(x) for x in np.linspace(0, total - 1, min(total, budget), dtype=int))


def _read_selected(video_path: Path, budget: int) -> tuple[tuple[int, ...], list[np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(total, budget)
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"cannot decode CLEVRER frame {index}: {video_path}")
        frames.append(frame)
    capture.release()
    return indices, frames


def _components(
    frame: np.ndarray, background: np.ndarray, sampled_frame: int,
    source_frame: int, config: Mapping[str, Any],
) -> list[CVObservation]:
    diff = cv2.cvtColor(cv2.absdiff(frame, background), cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    mask = (
        (diff > int(config["foreground_difference_threshold"]))
        & (
            (hsv[:, :, 1] > int(config["foreground_saturation_threshold"]))
            | (gray + int(config["dark_object_margin"]) < background_gray)
        )
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output: list[CVObservation] = []
    for contour in contours:
        area = int(round(cv2.contourArea(contour)))
        if not int(config["minimum_component_area"]) <= area <= int(config["maximum_component_area"]):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        x, y, w, h = (int(v) for v in cv2.boundingRect(contour))
        component = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        pixels = hsv[component.astype(bool)]
        hue, saturation, value = (float(np.median(pixels[:, i])) for i in range(3))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        output.append(CVObservation(
            sampled_frame=sampled_frame, source_frame=source_frame,
            center_x=float(moments["m10"] / moments["m00"]),
            center_y=float(moments["m01"] / moments["m00"]), area=area,
            bbox=(x, y, w, h), radius=float(np.sqrt(area / np.pi)),
            hue=hue, saturation=saturation, value=value,
            contour_circularity=max(0.0, min(1.0, circularity)),
        ))
    return sorted(output, key=lambda row: (row.center_x, row.center_y))


def _distance(left: CVObservation, right: CVObservation) -> float:
    return float(np.hypot(left.center_x - right.center_x, left.center_y - right.center_y))


def _track(detections: Sequence[Sequence[CVObservation]], config: Mapping[str, Any]) -> list[list[CVObservation]]:
    tracks: list[list[CVObservation]] = []
    maximum_distance = float(config["maximum_track_distance_px"])
    maximum_gap = int(config["maximum_track_gap"])
    for frame_index, rows in enumerate(detections):
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            gap = frame_index - track[-1].sampled_frame
            if gap < 1 or gap > maximum_gap + 1:
                continue
            for row_index, row in enumerate(rows):
                distance = _distance(track[-1], row)
                area_ratio = max(track[-1].area, row.area) / max(1, min(track[-1].area, row.area))
                cost = distance + 12.0 * max(0.0, area_ratio - 1.0)
                if distance <= maximum_distance * gap and area_ratio <= 3.0:
                    candidates.append((cost, track_index, row_index))
        used_tracks: set[int] = set()
        used_rows: set[int] = set()
        for _, track_index, row_index in sorted(candidates):
            if track_index in used_tracks or row_index in used_rows:
                continue
            tracks[track_index].append(rows[row_index])
            used_tracks.add(track_index); used_rows.add(row_index)
        for row_index, row in enumerate(rows):
            if row_index not in used_rows:
                tracks.append([row])
    minimum_support = max(3, len(detections) // 8)
    return [track for track in tracks if len(track) >= minimum_support]


def _color(hue: float, saturation: float, value: float) -> str:
    if saturation < 35:
        return "gray" if value >= 55 else "brown"
    # OpenCV hue is in [0,180).  Boundaries are fixed before evaluation.
    if hue < 8 or hue >= 172: return "red"
    if hue < 18: return "brown"
    if hue < 38: return "yellow"
    if hue < 78: return "green"
    if hue < 100: return "cyan"
    if hue < 132: return "blue"
    if hue < 172: return "purple"
    return "red"


def _summarize_track(index: int, rows: Sequence[CVObservation]) -> CVTrack:
    hues = np.asarray([row.hue for row in rows])
    saturations = np.asarray([row.saturation for row in rows])
    values = np.asarray([row.value for row in rows])
    circularity = float(np.median([row.contour_circularity for row in rows]))
    widths = np.asarray([row.bbox[2] for row in rows], dtype=float)
    heights = np.asarray([row.bbox[3] for row in rows], dtype=float)
    aspect = float(np.median(widths / np.maximum(heights, 1)))
    shape = "sphere" if circularity >= .72 else ("cylinder" if aspect < .86 else "cube")
    material = "metal" if float(np.std(values)) >= 28.0 else "rubber"
    stability = 1.0 / (1.0 + float(np.std(hues)) / 12.0 + float(np.std(saturations)) / 40.0)
    confidence = max(0.0, min(1.0, stability * min(1.0, len(rows) / 12.0)))
    return CVTrack(
        track_id=f"T{index}", observations=tuple(rows),
        color=_color(float(np.median(hues)), float(np.median(saturations)), float(np.median(values))),
        shape=shape, material=material, attribute_confidence=confidence,
    )


def _events(tracks: Sequence[CVTrack], sampled_count: int, config: Mapping[str, Any]) -> tuple[CVEvent, ...]:
    events: list[CVEvent] = []
    for track in tracks:
        first, last = track.observations[0], track.observations[-1]
        if first.sampled_frame > 0:
            events.append(CVEvent("ENTER", track.track_id, None, first.sampled_frame, first.source_frame, .8))
        if last.sampled_frame < sampled_count - 1:
            events.append(CVEvent("EXIT", track.track_id, None, last.sampled_frame, last.source_frame, .8))
    by_track = [{row.sampled_frame: row for row in track.observations} for track in tracks]
    multiplier = float(config["collision_radius_multiplier"])
    for left_index in range(len(tracks)):
        for right_index in range(left_index + 1, len(tracks)):
            shared = sorted(set(by_track[left_index]) & set(by_track[right_index]))
            if len(shared) < 3:
                continue
            distances = [
                _distance(by_track[left_index][frame], by_track[right_index][frame])
                for frame in shared
            ]
            position = int(np.argmin(distances)); frame = shared[position]
            left = by_track[left_index][frame]; right = by_track[right_index][frame]
            touching = distances[position] <= multiplier * (left.radius + right.radius)
            local_minimum = 0 < position < len(shared) - 1 and (
                distances[position] <= distances[position - 1]
                and distances[position] <= distances[position + 1]
            )
            if touching and local_minimum:
                margin = max(0.0, 1.0 - distances[position] / max(1.0, left.radius + right.radius))
                events.append(CVEvent(
                    "COLLIDE", tracks[left_index].track_id, tracks[right_index].track_id,
                    frame, left.source_frame, min(1.0, .55 + margin),
                ))
    return tuple(sorted(events, key=lambda row: (row.sampled_frame, row.kind, row.subject_track_id)))


def ground_clevrer_video(video_path: Path, config: Mapping[str, Any]) -> CLEVRERCVReceipt:
    if config.get("status") != "FROZEN_BEFORE_CLEVRER_RAW_VIDEO_RESERVE_SELECTION":
        raise ValueError("CLEVRER CV grounder config is not frozen")
    if "NO_ANNOTATIONS" not in str(config.get("authority")):
        raise ValueError("CLEVRER CV grounder authority drift")
    indices, frames = _read_selected(video_path, int(config["frame_budget"]))
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    detections = [
        _components(frame, background, index, indices[index], config)
        for index, frame in enumerate(frames)
    ]
    raw_tracks = _track(detections, config)
    tracks = tuple(_summarize_track(index, rows) for index, rows in enumerate(raw_tracks))
    events = _events(tracks, len(frames), config)
    body = {
        "video_sha256": _file_sha(video_path),
        "grounder_config_sha256": stable_hash(config),
        "selected_frame_indices": indices,
        "selected_frame_sha256s": tuple(_frame_sha(frame) for frame in frames),
        "tracks": tuple(tracks), "events": events, "provider_calls": 0,
        "question_read": False, "official_annotation_read": False,
        "functional_program_read": False, "answer_read": False,
        "source_controller_read": False,
    }
    serializable = {
        **body,
        "tracks": [asdict(row) for row in tracks],
        "events": [asdict(row) for row in events],
    }
    return CLEVRERCVReceipt(**body, receipt_sha256=stable_hash(serializable))


__all__ = [
    "CLEVRERCVReceipt", "CVEvent", "CVObservation", "CVTrack",
    "ground_clevrer_video",
]
