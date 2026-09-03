"""Deterministic target-native tracking and score fusion for AGQA SGDET.

The functions in this module consume only prediction-only neural receipts.
They never open AGQA answers, functional programs, or Action Genome boxes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .agqa_query_grounder_v2 import EntityTrack
from .agqa_query_object_grounder import canonical_object_label


PREDICATE_CHANNELS: Mapping[str, tuple[tuple[str, str], ...]] = {
    # TEMPURA spatial logits describe object -> person; AGQA query wording
    # describes person -> object, hence the inverse names below.
    "beneath": (("spatial_object_to_person", "above"),),
    "below": (("spatial_object_to_person", "above"),),
    "above": (("spatial_object_to_person", "beneath"),),
    "in front of": (("spatial_object_to_person", "behind"),),
    "behind": (("spatial_object_to_person", "in_front_of"),),
    "on the side of": (("spatial_object_to_person", "on_the_side_of"),),
    "in": (("spatial_object_to_person", "in"),),
    "touching": (("contact_person_to_object", "touching"),),
    "carrying": (("contact_person_to_object", "carrying"),),
    "holding": (("contact_person_to_object", "holding"),),
    "wearing": (("contact_person_to_object", "wearing"),),
    "sitting on": (("contact_person_to_object", "sitting_on"),),
    "standing on": (("contact_person_to_object", "standing_on"),),
    "leaning on": (("contact_person_to_object", "leaning_on"),),
    "watching": (("attention_person_to_object", "looking_at"),),
    "looking at": (("attention_person_to_object", "looking_at"),),
    "related to": (
        ("attention_person_to_object", "looking_at"),
        ("contact_person_to_object", "carrying"),
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
        ("contact_person_to_object", "wearing"),
        ("contact_person_to_object", "sitting_on"),
        ("contact_person_to_object", "standing_on"),
        ("contact_person_to_object", "leaning_on"),
        ("contact_person_to_object", "lying_on"),
        ("contact_person_to_object", "covered_by"),
    ),
    # Action-state signatures are used only as one frozen neural view and are
    # fused with the independent SlowFast action view downstream.
    "taking": (("contact_person_to_object", "carrying"),
               ("contact_person_to_object", "holding")),
    "putting down": (("contact_person_to_object", "carrying"),
                     ("contact_person_to_object", "holding"),
                     ("contact_person_to_object", "touching")),
    "opening": (("contact_person_to_object", "holding"),
                ("contact_person_to_object", "touching")),
    "closing": (("contact_person_to_object", "holding"),
                ("contact_person_to_object", "touching")),
    "grasping": (("contact_person_to_object", "holding"),
                 ("contact_person_to_object", "touching")),
    "throwing": (("contact_person_to_object", "carrying"),
                 ("contact_person_to_object", "holding")),
    "washing": (("contact_person_to_object", "wiping"),
                ("contact_person_to_object", "touching")),
    "tidying": (("contact_person_to_object", "holding"),
                ("contact_person_to_object", "touching")),
    "working on": (("attention_person_to_object", "looking_at"),
                   ("contact_person_to_object", "touching")),
    "snuggling": (("contact_person_to_object", "holding"),
                  ("contact_person_to_object", "covered_by")),
    "undressing": (("contact_person_to_object", "wearing"),),
}


@dataclass(frozen=True)
class StableTrackCompilation:
    tracks: tuple[EntityTrack, ...]
    detection_to_track: Mapping[int, str]
    retained_detection_indices: frozenset[int]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(x) for x in left)
    rx1, ry1, rx2, ry2 = (float(x) for x in right)
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    la = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = la + ra - intersection
    return intersection / union if union > 0 else 0.0


def _affinity(left: Sequence[float], right: Sequence[float]) -> float:
    overlap = _iou(left, right)
    lx1, ly1, lx2, ly2 = (float(x) for x in left)
    rx1, ry1, rx2, ry2 = (float(x) for x in right)
    lc = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
    rc = ((rx1 + rx2) / 2, (ry1 + ry2) / 2)
    scale = max(
        math.hypot(lx2 - lx1, ly2 - ly1),
        math.hypot(rx2 - rx1, ry2 - ry1), 1.0,
    )
    center = max(0.0, 1.0 - math.hypot(lc[0] - rc[0], lc[1] - rc[1]) / (2.0 * scale))
    la = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    ra = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    size = min(la, ra) / max(la, ra)
    return 0.55 * overlap + 0.35 * center + 0.10 * size


def build_stable_tracks(
    video_row: Mapping[str, object], *, minimum_object_score: float = 0.05,
    within_frame_nms_iou: float = 0.70, minimum_track_affinity: float = 0.15,
    maximum_track_gap: int = 6,
) -> StableTrackCompilation:
    """Associate same-class SGDET boxes without using annotations or questions."""
    if not 0 <= minimum_object_score <= 1:
        raise ValueError("minimum_object_score must be in [0,1]")
    frame_count = int(video_row["model_visible_frame_count"])
    raw = [dict(row) for row in video_row["objects"]]
    by_frame_label: dict[tuple[int, str], list[dict]] = defaultdict(list)
    person_rows: list[dict] = []
    for row in raw:
        frame = int(row["sampled_frame_index"])
        if not 0 <= frame < frame_count:
            raise ValueError("SGDET detection frame exceeds model-visible frames")
        label = canonical_object_label(str(row["label"]))
        row["canonical_label"] = label
        if label == "person":
            person_rows.append(row)
        elif float(row["score"]) >= minimum_object_score:
            by_frame_label[(frame, label)].append(row)

    detection_to_track: dict[int, str] = {}
    person_best: dict[int, dict] = {}
    for row in person_rows:
        frame = int(row["sampled_frame_index"])
        if frame not in person_best or float(row["score"]) > float(person_best[frame]["score"]):
            person_best[frame] = row
        detection_to_track[int(row["detection_index"])] = "T0"
    person_frames = tuple(sorted(person_best)) or (0,)
    tracks_internal: list[dict] = [{
        "track_id": "T0", "label": "person", "aliases": set(),
        "rows": list(person_best.values()), "last_frame": max(person_frames),
        "last_bbox": person_best[max(person_frames)]["bbox_xyxy"] if person_best else (0, 0, 1, 1),
    }]
    retained: set[int] = {int(row["detection_index"]) for row in person_best.values()}
    suppressed_to_kept: dict[int, int] = {}

    for (frame, label), rows in sorted(by_frame_label.items()):
        kept: list[dict] = []
        for row in sorted(rows, key=lambda x: (-float(x["score"]), int(x["detection_index"]))):
            duplicate = next((prior for prior in kept
                              if _iou(prior["bbox_xyxy"], row["bbox_xyxy"]) >= within_frame_nms_iou), None)
            if duplicate is None:
                kept.append(row)
            else:
                suppressed_to_kept[int(row["detection_index"])] = int(duplicate["detection_index"])

        active = [track for track in tracks_internal[1:]
                  if track["label"] == label and frame - int(track["last_frame"]) <= maximum_track_gap]
        pairs = sorted(
            ((_affinity(track["last_bbox"], row["bbox_xyxy"]), track, row)
             for track in active for row in kept),
            key=lambda value: (-value[0], value[1]["track_id"], int(value[2]["detection_index"])),
        )
        used_tracks: set[str] = set(); used_detections: set[int] = set()
        for affinity, track, row in pairs:
            index = int(row["detection_index"])
            if affinity < minimum_track_affinity or track["track_id"] in used_tracks or index in used_detections:
                continue
            track["rows"].append(row); track["last_frame"] = frame; track["last_bbox"] = row["bbox_xyxy"]
            detection_to_track[index] = track["track_id"]
            retained.add(index); used_tracks.add(track["track_id"]); used_detections.add(index)
        for row in kept:
            index = int(row["detection_index"])
            if index in used_detections:
                continue
            track_id = f"T{len(tracks_internal)}"
            tracks_internal.append({
                "track_id": track_id, "label": label,
                "aliases": {str(row["label"])} if canonical_object_label(str(row["label"])) != str(row["label"]) else set(),
                "rows": [row], "last_frame": frame, "last_bbox": row["bbox_xyxy"],
            })
            detection_to_track[index] = track_id; retained.add(index)

    for suppressed, kept in suppressed_to_kept.items():
        if kept in detection_to_track:
            detection_to_track[suppressed] = detection_to_track[kept]
    tracks = []
    for track in tracks_internal:
        rows = track["rows"]
        evidence = tuple(sorted({int(row["sampled_frame_index"]) for row in rows}))
        confidence = max((float(row["score"]) for row in rows), default=0.0)
        value = EntityTrack(
            track_id=str(track["track_id"]), canonical_label=str(track["label"]),
            aliases=tuple(sorted(alias for alias in track["aliases"] if alias != track["label"])),
            evidence_frames=evidence or person_frames, confidence=confidence,
        )
        value.validate(frame_count); tracks.append(value)
    return StableTrackCompilation(tuple(tracks), detection_to_track, frozenset(retained))


def relation_track_candidates(
    video_row: Mapping[str, object], tracks: StableTrackCompilation, *,
    predicate: str, lower_frame: int, upper_frame: int,
) -> tuple[dict, ...]:
    """Rank stable object tracks by frozen SGDET object×relation evidence."""
    specs = PREDICATE_CHANNELS.get(predicate.casefold().strip(), ())
    objects = {int(row["detection_index"]): row for row in video_row["objects"]}
    best: dict[str, dict] = {}
    for relation in video_row["relations"]:
        frame = int(relation["sampled_frame_index"])
        if not lower_frame <= frame <= upper_frame:
            continue
        detection = int(relation["object_detection_index"])
        track_id = tracks.detection_to_track.get(detection)
        if track_id is None or detection not in objects:
            continue
        relation_score = max((float(relation[channel][name]) for channel, name in specs), default=0.0)
        object_score = float(objects[detection]["score"])
        score = relation_score * object_score
        row = {
            "track_id": track_id,
            "candidate_label": canonical_object_label(str(relation["object_label"])),
            "score": score, "relation_score": relation_score, "object_score": object_score,
            "sampled_frame_index": frame,
            "original_frame_index": int(relation["original_frame_index"]),
            "object_detection_index": detection,
        }
        if track_id not in best or score > float(best[track_id]["score"]):
            best[track_id] = row
    return tuple(sorted(best.values(), key=lambda row: (
        -float(row["score"]), str(row["candidate_label"]), str(row["track_id"]),
    )))


def reciprocal_rank_fusion(
    primary: Sequence[Mapping[str, object]], secondary: Sequence[Mapping[str, object]], *,
    primary_weight: float = 0.60,
) -> tuple[dict, ...]:
    """Fuse independent label rankings without fitting target outcomes."""
    if not 0 <= primary_weight <= 1:
        raise ValueError("primary_weight must be in [0,1]")
    scores: dict[str, float] = defaultdict(float)
    provenance: dict[str, set[str]] = defaultdict(set)
    track_by_label: dict[str, str] = {}
    evidence_by_label: dict[str, tuple[int, ...]] = {}
    for source, rows, weight in (
        ("sgdet", primary, primary_weight), ("slowfast", secondary, 1.0 - primary_weight),
    ):
        seen: set[str] = set()
        rank = 0
        for row in rows:
            label = canonical_object_label(str(row["candidate_label"]))
            if label in seen:
                continue
            seen.add(label); rank += 1
            scores[label] += weight / rank; provenance[label].add(source)
            if source == "sgdet" or label not in track_by_label:
                if row.get("track_id") is not None:
                    track_by_label[label] = str(row["track_id"])
                if row.get("sampled_frame_index") is not None:
                    evidence_by_label[label] = (int(row["sampled_frame_index"]),)
    output = [{
        "candidate_label": label, "score": score,
        "sources": sorted(provenance[label]), "track_id": track_by_label.get(label),
        "evidence_frames": list(evidence_by_label.get(label, ())),
    } for label, score in scores.items()]
    return tuple(sorted(output, key=lambda row: (-float(row["score"]), str(row["candidate_label"]))))


__all__ = [
    "PREDICATE_CHANNELS", "StableTrackCompilation", "build_stable_tracks",
    "reciprocal_rank_fusion", "relation_track_candidates",
]
