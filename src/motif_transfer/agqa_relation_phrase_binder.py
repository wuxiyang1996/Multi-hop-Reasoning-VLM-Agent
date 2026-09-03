"""Bind answer-blind relation phrase regions to public-ontology tracks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping, Sequence

from .agqa_open_vocabulary_grounder import Detection, PhraseDetection


_INVERSE_SPATIAL = {
    "above": ("object below a person", "surface below a person"),
    "behind": ("object in front of a person",),
    "beneath": ("object above a person", "object a person is under"),
    "in": ("container around a person", "place a person is inside"),
    "in front of": ("object behind a person",),
    "on the side of": ("object beside a person", "object next to a person"),
    "sitting on": ("object a person is sitting on",),
    "standing on": ("object a person is standing on",),
    "leaning on": ("object a person is leaning on",),
    "lying on": ("object a person is lying on",),
}


def relation_query_phrases(predicate: str) -> tuple[str, ...]:
    """Translate a typed target relation into answer-free referring expressions."""
    value = " ".join(str(predicate).replace("_", " ").casefold().split())
    if value in _INVERSE_SPATIAL:
        return _INVERSE_SPATIAL[value]
    return (f"object a person is {value}", f"{value} object")


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / (la + ra - intersection) if la + ra > intersection else 0.0


def _affinity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    overlap = _iou(left, right)
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    diagonal = max(1.0, hypot(max(left[2], right[2]), max(left[3], right[3])))
    proximity = max(0.0, 1.0 - hypot(lx - rx, ly - ry) / diagonal)
    return max(overlap, proximity * 0.35)


@dataclass(frozen=True)
class PhraseTrackBinding:
    label: str
    score: float
    evidence_frames: tuple[int, ...]
    runner_up_score: float


def bind_phrase_regions_to_tracks(
    phrases: Sequence[PhraseDetection], ontology: Sequence[Detection], *,
    excluded_labels: frozenset[str] = frozenset({"person"}),
) -> PhraseTrackBinding | None:
    """Aggregate same-frame region/track agreement and fail closed on no match."""
    evidence: dict[str, dict[int, float]] = {}
    for phrase in phrases:
        for candidate in ontology:
            if candidate.frame_index != phrase.frame_index or candidate.label in excluded_labels:
                continue
            score = phrase.confidence * candidate.confidence * _affinity(
                phrase.bbox_xyxy, candidate.bbox_xyxy)
            if score <= 0:
                continue
            frame_scores = evidence.setdefault(candidate.label, {})
            frame_scores[candidate.frame_index] = max(
                frame_scores.get(candidate.frame_index, 0.0), score)
    ranked = []
    for label, frame_scores in evidence.items():
        best = sorted(frame_scores.values(), reverse=True)[:3]
        recurrence = min(1.0, len(frame_scores) / 2)
        ranked.append((sum(best) / len(best) * (0.75 + 0.25 * recurrence), label, frame_scores))
    ranked.sort(reverse=True)
    if not ranked:
        return None
    score, label, frame_scores = ranked[0]
    return PhraseTrackBinding(
        label=label, score=score,
        evidence_frames=tuple(sorted(frame_scores, key=frame_scores.get, reverse=True)[:4]),
        runner_up_score=ranked[1][0] if len(ranked) > 1 else 0.0,
    )


def slowfast_relation_frame_indices(
    action_row: Mapping[str, object], *, temporal_operator: str,
    frame_count: int = 48, inspection_frames: int = 8,
) -> tuple[int, ...]:
    """Choose a relation-search window from frozen coarse action scores."""
    if frame_count < 2 or inspection_frames < 1:
        raise ValueError("invalid frame sampling configuration")
    centers = []
    for obligation in action_row.get("obligations", ()):  # type: ignore[union-attr]
        if obligation.get("mapping_status") != "EXACT_PUBLIC_ACTION_CLASS":
            continue
        scores = tuple(max(0.0, float(x)) for x in obligation.get("window_scores", ()))
        if not scores or sum(scores) <= 0:
            continue
        view_centers = tuple(
            (float(view[0]) + float(view[-1])) / 2
            for view in action_row.get("native_frame_index_views", ())  # type: ignore[union-attr]
        )
        if len(view_centers) != len(scores):
            continue
        centers.append(sum(score * center for score, center in zip(scores, view_centers)) / sum(scores))
    operator = str(temporal_operator).upper()
    if not centers or operator not in {"BEFORE", "AFTER", "WHILE", "BETWEEN"}:
        low, high = 0.0, float(frame_count - 1)
    elif operator == "BEFORE":
        low, high = 0.0, min(centers)
    elif operator == "AFTER":
        low, high = max(centers), float(frame_count - 1)
    elif operator == "WHILE":
        center = centers[0]
        low, high = max(0.0, center - 8.0), min(float(frame_count - 1), center + 8.0)
    else:
        low, high = (min(centers), max(centers)) if len(centers) >= 2 else (0.0, float(frame_count - 1))
    if high - low < 2:
        low, high = max(0.0, low - 1), min(float(frame_count - 1), high + 1)
    if inspection_frames == 1:
        return (round((low + high) / 2),)
    return tuple(dict.fromkeys(round(low + (high - low) * index / (inspection_frames - 1))
                               for index in range(inspection_frames)))


__all__ = [
    "PhraseTrackBinding", "bind_phrase_regions_to_tracks", "relation_query_phrases",
    "slowfast_relation_frame_indices",
]
