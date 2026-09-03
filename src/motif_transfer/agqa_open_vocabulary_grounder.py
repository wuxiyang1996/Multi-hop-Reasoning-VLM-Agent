"""Frozen open-vocabulary entity acquisition for AGQA raw-video receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from .agqa_query_grounder_v2 import EntityTrack
from .agqa_query_object_grounder import canonical_object_label


_MODEL = None
_PROCESSOR = None
_MODEL_ID = None


@dataclass(frozen=True)
class Detection:
    frame_index: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class PhraseDetection:
    """A raw answer-blind referring-expression region."""

    frame_index: int
    phrase: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def _associate(detections: Sequence[Detection], *, maximum_tracks: int) -> tuple[EntityTrack, ...]:
    groups: list[list[Detection]] = []
    for row in sorted(detections, key=lambda x: (x.frame_index, -x.confidence)):
        candidates = [group for group in groups if group[0].label == row.label]
        match = max(candidates, key=lambda g: _iou(g[-1].bbox_xyxy, row.bbox_xyxy), default=None)
        if match is not None and _iou(match[-1].bbox_xyxy, row.bbox_xyxy) >= 0.15:
            match.append(row)
        else:
            groups.append([row])
    groups.sort(key=lambda g: (len({x.frame_index for x in g}), max(x.confidence for x in g)), reverse=True)
    first_per_label = []
    remaining = []
    seen = set()
    for group in groups:
        if group[0].label not in seen:
            first_per_label.append(group); seen.add(group[0].label)
        else:
            remaining.append(group)
    first_per_label.sort(key=lambda group: group[0].label != "person")
    groups = first_per_label + remaining
    tracks = []
    for group in groups[:maximum_tracks]:
        frames = tuple(sorted({row.frame_index for row in group}))
        tracks.append(EntityTrack(
            track_id=f"T{len(tracks)}", canonical_label=group[0].label, aliases=(),
            evidence_frames=frames[:6], confidence=max(row.confidence for row in group),
        ))
    return tuple(tracks)


def detect_ontology_tracks(
    frames: Sequence[Image.Image], *, frame_indices: Sequence[int], ontology: Sequence[str],
    query_terms: Sequence[str] | None = None,
    model_id: str = "IDEA-Research/grounding-dino-base", box_threshold: float = 0.18,
    text_threshold: float = 0.18, maximum_tracks: int = 12,
    ontology_chunk_size: int | None = None,
) -> tuple[tuple[EntityTrack, ...], tuple[Detection, ...]]:
    """Detect all public ontology classes without question, answer, or source access."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    global _MODEL, _PROCESSOR, _MODEL_ID
    if _MODEL is None:
        _PROCESSOR = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.eval(); _MODEL_ID = model_id
    if _MODEL_ID != model_id:
        raise ValueError("one detector process may use only one frozen model identity")
    processor, model = _PROCESSOR, _MODEL
    detections = []
    size = int(ontology_chunk_size or len(ontology))
    if size < 1:
        raise ValueError("ontology_chunk_size must be positive")
    terms = tuple(query_terms or ontology)
    queries = [tuple(terms[start:start + size]) for start in range(0, len(terms), size)]
    for frame_index in tuple(dict.fromkeys(int(x) for x in frame_indices)):
        for labels_in_query in queries:
            query = ". ".join(labels_in_query) + "."
            inputs = processor(images=frames[frame_index], text=query, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            sizes = torch.tensor([frames[frame_index].size[::-1]], device=model.device)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=sizes)[0]
            labels = result.get("text_labels", result.get("labels", ()))
            for box, score, raw_label in zip(result["boxes"], result["scores"], labels):
                label = canonical_object_label(str(raw_label))
                if label not in ontology:
                    continue
                detections.append(Detection(
                    frame_index, label, float(score), tuple(float(x) for x in box.tolist())))
    return _associate(detections, maximum_tracks=maximum_tracks), tuple(detections)


def detect_relation_phrase_regions(
    frames: Sequence[Image.Image], *, frame_indices: Sequence[int],
    phrases: Sequence[str], model_id: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.12, text_threshold: float = 0.12,
) -> tuple[PhraseDetection, ...]:
    """Ground public-question relation phrases without resolving object labels."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    global _MODEL, _PROCESSOR, _MODEL_ID
    if _MODEL is None:
        _PROCESSOR = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.eval(); _MODEL_ID = model_id
    if _MODEL_ID != model_id:
        raise ValueError("one detector process may use only one frozen model identity")
    clean_phrases = tuple(dict.fromkeys(str(value).strip().casefold() for value in phrases if str(value).strip()))
    detections = []
    for frame_index in tuple(dict.fromkeys(int(x) for x in frame_indices)):
        for phrase in clean_phrases:
            inputs = _PROCESSOR(
                images=frames[frame_index], text=phrase.rstrip(".") + ".",
                return_tensors="pt").to(_MODEL.device)
            with torch.no_grad():
                outputs = _MODEL(**inputs)
            sizes = torch.tensor([frames[frame_index].size[::-1]], device=_MODEL.device)
            result = _PROCESSOR.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=sizes)[0]
            for box, score in zip(result["boxes"], result["scores"]):
                detections.append(PhraseDetection(
                    frame_index=frame_index, phrase=phrase, confidence=float(score),
                    bbox_xyxy=tuple(float(x) for x in box.tolist())))
    return tuple(detections)


__all__ = [
    "Detection", "PhraseDetection", "detect_ontology_tracks",
    "detect_relation_phrase_regions",
]
