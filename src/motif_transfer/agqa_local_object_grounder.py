"""Local neural object corroboration for AGQA relation receipts.

The VLM supplies an action/relation interval.  A frozen COCO YOLOX detector
independently labels objects in that interval.  A small target-native ontology
may canonicalize two agreeing fine-grained labels (for example VLM ``pan`` and
COCO ``bowl``) to a broader answer vocabulary such as ``dish``.  The detector
never sees the question, answer, program, candidates, or source identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .agqa_active_frame_grounder import (
    AGQAOperandReceipt,
    parse_operand_receipt,
)
from .contracts import stable_hash


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)
GENERIC_OBJECTS = frozenset({
    "", "object", "unknown", "unknown object", "an unknown object", "item",
    "thing",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AGQALocalDetection:
    frame_index: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["bbox_xyxy"] = list(self.bbox_xyxy)
        return row


@dataclass(frozen=True)
class AGQALocalObjectReceipt:
    detector: str
    model_sha256: str
    inspected_frame_indices: tuple[int, ...]
    detections: tuple[AGQALocalDetection, ...]
    question_read: bool
    answer_read: bool
    functional_program_read: bool
    answer_candidates_read: bool
    source_identity_read: bool
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "model_sha256": self.model_sha256,
            "inspected_frame_indices": list(self.inspected_frame_indices),
            "detections": [row.as_dict() for row in self.detections],
            "question_read": self.question_read,
            "answer_read": self.answer_read,
            "functional_program_read": self.functional_program_read,
            "answer_candidates_read": self.answer_candidates_read,
            "source_identity_read": self.source_identity_read,
            "receipt_sha256": self.receipt_sha256,
        }


class _YoloX:
    def __init__(self, model_path: Path, *, confidence: float, nms: float):
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.confidence = float(confidence)
        self.nms = float(nms)
        grids, strides = [], []
        for stride in (8, 16, 32):
            size = 640 // stride
            xv, yv = np.meshgrid(np.arange(size), np.arange(size))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            strides.append(np.full((*grid.shape[:2], 1), stride))
        self.grids = np.concatenate(grids, axis=1)
        self.strides = np.concatenate(strides, axis=1)

    def infer(self, image: Image.Image) -> list[tuple[str, float, tuple[float, ...]]]:
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        ratio = min(640 / height, 640 / width)
        resized = cv2.resize(
            rgb, (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded = np.ones((640, 640, 3), dtype=np.float32) * 114.0
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = np.transpose(padded, (2, 0, 1))[np.newaxis, :, :, :]
        self.net.setInput(blob)
        output = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        output[:, :2] = (output[:, :2] + self.grids[0]) * self.strides[0]
        output[:, 2:4] = np.exp(output[:, 2:4]) * self.strides[0]
        boxes = np.ones_like(output[:, :4])
        boxes[:, 0] = output[:, 0] - output[:, 2] / 2
        boxes[:, 1] = output[:, 1] - output[:, 3] / 2
        boxes[:, 2] = output[:, 0] + output[:, 2] / 2
        boxes[:, 3] = output[:, 1] + output[:, 3] / 2
        scores = output[:, 4:5] * output[:, 5:]
        max_scores = np.max(scores, axis=1)
        classes = np.argmax(scores, axis=1)
        keep = cv2.dnn.NMSBoxesBatched(
            boxes.tolist(), max_scores.tolist(), classes.tolist(),
            self.confidence, self.nms,
        )
        results = []
        for index in keep:
            class_id = int(classes[index])
            xyxy = tuple(float(value / ratio) for value in boxes[index])
            results.append((COCO_CLASSES[class_id], float(max_scores[index]), xyxy))
        return results


def inspection_indices(receipt: AGQAOperandReceipt, *, maximum: int = 12) -> tuple[int, ...]:
    indices = set()
    for row in receipt.observations:
        if row.start_frame is not None and row.end_frame is not None:
            indices.update(range(row.start_frame, row.end_frame + 1))
        indices.update(row.evidence_frames)
    ordered = sorted(indices)
    if len(ordered) <= maximum:
        return tuple(ordered)
    positions = np.linspace(0, len(ordered) - 1, maximum)
    return tuple(dict.fromkeys(ordered[round(float(position))] for position in positions))


def detect_objects(
    frames: Sequence[Image.Image], *, frame_indices: Sequence[int],
    model_path: Path, expected_model_sha256: str,
    confidence_threshold: float = 0.08, nms_threshold: float = 0.5,
) -> AGQALocalObjectReceipt:
    if _sha256(model_path) != expected_model_sha256:
        raise ValueError("AGQA local object-detector model hash mismatch")
    indices = tuple(dict.fromkeys(int(value) for value in frame_indices))
    if not indices or any(not 0 <= value < len(frames) for value in indices):
        raise ValueError("AGQA local object-detector frame indices are invalid")
    model = _YoloX(
        model_path, confidence=confidence_threshold, nms=nms_threshold,
    )
    detections = []
    for index in indices:
        for label, confidence, box in model.infer(frames[index]):
            detections.append(AGQALocalDetection(index, label, confidence, box))
    core = {
        "detector": "opencv_zoo_yolox_s_int8_coco_v2022nov",
        "model_sha256": expected_model_sha256,
        "inspected_frame_indices": list(indices),
        "detections": [row.as_dict() for row in detections],
        "question_read": False,
        "answer_read": False,
        "functional_program_read": False,
        "answer_candidates_read": False,
        "source_identity_read": False,
    }
    return AGQALocalObjectReceipt(
        detector=core["detector"], model_sha256=expected_model_sha256,
        inspected_frame_indices=indices, detections=tuple(detections),
        question_read=False, answer_read=False, functional_program_read=False,
        answer_candidates_read=False, source_identity_read=False,
        receipt_sha256=stable_hash(core),
    )


def refine_query_object_receipt(
    receipt: AGQAOperandReceipt, detector: AGQALocalObjectReceipt,
) -> tuple[AGQAOperandReceipt, tuple[str, ...]]:
    """Fuse action-localized VLM objects with independent COCO detections."""

    labels_by_frame: dict[int, set[str]] = {}
    for row in detector.detections:
        labels_by_frame.setdefault(row.frame_index, set()).add(row.label)
    retained = []
    canonicalizations = list(receipt.canonicalizations)
    for row in receipt.observations:
        raw_object = row.object.strip().casefold()
        if raw_object in GENERIC_OBJECTS:
            continue
        support = set().union(*(
            labels_by_frame.get(index, set())
            for index in range(
                row.start_frame or 0,
                (row.end_frame if row.end_frame is not None else -1) + 1,
            )
        ))
        object_name = row.object
        if raw_object in {"pan", "plate"} and "bowl" in support:
            object_name = "dish"
            canonicalizations.append(
                f"{row.occurrence_id}:VLM_{raw_object.upper()}_PLUS_COCO_BOWL_TO_DISH"
            )
        retained.append({
            **row.as_dict(),
            "occurrence_id": f"O{len(retained)}",
            "object": object_name,
        })
    if not retained:
        return receipt, ()
    payload = {
        "operand_role": receipt.operand_role,
        "requested_operand": receipt.requested_operand,
        "observations": retained,
        "coverage": receipt.coverage,
        "uncertainties": list(receipt.uncertainties),
        "canonicalizations": canonicalizations,
    }
    refined = parse_operand_receipt(
        payload, expected_role=receipt.operand_role,
        expected_operand=receipt.requested_operand,
        frame_count=receipt.frame_count,
    )
    new_markers = tuple(
        value for value in refined.canonicalizations
        if value not in receipt.canonicalizations
    )
    return refined, new_markers


__all__ = [
    "AGQALocalDetection", "AGQALocalObjectReceipt", "COCO_CLASSES",
    "detect_objects", "inspection_indices", "refine_query_object_receipt",
]
