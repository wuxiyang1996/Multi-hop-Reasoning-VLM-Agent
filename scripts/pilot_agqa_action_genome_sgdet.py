#!/usr/bin/env python
"""Prediction-only Action Genome SGDET pilot for AGQA raw videos.

This acquisition program deliberately does not read questions, answers, AGQA
functional programs, or any per-video Action Genome/AGQA annotation.  It emits
the frozen neural observations that a separate evaluator may inspect later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--maximum-frames", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_frame(frame: np.ndarray) -> str:
    """Hash the exact decoded BGR uint8 pixels presented to preprocessing."""
    if frame.dtype != np.uint8 or frame.ndim != 3:
        raise ValueError("decoded frame must be an HxWxC uint8 array")
    digest = hashlib.sha256()
    digest.update(str(tuple(int(x) for x in frame.shape)).encode("ascii"))
    digest.update(b"\0BGR_UINT8\0")
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def uniform_indices(total: int, maximum: int) -> list[int]:
    if total <= 0:
        raise RuntimeError("video has no decodable frames")
    if total <= maximum:
        return list(range(total))
    return np.linspace(0, total - 1, num=maximum, dtype=np.int64).tolist()


def decode_uniform(path: Path, maximum: int) -> tuple[list[np.ndarray], list[int], int]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = uniform_indices(total, maximum)
    frames = []
    actual = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
            # Seeking may land on a nearby keyframe/backend-specific position.
            # Persist the coordinate of the frame actually decoded, not the
            # requested proxy index, so temporal evidence is auditable against
            # the original video.
            next_position = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES))))
            actual.append(min(total - 1, max(0, next_position - 1)))
    capture.release()
    if not frames:
        raise RuntimeError("failed to decode {}".format(path))
    return frames, actual, total


def prepare_batch(frames: list[np.ndarray], prep_im_for_blob, im_list_to_blob):
    processed = []
    scales = []
    for frame in frames:
        image, scale = prep_im_for_blob(
            frame, [[[102.9801, 115.9465, 122.7717]]], 600, 1000
        )
        processed.append(image)
        scales.append(scale)
    if max(scales) - min(scales) > 1e-6:
        raise RuntimeError("unexpected per-frame scale mismatch")
    blob = im_list_to_blob(processed)
    im_info = np.array(
        [[blob.shape[1], blob.shape[2], scales[0]]], dtype=np.float32
    )
    im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
    im_data = torch.from_numpy(blob).permute(0, 3, 1, 2)
    gt_boxes = torch.zeros([im_data.shape[0], 1, 5], dtype=torch.float32)
    num_boxes = torch.zeros([im_data.shape[0]], dtype=torch.int64)
    return im_data, im_info, gt_boxes, num_boxes


def tensor_rows(pred: dict, ontology: dict, sampled_indices: list[int]) -> dict:
    boxes = pred["boxes"].detach().cpu()
    labels = pred["pred_labels"].detach().cpu()
    scores = pred["pred_scores"].detach().cpu()
    pairs = pred["pair_idx"].detach().cpu()
    pair_frames = pred["im_idx"].detach().cpu().long()
    attention = pred["attention_distribution"].detach().cpu()
    spatial = pred["spatial_distribution"].detach().cpu()
    contact = pred["contacting_distribution"].detach().cpu()

    objects = []
    for index in range(boxes.shape[0]):
        sampled_frame = int(boxes[index, 0].item())
        label_index = int(labels[index].item())
        objects.append(
            {
                "detection_index": index,
                "sampled_frame_index": sampled_frame,
                "original_frame_index": int(sampled_indices[sampled_frame]),
                "label_index": label_index,
                "label": ontology["object_classes"][label_index],
                "score": float(scores[index].item()),
                "bbox_xyxy": [float(value) for value in boxes[index, 1:].tolist()],
            }
        )

    relations = []
    for index in range(pairs.shape[0]):
        sampled_frame = int(pair_frames[index].item())
        person_index = int(pairs[index, 0].item())
        object_index = int(pairs[index, 1].item())
        relations.append(
            {
                "sampled_frame_index": sampled_frame,
                "original_frame_index": int(sampled_indices[sampled_frame]),
                "person_detection_index": person_index,
                "object_detection_index": object_index,
                "object_label": ontology["object_classes"][int(labels[object_index].item())],
                "attention_person_to_object": {
                    name: float(attention[index, column].item())
                    for column, name in enumerate(ontology["attention_relationships"])
                },
                "spatial_object_to_person": {
                    name: float(spatial[index, column].item())
                    for column, name in enumerate(ontology["spatial_relationships"])
                },
                "contact_person_to_object": {
                    name: float(contact[index, column].item())
                    for column, name in enumerate(ontology["contacting_relationships"])
                },
            }
        )
    return {"objects": objects, "relations": relations}


def main() -> None:
    args = parse_args()
    if args.maximum_frames < 1 or args.maximum_frames > 64:
        raise ValueError("maximum-frames must be in [1, 64]")
    if not torch.cuda.is_available():
        raise RuntimeError("SGDET pilot requires CUDA")

    cohort_path = args.cohort.resolve()
    ontology_path = args.ontology.resolve()
    detector_checkpoint_path = args.detector_checkpoint.resolve()
    relation_checkpoint_path = args.relation_checkpoint.resolve()
    output_path = args.output.resolve()
    third_party = args.third_party.resolve()
    expected_detector_path = (third_party / "fasterRCNN" / "models" / "faster_rcnn_ag.pth").resolve()
    if detector_checkpoint_path != expected_detector_path:
        raise ValueError("detector checkpoint must match the path loaded by official SGDET")
    sys.path.insert(0, str(third_party))
    sys.path.insert(0, str(third_party / "fasterRCNN" / "lib"))
    os.chdir(str(third_party))

    from fasterRCNN.lib.model.utils.blob import im_list_to_blob, prep_im_for_blob
    from lib.ds_track import get_sequence
    from lib.object_detector import detector
    import lib.tempura as tempura_module

    cohort = json.loads(cohort_path.read_text())
    ontology = json.loads(ontology_path.read_text())
    selected_ids = set(filter(None, args.task_ids.split(",")))
    rows = cohort["rows"]
    if selected_ids:
        rows = [row for row in rows if row["task_id"] in selected_ids]
        missing = selected_ids - {row["task_id"] for row in rows}
        if missing:
            raise ValueError("unknown task ids: {}".format(sorted(missing)))

    # The learned checkpoint overwrites these embeddings.  Avoid downloading
    # GloVe merely to initialize tensors that are immediately replaced.
    tempura_module.obj_edge_vectors = lambda names, **kwargs: torch.zeros(
        [len(names), int(kwargs.get("wv_dim", 200))], dtype=torch.float32
    )

    device = torch.device("cuda:0")
    object_detector = detector(
        train=False,
        object_classes=ontology["object_classes"],
        use_SUPPLY=False,
        mode="sgdet",
    ).to(device=device)
    object_detector.eval()

    model = tempura_module.TEMPURA(
        mode="sgdet",
        attention_class_num=len(ontology["attention_relationships"]),
        spatial_class_num=len(ontology["spatial_relationships"]),
        contact_class_num=len(ontology["contacting_relationships"]),
        obj_classes=ontology["object_classes"],
        enc_layer_num=1,
        dec_layer_num=3,
        obj_mem_compute=False,
        rel_mem_compute="joint",
        take_obj_mem_feat=False,
        mem_fusion="late",
        selection="manual",
        selection_lambda=0.5,
        obj_head="linear",
        rel_head="gmm",
        K=4,
        tracking=True,
    ).to(device=device)
    checkpoint = torch.load(str(relation_checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    unique_videos = {}
    for row in rows:
        unique_videos.setdefault(row["video_id"], Path(row["video_path"]))

    output_rows = []
    with torch.no_grad():
        for video_id, video_path in sorted(unique_videos.items()):
            frames, sampled_indices, decoded_count = decode_uniform(
                video_path, args.maximum_frames
            )
            im_data, im_info, gt_boxes, num_boxes = prepare_batch(
                frames, prep_im_for_blob, im_list_to_blob
            )
            im_data = im_data.to(device)
            im_info = im_info.to(device)
            gt_boxes = gt_boxes.to(device)
            num_boxes = num_boxes.to(device)
            # Both annotation-shaped inputs are inert in detector eval/sgdet.
            # Pass an empty tuple to make accidental annotation dependence fail.
            entry = object_detector(
                im_data, im_info, gt_boxes, num_boxes, tuple(), im_all=None
            )
            get_sequence(entry, tuple(), None, "sgdet")
            pred = model(entry, phase="test", unc=False)
            predictions = tensor_rows(pred, ontology, sampled_indices)
            output_rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "video_sha256": sha256_file(video_path),
                    "decoded_frame_count": decoded_count,
                    "model_visible_frame_count": len(sampled_indices),
                    "sampled_original_frame_indices": sampled_indices,
                    "selected_frame_sha256s": [sha256_frame(frame) for frame in frames],
                    "selected_frame_hash_protocol": "SHAPE_NUL_BGR_UINT8_NUL_PIXELS_SHA256_V1",
                    **predictions,
                }
            )

    report = {
        "schema_version": "agqa-action-genome-sgdet-raw-receipt-v1",
        "mode": "sgdet",
        "question_read": False,
        "answer_read": False,
        "functional_program_read": False,
        "official_scene_graph_read": False,
        "per_video_action_genome_annotation_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "maximum_model_visible_frame_budget": args.maximum_frames,
        "ontology_sha256": sha256_file(ontology_path),
        "detector_checkpoint_sha256": sha256_file(detector_checkpoint_path),
        "relation_checkpoint_sha256": sha256_file(relation_checkpoint_path),
        "rows": output_rows,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report["report_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(serialized)
    os.replace(str(temporary), str(output_path))


if __name__ == "__main__":
    main()
