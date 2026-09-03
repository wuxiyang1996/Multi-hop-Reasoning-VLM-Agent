#!/usr/bin/env python3
"""Outcome-blind probe of a frozen Charades action recognizer for Layer B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from pytorchvideo.models.slowfast import create_slowfast

from motif_transfer.agqa_semantic_slots import action_anchor_obligations
from motif_transfer.agqa_temporal_sampling import native_index_views
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_frame_grounding_v2 import _sample_video
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_frame(frame: Image.Image) -> str:
    return stable_hash({
        "mode": frame.mode, "size": frame.size,
        "pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    })


def _classes(path: Path) -> tuple[tuple[str, str], ...]:
    rows = []
    for line in path.read_text().splitlines():
        class_id, phrase = line.strip().split(" ", 1)
        rows.append((class_id.casefold(), phrase.casefold()))
    if tuple(class_id for class_id, _ in rows) != tuple(f"c{i:03d}" for i in range(157)):
        raise ValueError("Charades classes must be ordered c000..c156")
    return tuple(rows)


def _phrase_to_class_id(phrase: str, ontology: dict[str, str]) -> str | None:
    exact = {
        str(value).casefold().strip(): str(key).casefold()
        for key, value in ontology.items()
        if str(key).casefold().startswith("c")
    }
    class_id = exact.get(phrase.casefold().strip())
    if class_id is None:
        return None
    if not class_id[1:].isdigit() or not 0 <= int(class_id[1:]) < 157:
        raise ValueError(f"invalid Charades class ID: {class_id}")
    return class_id


def _clip_tensor(frames, indices: tuple[int, ...], device: torch.device) -> list[torch.Tensor]:
    arrays = [np.asarray(frames[index], dtype=np.uint8) for index in indices]
    fast = torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2).float() / 255.0
    # Official frozen checkpoint configuration: RGB input is reversed to BGR,
    # shorter side 256, center crop 256, mean/std 0.45/0.225.
    fast = fast[[2, 1, 0]]
    _, _, height, width = fast.shape
    scale = 256.0 / min(height, width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    fast = F.interpolate(
        fast.permute(1, 0, 2, 3),
        size=(resized_height, resized_width), mode="bilinear", align_corners=False,
    ).permute(1, 0, 2, 3)
    top, left = (resized_height - 256) // 2, (resized_width - 256) // 2
    fast = fast[:, :, top:top + 256, left:left + 256]
    fast = (fast - 0.45) / 0.225
    slow_indices = torch.linspace(0, fast.shape[1] - 1, 8).long()
    slow = torch.index_select(fast, 1, slow_indices)
    return [slow.unsqueeze(0).to(device), fast.unsqueeze(0).to(device)]


def _dense_temporal_views(path: Path) -> tuple[list[list[Image.Image]], list[tuple[int, ...]], int]:
    """Decode ten official-style 32-frame views at native stride two."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {path}")
    span = 1 + (32 - 1) * 2
    starts = np.linspace(0, max(0, total - span), 10).round().astype(int)
    index_views = [tuple(min(total - 1, int(start) + offset * 2) for offset in range(32)) for start in starts]
    decoded: dict[int, Image.Image] = {}
    for index in sorted({index for view in index_views for index in view}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            capture.release()
            raise RuntimeError(f"failed decoding {path} at native frame {index}")
        decoded[index] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return [[decoded[index] for index in view] for view in index_views], index_views, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--sampling", choices=("uniform48", "dense10x32"), default="uniform48")
    parser.add_argument("--store-all-scores", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("action-model probe output is immutable")

    cohort = json.loads(args.cohort.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    grounding = json.loads(args.grounding.read_text())
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard configuration")
    video_ids = sorted({str(public[str(row["task_id"])]["video_id"])
                        for row in grounding["rows"]})
    selected_videos = {video_id for index, video_id in enumerate(video_ids)
                       if index % args.shard_count == args.shard_index}
    grounding_rows = [row for row in grounding["rows"]
                      if str(public[str(row["task_id"])]["video_id"]) in selected_videos]
    semantics = json.loads(args.semantic_runtime.read_text())
    semantic_by_id = {
        str(row["task_id"]): _semantic(row["receipt"])
        for row in semantics["rows"]
    }
    classes = _classes(args.classes)
    ontology = json.loads(args.ontology.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_slowfast(model_num_class=157, head_activation=None)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)

    rows = []
    uniform_windows = (tuple(range(0, 32)), tuple(range(8, 40)), tuple(range(16, 48)))
    recorded_windows = None
    for grounding_row in grounding_rows:
        task_id = str(grounding_row["task_id"])
        semantic = semantic_by_id[task_id]
        if args.sampling == "uniform48":
            frames, _, metadata = _sample_video(
                Path(public[task_id]["video_path"]), frame_count=48, max_side=448)
            clip_frames = [frames, frames, frames]
            local_indices = uniform_windows
            source_frame_count = int(metadata["source_frame_count"])
            native = tuple(int(index) for index in metadata["sampled_native_frame_indices"])
            windows = native_index_views(native, uniform_windows)
            presented = {int(index): frame for index, frame in zip(native, frames)}
        else:
            clip_frames, windows, source_frame_count = _dense_temporal_views(Path(public[task_id]["video_path"]))
            local_indices = (tuple(range(32)),) * len(clip_frames)
            presented = {}
            for one_clip, one_window in zip(clip_frames, windows):
                for frame, native_index in zip(one_clip, one_window):
                    presented.setdefault(int(native_index), frame)
        if recorded_windows is None:
            recorded_windows = len(windows)
        window_scores = []
        with torch.inference_mode():
            for one_clip, indices in zip(clip_frames, local_indices):
                logits = model(_clip_tensor(one_clip, indices, device))[0]
                window_scores.append(torch.sigmoid(logits).cpu())
        scores = torch.stack(window_scores)
        obligations = []
        for phrase, slot_id in action_anchor_obligations(semantic):
            class_id = _phrase_to_class_id(phrase, ontology)
            if class_id is None:
                obligations.append({
                    "phrase": phrase,
                    "slot_id": slot_id,
                    "mapping_status": "UNMAPPED_NO_EXACT_PUBLIC_ACTION_CLASS",
                })
                continue
            index = int(class_id[1:])
            obligations.append({
                "phrase": phrase,
                "slot_id": slot_id,
                "class_id": class_id,
                "checkpoint_class_phrase": classes[index][1],
                "mapping_status": "EXACT_PUBLIC_ACTION_CLASS",
                "window_scores": [float(value) for value in scores[:, index]],
                "max_score": float(scores[:, index].max()),
                "argmax_window": int(scores[:, index].argmax()),
            })
        top_scores, top_indices = torch.topk(scores.max(dim=0).values, k=10)
        output_row = {
            "task_id": task_id,
            "video_id": str(public[task_id]["video_id"]),
            "video_sha256": _sha_file(Path(public[task_id]["video_path"])),
            "source_frame_count": source_frame_count,
            "native_frame_index_views": [list(window) for window in windows],
            "presented_frame_receipts": [
                {"native_frame_index": index, "frame_sha256": _sha_frame(presented[index])}
                for index in sorted(presented)
            ],
            "obligations": obligations,
            "top10": [
                {"class_id": classes[int(index)][0], "phrase": classes[int(index)][1], "score": float(score)}
                for score, index in zip(top_scores, top_indices)
            ],
        }
        if args.store_all_scores:
            output_row["all_class_scores"] = [
                {
                    "class_id": class_id,
                    "phrase": phrase,
                    "window_scores": [float(value) for value in scores[:, index]],
                    "max_score": float(scores[:, index].max()),
                    "argmax_window": int(scores[:, index].argmax()),
                }
                for index, (class_id, phrase) in enumerate(classes)
            ]
        rows.append(output_row)
        print(json.dumps({"task_id": task_id, "obligations": len(obligations)}), flush=True)

    body = {
        "schema_version": "agqa-layer-b-frozen-charades-action-probe-v1",
        "status": "OUTCOME_BLIND_INTRINSIC_PROBE_COMPLETE",
        "checkpoint_sha256": _sha_file(args.checkpoint),
        "classes_sha256": _sha_file(args.classes),
        "ontology_sha256": _sha_file(args.ontology),
        "source": "FAIR_PYTORCHVIDEO_SLOWFAST_R50_CHARADES_FROZEN",
        "sampling": args.sampling,
        # ``uniform48`` decodes 48 distinct source frames, then presents three
        # overlapping 32-frame clips to SlowFast.  Keep acquisition bandwidth
        # separate from model compute instead of under-reporting the latter.
        "unique_sampled_frame_budget": 48 if args.sampling == "uniform48" else 320,
        "frame_presentation_budget": 96 if args.sampling == "uniform48" else 320,
        "temporal_views": recorded_windows,
        "all_class_scores_stored": args.store_all_scores,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "answers_read": False,
        "official_program_read": False,
        "official_scene_graph_read": False,
        "rows": rows,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
