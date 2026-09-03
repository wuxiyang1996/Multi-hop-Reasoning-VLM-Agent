#!/usr/bin/env python3
"""Freeze question-independent dense captions from raw AGQA videos.

This is a target-native perception diagnostic.  It never opens AGQA answers,
scene graphs, functional programs, questions, or source-controller artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _sample_video_range


PROMPT = (
    "Partition and identify events by dividing the video into a series of "
    "non-overlapping segments. Give the start and end time for every event in "
    "chronological order, covering the whole video. For each event, describe "
    "all visible person-object interactions and directed spatial relations in "
    "concrete language. Do not answer any question. Format each line as: "
    "From <start time> to <end time>, <detailed description>."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--proposal-grounding", type=Path, required=True)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--model", default="andaba/TEMPURA-Qwen2.5-VL-3B-s2")
    parser.add_argument("--frame-count", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("dense-caption pilot output is immutable")
    if not 1 <= args.frame_count <= 64:
        raise ValueError("frame count must stay inside the frozen 64-frame ceiling")

    vendor_src = str(args.third_party.resolve())
    if vendor_src not in sys.path:
        sys.path.insert(0, vendor_src)
    from qwen_vl_utils import process_vision_info  # noqa: PLC0415

    cohort = json.loads(args.cohort.read_text())
    proposals = json.loads(args.proposal_grounding.read_text())
    task_ids = {str(value["task_id"]) for value in proposals["rows"]}
    public_rows = [value for value in cohort["rows"] if str(value["task_id"]) in task_ids]
    # Deliberately project away question/semantic fields before model inference.
    videos = {}
    for row in public_rows:
        videos[str(row["video_id"])] = Path(row["video_path"])

    processor = AutoProcessor.from_pretrained(args.model, padding_side="left", use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval()
    outputs = []
    for video_id, video_path in sorted(videos.items()):
        frames, seconds, metadata = _sample_video_range(
            video_path, frame_count=args.frame_count, max_side=448,
            start_second=0.0, end_second=None,
        )
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "timestamps": seconds,
             "fps": 1.0, "min_pixels": 224 * 224, "max_pixels": 448 * 448},
            {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        frame_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        outputs.append({"video_id": video_id, "video_sha256": _file_sha256(video_path),
                        "sampling_metadata": metadata,
                        "sampled_seconds": seconds, "sampled_frame_sha256s": frame_hashes,
                        "caption": caption})
        print(json.dumps({"video_id": video_id, "caption_chars": len(caption)}), flush=True)

    body = {
        "schema_version": "agqa-tempura-dense-caption-pilot-v1",
        "status": "DENSE_CAPTIONS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "model": args.model, "prompt_sha256": stable_hash(PROMPT),
        "cohort_sha256": cohort["cohort_sha256"],
        "proposal_report_sha256": proposals["report_sha256"],
        "frame_budget": args.frame_count, "videos": outputs,
        "question_read": False, "answer_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "source_controller_read": False, "target_outcome_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
