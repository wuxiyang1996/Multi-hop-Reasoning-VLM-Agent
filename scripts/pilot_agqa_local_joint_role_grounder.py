#!/usr/bin/env python3
"""Locally bind one anonymous detector track to an AGQA question role."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _panels, _sample_video_range


SYSTEM = """Bind one anonymous visual track to the exact role requested by the question. Boxes and detector labels are fallible proposals, not facts. Respect action, relation, ordering, and temporal scope. Select a track only when its pixels show that exact role; otherwise ABSTAIN. Do not answer the question and do not emit an object name. Return only JSON: {\"selected_track_id\":\"T0|T1|...|ABSTAIN\",\"confidence\":0.0}."""


def _decode(text: str, valid: set[str]) -> tuple[str, float]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return "ABSTAIN", 0.0
    try:
        value = json.loads(match.group(0))
        track = str(value["selected_track_id"]).upper()
        confidence = float(value["confidence"])
        if track not in valid | {"ABSTAIN"} or not 0 <= confidence <= 1:
            raise ValueError
        return track, confidence
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "ABSTAIN", 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--localized-grounding", type=Path)
    parser.add_argument("--detector-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("joint role-grounder output is immutable")
    cohort = json.loads(args.cohort.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    localized = (json.loads(args.localized_grounding.read_text())
                 if args.localized_grounding else None)
    windows = ({str(row["task_id"]): row["video_metadata"] for row in localized["rows"]}
               if localized else {})
    detector = json.loads(args.detector_grounding.read_text())
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    palette = ("red", "lime", "cyan", "yellow", "magenta", "orange",
               "deepskyblue", "violet", "chartreuse", "gold", "pink", "white")
    rows = []
    sources = detector["rows"][:args.max_tasks] if args.max_tasks else detector["rows"]
    for source in sources:
        task_id = str(source["task_id"]); row = public[task_id]
        metadata = windows.get(task_id)
        local = metadata.get("localized", metadata) if metadata else None
        frames, seconds, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=64, max_side=800,
            start_second=float(local["sample_start_second"]) if local else 0.0,
            end_second=float(local["sample_end_second"]) if local else None)
        tracks = [track for track in source["tracks"] if track["canonical_label"] != "person"]
        indices = source["inspection_indices"]
        selected = [frames[index].copy() for index in indices]
        label_to_track = {track["canonical_label"]: track["track_id"] for track in tracks}
        track_to_color = {track["track_id"]: palette[i % len(palette)]
                          for i, track in enumerate(tracks)}
        for column, frame_index in enumerate(indices):
            draw = ImageDraw.Draw(selected[column])
            for detection in source["detections"]:
                if detection["frame_index"] != frame_index:
                    continue
                track_id = label_to_track.get(detection["label"])
                if track_id is None:
                    continue
                box = tuple(detection["bbox_xyxy"])
                draw.rectangle(box, outline=track_to_color[track_id], width=5)
                draw.text((box[0] + 3, box[1] + 3), track_id,
                          fill=track_to_color[track_id], stroke_width=2, stroke_fill="black")
        panel_bytes = _panels(selected, [seconds[index] for index in indices],
                              frames_per_panel=4, frame_width=192, quality=88)
        panels = [Image.open(BytesIO(blob)).convert("RGB") for blob in panel_bytes]
        proposal_table = ", ".join(
            f"{track['track_id']}={track['canonical_label']}"
            for track in tracks)
        prompt = (SYSTEM + f"\nQuestion scope (do not answer): {row['question']}\n"
                  f"Fallible proposal table: {proposal_table}")
        content = [{"type": "image", "image": panel} for panel in panels]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = processor(text=[text], images=panels, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=True)
        decoded = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        chosen, confidence = _decode(decoded, {track["track_id"] for track in tracks})
        candidates = [{"track_id": track["track_id"], "label": track["canonical_label"],
                       "status": "SUPPORTED" if track["track_id"] == chosen else "REFUTED",
                       "confidence": confidence if track["track_id"] == chosen else 1.0}
                      for track in tracks]
        rows.append({"task_id": task_id, "selected_track_id": chosen,
                     "confidence": confidence, "candidates": candidates,
                     "raw_response": decoded})
        print(json.dumps({"task_id": task_id, "selected_track_id": chosen,
                          "confidence": confidence}), flush=True)
    body = {"schema_version": "agqa-local-joint-role-grounder-pilot-v1",
            "status": "LOCAL_ROLE_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
            "model": args.model, "binding_mode": "anonymous_joint_single_or_abstain",
            "temporal_sampling": "localized" if localized else "uniform_whole_video",
            "detector_report_sha256": detector["report_sha256"], "rows": rows,
            "answer_read": False, "official_scene_graph_read": False,
            "functional_program_read": False, "source_controller_read": False}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
