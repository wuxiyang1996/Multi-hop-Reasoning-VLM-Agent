#!/usr/bin/env python3
"""Candidate-isolated local Qwen3-VL role verification over GroundingDINO tracks."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import ImageDraw, Image
import torch
from transformers import AutoProcessor

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _panels, _sample_video_range


SYSTEM = """Verify exactly one detector proposal against the video pixels and question scope. The red box is fallible, not proof of its label. SUPPORTED requires both (1) the box visibly contains the named candidate and (2) that same entity fills the exact action/relation role at the exact temporal scope. Use UNKNOWN for ambiguity. Never answer the question and never name another candidate. Return only JSON: {\"status\":\"SUPPORTED|REFUTED|UNKNOWN\",\"confidence\":0.0}."""


def _decode_status(text: str) -> tuple[str, float]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return "UNKNOWN", 0.0
    try:
        value = json.loads(match.group(0)); status = str(value["status"]).upper()
        confidence = float(value["confidence"])
        if status not in {"SUPPORTED", "REFUTED", "UNKNOWN"} or not 0 <= confidence <= 1:
            raise ValueError
        return status, confidence
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "UNKNOWN", 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--localized-grounding", type=Path, required=True)
    parser.add_argument("--detector-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("local role-verifier output is immutable")
    cohort = json.loads(args.cohort.read_text()); public = {x["task_id"]: x for x in cohort["rows"]}
    localized = json.loads(args.localized_grounding.read_text())
    windows = {x["task_id"]: x["video_metadata"] for x in localized["rows"]}
    detector = json.loads(args.detector_grounding.read_text())
    if "Qwen3.5" in args.model:
        from transformers import Qwen3_5ForConditionalGeneration
        model_class = Qwen3_5ForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = model_class.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    rows = []
    for source in detector["rows"]:
        task_id = str(source["task_id"]); row = public[task_id]; metadata = windows[task_id]
        local = metadata.get("localized", metadata)
        frames, seconds, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=64, max_side=800,
            start_second=float(local["sample_start_second"]),
            end_second=float(local["sample_end_second"]))
        detections = source["detections"]; indices = source["inspection_indices"]
        candidates = []
        for track in source["tracks"]:
            label = track["canonical_label"]
            if label == "person":
                continue
            selected = [frames[x].copy() for x in indices]
            for column, frame_index in enumerate(indices):
                draw = ImageDraw.Draw(selected[column])
                for detection in detections:
                    if detection["frame_index"] != frame_index or detection["label"] != label:
                        continue
                    x1, y1, x2, y2 = detection["bbox_xyxy"]
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
            panel_bytes = _panels(selected, [seconds[x] for x in indices],
                                  frames_per_panel=len(selected), frame_width=160, quality=82)[0]
            panel = Image.open(BytesIO(panel_bytes)).convert("RGB")
            prompt = (SYSTEM + f"\nQuestion scope (do not answer): {row['question']}\n"
                      f"Single candidate label: {label}. Verify only this candidate.")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": panel}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = processor(text=[text], images=[panel], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=True)
            decoded = processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)[0]
            status, confidence = _decode_status(decoded)
            candidates.append({"track_id": track["track_id"], "label": label,
                               "status": status, "confidence": confidence,
                               "raw_response": decoded})
        rows.append({"task_id": task_id, "candidates": candidates})
        print(json.dumps({"task_id": task_id,
                          "supported": [x["label"] for x in candidates if x["status"] == "SUPPORTED"]}), flush=True)
    body = {"schema_version": "agqa-local-role-verifier-pilot-v1",
            "status": "LOCAL_ROLE_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
            "model": args.model, "detector_report_sha256": detector["report_sha256"],
            "rows": rows, "answer_read": False, "official_scene_graph_read": False,
            "functional_program_read": False, "source_controller_read": False}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
