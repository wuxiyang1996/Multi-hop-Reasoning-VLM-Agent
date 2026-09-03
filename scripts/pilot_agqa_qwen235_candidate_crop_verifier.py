#!/usr/bin/env python3
"""Answer-blind high-resolution verification of dense detector candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_query_grounder_v2 import requested_query_role
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _panels, _provider_json_call,
    _sample_video_range,
)
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


SYSTEM = """You are a frozen candidate-isolated video grounding tool. Verify only the supplied typed visual proposition about the named public-ontology candidate inside RED boxes. BLUE boxes are person proposals. Boxes and labels are fallible proposals, not facts. SUPPORTED requires visible candidate identity AND visible proposition evidence; REFUTED requires contradictory pixels; otherwise UNKNOWN. Do not reconstruct or solve a question, emit an answer, compare candidates, infer a functional program, or use source-domain knowledge. Return only the schema."""


def _typed_proposition(predicate: str, candidate: str) -> str:
    relation = {
        "beneath": "the RED candidate is visibly above the person (the person is below it)",
        "in front of": "the person is visibly in front of the RED candidate in scene depth",
        "behind": "the person is visibly behind the RED candidate in scene depth",
        "above": "the RED candidate is visibly below the person (the person is above it)",
        "touching": "the person visibly touches the RED candidate",
        "carrying": "the person visibly carries the RED candidate",
        "holding": "the person visibly holds the RED candidate",
        "washing": "the person visibly washes the RED candidate",
    }
    return relation.get(predicate.casefold(), f"the person visibly {predicate} the RED candidate") + f" ({candidate})"


def _response_format(frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_candidate_crop_verification_v1", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {
                       "status": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "UNKNOWN"]},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                       "evidence_frames": {"type": "array", "maxItems": 4,
                                           "items": {"type": "integer", "enum": frame_ids}},
                       "visual_observation": {"type": "string"}},
                   "required": ["status", "confidence", "evidence_frames", "visual_observation"]}}}


def _fail_closed_provider_call(**kwargs):
    try:
        return _provider_json_call(**kwargs)
    except ValueError as exc:
        payload = {"status": "UNKNOWN", "confidence": 0.0,
                   "evidence_frames": [],
                   "visual_observation": f"provider_schema_failure:{type(exc).__name__}"}
        usage = {"model": str(kwargs["model"]["id"]), "finish_reason": "schema_failure",
                 "prompt_tokens": 0, "completion_tokens": 0, "reported_cost_usd": 0.0,
                 "response_sha256": stable_hash(payload)}
        return payload, usage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--detector-grounding", type=Path, required=True)
    parser.add_argument("--temporal-proposals", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--minimum-detection-confidence", type=float, default=.12)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("candidate-crop output is immutable")

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    detector = json.loads(args.detector_grounding.read_text())
    temporal = json.loads(args.temporal_proposals.read_text())
    if cohort["cohort_sha256"] != runtime["cohort_sha256"]:
        raise ValueError("cohort/runtime mismatch")
    public = {str(value["task_id"]): value for value in cohort["rows"]}
    semantics = {str(value["task_id"]): _semantic(value["receipt"])
                 for value in runtime["rows"]}
    windows = {str(value["task_id"]): value for value in temporal["rows"]}
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                    timeout=300, max_retries=2)
    model = {"id": args.model, "omit_temperature": True}
    outputs = []; total_cost = 0.0; calls = 0
    sources = detector["rows"][:args.max_tasks] if args.max_tasks else detector["rows"]
    for source in sources:
        task_id = str(source["task_id"]); row = public[task_id]
        semantic = semantics[task_id]; role = requested_query_role(semantic)
        temporal_row = windows[task_id]
        predicate = str(temporal_row["predicate"])
        temporal_operator = str(temporal_row["temporal_operator"])
        anchors = [str(value["phrase"]) for value in temporal_row.get("action_obligations", ())]
        original_window = [int(value) for value in temporal_row["inspection_indices"]]
        lower = round(min(original_window) / 47 * 63)
        upper = round(max(original_window) / 47 * 63)
        # The frozen action model partitions the video into three windows.  Its
        # argmax is stronger temporal evidence than the older phrase-binder's
        # fallback VIDEO tag.  This uses no answer or task outcome.
        if anchors:
            anchor_index = int(temporal_row["action_obligations"][0]["argmax_window"])
            anchor_lower = (0, 21, 42)[anchor_index]
            anchor_upper = (21, 42, 63)[anchor_index]
            if temporal_operator == "BEFORE":
                lower, upper = 0, anchor_lower
            elif temporal_operator == "AFTER":
                lower, upper = anchor_upper, 63
            else:
                lower, upper = anchor_lower, anchor_upper
            if lower == upper:
                lower, upper = anchor_lower, anchor_upper
        frames, seconds, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=64, max_side=800,
            start_second=0.0, end_second=None)
        by_label = {}
        person_by_frame = {}
        for detection in source["detections"]:
            frame_index = int(detection["frame_index"])
            if not lower <= frame_index <= upper:
                continue
            if str(detection["label"]) == "person":
                current = person_by_frame.get(frame_index)
                if current is None or float(detection["confidence"]) > float(current["confidence"]):
                    person_by_frame[frame_index] = detection
                continue
            by_label.setdefault(str(detection["label"]), []).append(detection)
        candidates = []
        for label, detections in sorted(by_label.items()):
            eligible = [value for value in detections
                        if float(value["confidence"]) >= args.minimum_detection_confidence]
            if not eligible:
                continue
            best_by_frame = {}
            for value in eligible:
                frame_index = int(value["frame_index"])
                current = best_by_frame.get(frame_index)
                if current is None or float(value["confidence"]) > float(current["confidence"]):
                    best_by_frame[frame_index] = value
            chosen_detections = sorted(
                best_by_frame.values(), key=lambda value: float(value["confidence"]), reverse=True)[:4]
            chosen_detections.sort(key=lambda value: int(value["frame_index"]))
            frame_ids = [int(value["frame_index"]) for value in chosen_detections]
            selected = [frames[value].copy() for value in frame_ids]
            for image, frame_index, candidate_detection in zip(selected, frame_ids, chosen_detections):
                draw = ImageDraw.Draw(image)
                draw.text((8, 8), f"F{frame_index} t={seconds[frame_index]:.2f}s",
                          fill="white", stroke_width=3, stroke_fill="black")
                draw.rectangle(tuple(candidate_detection["bbox_xyxy"]), outline="red", width=7)
                person = person_by_frame.get(frame_index)
                if person is not None:
                    draw.rectangle(tuple(person["bbox_xyxy"]), outline="deepskyblue", width=5)
            panels = _panels(selected, [seconds[value] for value in frame_ids],
                             frames_per_panel=2, frame_width=448, quality=90)
            proposition = _typed_proposition(predicate, label)
            prompt = (
                f"Candidate label in RED: {label}\nTyped proposition: {proposition}\n"
                f"Requested predicate: {predicate}\n"
                f"Requested role: {role}\nTemporal operator: {temporal_operator}\n"
                "The displayed interval has already been resolved by a separate frozen action tool; "
                "do not re-evaluate the anchor or temporal ordering.\n"
                f"Allowed sampled frame IDs: {frame_ids}\n"
                "Verify this candidate only. Use UNKNOWN unless the displayed pixels establish the exact role."
            )
            panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
            core = {"protocol": "AGQA_QGV2_QWEN235_CANDIDATE_CROP_V3",
                    "task_id": task_id, "question_sha256": row["question_sha256"],
                    "semantic_receipt_sha256": semantic.receipt_sha256,
                    "detector_report_sha256": detector["report_sha256"],
                    "temporal_report_sha256": temporal["report_sha256"],
                    "candidate": label, "predicate": predicate, "role": role,
                    "proposition": proposition, "temporal_operator": temporal_operator,
                    "anchor_phrases": anchors, "panel_sha256s": panel_hashes,
                    "model": model}
            payload, usage, reused = _cached_provider_call(
                cache_dir=args.cache_dir, call_name=f"candidate_{task_id}_{label}",
                input_core=core,
                invoke=lambda: _fail_closed_provider_call(
                    client=client, model=model, system=SYSTEM,
                    content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                    max_tokens=220, response_format=_response_format(frame_ids)))
            evidence = sorted(set(int(value) for value in payload["evidence_frames"]))
            status = str(payload["status"]); confidence = float(payload["confidence"])
            if status == "SUPPORTED" and not evidence:
                status = "UNKNOWN"; confidence = 0.0
            candidates.append({"candidate_label": label, "status": status,
                               "confidence": confidence, "evidence_frames": evidence,
                               "detection_max_confidence": max(float(x["confidence"]) for x in eligible),
                               "raw_payload": payload, "usage": usage, "cache_reused": reused})
            total_cost += float(usage.get("reported_cost_usd", 0.0)); calls += int(not reused)
        outputs.append({"task_id": task_id, "requested_role": role,
                        "predicate": predicate, "temporal_operator": temporal_operator,
                        "mapped_dense_window": [lower, upper], "candidates": candidates})
        print(json.dumps({"task_id": task_id, "n_candidates": len(candidates),
                          "supported": [x["candidate_label"] for x in candidates
                                        if x["status"] == "SUPPORTED"]}), flush=True)
    body = {"schema_version": "agqa-qwen235-candidate-crop-pilot-v1",
            "status": "CANDIDATE_CROP_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
            "model": args.model, "frame_budget": 64,
            "minimum_detection_confidence": args.minimum_detection_confidence,
            "detector_report_sha256": detector["report_sha256"],
            "temporal_report_sha256": temporal["report_sha256"],
            "semantic_runtime_sha256": runtime["runtime_sha256"],
            "rows": outputs, "provider_calls": calls, "reported_cost_usd": total_cost,
            "answer_read": False, "official_scene_graph_read": False,
            "functional_program_read": False, "source_controller_read": False,
            "target_outcome_read": False}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
