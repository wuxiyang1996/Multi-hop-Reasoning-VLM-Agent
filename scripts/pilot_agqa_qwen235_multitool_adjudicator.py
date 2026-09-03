#!/usr/bin/env python3
"""Answer-blind VLM adjudication over frozen SGDET/SlowFast candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _panels, _provider_json_call,
    _sample_video_range,
)


SYSTEM = """You are a frozen target-native video grounding tool. The public question defines a perceptual time window and a requested typed relation/action role. Select at most one supplied candidate ID that visibly fills that role. Candidate labels, boxes, and neural tool scores are fallible proposals, not facts. Use temporal order and pixels. Return ABSTAIN if the displayed evidence cannot distinguish the candidates. Never emit an object-name answer, functional program, source-domain fact, controller decision, or correctness judgment. Return only the required schema."""


ALIASES = {
    "paper/notebook": "paper", "phone/camera": "phone",
    "sofa/couch": "sofa", "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
}


def canonical(value: str) -> str:
    return ALIASES.get(str(value), str(value))


def response_format(candidate_ids: list[str], frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_multitool_candidate_binding_v1", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {
                       "selected_candidate_id": {"type": "string", "enum": candidate_ids + ["ABSTAIN"]},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                       "evidence_frames": {"type": "array", "maxItems": 4,
                                           "items": {"type": "integer", "enum": frame_ids}},
                       "visual_observation": {"type": "string"}},
                   "required": ["selected_candidate_id", "confidence", "evidence_frames", "visual_observation"]}}}


def fail_closed_call(**kwargs):
    try:
        return _provider_json_call(**kwargs)
    except ValueError as exc:
        payload = {"selected_candidate_id": "ABSTAIN", "confidence": 0.0,
                   "evidence_frames": [], "visual_observation": "schema_failure:" + type(exc).__name__}
        usage = {"model": str(kwargs["model"]["id"]), "finish_reason": "schema_failure",
                 "prompt_tokens": 0, "completion_tokens": 0, "reported_cost_usd": 0.0,
                 "response_sha256": stable_hash(payload)}
        return payload, usage


def nearest_sample(native_index: int, sampled_native: list[int]) -> int:
    return min(range(len(sampled_native)), key=lambda index: abs(sampled_native[index] - native_index))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet-raw", type=Path, required=True)
    parser.add_argument("--sgdet-bindings", type=Path, required=True)
    parser.add_argument("--action-bindings", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("multitool adjudication receipt is immutable")
    cohort = json.loads(args.cohort.read_text())
    raw = json.loads(args.sgdet_raw.read_text())
    sgdet = json.loads(args.sgdet_bindings.read_text())
    action = json.loads(args.action_bindings.read_text())
    if any(raw[key] for key in (
        "answer_read", "functional_program_read", "official_scene_graph_read",
        "per_video_action_genome_annotation_read", "source_controller_read", "target_outcome_read")):
        raise ValueError("raw SGDET receipt violates isolation contract")
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in raw["rows"]}
    action_by_task = {str(row["task_id"]): row for row in action["rows"]}
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=2)
    model = {"id": args.model, "omit_temperature": True}
    sources = sgdet["rows"][:args.max_tasks] if args.max_tasks else sgdet["rows"]
    palette = ("red", "lime", "cyan", "yellow", "magenta", "orange", "deepskyblue", "violet")
    outputs = []; calls = 0; total_cost = 0.0
    for source in sources:
        task_id = str(source["task_id"]); row = public[task_id]
        video_id = str(source["video_id"]); video = raw_by_video[video_id]
        action_row = action_by_task[task_id]
        pool = {}
        for candidate in source["candidates"][:5]:
            label = canonical(candidate["candidate_label"])
            pool.setdefault(label, {"label": label, "sgdet": candidate, "action": None})
        for candidate in action_row["candidates"][:5]:
            label = canonical(candidate["candidate_label"])
            current = pool.setdefault(label, {"label": label, "sgdet": None, "action": None})
            current["action"] = candidate
        candidates = [pool[label] for label in sorted(pool)]
        candidate_ids = ["C{}".format(index) for index in range(len(candidates))]
        for candidate_id, candidate in zip(candidate_ids, candidates):
            candidate["candidate_id"] = candidate_id

        lower, upper = [int(value) for value in source["mapped_dense_window"]]
        evidence = {lower, (lower + upper) // 2, upper}
        sampled_native = [int(value) for value in video["sampled_original_frame_indices"]]
        for candidate in candidates:
            if candidate["sgdet"] is not None:
                evidence.add(nearest_sample(
                    int(candidate["sgdet"]["evidence_original_frame_index"]), sampled_native))
            if candidate["action"] is not None:
                center48 = (15.5, 23.5, 31.5)[candidate["action"]["eligible_window_indices"][0]]
                evidence.add(round(center48 / 47 * 63))
        frame_ids = sorted(evidence)
        if len(frame_ids) > 10:
            frame_ids = sorted(frame_ids, key=lambda value: (abs(value - (lower + upper) / 2), value))[:10]
            frame_ids.sort()
        frames, seconds, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=64, max_side=800,
            start_second=0.0, end_second=None)
        selected = [frames[index].copy() for index in frame_ids]
        id_by_label = {candidate["label"]: candidate["candidate_id"] for candidate in candidates}
        color_by_id = {candidate_id: palette[index % len(palette)]
                       for index, candidate_id in enumerate(candidate_ids)}
        objects_by_frame = {}
        for detected in video["objects"]:
            label = canonical(detected["label"])
            if label in id_by_label:
                objects_by_frame.setdefault(int(detected["sampled_frame_index"]), []).append((label, detected))
        for image, frame_id in zip(selected, frame_ids):
            draw = ImageDraw.Draw(image)
            draw.text((8, 8), "F{}".format(frame_id), fill="white", stroke_width=3, stroke_fill="black")
            for label, detected in objects_by_frame.get(frame_id, []):
                candidate_id = id_by_label[label]; color = color_by_id[candidate_id]
                box = tuple(float(value) for value in detected["bbox_xyxy"])
                draw.rectangle(box, outline=color, width=5)
                draw.text((box[0] + 2, box[1] + 2), candidate_id,
                          fill=color, stroke_width=2, stroke_fill="black")
        panels = _panels(selected, [seconds[index] for index in frame_ids],
                         frames_per_panel=2, frame_width=448, quality=90)
        table = []
        for candidate in candidates:
            sg_score = (candidate["sgdet"] or {}).get("joint_score")
            action_score = (candidate["action"] or {}).get("action_score")
            table.append("{}={} [sgdet={}, action={}]".format(
                candidate["candidate_id"], candidate["label"],
                "NA" if sg_score is None else "{:.4f}".format(float(sg_score)),
                "NA" if action_score is None else "{:.4f}".format(float(action_score))))
        prompt = (
            "Public question (perceptual scope only; do not answer with a noun): {}\n"
            "Requested outer query predicate: {}\n"
            "Frozen 64-frame query window: {}..{}\n"
            "Displayed frame IDs: {}\n"
            "Fallible candidate tools: {}\n"
            "Boxes are candidate IDs. Select the one candidate ID visibly filling the OUTER query role, or ABSTAIN."
        ).format(row["question"], source["predicate"], lower, upper, frame_ids, "; ".join(table))
        panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
        core = {"protocol": "AGQA_QGV2_QWEN235_MULTITOOL_ADJUDICATOR_V1",
                "task_id": task_id, "question_sha256": row["question_sha256"],
                "sgdet_binding_sha256": sgdet["report_sha256"],
                "action_binding_sha256": action["report_sha256"],
                "candidate_table_sha256": stable_hash(candidates),
                "panel_sha256s": panel_hashes, "model": model}
        payload, usage, reused = _cached_provider_call(
            cache_dir=args.cache_dir, call_name="multitool_" + task_id, input_core=core,
            invoke=lambda: fail_closed_call(
                client=client, model=model, system=SYSTEM,
                content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                max_tokens=220, response_format=response_format(candidate_ids, frame_ids)))
        selected_id = str(payload["selected_candidate_id"])
        selected_candidate = next((value for value in candidates
                                   if value["candidate_id"] == selected_id), None)
        evidence_frames = sorted(set(int(value) for value in payload["evidence_frames"]))
        confidence = float(payload["confidence"])
        if selected_candidate is None or not evidence_frames:
            selected_id = "ABSTAIN"; selected_candidate = None
            evidence_frames = []; confidence = 0.0
        calls += int(not reused); total_cost += float(usage.get("reported_cost_usd", 0.0))
        outputs.append({"task_id": task_id, "predicate": source["predicate"],
                        "status": "BOUND" if selected_candidate else "ABSTAIN",
                        "top_candidate": selected_candidate["label"] if selected_candidate else None,
                        "selected_candidate_id": selected_id, "confidence": confidence,
                        "evidence_frames": evidence_frames, "candidates": candidates,
                        "raw_payload": payload, "usage": usage, "cache_reused": reused})
        print(json.dumps({"task_id": task_id, "selected": selected_id,
                          "label": selected_candidate["label"] if selected_candidate else None,
                          "cost_usd": usage.get("reported_cost_usd", 0.0)}), flush=True)
    report = {"schema_version": "agqa-qwen235-multitool-adjudication-v1",
              "status": "MULTITOOL_BINDINGS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
              "model": model, "frame_budget": 64, "rows": outputs,
              "provider_calls": calls, "reported_cost_usd": total_cost,
              "answer_read": False, "official_scene_graph_read": False,
              "functional_program_read": False, "source_controller_read": False,
              "target_outcome_read": False}
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
