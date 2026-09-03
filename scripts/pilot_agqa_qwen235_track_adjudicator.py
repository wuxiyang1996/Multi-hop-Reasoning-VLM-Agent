#!/usr/bin/env python3
"""Answer-blind Qwen235 adjudication of frozen raw-video entity tracks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_query_grounder_v2 import (
    QueryCandidateEvidence,
    requested_query_role,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _semantic
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _panels, _provider_json_call,
    _sample_video_range,
)


SYSTEM = """You are a frozen raw-video track-role adjudicator. Select at most one supplied anonymous track that visibly fills the exact action or directed spatial-relation role requested by the public question, within the supplied temporal interval. Detector boxes and labels are fallible proposals. Use ABSTAIN when pixels or temporal order are insufficient. Return only a track_id, never an object-name answer. Do not infer or emit a task answer, functional program, scene graph, source domain, controller, or correctness."""


def _format(track_ids: list[str], frame_ids: list[int]) -> dict:
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "selected_track_id": {"type": "string", "enum": track_ids + ["ABSTAIN"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_frames": {"type": "array", "maxItems": 4,
                                "items": {"type": "integer", "enum": frame_ids}},
            "uncertainty": {"type": "string"},
        },
        "required": ["selected_track_id", "confidence", "evidence_frames", "uncertainty"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_anonymous_track_binding_v1", "strict": True, "schema": schema}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--proposal-grounding", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("track-adjudicator output is immutable")
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    if cohort["cohort_sha256"] != runtime["cohort_sha256"]:
        raise ValueError("cohort/runtime mismatch")
    semantics = {str(value["task_id"]): _semantic(value["receipt"])
                 for value in runtime["rows"]}
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    proposals = json.loads(args.proposal_grounding.read_text())
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                    timeout=300, max_retries=2)
    model = {"id": args.model, "omit_temperature": False}
    palette = ("red", "lime", "cyan", "yellow", "magenta", "orange",
               "deepskyblue", "violet", "chartreuse", "gold", "pink", "white")
    rows = []; total_cost = 0.0; calls = 0
    for source in proposals["rows"]:
        task_id = str(source["task_id"]); row = public[task_id]
        semantic = semantics[task_id]
        requested_role = requested_query_role(semantic)
        tracks = [value for value in source["tracks"] if value["canonical_label"] != "person"]
        track_ids = [str(value["track_id"]) for value in tracks]
        label_to_track = {value["canonical_label"]: str(value["track_id"]) for value in tracks}
        colors = {track_id: palette[index % len(palette)]
                  for index, track_id in enumerate(track_ids)}
        frames, seconds, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=48, max_side=800,
            start_second=0.0, end_second=None)
        indices = [int(value) for value in source["inspection_indices"]]
        selected = [frames[index].copy() for index in indices]
        for column, frame_index in enumerate(indices):
            draw = ImageDraw.Draw(selected[column])
            draw.text((8, 8), f"F{frame_index}", fill="white", stroke_width=3, stroke_fill="black")
            for detection in source["ontology_detections"]:
                if int(detection["frame_index"]) != frame_index:
                    continue
                track_id = label_to_track.get(str(detection["label"]))
                if track_id is None:
                    continue
                box = tuple(float(value) for value in detection["bbox_xyxy"])
                draw.rectangle(box, outline=colors[track_id], width=5)
                draw.text((box[0] + 2, box[1] + 2), track_id,
                          fill=colors[track_id], stroke_width=2, stroke_fill="black")
        panels = _panels(selected, [seconds[index] for index in indices],
                         frames_per_panel=4, frame_width=224, quality=86)
        table = ", ".join(f"{value['track_id']}={value['canonical_label']}"
                          for value in tracks)
        prompt = (
            f"Public question (perceptual scope only; do not answer): {row['question']}\n"
            f"Requested typed predicate: {source['predicate']}\n"
            f"Requested typed role: {requested_role}\n"
            f"Temporal operator: {source['temporal_operator']}\n"
            f"Displayed original frame IDs: {indices}\n"
            f"Fallible public-ontology proposals: {table}\n"
            "Return the single anonymous track_id filling the requested role, or ABSTAIN."
        )
        panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
        core = {"protocol": "AGQA_QGV2_QWEN235_TRACK_ADJUDICATOR_V1",
                "task_id": task_id, "question_sha256": row["question_sha256"],
                "semantic_receipt_sha256": semantic.receipt_sha256,
                "proposal_report_sha256": proposals["report_sha256"],
                "track_table_sha256": stable_hash(tracks), "panel_sha256s": panel_hashes,
                "model": model}
        payload, usage, reused = _cached_provider_call(
            cache_dir=args.cache_dir, call_name=f"track_{task_id}", input_core=core,
            invoke=lambda: _provider_json_call(
                client, model=model, system=SYSTEM,
                content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                max_tokens=300, response_format=_format(track_ids, indices)))
        chosen = str(payload["selected_track_id"])
        confidence = float(payload["confidence"])
        evidence = tuple(sorted(set(int(value) for value in payload["evidence_frames"])))
        if chosen == "ABSTAIN" or not evidence:
            chosen = "ABSTAIN"; confidence = 0.0; evidence = ()
        candidates = [QueryCandidateEvidence(
            track_id=str(value["track_id"]), requested_role=requested_role,
            status="SUPPORTED" if str(value["track_id"]) == chosen else "REFUTED",
            confidence=confidence if str(value["track_id"]) == chosen else 1.0,
            evidence_frames=evidence if str(value["track_id"]) == chosen else (indices[0],),
        ) for value in tracks]
        total_cost += float(usage.get("reported_cost_usd", 0.0)); calls += int(not reused)
        rows.append({"task_id": task_id, "requested_role": requested_role,
                     "selected_track_id": chosen,
                     "selected_label_for_executor": next(
                         (value["canonical_label"] for value in tracks
                          if str(value["track_id"]) == chosen), None),
                     "confidence": confidence, "evidence_frames": list(evidence),
                     "candidates": [asdict(value) for value in candidates],
                     "raw_payload": payload, "usage": usage, "cache_reused": reused})
        print(json.dumps({"task_id": task_id, "track": chosen,
                          "confidence": confidence, "cost_usd": usage.get("reported_cost_usd", 0.0)}), flush=True)
    body = {"schema_version": "agqa-qwen235-track-adjudicator-pilot-v1",
            "status": "TRACK_ADJUDICATION_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
            "model": args.model, "proposal_report_sha256": proposals["report_sha256"],
            "semantic_runtime_sha256": runtime["runtime_sha256"],
            "frame_budget": 48, "rows": rows, "provider_calls": calls,
            "reported_cost_usd": total_cost, "answer_read": False,
            "official_scene_graph_read": False, "functional_program_read": False,
            "source_controller_read": False, "target_outcome_read": False}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
