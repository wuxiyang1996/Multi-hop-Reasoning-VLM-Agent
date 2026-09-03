#!/usr/bin/env python3
"""Outcome-blind second-pass verification of Layer-B event hypotheses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI

from motif_transfer.agqa_layer_b_contracts import (
    GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt,
)
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _cached_provider_call, _panel_content, _provider_json_call
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video
from scripts.collect_agqa_layer_b_local_grounding import _frame_hash, _slot_prompt
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


SYSTEM = """You are a frozen, answer-blind video evidence verifier. You receive chronological proxy frames, operator-free semantic slots, and candidate event hypotheses from another frozen visual tool. Verify each candidate independently from pixels. Return JSON only as {"adjudications":[{"event_id":"E0","status":"SUPPORTED|REFUTED|UNKNOWN","start_frame":0,"end_frame":3,"evidence_frames":[0,3],"confidence":0.8,"reason":"visible evidence only"}]}. Rules:
- Return exactly one adjudication for every supplied event_id and never invent a new event.
- SUPPORTED means the predicate/object and temporal interval are visibly supported. Correct its proxy-frame boundaries when needed and cite 1-3 in-interval frames.
- REFUTED means pixels visibly contradict the candidate. UNKNOWN means sampled pixels are insufficient. REFUTED/UNKNOWN may use an empty evidence_frames list.
- Do not answer the question, compare answer choices, execute logic, infer a functional program, read an official scene graph, or emit controller actions.
- Never use absence from sampled frames as proof of REFUTED; use UNKNOWN unless contradictory pixels are visible."""


def _validate(payload: object, *, candidates: tuple[str, ...], frame_count: int) -> dict[str, dict]:
    if not isinstance(payload, dict) or set(payload) != {"adjudications"}:
        raise ValueError("adjudicator payload must contain only adjudications")
    rows = payload["adjudications"]
    if not isinstance(rows, list) or len(rows) != len(candidates):
        raise ValueError("adjudicator must return one row per event")
    output: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("adjudication row must be an object")
        event_id = str(row.get("event_id", "")); status = str(row.get("status", ""))
        if event_id not in candidates or event_id in output:
            raise ValueError("adjudication event IDs must exactly match candidates")
        if status not in {"SUPPORTED", "REFUTED", "UNKNOWN"}:
            raise ValueError("invalid adjudication status")
        confidence = float(row.get("confidence", -1))
        if not 0 <= confidence <= 1:
            raise ValueError("invalid adjudication confidence")
        evidence = tuple(sorted(set(int(value) for value in row.get("evidence_frames", ()))))
        start = int(row.get("start_frame", 0)); end = int(row.get("end_frame", 0))
        if not 0 <= start <= end < frame_count:
            raise ValueError("adjudication interval outside proxy frames")
        if status == "SUPPORTED" and (
            not evidence or len(evidence) > 3 or any(value < start or value > end for value in evidence)
        ):
            raise ValueError("supported adjudication requires in-interval evidence")
        output[event_id] = {
            "event_id": event_id, "status": status, "start_frame": start,
            "end_frame": end, "evidence_frames": list(evidence),
            "confidence": confidence, "reason": str(row.get("reason", "")),
        }
    if set(output) != set(candidates):
        raise ValueError("adjudication omitted candidates")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--base-grounding", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("adjudicated grounding output is immutable")

    cohort = json.loads(args.cohort.read_text()); base = json.loads(args.base_grounding.read_text())
    if base["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("base grounding is not frozen")
    if base["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("cohort/base mismatch")
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=2)
    model = {"id": args.model, "omit_temperature": True}
    backend = stable_hash({
        "protocol": "INDEPENDENT_PIXEL_EVENT_ADJUDICATION_V1", "model": model,
        "system": SYSTEM, "base_grounder_sha256": base["report_sha256"],
        "sampling": "REPLAY_EXACT_BASE_UNIFORM_PROXY_FRAMES",
    })

    rows = []; calls = 0; total_cost = 0.0; incremental_cost = 0.0
    for raw in base["rows"]:
        task_id = str(raw["task_id"]); old = _grounding(raw["grounding_receipt"])
        semantic = _semantic(raw["semantic_receipt"]); public_row = public[task_id]
        frame_count = len(old.selected_frame_sha256s)
        frames, seconds, metadata = _sample_video(Path(public_row["video_path"]), frame_count=frame_count, max_side=448)
        replay_hashes = tuple(_frame_hash(frame) for frame in frames)
        if replay_hashes != old.selected_frame_sha256s:
            raise ValueError(f"{task_id}: adjudicator did not replay exact base frames")
        panels = _panels(frames, seconds, {"frames_per_panel": 6, "panel_frame_width": 224, "jpeg_quality": 82})
        candidates = tuple(event.event_id for event in old.events)
        if not candidates:
            adjudications = {}; usage = {"reported_cost_usd": 0.0}; reused = True; provider_error = None
        else:
            event_payload = [asdict(event) for event in old.events]
            text = (
                f"Question for perceptual relevance only (never answer it): {public_row['question']}\n"
                f"Operator-free semantic slots:\n{_slot_prompt(semantic)}\n"
                f"Candidate event hypotheses:\n{json.dumps(event_payload, separators=(',', ':'))}"
            )
            content = [{"type": "text", "text": text}] + _panel_content(panels)
            core = {
                "prompt_version": "AGQA_LAYER_B_EVENT_ADJUDICATION_V1", "model": model,
                "task_id": task_id, "question_sha256": public_row["question_sha256"],
                "semantic_receipt_sha256": semantic.receipt_sha256,
                "base_grounding_receipt_sha256": old.receipt_sha256,
                "panel_sha256s": [hashlib.sha256(panel).hexdigest() for panel in panels],
                "max_tokens": args.max_tokens,
            }
            provider_error = None
            try:
                payload, usage, reused = _cached_provider_call(
                    cache_dir=args.cache_dir, call_name=f"adjudicate_{task_id}", input_core=core,
                    invoke=lambda: _provider_json_call(
                        client, model=model, system=SYSTEM, content=content,
                        max_tokens=args.max_tokens, response_format={"type": "json_object"},
                    ),
                )
                adjudications = _validate(payload, candidates=candidates, frame_count=frame_count)
            except Exception as exc:
                # Fail closed: malformed verification cannot certify any event.
                provider_error = f"{type(exc).__name__}:{exc}"
                adjudications = {
                    event_id: {"event_id": event_id, "status": "UNKNOWN", "start_frame": 0,
                               "end_frame": 0, "evidence_frames": [], "confidence": 0.0,
                               "reason": "PROVIDER_OR_CONTRACT_FAILURE"}
                    for event_id in candidates
                }
                usage = {"reported_cost_usd": 0.0}; reused = False
            calls += int(not reused); total_cost += float(usage.get("reported_cost_usd", 0.0))
            incremental_cost += float(usage.get("reported_cost_usd", 0.0)) if not reused else 0.0

        events = []
        for event in old.events:
            decision = adjudications[event.event_id]
            if decision["status"] != "SUPPORTED":
                continue
            events.append(GroundedEvent(
                event_id=f"E{len(events)}", subject=event.subject, predicate=event.predicate,
                object=event.object, start_frame=decision["start_frame"], end_frame=decision["end_frame"],
                evidence_frames=tuple(decision["evidence_frames"]),
                confidence=min(event.confidence, decision["confidence"]),
                semantic_slot_ids=event.semantic_slot_ids,
            ))
        receipt = RawVideoEventGraphReceipt.create(
            task_id=task_id, video_sha256=old.video_sha256,
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=old.selected_frame_indices,
            selected_frame_sha256s=old.selected_frame_sha256s,
            events=events, grounder_backend_sha256=backend,
            frame_budget=int(base["frame_budget"]) + frame_count,
            provider_calls=old.provider_calls + (1 if candidates else 0),
        )
        state = LayerBTaskStateReceipt.create(semantic, receipt)
        row = dict(raw)
        row.update(
            grounding_receipt=asdict(receipt), task_state_receipt=asdict(state),
            event_adjudications=list(adjudications.values()), adjudicator_usage=usage,
            adjudicator_cache_reused=reused, adjudicator_provider_error=provider_error,
            adjudicator_replay_metadata=metadata,
        )
        rows.append(row)
        print(json.dumps({"task_id": task_id, "before": len(old.events), "after": len(events),
                          "cost_usd": usage.get("reported_cost_usd", 0.0)}), flush=True)

    body = {key: value for key, value in base.items()
            if key not in {"rows", "report_sha256", "grounder_backend_sha256", "frame_budget", "provider_calls"}}
    body.update({
        "schema_version": "agqa-layer-b-adjudicated-grounding-v1",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES", "rows": rows,
        "grounder_backend_sha256": backend,
        "frame_budget": int(base["frame_budget"]) + int(base["rows"][0]["grounding_receipt"]["frame_budget"]),
        "provider_calls": int(base.get("provider_calls", 0)) + calls,
        "base_grounding_report_sha256": base["report_sha256"], "adjudicator_model": args.model,
        "reported_receipt_provider_cost_usd": float(base.get("reported_receipt_provider_cost_usd", 0.0)) + total_cost,
        "incremental_provider_cost_usd": incremental_cost,
        "all_harness_arms_share_exact_receipts": True,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
    })
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "calls": calls, "incremental_cost_usd": incremental_cost,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
