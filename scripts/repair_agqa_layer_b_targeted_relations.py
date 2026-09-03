#!/usr/bin/env python3
"""Outcome-blind focused relation/object grounding over frozen VLM frames."""

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
from motif_transfer.agqa_semantic_slots import relation_grounding_obligations
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _provider_json_call,
)
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _frame_hash(frame) -> str:
    return stable_hash({
        "mode": frame.mode, "size": frame.size,
        "pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    })


def _response_format(frame_count: int) -> dict:
    observation = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "object": {"type": "string"},
            "start_frame": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            "end_frame": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            "evidence_frames": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["object", "start_frame", "end_frame", "evidence_frames", "confidence"],
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "observations": {"type": "array", "maxItems": 8, "items": observation},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["observations", "uncertainties"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_exact_relation_objects_v1", "strict": True, "schema": schema,
    }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("targeted relation grounding output is immutable")

    cohort = json.loads(args.cohort.read_text())
    base = json.loads(args.input.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=2)
    model = {"id": args.model}
    system = (
        "You are an answer-blind exact visual relation grounding tool. Given one visual predicate and "
        "chronological video frames, list every directly visible object that stands in that exact relation "
        "to the person, with supported intervals. Use concrete short object names. Reject implied, hidden, "
        "uncertain, merely nearby, or semantically related objects. Never answer a question, choose an option, "
        "infer temporal logic, or emit a functional program."
    )
    backend = stable_hash({
        "upstream": base["grounder_backend_sha256"], "model": args.model,
        "tool": "SEMANTIC_TARGETED_EXACT_RELATION_OBJECTS_V1",
        "frame_budget": base["frame_budget"], "system": system,
    })
    rows = []
    calls = 0
    incremental_cost = 0.0
    total_cost = float(base.get("reported_receipt_provider_cost_usd", 0.0))
    for raw in base["rows"]:
        task_id = str(raw["task_id"])
        semantic = _semantic(raw["semantic_receipt"])
        old = _grounding(raw["grounding_receipt"])
        frames, seconds, metadata = _sample_video(
            Path(public[task_id]["video_path"]), frame_count=base["frame_budget"], max_side=448,
        )
        if tuple(_frame_hash(frame) for frame in frames) != old.selected_frame_sha256s:
            raise ValueError("relation tool did not receive the exact frozen VLM frames")
        panels = _panels(frames, seconds, {
            "frames_per_panel": 6, "panel_frame_width": 224, "jpeg_quality": 82,
        })
        events = list(old.events)
        receipts = []
        existing = {
            (event.predicate.casefold().strip(), event.object.casefold().strip(),
             event.start_frame, event.end_frame)
            for event in events
        }
        for predicate, slot_id in relation_grounding_obligations(semantic):
            content = [{
                "type": "text",
                "text": f"Exact visual predicate to ground: {predicate}\nReturn related objects only. Do not answer any question.",
            }] + _panel_content(panels)
            core = {
                "prompt_version": "AGQA_TARGETED_RELATION_OBJECTS_V1",
                "model": model, "task_id": task_id, "predicate": predicate,
                "slot_id": slot_id,
                "panel_sha256s": [hashlib.sha256(panel).hexdigest() for panel in panels],
                "frame_budget": base["frame_budget"],
            }
            payload, usage, reused = _cached_provider_call(
                cache_dir=args.cache_dir,
                call_name=f"{task_id}_{stable_hash([predicate, slot_id])[:10]}",
                input_core=core,
                invoke=lambda: _provider_json_call(
                    client, model=model, system=system, content=content,
                    max_tokens=900, response_format=_response_format(base["frame_budget"]),
                ),
            )
            calls += int(not reused)
            cost = float(usage.get("reported_cost_usd", 0.0))
            total_cost += cost
            if not reused:
                incremental_cost += cost
            accepted = 0
            rejected = []
            for observation in payload["observations"]:
                obj = str(observation["object"]).casefold().strip()
                key_tuple = (
                    predicate.casefold().strip(), obj,
                    int(observation["start_frame"]), int(observation["end_frame"]),
                )
                if not obj or obj in {"object", "something", "unknown", "none", "n/a"}:
                    rejected.append("NON_CONCRETE_OBJECT")
                    continue
                if key_tuple in existing:
                    rejected.append("EXACT_DUPLICATE")
                    continue
                try:
                    event = GroundedEvent(
                        f"E{len(events)}", "person", predicate, obj,
                        key_tuple[2], key_tuple[3],
                        tuple(sorted(set(int(value) for value in observation["evidence_frames"]))),
                        float(observation["confidence"]), (slot_id,),
                    )
                    event.validate(len(frames))
                except Exception as exc:
                    rejected.append(f"{type(exc).__name__}:{exc}")
                    continue
                events.append(event)
                existing.add(key_tuple)
                accepted += 1
            receipts.append({
                "predicate": predicate, "slot_id": slot_id,
                "payload": payload, "usage": usage, "cache_reused": reused,
                "accepted_observations": accepted, "rejected": rejected,
            })
        receipt = RawVideoEventGraphReceipt.create(
            task_id=task_id, video_sha256=old.video_sha256,
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=old.selected_frame_indices,
            selected_frame_sha256s=old.selected_frame_sha256s,
            events=events, grounder_backend_sha256=backend,
            frame_budget=old.frame_budget, provider_calls=old.provider_calls + len(receipts),
        )
        state = LayerBTaskStateReceipt.create(semantic, receipt)
        row = dict(raw)
        row.update(
            grounding_receipt=asdict(receipt), task_state_receipt=asdict(state),
            targeted_relation_receipts=receipts,
        )
        rows.append(row)
        print(json.dumps({
            "task_id": task_id, "obligations": len(receipts),
            "added_events": len(events) - len(old.events),
        }), flush=True)

    body = {
        key: value for key, value in base.items()
        if key not in {
            "rows", "report_sha256", "grounder_backend_sha256", "provider_calls",
            "reported_receipt_provider_cost_usd", "incremental_provider_cost_usd",
        }
    }
    body.update({
        "rows": rows, "grounder_backend_sha256": backend,
        "provider_calls": sum(row["grounding_receipt"]["provider_calls"] for row in rows),
        "reported_receipt_provider_cost_usd": total_cost,
        "incremental_provider_cost_usd": incremental_cost,
        "targeted_relation_tool": True, "additional_provider_calls": calls,
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
    })
    body["report_sha256"] = stable_hash(body)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": len(rows), "calls": calls,
        "incremental_cost_usd": incremental_cost,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
