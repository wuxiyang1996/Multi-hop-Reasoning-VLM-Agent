#!/usr/bin/env python3
"""Freeze shared raw-video event graphs for an outcome-blind Layer-B cohort."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from motif_transfer.agqa_layer_b_contracts import (
    AGQASemanticSlotReceipt, GroundedEvent, LayerBTaskStateReceipt,
    RawVideoEventGraphReceipt, SemanticSlotNode,
)
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video


SYSTEM = """You are a frozen, answer-blind video perception tool. Ground the supplied operator-free semantic slots in chronological image panels and return JSON only:
{"events":[{"event_id":"E0","subject":"person","predicate":"hold","object":"blanket","start_frame":2,"end_frame":8,"evidence_frames":[2,5,8],"confidence":0.82,"semantic_slot_ids":["S3"]}],"uncertainties":["..."]}

Rules:
- F0..F23 are proxy-frame indices, in chronological order.
- Emit visually supported entity, action, relation, state, and temporal events needed by the semantic slots. Bind every event to 1-6 supplied slot IDs. Bind only the most specific slots whose kind is LITERAL, ACTION, RELATION, or ENTITY. Never bind QUERY_GOAL, LOGICAL_CONSTRAINT, ORDINAL_CONSTRAINT, TEMPORAL_CONSTRAINT, DURATION_CONSTRAINT, or CHOICE slots.
- When a supplied literal names the observed predicate or object, copy that literal spelling into predicate/object; do not paraphrase it.
- Include at most 24 events. Cite 1-3 evidence frames within each event interval. Preserve repeated occurrences as separate events.
- Do not choose between alternatives, execute logic, answer the question, or report correctness.
- Never emit an answer, selected choice, official scene graph, functional program, VM operator, source game, or controller decision.
- If pixels are insufficient, omit the unsupported event and explain only in uncertainties."""

CAPTION_SYSTEM = """You are a frozen answer-blind video perception tool. Describe the chronological video panels as a compact timeline. Mention only visibly supported actions, objects, relations, state changes, repeated occurrences, and approximate F0..F23 intervals. Do not receive or answer any question. Do not infer hidden intent, correctness, a functional program, a scene graph, a source game, or a controller action."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode()


def _contains_forbidden_key(value: Any) -> bool:
    forbidden = {
        "answer", "correct", "functional_program", "gold", "operator_sequence",
        "program", "selected_option", "sg_grounding", "source_game", "target_outcome",
    }
    if isinstance(value, dict):
        return any(str(key).casefold() in forbidden or _contains_forbidden_key(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("grounder omitted a JSON object")
    payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("grounder JSON must contain an events list")
    if _contains_forbidden_key(payload):
        raise ValueError("grounder crossed the answer/program authority boundary")
    if len(payload["events"]) > 24:
        raise ValueError("grounder exceeded the event budget")
    return payload


def _semantic_receipt(row: dict[str, Any]) -> AGQASemanticSlotReceipt:
    raw = row.get("receipt")
    if row.get("status") != "SEMANTIC_SLOTS_FROZEN" or not isinstance(raw, dict):
        raise ValueError("grounding requires frozen valid semantic slots")
    slots = tuple(SemanticSlotNode(
        slot_id=str(slot["slot_id"]), kind=str(slot["kind"]), surface=str(slot["surface"]),
        children=tuple(slot.get("children", ())),
        attributes=tuple(tuple(pair) for pair in slot.get("attributes", ())),
    ) for slot in raw["slots"])
    receipt = AGQASemanticSlotReceipt(**{**raw, "slots": slots})
    receipt.validate()
    return receipt


def _slot_prompt(semantic: AGQASemanticSlotReceipt) -> str:
    lines = [f"Root: {semantic.root_slot_id}; expected answer type: {semantic.answer_kind}"]
    for slot in semantic.slots:
        attrs = ", ".join(f"{key}={value}" for key, value in slot.attributes) or "none"
        children = ",".join(slot.children) or "none"
        lines.append(
            f"{slot.slot_id} | kind={slot.kind} | meaning={slot.surface} | "
            f"children={children} | attributes={attrs}"
        )
    return "\n".join(lines)


def _canonical_slot_bindings(raw_event: dict[str, Any], semantic: AGQASemanticSlotReceipt) -> tuple[str, ...]:
    """Deterministically project model bindings onto perceptual lexical slots."""
    generic = {"frame", "relation", "relations", "object", "objects", "video", "class", "action", "actions"}
    predicate = re.sub(r"\s+", " ", str(raw_event.get("predicate", "")).casefold()).strip()
    obj = re.sub(r"\s+", " ", str(raw_event.get("object", "")).casefold()).strip()
    targets = {value for value in (predicate, obj, f"{predicate} {obj}".strip()) if value}
    allowed = [slot for slot in semantic.slots
               if slot.kind in {"LITERAL", "ENTITY", "ACTION", "RELATION"}
               and slot.surface.casefold().strip() not in generic]
    lexical = []
    for slot in allowed:
        surface = re.sub(r"\s+", " ", slot.surface.casefold()).strip()
        exact = surface in targets
        related = any(surface in target or target in surface for target in targets)
        if exact or related:
            lexical.append((0 if exact else 1, len(surface), slot.slot_id))
    if lexical:
        return tuple(slot_id for _, _, slot_id in sorted(lexical)[:6])
    supplied = [str(value) for value in raw_event.get("semantic_slot_ids", ())]
    allowed_ids = {slot.slot_id for slot in allowed}
    return tuple(dict.fromkeys(value for value in supplied if value in allowed_ids))[:6]


def _frame_hash(frame: Any) -> str:
    return stable_hash({
        "mode": frame.mode, "size": frame.size,
        "pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    })


def main() -> int:
    # Keep pure receipt/prompt helpers importable by API grounders on hosts
    # without the local GPU runtime installed.
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--caption-first", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Layer-B grounding output is immutable")
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    if runtime["status"] != "SEMANTIC_RUNTIME_FROZEN_BEFORE_VIDEO_OR_OUTCOME":
        raise ValueError("semantic runtime is not frozen")
    if runtime["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("semantic runtime/cohort mismatch")
    if any(runtime.get(key) for key in ("answer_read", "functional_program_read", "scene_graph_read")):
        raise ValueError("semantic runtime crossed target authority boundary")
    semantics = {str(row["task_id"]): _semantic_receipt(row) for row in runtime["rows"]}
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    backend_sha = stable_hash({
        "model": args.model, "model_commit": getattr(config, "_commit_hash", None),
        "architectures": getattr(config, "architectures", None), "system": SYSTEM,
        "thinking": False, "temperature": 0, "frame_count": args.frame_count,
        "sampling": "uniform_full_video", "frames_per_panel": 6,
        "panel_frame_width": 224, "jpeg_quality": 82,
        "caption_first": args.caption_first,
        "caption_system": CAPTION_SYSTEM if args.caption_first else None,
    })
    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model, dtype="bfloat16", max_model_len=16384,
        gpu_memory_utilization=.90, trust_remote_code=True,
        limit_mm_per_prompt={"image": (args.frame_count + 5) // 6},
    )
    sampling = SamplingParams(temperature=0, max_tokens=2048)
    caption_sampling = SamplingParams(temperature=0, max_tokens=512)
    frozen_rows = []
    for position, row in enumerate(cohort["rows"]):
        task_id = str(row["task_id"])
        semantic = semantics[task_id]
        checkpoint_path = (args.checkpoint_dir / f"{stable_hash(task_id)}.json"
                           if args.checkpoint_dir else None)
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            if (checkpoint.get("grounder_backend_sha256") != backend_sha
                    or checkpoint.get("semantic_receipt_sha256") != semantic.receipt_sha256):
                raise ValueError("grounding checkpoint provenance mismatch")
            frozen_rows.append(checkpoint["row"])
            print(json.dumps({"position": position + 1, "rows": len(cohort["rows"]),
                              "task_id": task_id, "checkpoint_reused": True}), flush=True)
            continue
        video_path = Path(row["video_path"])
        frames, seconds, metadata = _sample_video(
            video_path, frame_count=args.frame_count, max_side=args.max_side,
        )
        panels = _panels(frames, seconds, {
            "frames_per_panel": 6, "panel_frame_width": 224, "jpeg_quality": 82,
        })
        panel_content: list[dict[str, Any]] = []
        for panel_index, panel in enumerate(panels):
            panel_content.extend((
                {"type": "text", "text": f"Chronological panel {panel_index + 1}:"},
                {"type": "image_url", "image_url": {"url": _data_url(panel)}},
            ))
        timeline = None
        if args.caption_first:
            caption_messages = [
                {"role": "system", "content": CAPTION_SYSTEM},
                {"role": "user", "content": panel_content},
            ]
            caption_output = llm.chat(
                caption_messages, sampling_params=caption_sampling,
                chat_template_kwargs={"enable_thinking": False}, use_tqdm=False,
            )
            timeline = caption_output[0].outputs[0].text.strip()
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Question for perceptual relevance only (never answer it): {row['question']}\n"
                f"Frozen semantic slots:\n{_slot_prompt(semantic)}\n"
                + (f"Independent answer-blind timeline:\n{timeline}" if timeline else "")
            ),
        }]
        content.extend(panel_content)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        output = llm.chat(
            messages, sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": False}, use_tqdm=False,
        )
        raw_response = output[0].outputs[0].text
        response_rejection = None
        try:
            payload = _parse_json(raw_response)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # A malformed/authority-violating response contributes no event;
            # it never receives a repair call or a different frame budget.
            response_rejection = f"{type(exc).__name__}:{exc}"
            payload = {"events": [], "uncertainties": ["GROUNDING_RESPONSE_REJECTED"]}
        events = []; rejected_events = []
        known_slot_ids = {slot.slot_id for slot in semantic.slots}
        perceptual_slot_ids = {slot.slot_id for slot in semantic.slots
                               if slot.kind in {"LITERAL", "ENTITY", "ACTION", "RELATION"}}
        for event_index, raw_event in enumerate(payload["events"]):
            try:
                candidate = GroundedEvent(
                    event_id=f"E{len(events)}", subject=str(raw_event.get("subject", "person")),
                    predicate=str(raw_event.get("predicate", "")), object=str(raw_event.get("object", "")),
                    start_frame=int(raw_event["start_frame"]), end_frame=int(raw_event["end_frame"]),
                    evidence_frames=tuple(sorted(set(int(value) for value in raw_event["evidence_frames"]))),
                    confidence=float(raw_event.get("confidence", 0.0)),
                    semantic_slot_ids=_canonical_slot_bindings(raw_event, semantic),
                )
                candidate.validate(len(frames))
                if not set(candidate.semantic_slot_ids) <= known_slot_ids:
                    raise ValueError("event binds a semantic slot absent from the frozen parser receipt")
                if not set(candidate.semantic_slot_ids) <= perceptual_slot_ids:
                    raise ValueError("event binds a non-perceptual semantic slot")
                events.append(candidate)
            except (KeyError, TypeError, ValueError) as exc:
                # Never repair or clamp claimed pixel evidence. Invalid model
                # events fail closed while the shared fallback remains usable.
                rejected_events.append({
                    "raw_event_index": event_index, "reason": f"{type(exc).__name__}:{exc}",
                    "raw_event_sha256": stable_hash(raw_event),
                })
        grounding = RawVideoEventGraphReceipt.create(
            task_id=task_id, video_sha256=_sha_file(video_path),
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=tuple(range(len(frames))),
            selected_frame_sha256s=tuple(_frame_hash(frame) for frame in frames),
            events=events, grounder_backend_sha256=backend_sha,
            frame_budget=args.frame_count, provider_calls=0,
        )
        state = LayerBTaskStateReceipt.create(semantic, grounding)
        frozen_row = {
            "task_id": task_id, "video_id": row["video_id"],
            "semantic_receipt": asdict(semantic), "grounding_receipt": asdict(grounding),
            "task_state_receipt": asdict(state), "raw_response": raw_response,
            "answer_blind_timeline": timeline,
            "uncertainties": payload.get("uncertainties", []), "video_metadata": metadata,
            "response_rejection": response_rejection,
            "rejected_events": rejected_events,
            "panel_sha256s": [hashlib.sha256(panel).hexdigest() for panel in panels],
        }
        frozen_rows.append(frozen_row)
        if checkpoint_path:
            checkpoint_path.write_text(json.dumps({
                "schema_version": "agqa-layer-b-grounding-checkpoint-v1",
                "grounder_backend_sha256": backend_sha,
                "semantic_receipt_sha256": semantic.receipt_sha256,
                "row": frozen_row,
            }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"position": position + 1, "rows": len(cohort["rows"]),
                          "task_id": task_id, "events": len(events),
                          "rejected_events": len(rejected_events)}), flush=True)
    body = {
        "schema_version": "agqa-layer-b-local-grounding-v1",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "grounder_backend_sha256": backend_sha, "model": args.model,
        "frame_budget": args.frame_count, "provider_calls": 0, "rows": frozen_rows,
        "model_invocations_per_task": 2 if args.caption_first else 1,
        "invalid_events_fail_closed_without_clamping": True,
        "rejected_event_count": sum(len(row["rejected_events"]) for row in frozen_rows),
        "rejected_response_count": sum(row["response_rejection"] is not None for row in frozen_rows),
        "all_harness_arms_share_exact_receipts": True, "answer_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "source_controller_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "rows": len(frozen_rows),
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
