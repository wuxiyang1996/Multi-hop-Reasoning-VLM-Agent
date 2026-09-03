#!/usr/bin/env python3
"""Freeze a neural-only/fallback actor over the shared Layer-B event graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from transformers import AutoConfig
from vllm import LLM, SamplingParams

from motif_transfer.contracts import stable_hash


SYSTEM = """You are the frozen shared target actor for video QA. You receive a question, operator-free semantic slots, and pixel-grounded events. Return JSON only as {"answer":"..."}. Answer with the shortest AGQA-compatible value (for example yes, no, before, after, an object phrase, or an action phrase). Do not invent unseen evidence. You have no official scene graph, functional program, gold answer, source game, or symbolic Harness."""


def _parse_answer(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end >= start:
        try:
            payload = json.loads(cleaned[start:end + 1])
            answer = payload.get("answer")
        except (json.JSONDecodeError, AttributeError):
            answer = cleaned
    else:
        answer = cleaned
    answer = re.sub(r"\s+", " ", str(answer or "").strip().casefold())
    return answer or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Layer-B shared fallback is immutable")
    cohort = json.loads(args.cohort.read_text())
    grounding = json.loads(args.grounding.read_text())
    if grounding["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("raw grounding is not frozen")
    if grounding["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("fallback cohort/grounding mismatch")
    questions = {str(row["task_id"]): str(row["question"]) for row in cohort["rows"]}
    conversations = []
    task_ids = []
    for row in grounding["rows"]:
        task_id = str(row["task_id"]); task_ids.append(task_id)
        semantic = row["semantic_receipt"]
        slots = [{
            "slot_id": slot["slot_id"], "kind": slot["kind"], "surface": slot["surface"],
            "children": slot["children"], "attributes": slot["attributes"],
        } for slot in semantic["slots"]]
        events = [{
            "event_id": event["event_id"], "subject": event["subject"],
            "predicate": event["predicate"], "object": event["object"],
            "start_frame": event["start_frame"], "end_frame": event["end_frame"],
            "evidence_frames": event["evidence_frames"], "confidence": event["confidence"],
            "semantic_slot_ids": event["semantic_slot_ids"],
        } for event in row["grounding_receipt"]["events"]]
        user = json.dumps({
            "question": questions[task_id], "answer_kind": semantic["answer_kind"],
            "root_slot_id": semantic["root_slot_id"], "semantic_slots": slots,
            "pixel_grounded_events": events,
        }, separators=(",", ":"), sort_keys=True)
        conversations.append([
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
        ])
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    actor_sha = stable_hash({
        "model": args.model, "model_commit": getattr(config, "_commit_hash", None),
        "system": SYSTEM, "temperature": 0, "thinking": False,
        "input_grounding_sha256": grounding["report_sha256"],
    })
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=.90, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for messages in conversations]
    outputs = llm.generate(
        prompts, SamplingParams(temperature=0, max_tokens=128), use_tqdm=True,
    )
    rows = []
    for task_id, output in zip(task_ids, outputs):
        raw = output.outputs[0].text
        rows.append({"task_id": task_id, "prediction": _parse_answer(raw), "raw_response": raw})
    body = {
        "schema_version": "agqa-layer-b-shared-fallback-v1",
        "status": "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES",
        "cohort_sha256": cohort["cohort_sha256"],
        "grounding_report_sha256": grounding["report_sha256"],
        "actor_backend_sha256": actor_sha, "model": args.model, "rows": rows,
        "shared_by_all_five_arms": True, "answer_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "source_controller_read": False, "provider_calls": 0,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "rows": len(rows),
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
