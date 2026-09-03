#!/usr/bin/env python3
"""Collect arm-shared, answer-blind atomic visual evidence for Layer-B.

Every provider call sees exactly one operator-free proposition.  It cannot see
competing claims, a candidate slot, source capabilities, an official program,
or an answer.  Thus this is a frozen target-native grounder, not a controller.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI

from motif_transfer.agqa_layer_b_epistemic import (
    AtomicVisualClaimDecision, AtomicVisualClaimReceipt, extract_atomic_claims,
)
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _provider_json_call,
)
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video
from scripts.collect_agqa_layer_b_local_grounding import _frame_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


SYSTEM = """You are a frozen, answer-blind video proposition grounder. Verify exactly one proposition against chronological frames from the full video. The proposition inherits every temporal, ordinal, reference-resolution, and localization constraint stated in the question: evidence that it happened somewhere else or at another time does not support it. Return JSON only with status, confidence, evidence_frames, and rationale. SUPPORTED means the scoped proposition is visibly established. REFUTED requires visible contradictory evidence within the same scoped query, never mere absence from sampled frames. Otherwise return UNKNOWN. Do not answer the question, compare alternatives, infer or emit symbolic operators, use an official scene graph, or emit any answer/correct/program/source fields."""


def _response_format(frame_count: int) -> dict:
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "UNKNOWN"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_frames": {
                # Alibaba's structured-output subset rejects uniqueItems; the
                # local parser below still canonicalizes and validates IDs.
                "type": "array", "maxItems": 3,
                "items": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            },
            "rationale": {"type": "string"},
        },
        "required": ["status", "confidence", "evidence_frames", "rationale"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_layer_b_atomic_visual_claim_v1", "strict": True, "schema": schema,
    }}


def _validate(payload: object, *, claim_id: str, frame_hashes: tuple[str, ...]) -> AtomicVisualClaimDecision:
    if not isinstance(payload, dict) or set(payload) != {
        "status", "confidence", "evidence_frames", "rationale",
    }:
        raise ValueError("atomic claim payload has invalid keys")
    status = str(payload["status"])
    if status not in {"SUPPORTED", "REFUTED", "UNKNOWN"}:
        raise ValueError("invalid atomic claim status")
    confidence = float(payload["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("invalid atomic claim confidence")
    indices = tuple(sorted(set(int(value) for value in payload["evidence_frames"])))
    if len(indices) > 3 or any(index < 0 or index >= len(frame_hashes) for index in indices):
        raise ValueError("atomic claim evidence is outside frozen frames")
    if status in {"SUPPORTED", "REFUTED"} and not indices:
        raise ValueError("decisive atomic claims require cited pixels")
    return AtomicVisualClaimDecision(
        claim_id=claim_id, status=status, confidence=confidence,
        evidence_frame_indices=indices,
        evidence_frame_sha256s=tuple(frame_hashes[index] for index in indices),
        rationale=str(payload["rationale"]).strip(),
    )


def _one(
    *, row: dict, public: dict, model: str, key: str, cache_dir: Path,
    frame_count: int, max_tokens: int, backend: str,
) -> dict:
    semantic = _semantic(row["semantic_receipt"])
    grounding = _grounding(row["grounding_receipt"])
    claims = extract_atomic_claims(semantic)
    frames, seconds, metadata = _sample_video(
        Path(public["video_path"]), frame_count=frame_count, max_side=448,
    )
    frame_hashes = tuple(_frame_hash(frame) for frame in frames)
    panels = _panels(frames, seconds, {
        "frames_per_panel": 6, "panel_frame_width": 224, "jpeg_quality": 82,
    })
    client = OpenAI(
        api_key=key, base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=2,
    )
    decisions = []; raw_payloads = {}; usage_rows = {}; errors = {}
    for claim in claims:
        prompt = (
            "Question supplies perceptual context only; never answer it:\n"
            f"{public['question']}\n\nSingle scoped proposition to verify independently:\n"
            "At the exact temporal/ordinal/reference scope requested by the question, "
            f"the following branch is satisfied: {claim.proposition}\n\n"
            "Do not mark SUPPORTED merely because its action/object occurs elsewhere in "
            f"the video. Frame IDs are F0..F{frame_count - 1}."
        )
        content = [{"type": "text", "text": prompt}] + _panel_content(panels)
        core = {
            "protocol": "AGQA_LAYER_B_SCOPED_ATOMIC_CLAIM_V2", "model": model,
            "task_id": semantic.task_id, "claim": asdict(claim),
            "question_sha256": public["question_sha256"],
            "semantic_receipt_sha256": semantic.receipt_sha256,
            "raw_event_graph_receipt_sha256": grounding.receipt_sha256,
            "panel_sha256s": [hashlib.sha256(panel).hexdigest() for panel in panels],
            "frame_hashes": frame_hashes, "max_tokens": max_tokens,
        }
        try:
            payload, usage, reused = _cached_provider_call(
                cache_dir=cache_dir,
                call_name=f"claim_{semantic.task_id}_{claim.claim_id}", input_core=core,
                invoke=lambda: _provider_json_call(
                    client, model={"id": model, "omit_temperature": True}, system=SYSTEM,
                    content=content, max_tokens=max_tokens,
                    response_format=_response_format(frame_count),
                ),
            )
            decision = _validate(payload, claim_id=claim.claim_id, frame_hashes=frame_hashes)
            error = None
        except Exception as exc:
            payload = {}; usage = {"reported_cost_usd": 0.0}; reused = False
            decision = AtomicVisualClaimDecision(
                claim.claim_id, "UNKNOWN", 0.0, (), (), "PROVIDER_OR_CONTRACT_FAILURE",
            )
            error = f"{type(exc).__name__}:{exc}"
        decisions.append(decision); raw_payloads[claim.claim_id] = payload
        usage_rows[claim.claim_id] = usage; errors[claim.claim_id] = error
    receipt = AtomicVisualClaimReceipt.create(
        task_id=semantic.task_id, semantic_receipt_sha256=semantic.receipt_sha256,
        raw_event_graph_receipt_sha256=grounding.receipt_sha256,
        claims=claims, decisions=decisions, verifier_backend_sha256=backend,
        frame_budget=frame_count,
    )
    return {
        "task_id": semantic.task_id, "video_id": row["video_id"],
        "claim_receipt": asdict(receipt), "raw_payloads": raw_payloads,
        "usage": usage_rows, "provider_errors": errors, "video_metadata": metadata,
        "panel_sha256s": [hashlib.sha256(panel).hexdigest() for panel in panels],
        "each_call_saw_exactly_one_claim": True,
        "competing_claims_visible_to_provider": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    parser.add_argument("--frame-count", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--positions", default="all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("atomic claim output is immutable")
    cohort = json.loads(args.cohort.read_text()); grounding = json.loads(args.grounding.read_text())
    if grounding["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("base grounding is not frozen")
    if grounding["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("cohort/grounding mismatch")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    rows = grounding["rows"]
    positions = (
        list(range(len(rows))) if args.positions == "all"
        else [int(value) for value in args.positions.replace(":", ",").split(",") if value.strip()]
    )
    selected = [rows[index] for index in positions]
    backend = stable_hash({
        "protocol": "AGQA_LAYER_B_SCOPED_ATOMIC_CLAIM_V2", "system": SYSTEM,
        "model": args.model, "frame_count": args.frame_count,
        "sampling": "UNIFORM_FULL_VIDEO_REUSING_ALREADY_PRESENTED_FRAME_POLICY",
        "max_tokens": args.max_tokens, "base_grounding_sha256": grounding["report_sha256"],
    })
    outputs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _one, row=row, public=public[str(row["task_id"])], model=args.model,
                key=str(key), cache_dir=args.cache_dir, frame_count=args.frame_count,
                max_tokens=args.max_tokens, backend=backend,
            ): position for position, row in zip(positions, selected)
        }
        for future in as_completed(futures):
            result = future.result(); outputs.append((futures[future], result))
            statuses = [x["status"] for x in result["claim_receipt"]["decisions"]]
            print(json.dumps({"task_id": result["task_id"], "statuses": statuses}), flush=True)
    output_rows = [row for _, row in sorted(outputs)]
    costs = sum(float(usage.get("reported_cost_usd", 0.0))
                for row in output_rows for usage in row["usage"].values())
    body = {
        "schema_version": "agqa-layer-b-atomic-claim-grounding-v1",
        "status": "ATOMIC_VISUAL_CLAIMS_FROZEN_BEFORE_OUTCOMES",
        "cohort_sha256": cohort["cohort_sha256"],
        "base_grounding_report_sha256": grounding["report_sha256"],
        "verifier_backend_sha256": backend, "model": args.model,
        "frame_budget_per_task": args.frame_count, "positions": positions,
        "rows": output_rows, "reported_receipt_provider_cost_usd": costs,
        "all_harness_arms_share_exact_receipts": True,
        "answer_read": False, "functional_program_read": False,
        "official_scene_graph_read": False, "source_controller_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(output_rows), "cost_usd": costs,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
