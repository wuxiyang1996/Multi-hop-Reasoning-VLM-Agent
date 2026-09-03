#!/usr/bin/env python3
"""Verify AGQA overrides through explicit precondition-transition-effect evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.agqa_qwen235_selective_authorizer import authorize_source_override  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.visual_wrapper_bridge import build_video_registry, execute_video_intervention  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import _cached_provider_call, _panel_content, _panels, _provider_json_call, _sample_video_range  # noqa: E402
from scripts.collect_agqa2_qwen235_global_verifier_v5 import _direct_boolean  # noqa: E402
from scripts.collect_agqa2_qwen235_override_verifier_v1 import _sha256  # noqa: E402


def _response_format() -> dict:
    return {"type": "json_schema", "json_schema": {"name": "agqa_transition_evidence_v7", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "supported": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "entailed_state_change": {"type": "string"},
            "precondition_observed": {"type": "boolean"}, "transition_observed": {"type": "boolean"},
            "effect_observed": {"type": "boolean"}, "same_entity_binding": {"type": "boolean"},
            "before_evidence_frames": {"type": "array", "items": {"type": "integer"}},
            "transition_evidence_frames": {"type": "array", "items": {"type": "integer"}},
            "after_evidence_frames": {"type": "array", "items": {"type": "integer"}},
            "confounders": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["supported", "confidence", "entailed_state_change", "precondition_observed", "transition_observed", "effect_observed", "same_entity_binding", "before_evidence_frames", "transition_evidence_frames", "after_evidence_frames", "confounders"]
    }}}


def transition_authorized(payload: dict, *, threshold: float = 0.9) -> bool:
    return bool(payload.get("supported")) and float(payload.get("confidence", 0)) >= threshold and all(bool(payload.get(key)) for key in ("precondition_observed", "transition_observed", "effect_observed", "same_entity_binding"))


def collect(config_path: Path, keys_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    base_path, manifest_path = REPO_ROOT / config["base_report"], REPO_ROOT / config["manifest"]
    if _sha256(base_path) != config["base_report_file_sha256"] or _sha256(manifest_path) != config["manifest_file_sha256"]:
        raise ValueError("frozen input hash mismatch")
    base, manifest = json.loads(base_path.read_text()), json.loads(manifest_path.read_text())
    samples = {row["task_id"]: row for row in manifest["samples"]}
    key = runpy.run_path(str(keys_path)).get(config["model"]["api_key_name"])
    client = OpenAI(api_key=key, base_url=config["model"]["base_url"], timeout=240, max_retries=2)
    rows, usage = [], []
    n = int(config["tool_frame_budget"])
    for runtime in base["rows"]:
        source = authorize_source_override(runtime)
        disagreement = _direct_boolean(runtime["direct_response"]) != source["decision"]
        if not source["authorized"] or not disagreement:
            rows.append({"task_id": runtime["task_id"], "candidate_authorized": False, "verifier": None})
            continue
        sample = samples[runtime["task_id"]]
        frames, seconds, metadata = _sample_video_range(Path(sample["video_path"]), frame_count=n, max_side=512)
        duration = float(metadata["duration_seconds"])
        registry, _ = build_video_registry(frames, duration_seconds=duration, wrapper_root=config["wrapper_root"], required_tools=("sample_frames",))
        observed, receipt = execute_video_intervention(registry, frames, tool="sample_frames", arguments={"start_sec": 0.0, "end_sec": duration, "n": n})
        indices = receipt["proxy_frame_indices"]
        panels = _panels(observed, [seconds[index] for index in indices], frames_per_panel=4, frame_width=224, quality=80)
        target = str(runtime["grounding_receipt"]["operand_a"])
        system = (
            "You are a fail-closed intervention evidence verifier. For the exact target, first state the minimal visually observable state change it entails. "
            "Then require chronological evidence of the precondition, the actual transition, and the resulting effect on the same bound person/object. "
            "Mere final-state holding, proximity, preparation, aftermath, or a related action is insufficient. For sustained state predicates, onset plus sustained effect must be visible. "
            "supported may be true only when all four boolean evidence requirements are true. Frame IDs are local F0... Return only JSON."
        )
        content = [{"type": "text", "text": f"Exact target intervention/relation: {target}"}] + _panel_content(panels)
        core = {"prompt_version": "AGQA_TRANSITION_EVIDENCE_V7", "model": config["model"]["id"], "target": target, "tool_receipt": receipt, "frame_count": n}
        payload, call_usage, reused = _cached_provider_call(cache_dir=output_path.parent / "call_cache", call_name=f"transition_verify_{runtime['task_id']}", input_core=core, invoke=lambda: _provider_json_call(client, model=config["model"], system=system, content=content, max_tokens=700, response_format=_response_format()))
        usage.append(call_usage)
        rows.append({"task_id": runtime["task_id"], "candidate_authorized": True, "target_sha256": stable_hash(target), "tool_receipt": receipt, "verifier": payload, "transition_authorized_at_0_9": transition_authorized(payload), "usage": call_usage, "cache_reused": reused, "answer_read": False, "program_read": False, "scene_graph_read": False, "source_identity_read": False})
        print(f"verified {runtime['task_id']}", flush=True)
    body = {"schema_version": "agqa-transition-verifier-runtime-v7", "status": "FROZEN_RUNTIME_BEFORE_EVALUATOR", "config_sha256": stable_hash(config), "base_report_sha256": base["report_sha256"], "candidate_count": sum(row["candidate_authorized"] for row in rows), "provider_calls": len(usage), "reported_provider_cost_usd": sum(float(row["reported_cost_usd"]) for row in usage), "rows": rows}
    result = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--keys", default="/fs/gamma-projects/vlm-robot/keys.py", type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); result = collect(args.config, args.keys, args.output)
    print(json.dumps({key: result[key] for key in ("status", "candidate_count", "provider_calls", "reported_provider_cost_usd", "report_sha256")}, indent=2))


if __name__ == "__main__": main()
