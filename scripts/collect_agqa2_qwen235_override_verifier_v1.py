#!/usr/bin/env python3
"""Collect answer-blind wrapper-window verification for candidate overrides."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _cached_provider_call,
    _panel_content,
    _panels,
    _provider_json_call,
    _sample_video_range,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_skeptical_override_verifier_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "supported": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "observed_action": {"type": "string"},
                    "observed_object": {"type": "string"},
                    "evidence_frames": {"type": "array", "items": {"type": "integer"}},
                    "confounders": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["supported", "confidence", "observed_action", "observed_object", "evidence_frames", "confounders"],
            },
        },
    }


def collect(config_path: Path, keys_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    base_path = REPO_ROOT / config["base_report"]
    manifest_path = REPO_ROOT / config["manifest"]
    if _sha256(base_path) != config["base_report_file_sha256"]:
        raise ValueError("base report hash mismatch")
    if _sha256(manifest_path) != config["manifest_file_sha256"]:
        raise ValueError("manifest hash mismatch")
    base = json.loads(base_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    samples = {row["task_id"]: row for row in manifest["samples"]}
    key = runpy.run_path(str(keys_path)).get(config["model"]["api_key_name"])
    if not key:
        raise ValueError("OpenRouter API key is unavailable")
    client = OpenAI(api_key=key, base_url=config["model"]["base_url"], timeout=240, max_retries=2)
    rows = []
    usage = []
    for runtime in base["rows"]:
        candidate = authorize_source_override(runtime)
        if not candidate["authorized"]:
            rows.append({"task_id": runtime["task_id"], "candidate_authorized": False, "verifier": None})
            continue
        sample = samples[runtime["task_id"]]
        frames, seconds, metadata = _sample_video_range(
            Path(sample["video_path"]), frame_count=48, max_side=512,
        )
        events = [event for event in runtime["grounding_receipt"]["events"] if event["observability"] == "OBSERVED"]
        evidence = sorted({index for event in events for index in event["evidence_frames"]})
        if not evidence:
            raise ValueError(f"authorized candidate has no evidence: {runtime['task_id']}")
        start = max(0.0, seconds[min(evidence)] - 1.0)
        end = min(float(metadata["duration_seconds"]), seconds[max(evidence)] + 1.0)
        if end <= start:
            end = min(float(metadata["duration_seconds"]), start + 1.0)
        registry, _ = build_video_registry(
            frames, duration_seconds=float(metadata["duration_seconds"]),
            wrapper_root=config["wrapper_root"], required_tools=("sample_frames",),
        )
        focused, tool_receipt = execute_video_intervention(
            registry, frames, tool="sample_frames",
            arguments={"start_sec": start, "end_sec": end, "n": 12},
        )
        indices = tool_receipt["proxy_frame_indices"]
        focused_seconds = [seconds[index] for index in indices]
        panels = _panels(focused, focused_seconds, frames_per_panel=4, frame_width=256, quality=82)
        target = str(runtime["grounding_receipt"]["operand_a"])
        system = (
            "You are a skeptical visual evidence verifier. You see only one target action/relation and a focused chronological frame window. "
            "Mark supported=true only when the exact target is directly visible, not merely plausible. Reject related but different actions, static proximity, preparation, aftermath, and ambiguous object identity. "
            "Evidence frame IDs are local F0.. in the panels. Return only the required JSON."
        )
        content = [{"type": "text", "text": f"Target to verify exactly: {target}"}] + _panel_content(panels)
        input_core = {
            "prompt_version": "AGQA_QWEN235_SKEPTICAL_OVERRIDE_VERIFIER_V1",
            "model": config["model"]["id"], "target": target,
            "tool_receipt": tool_receipt,
            "panel_sha256": [stable_hash(panel.hex()) for panel in panels],
        }
        payload, call_usage, reused = _cached_provider_call(
            cache_dir=output_path.parent / "call_cache",
            call_name=f"verify_{runtime['task_id']}", input_core=input_core,
            invoke=lambda: _provider_json_call(
                client, model=config["model"], system=system, content=content,
                max_tokens=400, response_format=_response_format(),
            ),
        )
        usage.append(call_usage)
        rows.append({
            "task_id": runtime["task_id"], "candidate_authorized": True,
            "target_sha256": stable_hash(target), "tool_receipt": tool_receipt,
            "verifier": payload, "usage": call_usage, "cache_reused": reused,
            "answer_read": False, "program_read": False, "scene_graph_read": False,
            "source_identity_read": False,
        })
        print(f"verified {runtime['task_id']}", flush=True)
    body = {
        "schema_version": "agqa-qwen235-override-verifier-runtime-v1",
        "status": "FROZEN_RUNTIME_BEFORE_EVALUATOR",
        "config_sha256": stable_hash(config),
        "base_report_sha256": base["report_sha256"],
        "candidate_count": sum(row["candidate_authorized"] for row in rows),
        "provider_calls": len(usage),
        "reported_provider_cost_usd": sum(float(row["reported_cost_usd"]) for row in usage),
        "rows": rows,
    }
    result = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", default="/fs/gamma-projects/vlm-robot/keys.py", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.config, args.keys, args.output)
    print(json.dumps({key: result[key] for key in ("status", "candidate_count", "provider_calls", "reported_provider_cost_usd", "report_sha256")}, indent=2))


if __name__ == "__main__":
    main()
