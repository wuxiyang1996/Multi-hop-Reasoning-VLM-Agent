#!/usr/bin/env python3
"""Collect full-video high-coverage Qwen235 verification for source overrides."""

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
from scripts.collect_agqa2_qwen235_override_verifier_v1 import _response_format, _sha256  # noqa: E402


def _direct_boolean(value: object) -> str | None:
    first = str(value).strip().casefold().split(maxsplit=1)
    if not first:
        return None
    token = first[0].strip(".,:;!?()[]{}\"'")
    return token if token in {"yes", "no"} else None


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
    frame_count = int(config["tool_frame_budget"])
    for runtime in base["rows"]:
        candidate = authorize_source_override(runtime)
        direct_boolean = _direct_boolean(runtime["direct_response"])
        decision_disagrees = direct_boolean != candidate["decision"]
        if not candidate["authorized"] or not decision_disagrees:
            rows.append({"task_id": runtime["task_id"], "candidate_authorized": False, "decision_disagrees": decision_disagrees, "verifier": None})
            continue
        sample = samples[runtime["task_id"]]
        frames, seconds, metadata = _sample_video_range(Path(sample["video_path"]), frame_count=frame_count, max_side=512)
        duration = float(metadata["duration_seconds"])
        registry, _ = build_video_registry(frames, duration_seconds=duration, wrapper_root=config["wrapper_root"], required_tools=("sample_frames",))
        observed, tool_receipt = execute_video_intervention(registry, frames, tool="sample_frames", arguments={"start_sec": 0.0, "end_sec": duration, "n": frame_count})
        indices = tool_receipt["proxy_frame_indices"]
        panels = _panels(observed, [seconds[index] for index in indices], frames_per_panel=4, frame_width=224, quality=80)
        target = str(runtime["grounding_receipt"]["operand_a"])
        system = (
            "You are a skeptical full-video action verifier. Inspect the entire chronological coverage. "
            "supported=true only if the exact target action/relation is directly visible in at least two adjacent sampled frames. "
            "Reject preparation, aftermath, static proximity, a related but different action, and ambiguous object identity. "
            "Do not infer an unobserved event. Return only the required JSON."
        )
        content = [{"type": "text", "text": f"Exact target to verify across the full video: {target}"}] + _panel_content(panels)
        input_core = {"prompt_version": "AGQA_QWEN235_GLOBAL_96_VERIFIER_V5", "model": config["model"]["id"], "target": target, "tool_receipt": tool_receipt, "frame_count": frame_count}
        payload, call_usage, reused = _cached_provider_call(
            cache_dir=output_path.parent / "call_cache", call_name=f"global_verify_{runtime['task_id']}", input_core=input_core,
            invoke=lambda: _provider_json_call(client, model=config["model"], system=system, content=content, max_tokens=400, response_format=_response_format()),
        )
        usage.append(call_usage)
        rows.append({"task_id": runtime["task_id"], "candidate_authorized": True, "decision_disagrees": True, "target_sha256": stable_hash(target), "tool_receipt": tool_receipt, "verifier": payload, "usage": call_usage, "cache_reused": reused, "answer_read": False, "program_read": False, "scene_graph_read": False, "source_identity_read": False})
        print(f"verified {runtime['task_id']}", flush=True)
    body = {"schema_version": "agqa-qwen235-global-verifier-runtime-v5", "status": "FROZEN_RUNTIME_BEFORE_EVALUATOR", "config_sha256": stable_hash(config), "base_report_sha256": base["report_sha256"], "candidate_count": sum(row["candidate_authorized"] for row in rows), "provider_calls": len(usage), "reported_provider_cost_usd": sum(float(row["reported_cost_usd"]) for row in usage), "rows": rows}
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
