#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _media(path: Path) -> dict[str, Any]:
    row = _load(path)
    result = row["result"]
    trace = result.get("tool_trace") or []
    tool_errors = []
    for index, event in enumerate(trace):
        value = event.get("result") or {}
        if isinstance(value, dict) and value.get("error"):
            tool_errors.append({"event": index, "error": str(value["error"])})
    validation = result.get("validation") or {}
    return {
        "receipt": str(path),
        "receipt_sha256": _sha(path),
        "sample_id": row["sample_id"],
        "answer": result.get("answer"),
        "ground_truth": result.get("ground_truth"),
        "correct": result.get("correct"),
        "rounds": result.get("rounds"),
        "tool_events": len(trace),
        "head_used": result.get("head_used"),
        "validation_valid": validation.get("valid"),
        "validation_errors": validation.get("errors") or [],
        "tool_errors": tool_errors,
    }


def _browser(path: Path) -> dict[str, Any]:
    row = _load(path)
    return {
        "receipt": str(path),
        "receipt_sha256": _sha(path),
        "task_id": row["task_id"],
        "initial_state_hash": row["initial_state_hash"],
        "steps": len(row["steps"]),
        "actions": [step["action"] for step in row["steps"]],
        "total_reward": row["total_reward"],
        "success": row["success"],
        "terminated": row["terminated"],
        "invalid_actions": sum(bool(step.get("last_action_error")) for step in row["steps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/target_feasibility_v1"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/target_smoke_summary_v1.json"))
    args = parser.parse_args()
    root = args.run_root
    payload = {
        "schema_version": 1,
        "matrix": "4-domain/7-cell",
        "condition": "target_only",
        "visual_toolbench": [_media(root / "visual_toolbench_target_only_smoke0_ocr.json")],
        "tir_bench": [
            _media(root / "tir_bench_target_only_smoke0.json"),
            _media(root / "tir_bench_target_only_smoke1.json"),
        ],
        "video_holmes": [_media(root / "video_holmes_target_only_smoke0.json")],
        "miniwob": [_browser(root / "miniwob_target_only_smoke0.json")],
        "webshop": [_browser(root / "webshop_target_only_smoke0.json")],
        "alfworld_summary": _load(Path("docs/results/alfworld_target_feasibility_v1.json")),
        "claim_limit": "These are infrastructure and target-only policy smokes, not source-motif transfer results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
