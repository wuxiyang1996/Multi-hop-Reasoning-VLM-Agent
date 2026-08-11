#!/usr/bin/env python3
"""Freeze and independently confirm the V2 Sokoban effect program."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.sokoban_commit_skill import build_fresh_confirmation_plan
from motif_transfer.sokoban_effect_program import (
    build_effect_program,
    qualify_effect_program,
)


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--v1-target-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=96001)
    parser.add_argument("--episodes", type=int, default=24)
    args = parser.parse_args()
    target_report = _read(args.v1_target_report)
    if target_report.get("status") != "QUALIFICATION_FAILED_STOP_BEFORE_HELDOUT":
        raise SystemExit("V2 development receipt must bind the failed V1 qualification")
    if target_report.get("heldout_read") is not False:
        raise SystemExit("V1 held-out boundary was not preserved")
    source_plan = _read(args.source_plan)
    development_receipt = {
        "v1_target_report_path": str(args.v1_target_report.resolve()),
        "v1_target_report_file_sha256": _sha256(args.v1_target_report),
        "v1_target_report_sha256": str(target_report["report_sha256"]),
        "reason": "REMOVE_SOURCE_OPTION_OCCUPANCY_AFTER_PHASE_CONTROL_DOMINATED_V1",
    }
    artifact = build_effect_program(
        source_plan, development_receipt=development_receipt,
    )
    artifact_path = args.output_dir / "discovery_artifact.json"
    _write_new(artifact_path, artifact)
    fresh_plan = build_fresh_confirmation_plan(
        seeds=tuple(range(args.seed_start, args.seed_start + args.episodes)),
        snapshots_per_episode=4,
    )
    plan_path = args.output_dir / "fresh_confirmation_plan.json"
    _write_new(plan_path, fresh_plan)
    report = qualify_effect_program(fresh_plan, artifact)
    report_path = args.output_dir / "fresh_confirmation_report.json"
    _write_new(report_path, report)
    print(json.dumps({
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "fresh_plan": str(plan_path.resolve()),
        "fresh_plan_sha256": fresh_plan["plan_sha256"],
        "report": str(report_path.resolve()),
        "report_sha256": report["report_sha256"],
        "source_gate_passed": report["source_gate_passed"],
        "metrics": {
            name: {
                "accuracy": row["accuracy"],
                "selected_option_counts": row["selected_option_counts"],
            }
            for name, row in report["condition_metrics"].items()
        },
        "next_step": report["next_step"],
    }, indent=2, sort_keys=True))
    return 0 if report["source_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
