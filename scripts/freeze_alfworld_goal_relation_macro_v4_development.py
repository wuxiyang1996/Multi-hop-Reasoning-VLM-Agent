#!/usr/bin/env python3
"""Freeze the effect-gated V4 on the same consumed ALFWorld dev tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parent_path = REPO / "configs/alfworld_goal_relation_macro_v3_development.json"
    output = REPO / "configs/alfworld_goal_relation_macro_v4_development.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {output}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent.pop("config_sha256", None)
    body = parent | {
        "schema_version": "alfworld-goal-relation-macro-development-config-v4",
        "experiment_version": "EFFECT_GATED_RECURRENCE_V4",
        "claim_boundary": (
            "SAME 24 STRATIFIED ALREADY-CONSUMED ALFWORLD TRAIN-DEVELOPMENT "
            "TASKS AS V3; V4 REFINES APPLICABILITY AFTER INSPECTING V3 FAILURES; "
            "NOT CONFIRMATORY EVIDENCE; UNTOUCHED MULTIPLICITY RESERVE AND "
            "VALID_UNSEEN REMAIN UNREAD"
        ),
        "v3_failed_report": (
            "runs/alfworld_goal_relation_macro_v3_development/report.json"
        ),
        "v3_failed_report_file_sha256": _sha256(
            REPO / "runs/alfworld_goal_relation_macro_v3_development/report.json"
        ),
        "v4_runner_file_sha256": _sha256(
            REPO / "scripts/run_alfworld_goal_relation_macro_v4.py"
        ),
        "v4_target_runtime_file_sha256": _sha256(
            REPO / "src/motif_transfer/alfworld_goal_relation_macro_v4.py"
        ),
        "output": "runs/alfworld_goal_relation_macro_v4_development/report.json",
    }
    config = body | {"config_sha256": stable_hash(body)}
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "task_count": len(config["task_ids"]),
        "config_sha256": config["config_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
