#!/usr/bin/env python3
"""Freeze V10 on consumed ALFWorld development and immutable V9 controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_goal_acquisition_v10 import (  # noqa: E402
    AUTHENTIC, CARDINALITY_CONTROL, CEILING, EFFECT_CONTROL, GENERIC, RAW,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parent = REPO / "configs/alfworld_goal_relation_macro_v5_development.json"
    output = REPO / "configs/alfworld_goal_acquisition_v10_development.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen V10 config: {output}")
    config = json.loads(parent.read_text(encoding="utf-8"))
    config.pop("config_sha256", None)
    runner = REPO / "scripts/run_alfworld_goal_acquisition_v10.py"
    runtime = REPO / "src/motif_transfer/alfworld_goal_acquisition_v10.py"
    acquisition = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"
    confirmation = (
        REPO / "runs/sokoban_goal_acquisition_v1/fresh_confirmation_report.json"
    )
    reuse = REPO / "runs/alfworld_goal_acquisition_v9_development/report.json"
    config |= {
        "schema_version": "alfworld-goal-acquisition-development-config-v10",
        "experiment_version": (
            "TYPED_HANDLE_PRESERVING_ACQUISITION_CONSUMED_DEVELOPMENT_V10"
        ),
        "claim_boundary": (
            "CONSUMED_ALFWORLD_TRAIN_DEVELOPMENT_ONLY;V10_EXECUTES_AUTHENTIC_"
            "AND_TARGET_NATIVE_CEILING;UNCHANGED_RAW_AND_SOURCE_CONTROLS_REUSED_"
            "BY_FROZEN_V9_REPORT_HASH;NO_V8_QUALIFICATION_OR_FORMAL_TASK_RERUN;"
            "NO_CONFIRMATORY_TARGET_CLAIM"
        ),
        "source_acquisition_artifact": str(acquisition.relative_to(REPO)),
        "source_acquisition_artifact_file_sha256": _sha256(acquisition),
        "source_acquisition_confirmation": str(confirmation.relative_to(REPO)),
        "source_acquisition_confirmation_file_sha256": _sha256(confirmation),
        "reused_v9_report": str(reuse.relative_to(REPO)),
        "reused_v9_report_file_sha256": _sha256(reuse),
        "executed_conditions": [AUTHENTIC, CEILING],
        "reused_conditions": [RAW, CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC],
        "v10_runner_file_sha256": _sha256(runner),
        "v10_target_runtime_file_sha256": _sha256(runtime),
        "parent_v5_config_file_sha256": _sha256(parent),
        "output": "runs/alfworld_goal_acquisition_v10_development/report.json",
        "gates": dict(config["gates"]) | {
            "minimum_source_acquisition_groundings": 8,
        },
    }
    config["config_sha256"] = stable_hash(config)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
