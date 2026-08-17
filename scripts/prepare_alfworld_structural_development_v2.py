#!/usr/bin/env python3
"""Prepare a closed-loop config using only consumed ALFWorld development tasks."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_structural_runtime_v1 import CONDITIONS  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


OUTPUT = REPO / "configs/alfworld_structural_transfer_v2_development/manifest.json"
GROUNDING = REPO / "artifacts/alfworld_structural_grounder_v1/artifact.json.gz"
SOURCE = REPO / "configs/source_structural_v5c_frozen/programs/put_near.json"
PERMUTED = REPO / "configs/source_structural_v5c_frozen/programs/unlock_pickup.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with gzip.open(GROUNDING, "rt", encoding="utf-8") as handle:
        grounder = json.load(handle)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    permuted = json.loads(PERMUTED.read_text(encoding="utf-8"))
    task_ids = [
        *grounder["training_task_ids"], *grounder["qualification_task_ids"],
    ]
    integrity_paths = (
        "src/motif_transfer/alfworld_env.py",
        "src/motif_transfer/alfworld_structural_induction.py",
        "src/motif_transfer/alfworld_structural_runtime_v1.py",
        "src/motif_transfer/structural_delta_induction.py",
        "scripts/run_alfworld_structural_transfer_v1.py",
        str(GROUNDING.relative_to(REPO)), str(SOURCE.relative_to(REPO)),
        str(PERMUTED.relative_to(REPO)),
    )
    body = {
        "schema_version": "alfworld-structural-closed-loop-development-v2",
        "role": "CONSUMED_TARGET_DEVELOPMENT_CLOSED_LOOP_QUALIFICATION",
        "evaluation_mode": "DEVELOPMENT",
        "conditions": list(CONDITIONS),
        "grounder": {
            "path": str(GROUNDING.relative_to(REPO)),
            "grounder_sha256": grounder["grounder_sha256"],
        },
        "source_induced": {
            "path": str(SOURCE.relative_to(REPO)),
            "program_sha256": source["program_sha256"],
            "source_name_evaluator_label_only": "put_near",
        },
        "source_permuted": {
            "path": str(PERMUTED.relative_to(REPO)),
            "program_sha256": permuted["program_sha256"],
        },
        "target": {
            "alfworld_config": "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1/configs/alfworld_base_config.yaml",
            "alfworld_data": "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data",
            "split": "train", "seed": 2026081621, "max_steps": 180,
            "task_ids": task_ids,
        },
        "integrity": {"file_sha256": {
            relative: _sha((REPO / relative).resolve()) for relative in integrity_paths
        }},
    }
    config = body | {"config_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "config_sha256": config["config_sha256"], "tasks": len(task_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
