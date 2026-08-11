#!/usr/bin/env python3
"""Run the frozen causal-effect candidate on untouched source episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from motif_transfer.effect_option_qualification import collect_source_qualification
from motif_transfer.visual_intervention_receipts import (
    file_sha256,
    load_runtime_env_factory,
    runtime_file_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _runtime_receipt(runtime_root: Path) -> dict:
    local_files = (
        "scripts/run_source_effect_qualification.py",
        "src/motif_transfer/effect_option_qualification.py",
        "src/motif_transfer/causal_effect_options.py",
        "src/motif_transfer/visual_intervention_receipts.py",
    )
    return {
        "clean_repo_commit": _git_commit(REPO_ROOT),
        "clean_repo_files_sha256": {
            name: file_sha256(REPO_ROOT / name) for name in local_files
        },
        "source_runtime": runtime_file_receipt(runtime_root),
        "source_runtime_commit": _git_commit(runtime_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "working_tree_changes_may_exist": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("qualification", "held_out"), default="qualification",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    manifest = collect_source_qualification(
        plan, artifact, split=args.split, output_dir=args.output_dir,
        env_factory=load_runtime_env_factory(args.runtime_root), workers=args.workers,
        runtime_receipt=_runtime_receipt(args.runtime_root),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["summary"].get("next_step") == "STOP_PROTOCOL_FAILURE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
