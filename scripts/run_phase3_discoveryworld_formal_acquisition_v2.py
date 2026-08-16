#!/usr/bin/env python3
"""Validate the frozen V2 manifest, then run structured formal acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.run_phase3_discoveryworld_structured_acquisition_v2 as base  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def validate_formal_manifest(manifest: Mapping[str, Any]) -> None:
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("invalid formal manifest self-hash")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_PHASE3_V2_TARGET_RESET_OR_OUTCOME":
        raise ValueError("manifest is not frozen for V2 formal acquisition")
    if manifest.get("role") != "formal_reserve":
        raise ValueError("V2 formal manifest has the wrong role")
    tasks = manifest.get("tasks") or ()
    if len(tasks) != 24 or [int(row["seed"]) for row in tasks] != list(range(121, 145)):
        raise ValueError("V2 formal reserve must be exactly seeds121-144")
    qualification = manifest.get("structured_acquisition_qualification") or {}
    if qualification.get("status") != (
        "DISCOVERYWORLD_STRUCTURED_ACQUISITION_QUALIFICATION_PASSED"
    ):
        raise ValueError("structured acquisition qualification is not passed")
    if not all((qualification.get("gates") or {}).values()):
        raise ValueError("structured acquisition qualification has a failed gate")
    if manifest.get("formal_target_outcome_read_for_freeze") is not False:
        raise ValueError("formal outcome exposure attestation is missing")
    if manifest.get("formal_reserve_task_opened") is not False:
        raise ValueError("formal reserve was already opened at freeze")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    validate_formal_manifest(_read(args.config))
    old_argv = sys.argv
    try:
        sys.argv = [
            str(REPO / "scripts/run_phase3_discoveryworld_structured_acquisition_v2.py"),
            "--config", str(args.config), "--keys", str(args.keys),
            "--output-dir", str(args.output_dir),
        ]
        if args.workers is not None:
            sys.argv.extend(["--workers", str(args.workers)])
        for task_id in args.task_id or ():
            sys.argv.extend(["--task-id", str(task_id)])
        base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
