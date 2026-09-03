#!/usr/bin/env python3
"""Run V3: qualified Qwen target acquisition plus GPT target grounding."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.run_phase1_direct_discoveryworld_v2 as runner  # noqa: E402


runner.SCHEMA = "phase1-direct-discoveryworld-confirmation-v3"
runner.STATUS = "FROZEN_BEFORE_ANY_V3_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"


def prepare_forks(*, manifest, keys: Path, output_root: Path) -> Path:
    receipt = output_root / "preparation_receipt.json"
    if not receipt.is_file():
        log_path = output_root / "preparation.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/prepare_phase1_direct_discoveryworld_v3.py"),
                    "--manifest", str(REPO / "configs/phase1_direct_prospective_v3/discoveryworld_manifest.json"),
                    "--keys", str(keys),
                    "--output-root", str(output_root),
                ],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0 or not receipt.is_file():
            raise RuntimeError(
                f"DiscoveryWorld V3 preparation failed: exit={completed.returncode}"
            )
    return output_root / "frozen_forks"


runner.prepare_forks = prepare_forks


def _inject_defaults() -> None:
    if "--manifest" not in sys.argv:
        sys.argv.extend([
            "--manifest",
            str(REPO / "configs/phase1_direct_prospective_v3/discoveryworld_manifest.json"),
        ])
    if "--output-root" not in sys.argv:
        sys.argv.extend([
            "--output-root",
            str(REPO / "runs/phase1_direct_prospective_v3/discoveryworld"),
        ])


if __name__ == "__main__":
    _inject_defaults()
    raise SystemExit(runner.main())
