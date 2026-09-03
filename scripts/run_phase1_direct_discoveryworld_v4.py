#!/usr/bin/env python3
"""Run V4 with fail-closed neural/symbolic applicability completeness."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.discoveryworld_applicability_grounder_v4 import (  # noqa: E402
    call_applicability_complete_grounder,
    select_source_safe_candidate,
)
import scripts.run_discoveryworld_commit_recovery_v1 as matched_runner  # noqa: E402
import scripts.run_phase1_direct_discoveryworld_v2 as runner  # noqa: E402


runner.SCHEMA = "phase1-direct-discoveryworld-confirmation-v4"
runner.STATUS = "FROZEN_BEFORE_ANY_V4_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"
matched_runner.call_grounder = call_applicability_complete_grounder
matched_runner.select_candidate = select_source_safe_candidate


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
                    "--manifest", str(REPO / "configs/phase1_direct_prospective_v4/discoveryworld_manifest.json"),
                    "--keys", str(keys),
                    "--output-root", str(output_root),
                ],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0 or not receipt.is_file():
            raise RuntimeError(
                f"DiscoveryWorld V4 preparation failed: exit={completed.returncode}"
            )
    return output_root / "frozen_forks"


runner.prepare_forks = prepare_forks


def _inject_defaults() -> None:
    if "--manifest" not in sys.argv:
        sys.argv.extend([
            "--manifest",
            str(REPO / "configs/phase1_direct_prospective_v4/discoveryworld_manifest.json"),
        ])
    if "--output-root" not in sys.argv:
        sys.argv.extend([
            "--output-root",
            str(REPO / "runs/phase1_direct_prospective_v4/discoveryworld"),
        ])


if __name__ == "__main__":
    _inject_defaults()
    raise SystemExit(runner.main())
