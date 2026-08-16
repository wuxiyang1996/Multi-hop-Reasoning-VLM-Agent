#!/usr/bin/env python3
"""Freeze untouched DiscoveryWorld grounding qualification before reset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


QUALIFICATION_SEEDS = tuple(range(81, 89))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-config", required=True, type=Path)
    parser.add_argument("--development-summary", required=True, type=Path)
    parser.add_argument("--qualification-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    development = _read(args.development_config)
    summary = _read(args.development_summary)
    if summary.get("status") != "DISCOVERYWORLD_GROUNDING_QUALIFICATION_PASSED":
        raise SystemExit("development grounding gates did not pass")
    if summary.get("role") != "development":
        raise SystemExit("grounding evidence is not a development run")
    if not all(summary.get("gates", {}).values()):
        raise SystemExit("development summary contains a failed gate")
    if development.get("reads_target_success") is not False:
        raise SystemExit("development config can read target success")
    # Freeze the exact development thresholds.  No formal outcome has been
    # consulted, and qualification is restricted to new component-only tasks.
    qualification = {
        "schema_version": "phase3-discoveryworld-grounding-config-v1",
        "status": "FROZEN_BEFORE_QUALIFICATION_TASK_RESET",
        "role": "qualification",
        "reads_target_success": False,
        "claim_boundary": (
            "Untouched component qualification on Proteomics Easy seeds81-88; "
            "fixed horizon, no evaluator call, official success persistence, "
            "source program, or Phase-3 formal-reserve task."
        ),
        "tasks": [
            {"scenario": "Proteomics", "difficulty": "Easy", "seed": seed}
            for seed in QUALIFICATION_SEEDS
        ],
        "model": dict(development["model"]),
        "runtime": {
            **dict(development["runtime"]),
            "thread_id_base": 132000,
        },
        "qualification_gates": dict(development["qualification_gates"]),
    }
    _write(args.qualification_config, qualification)
    runtime_paths = (
        "src/motif_transfer/phase3_discoveryworld_grounding.py",
        "src/motif_transfer/discoveryworld_policy.py",
        "src/motif_transfer/discoveryworld_env.py",
        "scripts/run_phase3_discoveryworld_grounding_qualification_v1.py",
    )
    body = {
        "schema_version": "phase3-discoveryworld-grounding-freeze-v1",
        "status": "FROZEN_BEFORE_QUALIFICATION_TASK_RESET",
        "development_config_path": str(args.development_config),
        "development_config_file_sha256": _file_sha256(args.development_config),
        "development_summary_path": str(args.development_summary),
        "development_summary_file_sha256": _file_sha256(args.development_summary),
        "development_summary_sha256": summary["summary_sha256"],
        "development_schema_fallback_rate": summary["schema_fallback_rate"],
        "development_invalid_native_actions": summary["invalid_native_actions"],
        "qualification_config_path": str(args.qualification_config),
        "qualification_config_file_sha256": _file_sha256(args.qualification_config),
        "qualification_seeds": list(QUALIFICATION_SEEDS),
        "qualification_tasks_previously_executed": False,
        "frozen_qualification_gates": dict(qualification["qualification_gates"]),
        "runtime_file_sha256": {
            path: _file_sha256(REPO / path) for path in runtime_paths
        },
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
        "claim_boundary": qualification["claim_boundary"],
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write(args.manifest, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "qualification_config_file_sha256": manifest[
            "qualification_config_file_sha256"
        ],
        "qualification_seeds": list(QUALIFICATION_SEEDS),
    }, indent=2))


if __name__ == "__main__":
    main()
