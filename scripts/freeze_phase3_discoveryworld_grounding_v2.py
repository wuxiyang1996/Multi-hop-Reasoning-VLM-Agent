#!/usr/bin/env python3
"""Freeze the V2 DiscoveryWorld grounder before untouched qualification.

V2 is allowed to use only the failed V1 qualification tasks (seeds 81--88)
as development data.  This receipt binds the repaired runtime, the unchanged
qualification gates, and the next untouched component-only seeds before any
of those environments are reset.
"""

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


DEVELOPMENT_SEEDS = tuple(range(81, 89))
QUALIFICATION_SEEDS = tuple(range(89, 97))


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


def _task_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(row["seed"]) for row in config.get("tasks", ()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-config", required=True, type=Path)
    parser.add_argument("--development-summary", required=True, type=Path)
    parser.add_argument("--v1-failure-summary", required=True, type=Path)
    parser.add_argument("--qualification-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    development = _read(args.development_config)
    summary = _read(args.development_summary)
    v1_failure = _read(args.v1_failure_summary)
    if _task_seeds(development) != DEVELOPMENT_SEEDS:
        raise SystemExit("V2 development was not restricted to consumed seeds81-88")
    if summary.get("status") != "DISCOVERYWORLD_GROUNDING_QUALIFICATION_PASSED":
        raise SystemExit("V2 development grounding gates did not pass")
    if summary.get("role") != "development" or not all(summary.get("gates", {}).values()):
        raise SystemExit("V2 evidence is not a passing development-only run")
    if development.get("reads_target_success") is not False:
        raise SystemExit("V2 development config can read target success")
    if v1_failure.get("status") != "DISCOVERYWORLD_GROUNDING_QUALIFICATION_FAILED":
        raise SystemExit("V1 qualification failure evidence is missing")
    if dict(development["qualification_gates"]) != {
        "minimum_steps": 128,
        "maximum_schema_fallback_rate": 0.10,
        "maximum_invalid_native_actions": 0,
    }:
        raise SystemExit("qualification gates differ from the frozen V1 gates")

    qualification = {
        "schema_version": "phase3-discoveryworld-grounding-config-v2",
        "status": "FROZEN_BEFORE_V2_QUALIFICATION_TASK_RESET",
        "role": "qualification",
        "reads_target_success": False,
        "claim_boundary": (
            "Untouched component-only qualification of the V2 target-native "
            "grounder on Proteomics Easy seeds89-96; fixed horizon and frozen "
            "V1 qualification thresholds; no evaluator, official success "
            "persistence, source program, or Phase-3 formal-reserve task."
        ),
        "tasks": [
            {"scenario": "Proteomics", "difficulty": "Easy", "seed": seed}
            for seed in QUALIFICATION_SEEDS
        ],
        "model": dict(development["model"]),
        "runtime": {
            **dict(development["runtime"]),
            "thread_id_base": 134000,
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
        "schema_version": "phase3-discoveryworld-grounding-freeze-v2",
        "status": "FROZEN_BEFORE_V2_QUALIFICATION_TASK_RESET",
        "development_config_path": str(args.development_config),
        "development_config_file_sha256": _file_sha256(args.development_config),
        "development_summary_path": str(args.development_summary),
        "development_summary_file_sha256": _file_sha256(args.development_summary),
        "development_summary_sha256": summary["summary_sha256"],
        "development_schema_fallback_rate": summary["schema_fallback_rate"],
        "development_invalid_native_actions": summary["invalid_native_actions"],
        "v1_failure_summary_path": str(args.v1_failure_summary),
        "v1_failure_summary_file_sha256": _file_sha256(args.v1_failure_summary),
        "v1_failure_summary_sha256": v1_failure["summary_sha256"],
        "v1_schema_fallback_rate": v1_failure["schema_fallback_rate"],
        "qualification_config_path": str(args.qualification_config),
        "qualification_config_file_sha256": _file_sha256(args.qualification_config),
        "qualification_seeds": list(QUALIFICATION_SEEDS),
        "qualification_tasks_previously_executed": False,
        "development_seeds_disjoint_from_qualification": (
            set(DEVELOPMENT_SEEDS).isdisjoint(QUALIFICATION_SEEDS)
        ),
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
        "frozen_qualification_gates": manifest["frozen_qualification_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
