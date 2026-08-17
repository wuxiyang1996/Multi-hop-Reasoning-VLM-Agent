#!/usr/bin/env python3
"""Freeze fresh DiscoveryWorld structural-transfer tasks, code, and gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    validate_structural_program,
)
from motif_transfer.target_structural_induction import (  # noqa: E402
    validate_mlp_grounder, validate_target_program,
)


TARGET_REPORT = REPO / "runs/discoveryworld_structural_grounder_v1_development/report_v2.json"
SOURCE_MANIFEST = REPO / "configs/source_structural_v5c_frozen/manifest.json"
ACQUISITION_DIR = REPO / "runs/discoveryworld_structural_transfer_v1_acquisition"
MATCHED_DIR = REPO / "runs/discoveryworld_structural_transfer_v1_matched"
SEEDS = tuple(range(201, 213))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value); claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _official_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO.parent / "discoveryworld-official",
        check=True, text=True, capture_output=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    if ACQUISITION_DIR.exists() or MATCHED_DIR.exists():
        raise SystemExit("fresh DiscoveryWorld reserve was opened before freeze")

    report = _read(TARGET_REPORT); _self_hash(report, "report_sha256")
    if report.get("status") != "DISCOVERYWORLD_STRUCTURAL_GROUNDER_QUALIFIED":
        raise SystemExit("target-native structural grounder is not qualified")
    if not all(report.get("gates", {}).values()):
        raise SystemExit("target development contains a failed gate")
    validate_mlp_grounder(report["grounder"])
    validate_target_program(report["target_program"])
    source_manifest = _read(SOURCE_MANIFEST)
    _self_hash(source_manifest, "manifest_sha256")
    source_report_path = REPO / "runs/source_structural_v5c_fresh/report.json"
    source_report = _read(source_report_path)
    _self_hash(source_report, "report_sha256")
    if source_report.get("status") != "SOURCE_STRUCTURAL_FRESH_VALIDATED":
        raise SystemExit("fresh source structural programs are not validated")

    programs = {}
    for task, receipt in source_manifest["source_programs"].items():
        path = REPO / receipt["path"]
        if _sha(path) != receipt["file_sha256"]:
            raise SystemExit(f"source program changed: {task}")
        program = _read(path); validate_structural_program(program)
        programs[task] = dict(receipt)
    selected = str(report["selected_source_program"])
    permuted = str(report["source_permuted_control"])
    if selected == permuted or selected not in programs or permuted not in programs:
        raise SystemExit("invalid source/permuted program assignment")

    runtime_paths = (
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/structural_delta_induction.py",
        "src/motif_transfer/target_structural_induction.py",
        "src/motif_transfer/discoveryworld_structural_runtime_v1.py",
        "src/motif_transfer/discoveryworld_env.py",
        "src/motif_transfer/discoveryworld_policy.py",
        "src/motif_transfer/discoveryworld_sokoban_transfer.py",
        "src/motif_transfer/discoveryworld_structured_acquisition_v2.py",
        "src/motif_transfer/phase3_discoveryworld_grounding.py",
        "src/motif_transfer/phase3_discoveryworld_transfer.py",
        "scripts/run_phase3_discoveryworld_structured_acquisition_v2.py",
        "scripts/run_discoveryworld_structural_transfer_v1.py",
    )
    runtime_hashes = {path: _sha(REPO / path) for path in runtime_paths}
    tasks = [
        {
            "task_id": f"proteomics.easy.seed{seed}",
            "scenario": "Proteomics", "difficulty": "Easy", "seed": seed,
            "selected_target_previously_executed": False,
        }
        for seed in SEEDS
    ]
    model = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_name": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4.1-mini",
        "affordance_model": "openai/gpt-4.1",
        "temperature": 0,
        "maximum_output_tokens": 1200,
        "schema_attempts": 3,
    }
    body = {
        "schema_version": "discoveryworld-structural-transfer-manifest-v1",
        "status": "FROZEN_BEFORE_ANY_FRESH_TARGET_RESET_OR_OUTCOME",
        "role": "formal_reserve",
        "target_domain": "DiscoveryWorld/Proteomics/Easy",
        "tasks": tasks, "task_count": len(tasks),
        "model": model,
        "runtime": {
            "maximum_acquisition_steps": 24,
            "continuation_horizon": 8,
            "task_workers": 4,
            "thread_id_base": 310000,
        },
        "binding_model": model,
        "matched_runtime": {
            "recovery_horizon": 8,
            "task_workers": 4,
            "thread_id_base": 320000,
        },
        "conditions": [
            "neural_only", "source_induced", "source_permuted",
            "generic_scaffold", "target_native_ceiling",
        ],
        "target_development_report": {
            "path": _relative(TARGET_REPORT),
            "file_sha256": _sha(TARGET_REPORT),
            "report_sha256": report["report_sha256"],
            "grounder_sha256": report["grounder"]["grounder_sha256"],
            "target_program_sha256": report["target_program"]["program_sha256"],
            "fixed_grounding_threshold": report["grounder"]["threshold"],
            "selected_source_program": selected,
            "source_permuted_control": permuted,
        },
        "source_programs": programs,
        "source_validation": {
            "manifest_path": _relative(SOURCE_MANIFEST),
            "manifest_file_sha256": _sha(SOURCE_MANIFEST),
            "manifest_sha256": source_manifest["manifest_sha256"],
            "report_path": _relative(source_report_path),
            "report_file_sha256": _sha(source_report_path),
            "report_sha256": source_report["report_sha256"],
        },
        "transfer_semantics": {
            "shared": "ANONYMOUS_TYPED_GRAPH_EDIT_IR_AND_LEARNING_PROTOCOL",
            "source_reused": "QUALIFIED_ADD_SLOT_TO_REMOVE_SLOT_SUBGRAPH",
            "target_learned": (
                "COUNTED_OBSERVATION_RELATIONS_PARTIAL_ORDER_COMMIT_GUARD_"
                "AND_NATIVE_REALIZATION"
            ),
            "source_program_copied_as_target_controller": False,
            "target_native_neural_grounding": True,
        },
        "preregistered_gates": {
            "minimum_applicable_tasks": 10,
            "source_vs_neural_sign_p_max": 0.05,
            "negative_transfer_rate_max": 0.20,
            "minimum_source_permutation_behavior_contrasts": 10,
        },
        "failure_gates": {
            "any_runtime_or_hash_error": "FAIL",
            "any_formal_outcome_used_for_binding_selection_or_threshold": "FAIL",
            "source_not_strictly_above_neural_permuted_and_generic": "FAIL",
            "fewer_than_six_source_vs_neural_discordant_wins": "FAIL_SIGNIFICANCE",
        },
        "runtime_file_sha256": runtime_hashes,
        "official_environment_commit": _official_commit(),
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
        "all_formal_seeds_new_relative_to_consumed_phase3_seeds_45_144": True,
        "claim_boundary": (
            "Prospectively frozen seeds201-212. Shared anonymous structural IR; "
            "target-specific counted partial-order function and neural grounder "
            "learned only from consumed development seeds45-50; no formal reset, "
            "action, outcome, evaluator, or score read before freeze."
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"], "tasks": len(tasks),
        "selected_source_program": selected,
        "source_permuted_control": permuted,
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
