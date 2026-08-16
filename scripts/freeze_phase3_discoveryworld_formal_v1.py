#!/usr/bin/env python3
"""Freeze the untouched DiscoveryWorld Phase-3 five-arm reserve."""

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


SEEDS = tuple(range(97, 121))
PROGRAM_DIR = REPO / "configs/phase3_source_induction_v3/frozen_reserve/programs"
SOURCE_REPORT = REPO / "runs/phase3_typed_effect_source_reserve_v3/report.json"
GROUNDING_MANIFEST = (
    REPO / "configs/phase3_discoveryworld_typed_grounding_v3/qualification_manifest.json"
)
GROUNDING_REPORT = REPO / "runs/phase3_discoveryworld_typed_grounding_v3_qualification/report.json"
ACQUISITION_CONFIG = REPO / "configs/phase3_discoveryworld_grounding_v2/qualification.json"
DEVELOPMENT_SUMMARY = (
    REPO / "runs/phase3_discoveryworld_consumed_development_v14_fail_closed_acquisition_typed/summary.json"
)
LEGACY_TRANSPORT_CONFIG = (
    REPO / "runs/phase3_discoveryworld_consumed_development_v14_fail_closed_acquisition_typed/"
    "proteomics.easy.seed45/config.json"
)
RUNTIME_PATHS = (
    "src/motif_transfer/contracts.py",
    "src/motif_transfer/discoveryworld_env.py",
    "src/motif_transfer/discoveryworld_policy.py",
    "src/motif_transfer/discoveryworld_sokoban_transfer.py",
    "src/motif_transfer/discoveryworld_applicability_grounder_v4.py",
    "src/motif_transfer/phase3_discoveryworld_grounding.py",
    "src/motif_transfer/phase3_typed_effect_induction.py",
    "src/motif_transfer/phase3_attempt_runtime.py",
    "src/motif_transfer/phase3_source_portfolio.py",
    "src/motif_transfer/phase3_discoveryworld_transfer.py",
    "src/motif_transfer/phase3_discoveryworld_formal.py",
    "scripts/run_discoveryworld_target_only_v1.py",
    "scripts/run_discoveryworld_commit_recovery_v1.py",
    "scripts/run_phase3_discoveryworld_acquisition_v1.py",
    "scripts/prepare_phase3_discoveryworld_formal_forks_v1.py",
    "scripts/run_phase3_discoveryworld_formal_v1.py",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value); claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite frozen formal manifest")

    source_report = _read(SOURCE_REPORT)
    grounding_manifest = _read(GROUNDING_MANIFEST)
    grounding_report = _read(GROUNDING_REPORT)
    acquisition_config = _read(ACQUISITION_CONFIG)
    development = _read(DEVELOPMENT_SUMMARY)
    transport = _read(LEGACY_TRANSPORT_CONFIG)
    for value, field in (
        (source_report, "report_sha256"),
        (grounding_manifest, "manifest_sha256"),
        (grounding_report, "report_sha256"),
        (development, "summary_sha256"),
    ):
        _self_hash(value, field)
    if source_report.get("status") != "SOURCE_SPECIFIC_TYPED_EFFECT_APPLICABILITY_VALIDATED":
        raise SystemExit("typed source reserve is not validated")
    if grounding_report.get("status") != "DISCOVERYWORLD_TYPED_GROUNDING_QUALIFICATION_PASSED":
        raise SystemExit("typed DiscoveryWorld grounding is not qualified")
    if not all((grounding_report.get("gates") or {}).values()):
        raise SystemExit("typed DiscoveryWorld qualification has a failed gate")
    if development.get("complete_tasks") != 6:
        raise SystemExit("consumed development is incomplete")

    source_artifacts = []
    for path in sorted(PROGRAM_DIR.glob("*.json")):
        artifact = _read(path)
        _self_hash(artifact, "artifact_sha256")
        source_artifacts.append({
            "source_game": artifact["source_game"],
            "path": _relative(path),
            "file_sha256": _file_sha256(path),
            "artifact_sha256": artifact["artifact_sha256"],
            "program_sha256": artifact["typed_effect_program"]["program_sha256"],
            "selected_effect_type": artifact["typed_effect_program"]["selected_effect_type"],
            "qualification_status": artifact["typed_effect_program"]["status"],
        })
    if len(source_artifacts) != 6:
        raise SystemExit("expected exactly six frozen source artifacts")

    expected_outputs = [
        REPO / "runs/phase3_discoveryworld_formal_v1_acquisition" /
        f"proteomics.easy.seed{seed}.json" for seed in SEEDS
    ] + [
        REPO / "runs/phase3_discoveryworld_formal_v1_matched" /
        f"proteomics.easy.seed{seed}/matched_result.json" for seed in SEEDS
    ]
    if any(path.exists() for path in expected_outputs):
        raise SystemExit("formal reserve was already opened before freeze")
    for path in RUNTIME_PATHS:
        if not (REPO / path).is_file():
            raise SystemExit(f"missing formal runtime file: {path}")

    official = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO.parent / "discoveryworld-official",
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    tasks = [{
        "task_id": f"proteomics.easy.seed{seed}",
        "scenario": "Proteomics", "difficulty": "Easy", "seed": seed,
        "selected_target_previously_reset_or_executed": False,
    } for seed in SEEDS]
    body = {
        "schema_version": "phase3-discoveryworld-formal-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_PHASE3_TARGET_RESET_OR_OUTCOME",
        "target_domain": "DiscoveryWorld/Proteomics/Easy",
        "tasks": tasks, "task_count": len(tasks),
        "conditions": [
            "neural_only", "source_induced", "source_permuted",
            "generic_scaffold", "target_native_ceiling",
        ],
        "source_artifacts": source_artifacts,
        "source_portfolio_rule": (
            "SOURCE_CALIBRATED_ACCURACY_TIMES_TARGET_NATIVE_UNIQUE_ARGMAX_MARGIN;"
            "SOURCE_IDENTITY_NOT_A_FEATURE"
        ),
        "source_permuted_control": (
            "SAME_SELECTED_PROGRAM_WITH_DETERMINISTIC_NONIDENTITY_"
            "TARGET_CANDIDATE_EFFECT_BINDING_PERMUTATION"
        ),
        "acquisition_model": dict(acquisition_config["model"]),
        "acquisition_runtime": {
            "maximum_steps": 96, "include_vision": False,
            "task_workers": 4, "thread_id_base": 157000,
        },
        "typed_grounding_model": dict(grounding_manifest["model"]),
        "matched_runtime": {
            "task_workers": 4, "thread_id_base": 158000,
            "recovery_horizon": 8,
        },
        "fork_protocol": {
            "selection_rule": (
                "FIRST_POLICY_VISIBLE_STATE_WITH_THREE_MEASUREMENT_VECTORS_"
                "UNIQUE_ONE_OUTLIER_HELD_FLAG_AND_DERIVED_STATUE_VISIBLE"
            ),
            "forbidden_selection_fields": [
                "action_succeeded", "reward", "terminal", "official_success",
                "evaluation", "score", "scorecard", "recorded_next_action",
            ],
            "recovery_horizon": 8,
            "required_acquisition_measurement_vectors": 3,
            "required_acquisition_outlier_candidates": 1,
            "require_held_flag": True,
            "require_derived_target_statue_visible": True,
        },
        "analysis_protocol": {
            "estimand": "INTENTION_TO_TREAT_OVER_ALL_24_UNTOUCHED_TASKS",
            "primary_endpoint": (
                "PROGRAM_ALIGNED_HORIZON_SHARED_BY_ALL_ARMS_PER_TASK;"
                "H1=1,H4=4,H8=8,PERSISTENCE=8"
            ),
            "paired_test": "EXACT_TWO_SIDED_SIGN_TEST_DISCARDING_TIES",
            "negative_transfer_rate": "LOSSES_DIVIDED_BY_WINS_PLUS_LOSSES",
            "secondary_endpoints": ["H1", "H2", "H4", "H8"],
        },
        "frozen_gates": {
            "minimum_applicable_tasks": 16,
            "maximum_acquisition_schema_fallback_rate": 0.10,
            "source_vs_neural_exact_sign_p_max": 0.05,
            "source_vs_neural_negative_transfer_rate_max": 0.25,
            "minimum_admitted_source_permuted_first_selection_contrasts": 12,
            "maximum_formal_binder_repair_rate": 0.10,
            "maximum_formal_grounder_repair_rate": 0.10,
            "source_induced_strictly_improves_neural_only": True,
            "source_induced_strictly_beats_source_permuted": True,
            "source_induced_strictly_beats_generic_scaffold": True,
            "target_native_ceiling_not_below_source_induced": True,
            "zero_target_outcome_use_for_grounding_or_fork_selection": True,
        },
        "failure_gates": {
            "any_runtime_error": "FAIL",
            "any_receipt_or_hash_mismatch": "FAIL",
            "missing_structural_fork": "FAIL_APPLICABILITY_COVERAGE",
            "formal_result_used_to_change_protocol": "FAIL_LEAKAGE",
            "source_not_above_permuted": "FAIL_SOURCE_SPECIFIC_TARGET_CONTROL",
        },
        "legacy_runner_transport_contract": dict(transport["source_contract"]),
        "target_native_spatial_realizer": dict(
            transport["target_native_spatial_realizer"]
        ),
        "selector": dict(transport["selector"]),
        "source_validation": {
            "path": _relative(SOURCE_REPORT),
            "file_sha256": _file_sha256(SOURCE_REPORT),
            "report_sha256": source_report["report_sha256"],
        },
        "grounding_qualification": {
            "manifest_path": _relative(GROUNDING_MANIFEST),
            "manifest_sha256": grounding_manifest["manifest_sha256"],
            "report_path": _relative(GROUNDING_REPORT),
            "report_sha256": grounding_report["report_sha256"],
            "frozen_repair_threshold": 0.10,
        },
        "consumed_development": {
            "path": _relative(DEVELOPMENT_SUMMARY),
            "summary_sha256": development["summary_sha256"],
            "excluded_from_formal_estimates": True,
        },
        "runtime_file_sha256": {
            path: _file_sha256(REPO / path) for path in RUNTIME_PATHS
        },
        "official_environment_commit": official,
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
        "claim_boundary": (
            "Untouched seeds97-120. Source programs, target grounding, fork "
            "selection, endpoints, applicability, negative-transfer, and "
            "failure gates were frozen before any formal target reset."
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "tasks": len(tasks),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
