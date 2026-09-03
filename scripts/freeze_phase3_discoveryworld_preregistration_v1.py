#!/usr/bin/env python3
"""Freeze the untouched Phase-3 DiscoveryWorld cohort and analysis gates."""

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
from motif_transfer.phase3_source_applicability import (  # noqa: E402
    maximum_profile_contrast_derangement,
)


PROGRAM_DIR = REPO / "configs/phase3_source_induction_v1/frozen_confirmation/programs"
SOURCE_CONFIRMATION = REPO / "runs/phase3_source_confirmation_v1/report.json"
GROUNDING_MANIFEST = REPO / "configs/phase3_discoveryworld_grounding_v2/qualification_manifest.json"
GROUNDING_SUMMARY = REPO / "runs/phase3_discoveryworld_grounding_v2_qualification/summary.json"
GROUNDING_CONFIG = REPO / "configs/phase3_discoveryworld_grounding_v2/qualification.json"
DEVELOPMENT_SUMMARY = REPO / "runs/phase3_discoveryworld_consumed_development_v5/summary.json"
SEEDS = tuple(range(97, 121))
SOURCE_ORDER = (
    "tetris",
    "candy_crush",
    "gymv_columns",
    "gymv_streets_of_rage_2",
    "gymv_thunder_force_iii",
    "gymv_strider",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _official_commit() -> str:
    checkout = REPO.parent / "discoveryworld-official"
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing preregistration")

    confirmation = _read(SOURCE_CONFIRMATION)
    grounding_manifest = _read(GROUNDING_MANIFEST)
    grounding_summary = _read(GROUNDING_SUMMARY)
    grounding_config = _read(GROUNDING_CONFIG)
    development = _read(DEVELOPMENT_SUMMARY)
    _validate_self_hash(confirmation, "report_sha256")
    _validate_self_hash(grounding_manifest, "manifest_sha256")
    _validate_self_hash(grounding_summary, "summary_sha256")
    _validate_self_hash(development, "summary_sha256")
    if confirmation.get("status") != "PHASE3_SOURCE_ONLY_INDUCTION_PROSPECTIVELY_CONFIRMED":
        raise SystemExit("source-only induction is not prospectively confirmed")
    if grounding_summary.get("status") != "DISCOVERYWORLD_GROUNDING_QUALIFICATION_PASSED":
        raise SystemExit("DiscoveryWorld V2 grounding is not qualified")
    if development.get("status") != "PHASE3_DISCOVERYWORLD_DEVELOPMENT_RUNTIME_COMPLETE":
        raise SystemExit("consumed Phase-3 development is incomplete")
    if int(development.get("complete_tasks", 0)) != 6:
        raise SystemExit("consumed Phase-3 development did not cover six tasks")

    artifacts = {
        path.stem: _read(path) for path in sorted(PROGRAM_DIR.glob("*.json"))
    }
    if set(artifacts) != set(SOURCE_ORDER):
        raise SystemExit("expected exactly the six frozen source artifacts")
    permutation = maximum_profile_contrast_derangement(artifacts)
    source_artifacts = {}
    for game in SOURCE_ORDER:
        path = PROGRAM_DIR / f"{game}.json"
        artifact = artifacts[game]
        _validate_self_hash(artifact, "artifact_sha256")
        source_artifacts[game] = {
            "path": _relative(path),
            "file_sha256": _file_sha256(path),
            "artifact_sha256": artifact["artifact_sha256"],
            "authentic_program_sha256": artifact["authentic_program"]["program_sha256"],
            "source_profile_sha256": artifact["source_only_profile"]["profile_sha256"],
            "permuted_control_source": permutation[game],
        }

    runtime_paths = (
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/discoveryworld_env.py",
        "src/motif_transfer/discoveryworld_policy.py",
        "src/motif_transfer/discoveryworld_sokoban_transfer.py",
        "src/motif_transfer/discoveryworld_applicability_grounder_v4.py",
        "src/motif_transfer/phase3_discoveryworld_grounding.py",
        "src/motif_transfer/phase3_source_induction.py",
        "src/motif_transfer/phase3_source_applicability.py",
        "src/motif_transfer/phase3_attempt_runtime.py",
        "src/motif_transfer/phase3_discoveryworld_transfer.py",
        "scripts/run_discoveryworld_target_only_v1.py",
        "scripts/run_discoveryworld_commit_recovery_v1.py",
        "scripts/run_phase3_discoveryworld_acquisition_v1.py",
    )
    runtime_hashes = {
        path: _file_sha256(REPO / path) for path in runtime_paths
    }
    model = dict(grounding_config["model"])
    tasks = []
    for index, seed in enumerate(SEEDS):
        source = SOURCE_ORDER[index % len(SOURCE_ORDER)]
        tasks.append({
            "task_id": f"proteomics.easy.seed{seed}",
            "scenario": "Proteomics", "difficulty": "Easy", "seed": seed,
            "source_game": source,
            "source_artifact": source_artifacts[source]["path"],
            "source_artifact_sha256": source_artifacts[source]["artifact_sha256"],
            "source_permuted_game": permutation[source],
            "selected_target_previously_executed": False,
        })

    body = {
        "schema_version": "phase3-discoveryworld-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_PHASE3_TARGET_RESET_OR_OUTCOME",
        "target_domain": "DiscoveryWorld/Proteomics/Easy",
        "tasks": tasks,
        "task_count": len(tasks),
        "source_assignment": "ROUND_ROBIN_FOUR_UNTOUCHED_SEEDS_PER_FROZEN_SOURCE",
        "source_artifacts": source_artifacts,
        "source_permutation": permutation,
        "conditions": [
            "neural_only", "source_induced", "source_permuted",
            "generic_scaffold", "target_native_ceiling",
        ],
        "acquisition_model": model,
        "acquisition_runtime": {
            "maximum_steps": 96, "include_vision": False,
            "task_workers": 4, "thread_id_base": 135000,
        },
        "fork_protocol": {
            "minimum_fork_after_episode_step": 1,
            "recovery_horizon": 8,
            "allowed_commit_actions": ["DROP", "PUT"],
            "minimum_binding_confidence": 0.8,
            "required_initial_position_candidates": 4,
            "required_initial_commit_candidates": 1,
            "require_initial_exact_commit_effect": "LOW",
            "require_authentic_and_permuted_source_admission": True,
            "require_authentic_permuted_first_selection_contrast": True,
            "selection_rule": (
                "FIRST_POLICY_VISIBLE_STATE_MEETING_ALL_FROZEN_STRUCTURAL_"
                "APPLICABILITY_PREDICATES"
            ),
            "forbidden_selection_fields": [
                "action_succeeded", "reward", "terminal", "official_success",
                "evaluation", "scorecard",
            ],
        },
        "analysis_protocol": {
            "estimand": "INTENTION_TO_TREAT_OVER_ALL_24_PREREGISTERED_TASKS",
            "inapplicable_task_rule": (
                "NO_MATCHED_SOURCE_ACTION;ALL_INTERVENTION_CONDITIONS_INHERIT_"
                "THE_RECORDED_NEURAL_ONLY_OUTCOME"
            ),
            "paired_test": "EXACT_TWO_SIDED_SIGN_TEST_DISCARDING_TIES",
            "negative_transfer_rate": "LOSSES_DIVIDED_BY_WINS_PLUS_LOSSES",
            "source_specific_control": (
                "MAXIMUM_SOURCE_PROFILE_CONTRAST_DERANGEMENT_FROZEN_WITHOUT_"
                "TARGET_OUTCOMES"
            ),
        },
        "frozen_gates": {
            "exact_24_acquisition_episodes_complete": True,
            "minimum_applicable_tasks": 16,
            "maximum_acquisition_schema_fallback_rate": 0.10,
            "maximum_invalid_native_actions": 0,
            "all_matched_forks_and_receipts_valid": True,
            "source_induced_strictly_improves_neural_only": True,
            "source_induced_vs_neural_exact_sign_p_max": 0.05,
            "source_induced_vs_neural_negative_transfer_rate_max": 0.25,
            "source_induced_strictly_beats_source_permuted": True,
            "source_induced_vs_permuted_exact_sign_p_max": 0.05,
            "source_induced_strictly_beats_generic_scaffold": True,
            "target_native_ceiling_not_below_source_induced": True,
            "all_six_sources_represented_among_applicable_tasks": True,
            "all_admitted_low_effect_source_permutations_change_first_trial": True,
            "zero_target_outcome_use_for_grounding_or_applicability": True,
            "zero_source_identity_features_in_runtime": True,
        },
        "failure_gates": {
            "any_runtime_error": "FAIL",
            "any_receipt_or_hash_mismatch": "FAIL",
            "any_formal_outcome_used_to_change_grounding_threshold_or_fork_rule": "FAIL",
            "fewer_than_six_discordant_source_induced_vs_neural_pairs": "FAIL_SIGNIFICANCE_GATE",
            "source_permuted_not_behaviorally_distinct_when_admitted_low_effect": "FAIL_SOURCE_SPECIFICITY",
        },
        "source_confirmation": {
            "path": _relative(SOURCE_CONFIRMATION),
            "file_sha256": _file_sha256(SOURCE_CONFIRMATION),
            "report_sha256": confirmation["report_sha256"],
        },
        "grounding_qualification": {
            "manifest_path": _relative(GROUNDING_MANIFEST),
            "manifest_file_sha256": _file_sha256(GROUNDING_MANIFEST),
            "manifest_sha256": grounding_manifest["manifest_sha256"],
            "summary_path": _relative(GROUNDING_SUMMARY),
            "summary_file_sha256": _file_sha256(GROUNDING_SUMMARY),
            "summary_sha256": grounding_summary["summary_sha256"],
            "frozen_schema_fallback_threshold": 0.10,
        },
        "consumed_development_evidence": {
            "path": _relative(DEVELOPMENT_SUMMARY),
            "file_sha256": _file_sha256(DEVELOPMENT_SUMMARY),
            "summary_sha256": development["summary_sha256"],
            "excluded_from_formal_estimates": True,
        },
        "runtime_file_sha256": runtime_hashes,
        "official_environment_commit": _official_commit(),
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
        "claim_boundary": (
            "Prospective untouched Phase-3 DiscoveryWorld test on seeds97-120. "
            "Source programs and source-permuted controls are frozen from source "
            "interventions; target-native neural grounding is qualified and frozen; "
            "no formal target reset, action, outcome, evaluator, or score was read "
            "before this preregistration."
        ),
    }
    payload = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "tasks": payload["task_count"],
        "manifest_sha256": payload["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
