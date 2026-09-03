#!/usr/bin/env python3
"""Freeze six untouched DiscoveryWorld confirmation cells after consumed V1."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    read_object,
    validate_manifest,
)


CONFIG_ROOT = REPO / "configs/phase1_direct_prospective_v2"
V1_MANIFEST_PATH = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
V2_MANIFEST_PATH = CONFIG_ROOT / "discoveryworld_manifest.json"
OFFICIAL_COMMIT = "fd591323920be0d3786ef350955de1945aa571e5"
SEEDS = tuple(range(27, 33))
SCHEMA = "phase1-direct-discoveryworld-confirmation-v2"
STATUS = "FROZEN_BEFORE_ANY_V2_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def _assert_fresh(task_ids: tuple[str, ...]) -> None:
    for task_id in task_ids:
        result = subprocess.run(
            ["rg", "-l", "-F", task_id, "runs", "docs/results"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        hits = [line for line in result.stdout.splitlines() if line]
        if hits:
            raise SystemExit(
                f"V2 target identity already appears in evidence: {task_id}: {hits[:3]}"
            )


def main() -> None:
    frozen_paths = [
        V2_MANIFEST_PATH,
        CONFIG_ROOT / "discoveryworld_target_manifest.json",
        CONFIG_ROOT / "discoveryworld_target_only.json",
        CONFIG_ROOT / "discoveryworld_protocol.json",
    ]
    if any(path.exists() for path in frozen_paths):
        raise SystemExit("DiscoveryWorld V2 frozen files already exist; refusing overwrite")

    v1 = read_object(V1_MANIFEST_PATH)
    validate_manifest(v1, repo=REPO)
    task_ids = tuple(f"proteomics.easy.seed{seed}" for seed in SEEDS)
    _assert_fresh(task_ids)
    tasks = [
        {
            "task_id": task_id,
            "scenario": "Proteomics",
            "difficulty": "Easy",
            "seed": seed,
        }
        for task_id, seed in zip(task_ids, SEEDS)
    ]

    target_manifest_body = {
        "schema_version": "discoveryworld-phase1-direct-reserve-v2",
        "status": "FROZEN_BEFORE_TARGET_RESET",
        "official_environment_commit": OFFICIAL_COMMIT,
        "roles": {"formal_reserve": tasks},
    }
    target_manifest = target_manifest_body | {
        "manifest_sha256": stable_hash(target_manifest_body),
    }
    target_manifest_path = CONFIG_ROOT / "discoveryworld_target_manifest.json"
    _write_once(target_manifest_path, target_manifest)

    model = {
        "api_key_name": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "hidden_reasoning_effort": "none",
        "maximum_output_tokens": 2000,
        "model": "openai/gpt-4.1-mini",
        "provider": "openrouter",
        "schema_attempts": 3,
        "temperature": 0,
    }
    target_config = {
        "schema_version": "discoveryworld-target-only-phase1-direct-v2",
        "claim_boundary": (
            "Six never-executed Proteomics Easy instances, seeds27-32, "
            "acquired target-only before any source routing."
        ),
        "manifest": str(target_manifest_path.relative_to(REPO)),
        "manifest_role": "formal_reserve",
        "model": {
            key: value for key, value in model.items()
            if key != "hidden_reasoning_effort"
        },
        "runtime": {"include_vision": False, "maximum_steps": 96},
    }
    target_config_path = CONFIG_ROOT / "discoveryworld_target_only.json"
    _write_once(target_config_path, target_config)

    source_contract = {
        "compact_receipt": "docs/results/sokoban_effect_program_v2_compact_receipt.json",
        "require_source_gate_passed": True,
        "source_confirmation_sha256": "d64606c916ce6e812ae1b920771d5175cb48983a2f141b2dab6f43d491a6c1ed",
        "source_program_sha256": "6b02dc1d7271bbd435e90539cedd7d56d04fcc1ad03798dd6dd06146d67f1fcd",
        "transferred_structure": (
            "DIRECT_PROGRESS_AVAILABLE_OR_ASSIGNMENT_IMPROVEMENT_AVAILABLE_"
            "THEN_COMMIT_AND_VERIFY_ELSE_POSITION_AND_RECOMPUTE"
        ),
    }
    conditions = [
        "target_native_myopic",
        "authentic_sokoban_effect_plus_target",
        "commit_availability_control_plus_target",
        "inverted_effect_control_plus_target",
        "position_prior_control_plus_target",
    ]
    protocol = {
        "schema_version": "discoveryworld-search-automaton-direct-v2",
        "status": "FORMAL_RESERVE_FROZEN_BEFORE_OPEN",
        "claim_boundary": (
            "Uniform six-source confirmation on fresh DiscoveryWorld seeds27-32 "
            "after V1 was consumed for transport/grounder diagnosis."
        ),
        "evaluation_stage": "FORMAL_RESERVE",
        "target_baseline_config": str(target_config_path.relative_to(REPO)),
        "task_ids": list(task_ids),
        "fork_rule": {
            "name": "FIRST_PREDECLARED_NATIVE_COMMIT_PROPOSAL",
            "allowed_commit_actions": ["DROP", "PUT"],
            "minimum_fork_after_episode_step": 1,
            "reads_action_success": False,
            "reads_evaluation_or_scorecard": False,
            "reads_terminal_outcome": False,
        },
        "conditions": conditions,
        "source_contract": source_contract,
        "selector": {
            "positive_effect_threshold": 0.65,
            "prerequisite_threshold": 0.9,
        },
        "model": model,
        "recovery_horizon": 8,
        "target_native_spatial_realizer": {
            "activation": "AFTER_SUCCESSFUL_TARGET_BOUND_TELEPORT",
            "enabled": True,
            "may_select_commit": False,
            "objective": "STRICT_WORST_CASE_RELATION_ERROR_REDUCTION_OVER_UI_COMPATIBLE_VECTORS",
            "schema_version": "discoveryworld-local-spatial-realization-v1",
            "shared_across_all_conditions": True,
        },
        "transport_compatibility": {
            "change": "APPEND_LOWERCASE_JSON_TOKEN_TO_EXISTING_SYSTEM_PROMPTS",
            "candidate_schema_changed": False,
            "parser_changed": False,
            "selector_changed": False,
            "symbolic_gate_changed": False,
        },
        "operational_disclosure": (
            "All six fresh cells are run under one frozen V2; V1 seeds21-26 "
            "remain consumed development evidence and are not counted."
        ),
    }
    protocol_path = CONFIG_ROOT / "discoveryworld_protocol.json"
    _write_once(protocol_path, protocol)

    sources = v1["sources"]
    cells = []
    for game, task in zip(SOURCE_GAMES, tasks):
        source = sources[game]
        cells.append({
            "cell_id": f"{game}__to__discoveryworld",
            "source_game": game,
            "target_domain": "discoveryworld",
            "target_task_id": task["task_id"],
            "target_task": task,
            "target_task_multiplicity": 1,
            "selected_target_previously_executed": False,
            "source_artifact": source["artifact"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_file_sha256": source["artifact_file_sha256"],
        })

    runtime_files = [
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/direct_prospective_matrix_v1.py",
        "src/motif_transfer/discoveryworld_env.py",
        "src/motif_transfer/discoveryworld_policy.py",
        "src/motif_transfer/discoveryworld_search_automaton_v16.py",
        "src/motif_transfer/discoveryworld_sokoban_transfer.py",
        "src/motif_transfer/frozen_motif_agent.py",
        "src/motif_transfer/search_automaton_transfer_v16.py",
        "scripts/freeze_phase1_direct_discoveryworld_v2.py",
        "scripts/run_discoveryworld_target_only_v1.py",
        "scripts/run_discoveryworld_target_only_v2.py",
        "scripts/freeze_discoveryworld_qualification_forks_v1.py",
        "scripts/run_discoveryworld_commit_recovery_v1.py",
        "scripts/run_phase1_direct_discoveryworld_v1.py",
        "scripts/run_phase1_direct_discoveryworld_v2.py",
    ]
    missing = [path for path in runtime_files if not (REPO / path).is_file()]
    if missing:
        raise SystemExit(f"missing V2 runtime files: {missing}")
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": (
            "Six direct prospective game-lineage-to-DiscoveryWorld executions "
            "that replace, rather than retry, the uniformly consumed V1 reserve."
        ),
        "parent_v1_manifest": str(V1_MANIFEST_PATH.relative_to(REPO)),
        "parent_v1_manifest_sha256": v1["manifest_sha256"],
        "parent_v1_manifest_file_sha256": file_sha256(V1_MANIFEST_PATH),
        "superseded_v1_domain": "discoveryworld",
        "supersession_reason": (
            "V1 exposed a Qwen structured-output failure and an operational "
            "same-path duplicate race; no V1 DiscoveryWorld cell is counted."
        ),
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "selected_after_v1_diagnosis_but_before_v2_target_open": True,
        "uniform_confirmation_for_all_six_sources": True,
        "conditions": {"discoveryworld": conditions},
        "target_manifest": str(target_manifest_path.relative_to(REPO)),
        "target_config": str(target_config_path.relative_to(REPO)),
        "protocol": str(protocol_path.relative_to(REPO)),
        "cells": cells,
        "runtime_file_sha256": {
            path: file_sha256(REPO / path) for path in runtime_files
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(V2_MANIFEST_PATH, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "cells": len(cells),
        "tasks": list(task_ids),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
