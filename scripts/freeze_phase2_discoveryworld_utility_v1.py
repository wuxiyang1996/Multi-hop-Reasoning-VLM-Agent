#!/usr/bin/env python3
"""Freeze the fresh powered DiscoveryWorld Phase-2 utility cohort."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES  # noqa: E402
from motif_transfer.phase2_discoveryworld_utility_v1 import (  # noqa: E402
    CONDITIONS,
    SCHEMA,
    STATUS,
    file_sha256,
    read_object,
)


ROOT = REPO / "configs/phase2_discoveryworld_utility_v1"
OUTPUT = ROOT / "manifest.json"
SEEDS = tuple(range(45, 81))


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_fresh(task_ids: tuple[str, ...]) -> None:
    for task_id in task_ids:
        completed = subprocess.run(
            ["rg", "-l", "-F", task_id, "runs", "docs/results", "configs"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(completed.stderr)
        if completed.stdout.strip():
            raise SystemExit(f"target identity already appears: {task_id}")


def main() -> None:
    frozen = [OUTPUT, ROOT / "target_manifest.json", ROOT / "target_only.json", ROOT / "protocol.json"]
    if any(path.exists() for path in frozen):
        raise SystemExit("DiscoveryWorld Phase-2 V1 is already frozen")
    task_ids = tuple(f"proteomics.easy.seed{seed}" for seed in SEEDS)
    assert_fresh(task_ids)

    phase1 = read_object(REPO / "configs/phase1_direct_prospective_v1/manifest.json")
    source_rows = phase1["sources"]
    tasks = []
    for index, (seed, task_id) in enumerate(zip(SEEDS, task_ids)):
        game = SOURCE_GAMES[index % len(SOURCE_GAMES)]
        source = source_rows[game]
        tasks.append({
            "task_id": task_id,
            "scenario": "Proteomics",
            "difficulty": "Easy",
            "seed": seed,
            "source_game": game,
            "source_artifact": source["artifact"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_file_sha256": source["artifact_file_sha256"],
            "selected_target_previously_executed": False,
        })

    target_manifest_body = {
        "schema_version": "phase2-discoveryworld-target-reserve-v1",
        "status": "FROZEN_BEFORE_TARGET_RESET",
        "official_environment_commit": "fd591323920be0d3786ef350955de1945aa571e5",
        "roles": {
            "formal_reserve": [
                {key: row[key] for key in ("task_id", "scenario", "difficulty", "seed")}
                for row in tasks
            ]
        },
    }
    target_manifest = target_manifest_body | {
        "manifest_sha256": stable_hash(target_manifest_body),
    }
    target_manifest_path = ROOT / "target_manifest.json"
    write_once(target_manifest_path, target_manifest)

    target_config = {
        "schema_version": "phase2-discoveryworld-target-only-v1",
        "claim_boundary": (
            "Fresh source-blind Qwen target acquisition on Proteomics Easy seeds45-80; "
            "historical seeds and outcomes are excluded."
        ),
        "manifest": str(target_manifest_path.relative_to(REPO)),
        "manifest_role": "formal_reserve",
        "model": {
            "api_key_name": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "maximum_output_tokens": 1200,
            "model": "qwen/qwen3.5-35b-a3b",
            "provider": "openrouter",
            "schema_attempts": 3,
            "temperature": 0,
        },
        "runtime": {"include_vision": False, "maximum_steps": 96},
    }
    target_config_path = ROOT / "target_only.json"
    write_once(target_config_path, target_config)

    old_protocol = read_object(
        REPO / "configs/phase1_direct_prospective_v4/discoveryworld_protocol.json"
    )
    protocol = deepcopy(old_protocol)
    protocol.update({
        "schema_version": "phase2-discoveryworld-causal-utility-protocol-v1",
        "status": "FORMAL_RESERVE_FROZEN_BEFORE_OPEN",
        "claim_boundary": (
            "Powered fresh causal utility estimate on 36 Proteomics Easy forks; "
            "the 30 historical eligible forks used for power planning are excluded."
        ),
        "target_baseline_config": str(target_config_path.relative_to(REPO)),
        "task_ids": list(task_ids),
        "conditions": list(CONDITIONS),
    })
    protocol_path = ROOT / "protocol.json"
    write_once(protocol_path, protocol)

    runtime_files = (
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/discoveryworld_env.py",
        "src/motif_transfer/discoveryworld_policy.py",
        "src/motif_transfer/discoveryworld_sokoban_transfer.py",
        "src/motif_transfer/discoveryworld_applicability_grounder_v4.py",
        "src/motif_transfer/search_automaton_transfer_v16.py",
        "src/motif_transfer/sokoban_search_automaton_v16.py",
        "src/motif_transfer/direct_prospective_matrix_v1.py",
        "src/motif_transfer/phase2_discoveryworld_utility_v1.py",
        "scripts/run_discoveryworld_target_only_v1.py",
        "scripts/freeze_discoveryworld_qualification_forks_v1.py",
        "scripts/run_discoveryworld_commit_recovery_v1.py",
        "scripts/run_phase1_direct_discoveryworld_v1.py",
        "scripts/freeze_phase2_discoveryworld_utility_v1.py",
        "scripts/prepare_phase2_discoveryworld_utility_v1.py",
        "scripts/run_phase2_discoveryworld_utility_v1.py",
        "scripts/verify_phase2_discoveryworld_utility_v1.py",
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": (
            "Fresh causal utility of a source-derived positive-effect search controller "
            "with target-native neural grounding on Proteomics Easy seeds45-80 only."
        ),
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "historical_power_pilot": {
            "eligible_forks": 30,
            "authentic_vs_target_native_wins": 9,
            "authentic_vs_target_native_losses": 1,
            "included_in_primary_result": False,
            "purpose": "FIX_COHORT_SIZE_ONLY",
        },
        "primary_endpoint": {
            "metric": "official_success_at_end_of_eight_step_matched_recovery",
            "comparison": "authentic_sokoban_effect_plus_target_vs_target_native_myopic",
            "test": "exact_two_sided_paired_sign_test_on_discordant_tasks",
            "maximum_exact_two_sided_sign_p": 0.05,
            "maximum_discordant_negative_transfer_rate": 0.25,
        },
        "conditions": list(CONDITIONS),
        "target_manifest": str(target_manifest_path.relative_to(REPO)),
        "target_config": str(target_config_path.relative_to(REPO)),
        "protocol": str(protocol_path.relative_to(REPO)),
        "tasks": tasks,
        "source_assignment": "ROUND_ROBIN_SIX_TASKS_PER_INDEPENDENT_GAME_LINEAGE",
        "runtime_file_sha256": {
            relative: file_sha256(REPO / relative) for relative in runtime_files
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    write_once(OUTPUT, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "tasks": len(tasks),
        "source_counts": {
            game: sum(row["source_game"] == game for row in tasks) for game in SOURCE_GAMES
        },
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
