#!/usr/bin/env python3
"""Freeze disjoint replication reserves without reading task content or outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


OUTPUT_NAMES = (
    "alfworld_procedural_game_replication_v1_manifest.json",
    "alfworld_procedural_game_replication_v1_frozen.json",
    "webshop_sokoban_effect_replication_v1_frozen.json",
    "discoveryworld_replication_v1_manifest.json",
    "discoveryworld_target_only_replication_v1.json",
    "discoveryworld_sokoban_replication_v1_protocol.json",
    "tir_maze_topology_replication_v1_frozen.json",
    "four_domain_replication_v1_manifest.json",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def _task_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _task_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _task_strings(child)
    elif isinstance(value, str) and value.endswith("game.tw-pddl"):
        yield value


def _tir_split_ids(value: Any, valid_ids: set[str]) -> set[str]:
    observed: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "splits" and isinstance(child, Mapping):
                for rows in child.values():
                    if isinstance(rows, list):
                        observed.update(
                            str(row) for row in rows if str(row) in valid_ids
                        )
            observed.update(_tir_split_ids(child, valid_ids))
    elif isinstance(value, list):
        for child in value:
            observed.update(_tir_split_ids(child, valid_ids))
    return observed


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def freeze(
    *,
    output_dir: Path,
    alfworld_data: Path,
    tir_dataset_root: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    targets = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"replication freeze already exists: {existing}")

    # ALFWorld: exclude the three actually executed identity pools.  The 38
    # remaining valid_unseen IDs are partitioned before any reset into a
    # 32-task single-object replication and a six-task multiplicity reserve.
    alf_root = alfworld_data.resolve() / "json_2.1.1" / "valid_unseen"
    all_alf = {
        path.relative_to(alf_root).as_posix()
        for path in alf_root.rglob("game.tw-pddl")
    }
    if len(all_alf) != 134:
        raise ValueError(f"unexpected ALFWorld valid_unseen size: {len(all_alf)}")
    alf_consumed = set()
    for name in (
        "alfworld_v2_outcome_blind_pool.json",
        "sokoban_alfworld_effect_transfer_split_v2.json",
        "sokoban_alfworld_transfer_split_v1.json",
    ):
        alf_consumed.update(_task_strings(_read(REPO / "configs" / name)))
    remaining_alf = all_alf - alf_consumed
    multiplicity = sorted(
        (row for row in remaining_alf if row.startswith("pick_two_obj_and_place")),
        key=lambda row: _rank("alfworld-multiplicity-reserve-v1", row),
    )
    alf_replication = sorted(
        (row for row in remaining_alf if not row.startswith("pick_two_obj_and_place")),
        key=lambda row: _rank("alfworld-four-domain-replication-v1", row),
    )
    if len(alf_replication) != 32 or len(multiplicity) != 6:
        raise ValueError(
            "ALFWorld prospective partition changed: "
            f"replication={len(alf_replication)}, multiplicity={len(multiplicity)}"
        )
    alf_manifest = {
        "schema_version": "alfworld-procedural-game-replication-manifest-v1",
        "status": "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "selection_rule": (
            "Set difference over valid_unseen task identities only. Exclude the "
            "executed V2 adaptation pool and Sokoban V1/V2 evaluation splits; "
            "assign every remaining single-object task to replication and every "
            "remaining two-object task to the separately locked multiplicity reserve."
        ),
        "outcome_or_task_content_read_for_selection": False,
        "cells": {
            "alfworld_valid_unseen": {
                "splits": {
                    "held_out": alf_replication,
                    "multiplicity_formal_locked": multiplicity,
                }
            }
        },
        "counts": {
            "dataset": len(all_alf),
            "replication": len(alf_replication),
            "multiplicity_formal_locked": len(multiplicity),
        },
    }
    alf_manifest["manifest_sha256"] = stable_hash(alf_manifest)
    _write(targets[OUTPUT_NAMES[0]], alf_manifest)

    alf_config = _read(REPO / "configs/procedural_game_alfworld_v1_frozen.json")
    alf_config["claim_boundary"] = (
        "Independent prospective replication on all 32 never-opened remaining "
        "single-object ALFWorld valid_unseen identities. The six remaining "
        "two-object identities stay locked for multiplicity V1."
    )
    alf_config["target"]["qualification_manifest"] = (
        "configs/alfworld_procedural_game_replication_v1_manifest.json"
    )
    alf_config["target"]["qualification_report"] = (
        "runs/alfworld_procedural_game_replication_v1/replication_report.json"
    )
    alf_config["replication"] = {
        "role": "INDEPENDENT_FRESH_REPLICATION",
        "prior_final_report_used_for_policy_change": False,
        "multiplicity_tasks_in_this_run": 0,
        "expected_tasks": 32,
    }
    _write(targets[OUTPUT_NAMES[1]], alf_config)

    # WebShop: next contiguous deterministic IDs after the frozen 114-145 run.
    webshop = _read(REPO / "configs/webshop_sokoban_effect_transfer_v13_frozen.json")
    webshop_ids = [f"webshop.{index}" for index in range(146, 178)]
    webshop["artifact_role"] = "INDEPENDENT_FRESH_REPLICATION_V1"
    webshop["status"] = "FROZEN_BEFORE_ANY_GOAL_TEXT_OR_OUTCOME_FOR_IDS_146_177"
    webshop["selection_rule"] = (
        "Use the next 32 contiguous deterministic WebShop goal indices 146 "
        "through 177. No goal text, product identifier, trajectory, or outcome "
        "from these indices was read before this file was frozen."
    )
    webshop["task_ids"] = webshop_ids
    webshop["runtime"]["number_of_goals"] = 178
    webshop["claim_boundary"] = (
        "Independent prospective replication of the frozen Sokoban positive-"
        "effect route on deterministic local WebShop goals 146-177."
    )
    webshop["goal_text_read_or_run"] = False
    webshop["replication"] = {
        "role": "INDEPENDENT_FRESH_REPLICATION",
        "prior_final_report_used_for_policy_change": False,
        "expected_tasks": 32,
    }
    _write(targets[OUTPUT_NAMES[2]], webshop)

    # DiscoveryWorld: new Easy seeds, enumerated without environment reset.
    dw_tasks = [
        {"scenario": scenario, "difficulty": "Easy", "seed": seed}
        for seed in range(11, 21)
        for scenario in ("Space Sick", "Proteomics")
    ]
    dw_ids = [
        f"{row['scenario'].lower().replace(' ', '_')}.easy.seed{row['seed']}"
        for row in dw_tasks
    ]
    dw_manifest = {
        "schema_version": "discoveryworld-easy-replication-manifest-v1",
        "status": "FROZEN_BEFORE_ANY_SEED11_TO_SEED20_ROLLOUT",
        "official_environment_commit": "fd591323920be0d3786ef350955de1945aa571e5",
        "assignment_rule": (
            "Space Sick and Proteomics Easy seeds 11-20, enumerated before "
            "target rollout without reading world state, fork coverage, or outcome."
        ),
        "roles": {"formal_reserve": dw_tasks},
    }
    _write(targets[OUTPUT_NAMES[3]], dw_manifest)
    dw_target = _read(REPO / "configs/discoveryworld_target_only_v23_fresh_easy_formal.json")
    dw_target["schema_version"] = "discoveryworld-target-only-replication-v1"
    dw_target["manifest"] = "configs/discoveryworld_replication_v1_manifest.json"
    dw_target["claim_boundary"] = (
        "Independent target-only acquisition on 20 previously unrun Easy "
        "instances from Space Sick and Proteomics seeds11-20."
    )
    _write(targets[OUTPUT_NAMES[4]], dw_target)
    dw_protocol = _read(
        REPO / "configs/discoveryworld_sokoban_v23_fresh_easy_formal_protocol.json"
    )
    dw_protocol["schema_version"] = "discoveryworld-sokoban-easy-replication-v1"
    dw_protocol["status"] = "REPLICATION_RESERVE_PROTOCOL_FROZEN_BEFORE_OPEN"
    dw_protocol["candidate_commit"] = _git_head()
    dw_protocol["claim_boundary"] = (
        "Independent prospective replication on Space Sick and Proteomics Easy "
        "seeds11-20 with the unchanged source artifact, target policy, fork rule, "
        "five matched conditions, spatial realizer, and selector thresholds."
    )
    dw_protocol["target_baseline_config"] = (
        "configs/discoveryworld_target_only_replication_v1.json"
    )
    dw_protocol["task_ids"] = dw_ids
    dw_protocol["formal_gates"]["minimum_eligible_forks"] = 10
    dw_protocol["formal_gates"][
        "minimum_authentic_success_gain_vs_target_native"
    ] = 2
    dw_protocol["operational_disclosure"] = (
        "No scientific changes or selective retries are permitted after the "
        "first seed11 target reset. Transport retries must preserve the same "
        "config and memoized request identity."
    )
    _write(targets[OUTPUT_NAMES[5]], dw_protocol)

    # TIR: exclude every previously assigned TIR split ID and select 48 unseen
    # single-image maze IDs by a new namespace hash. Only schema/IDs are read.
    tir_path = tir_dataset_root.resolve() / "TIR-Bench.json"
    tir_rows = json.loads(tir_path.read_text(encoding="utf-8"))
    maze_rows = {
        str(row["id"]): row for row in tir_rows
        if row.get("task") == "maze" and not row.get("image_2")
    }
    used_tir: set[str] = set()
    for path in (REPO / "configs").glob("*tir*.json"):
        used_tir.update(_tir_split_ids(_read(path), set(maze_rows)))
    available_tir = sorted(
        set(maze_rows) - used_tir,
        key=lambda row: _rank("tir-maze-independent-replication-v1", row),
    )
    if len(available_tir) != 75:
        raise ValueError(f"unexpected unused TIR maze pool: {len(available_tir)}")
    tir_ids = available_tir[:48]
    tir = _read(REPO / "configs/tir_maze_topology_v2_frozen.json")
    tir["claim_boundary"]["heldout"] = (
        "Independent prospective replication on 48 never-assigned single-image "
        "TIR maze IDs using the unchanged Sokoban topology artifact, target "
        "neural binder, pixel executor, and controls."
    )
    tir["dataset"]["selection_contract"] = (
        "Collect every ID previously present in a TIR split, restrict the fixed "
        "dataset to single-image maze rows, then take the first 48 unseen IDs by "
        "sha256('tir-maze-independent-replication-v1\\0' + id). No prompt, "
        "image, answer, or model outcome is read for selection."
    )
    tir["splits"]["heldout"] = tir_ids
    tir["replication"] = {
        "role": "INDEPENDENT_FRESH_REPLICATION",
        "prior_final_report_used_for_policy_change": False,
        "previously_assigned_ids_excluded": len(used_tir),
        "available_before_selection": len(available_tir),
        "expected_tasks": len(tir_ids),
    }
    _write(targets[OUTPUT_NAMES[6]], tir)

    component_paths = list(targets.values())[:-1]
    master = {
        "schema_version": "four-domain-independent-replication-manifest-v1",
        "status": "FROZEN_BEFORE_ANY_REPLICATION_TARGET_RESET_OR_QUERY",
        "frozen_at_git_commit": _git_head(),
        "selection_saw_task_content_or_outcome": False,
        "domains": {
            "alfworld": {"tasks": 32, "config": component_paths[1].name},
            "webshop": {"tasks": 32, "config": component_paths[2].name},
            "discoveryworld": {"tasks": 20, "config": component_paths[5].name},
            "tir": {"tasks": 48, "config": component_paths[6].name},
        },
        "aggregate_estimand": {
            "primary": (
                "Each domain authentic paired wins must exceed losses; at least "
                "three of four domains must have a strict positive success delta; "
                "no domain may have a negative success delta."
            ),
            "secondary": (
                "Equal-domain-weight mean success-rate delta and pooled paired "
                "sign test are descriptive because task units are heterogeneous."
            ),
            "thresholds_or_tasks_may_change_after_open": False,
        },
        "locked_future_alfworld_multiplicity_ids": multiplicity,
        "component_file_sha256": {
            path.name: _sha256(path) for path in component_paths
        },
        "freezer_file_sha256": _sha256(Path(__file__)),
    }
    master["manifest_sha256"] = stable_hash(master)
    _write(targets[OUTPUT_NAMES[7]], master)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "configs")
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--tir-dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(
        output_dir=args.output_dir,
        alfworld_data=args.alfworld_data,
        tir_dataset_root=args.tir_dataset_root,
    ), indent=2))


if __name__ == "__main__":
    main()
