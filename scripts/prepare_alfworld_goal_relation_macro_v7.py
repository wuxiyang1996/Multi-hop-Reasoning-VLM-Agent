#!/usr/bin/env python3
"""Freeze, compile, and bind the four runnable untouched valid-unseen tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO.parent
SOURCE_ALFWORLD_DATA = (
    PROJECT_ROOT / "Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data"
)
SOURCE_SPLIT = SOURCE_ALFWORLD_DATA / "json_2.1.1/valid_unseen"
OUTPUT_DIR = REPO / "configs/alfworld_goal_relation_macro_v7"
GENERATED_DATA = (
    REPO / "runs/alfworld_goal_relation_macro_v7_compiled/alfworld_data"
)
NAMESPACE = "alfworld-goal-relation-macro-v7-compiled-valid-unseen"
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(task_id: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}\0{task_id}".encode()).hexdigest()


def main() -> int:
    if OUTPUT_DIR.exists() or GENERATED_DATA.exists():
        raise SystemExit("refusing to overwrite frozen/compiled V7 artifacts")
    v6_audit_path = (
        REPO / "configs/alfworld_goal_relation_macro_v6/identity_audit.json"
    )
    v6_audit = json.loads(v6_audit_path.read_text(encoding="utf-8"))
    candidates = sorted({
        task_id
        for task_ids in v6_audit["roles"].values()
        for task_id in task_ids
        if "Sliced" not in task_id
    }, key=_rank)
    if len(candidates) != 4:
        raise SystemExit(f"expected four non-sliced candidates, got {len(candidates)}")
    roles = {"qualification": candidates[:2], "formal": candidates[2:]}
    selection_body = {
        "schema_version": "alfworld-goal-relation-macro-v7-selection",
        "status": "FROZEN_BEFORE_COMPILATION_OR_POLICY_RESET",
        "namespace": NAMESPACE,
        "parent_identity_audit": str(v6_audit_path.relative_to(REPO)),
        "parent_identity_audit_sha256": _sha256(v6_audit_path),
        "selection_used_directory_identity_only": True,
        "selection_used_observation_or_policy_outcome": False,
        "official_unsupported_sliced_tasks_excluded": 3,
        "roles": roles,
        "ranked_task_id_sha256": [_rank(task_id) for task_id in candidates],
    }
    selection = selection_body | {
        "selection_sha256": stable_hash(selection_body),
    }
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    generated_split = GENERATED_DATA / "json_2.1.1/valid_unseen"
    generated_split.mkdir(parents=True)
    shutil.copytree(SOURCE_ALFWORLD_DATA / "logic", GENERATED_DATA / "logic")
    all_trajectories = sorted(SOURCE_SPLIT.glob("**/traj_data.json"))
    index_by_task = {
        str(path.parent.relative_to(SOURCE_SPLIT) / "game.tw-pddl"): index
        for index, path in enumerate(all_trajectories)
    }
    generator = (
        PROJECT_ROOT / "conda/envs/alfworld/bin/alfworld-generate"
    )
    for task_id in candidates:
        index = index_by_task[task_id]
        subprocess.run([
            str(generator),
            "--data_path", str(SOURCE_SPLIT),
            "--save_path", str(generated_split),
            "--domain", str(SOURCE_ALFWORLD_DATA / "logic/alfred.pddl"),
            "--grammar", str(SOURCE_ALFWORLD_DATA / "logic/alfred.twl2"),
            "--start", str(index),
            "--end", str(index + 1),
            "--seed", "20260817",
        ], check=True)
    generated_hashes = {}
    for task_id in candidates:
        game = generated_split / task_id
        payload = json.loads(game.read_text(encoding="utf-8"))
        if payload.get("solvable") is not True:
            raise SystemExit(f"compiled task is not solvable: {task_id}")
        generated_hashes[task_id] = _sha256(game)

    parent = json.loads(
        (REPO / "configs/alfworld_goal_relation_macro_v5_development.json")
        .read_text(encoding="utf-8")
    )
    parent.pop("config_sha256", None)
    runner = REPO / "scripts/run_alfworld_goal_relation_macro_v6.py"
    runtime = REPO / "src/motif_transfer/alfworld_goal_relation_macro_v5.py"
    configs = {}
    for role, task_ids in roles.items():
        body = parent | {
            "schema_version": f"alfworld-goal-relation-macro-{role}-config-v7",
            "status": "FROZEN_CONSUMED_DEVELOPMENT_BEFORE_OUTCOMES",
            "v7_protocol_status": "FROZEN_BEFORE_ANY_V7_POLICY_RESET",
            "experiment_version": "COMPILED_VALID_UNSEEN_FAIL_CLOSED_V7",
            "role": role,
            "claim_boundary": (
                "EXECUTION-UNTOUCHED VALID_UNSEEN MULTIPLICITY TASKS COMPILED "
                "AFTER IDENTITY FREEZE; COMPILER EXPERT USED ONLY TO ESTABLISH "
                "SOLVABILITY, NEVER AS POLICY INPUT; TWO-TASK ROLE IS "
                "MECHANISM REPLICATION ONLY, NOT A POWERED OOD CLAIM"
            ),
            "historical_identity_audit_sha256": v6_audit[
                "historical_identity_audit_sha256"
            ],
            "v7_selection_sha256": selection["selection_sha256"],
            "task_ids": task_ids,
            "alfworld_split": "eval_out_of_distribution",
            "alfworld_data": str(GENERATED_DATA),
            "seed": 618907,
            "v6_runner_file_sha256": _sha256(runner),
            "v5_target_runtime_file_sha256": _sha256(runtime),
            "generated_game_file_sha256": {
                task_id: generated_hashes[task_id] for task_id in task_ids
            },
            "compiler_solvability_used_for_selection": False,
            "compiler_walkthrough_exposed_to_policy": False,
            "output": f"runs/alfworld_goal_relation_macro_v7_{role}/report.json",
            "untouched_reserve_read_or_run": False,
            "valid_unseen_read_or_run": False,
            "gates": parent["gates"] | {
                "minimum_second_cycle_action_changes": 1,
            },
        }
        configs[role] = body | {"config_sha256": stable_hash(body)}
    configs["formal"]["qualification_config_sha256"] = configs[
        "qualification"
    ]["config_sha256"]
    configs["formal"]["qualification_report"] = configs["qualification"][
        "output"
    ]
    formal_body = dict(configs["formal"])
    formal_body.pop("config_sha256")
    configs["formal"]["config_sha256"] = stable_hash(formal_body)
    for role, config in configs.items():
        (OUTPUT_DIR / f"{role}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "selection_sha256": selection["selection_sha256"],
        "roles": roles,
        "generated_game_file_sha256": generated_hashes,
        "config_sha256": {
            role: config["config_sha256"] for role, config in configs.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
