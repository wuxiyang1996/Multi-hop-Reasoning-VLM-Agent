#!/usr/bin/env python3
"""Bind planner-compiled V7 identities before any V8 policy reset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO.parent
SOURCE_DATA = (
    PROJECT_ROOT / "Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data"
)
GENERATED_DATA = (
    REPO / "runs/alfworld_goal_relation_macro_v8_planner_compiler_crosscheck/"
    "alfworld_data"
)
OUTPUT_DIR = REPO / "configs/alfworld_goal_relation_macro_v8"
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT_DIR.exists():
        raise SystemExit(f"refusing to overwrite frozen V8 configs: {OUTPUT_DIR}")
    selection_path = REPO / "configs/alfworld_goal_relation_macro_v7/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "FROZEN_BEFORE_COMPILATION_OR_POLICY_RESET":
        raise SystemExit("V7 identity selection is not frozen")
    logic = GENERATED_DATA / "logic"
    if not logic.exists():
        shutil.copytree(SOURCE_DATA / "logic", logic)
    generated_split = GENERATED_DATA / "json_2.1.1/valid_unseen"
    task_ids = [
        task_id for rows in selection["roles"].values() for task_id in rows
    ]
    generated_hashes = {}
    compiler = {}
    for task_id in task_ids:
        game = generated_split / task_id
        payload = json.loads(game.read_text(encoding="utf-8"))
        if payload.get("solvable") is not True:
            raise SystemExit(f"planner compiler rejected frozen task: {task_id}")
        generated_hashes[task_id] = _sha256(game)
        compiler[task_id] = {
            "solvable": True,
            "walkthrough_length": len(payload.get("walkthrough") or []),
        }
    parent = json.loads(
        (REPO / "configs/alfworld_goal_relation_macro_v5_development.json")
        .read_text(encoding="utf-8")
    )
    parent.pop("config_sha256", None)
    runner = REPO / "scripts/run_alfworld_goal_relation_macro_v6.py"
    runtime = REPO / "src/motif_transfer/alfworld_goal_relation_macro_v5.py"
    configs = {}
    for role, role_ids in selection["roles"].items():
        body = parent | {
            "schema_version": f"alfworld-goal-relation-macro-{role}-config-v8",
            "status": "FROZEN_CONSUMED_DEVELOPMENT_BEFORE_OUTCOMES",
            "v8_protocol_status": "FROZEN_BEFORE_ANY_V8_POLICY_RESET",
            "experiment_version": (
                "PLANNER_COMPILED_VALID_UNSEEN_FAIL_CLOSED_V8"
            ),
            "role": role,
            "claim_boundary": (
                "TWO EXECUTION-UNTOUCHED VALID_UNSEEN MULTIPLICITY TASKS PER "
                "ROLE; IDENTITIES FROZEN BEFORE COMPILATION; PLANNER COMPILER "
                "USED ONLY FOR TEXTWORLD SOLVABILITY AND NEVER EXPOSED TO THE "
                "TRANSFER POLICY; MECHANISM REPLICATION, NOT POWERED OOD CLAIM"
            ),
            "historical_identity_audit_sha256": selection[
                "parent_identity_audit_sha256"
            ],
            "v7_selection_sha256": selection["selection_sha256"],
            "task_ids": role_ids,
            "alfworld_split": "eval_out_of_distribution",
            "alfworld_data": str(GENERATED_DATA),
            "seed": 618907,
            "v6_runner_file_sha256": _sha256(runner),
            "v5_target_runtime_file_sha256": _sha256(runtime),
            "generated_game_file_sha256": {
                task_id: generated_hashes[task_id] for task_id in role_ids
            },
            "planner_compiler_audit": {
                task_id: compiler[task_id] for task_id in role_ids
            },
            "compiler_solvability_used_for_selection": False,
            "compiler_walkthrough_exposed_to_policy": False,
            "output": f"runs/alfworld_goal_relation_macro_v8_{role}/report.json",
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
    OUTPUT_DIR.mkdir(parents=True)
    audit_body = {
        "schema_version": "alfworld-goal-relation-macro-compiler-audit-v8",
        "status": "COMPILED_BEFORE_ANY_TRANSFER_POLICY_RESET",
        "selection_sha256": selection["selection_sha256"],
        "compiler": "ALFWORLD_TEXTWORLD_PLANNER",
        "policy_outcome_read": False,
        "generated_game_file_sha256": generated_hashes,
        "compiler_results": compiler,
    }
    audit = audit_body | {"compiler_audit_sha256": stable_hash(audit_body)}
    (OUTPUT_DIR / "compiler_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for role, config in configs.items():
        (OUTPUT_DIR / f"{role}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "compiler_audit_sha256": audit["compiler_audit_sha256"],
        "roles": selection["roles"],
        "config_sha256": {
            role: config["config_sha256"] for role, config in configs.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
