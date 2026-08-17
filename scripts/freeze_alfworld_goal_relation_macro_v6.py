#!/usr/bin/env python3
"""Outcome-blind freeze of the last unreferenced valid-unseen multiplicity IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO.parent
DATA_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/"
    ".cache/alfworld_data/json_2.1.1/valid_unseen"
)
HISTORICAL_REPOS = (
    "Multi-hop-Reasoning-VLM-Agent",
    "Multi-hop-Reasoning-VLM-Agent-experiment-clean",
    "Multi-hop-Reasoning-VLM-Agent-github-main",
    "Multi-hop-Reasoning-VLM-Agent-source-fresh-v1",
    "Multi-hop-Reasoning-VLM-Agent-two-agent-clean",
)
NAMESPACE = "alfworld-goal-relation-macro-v6-qualification"
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(task_id: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}\0{task_id}".encode()).hexdigest()


def _historically_unreferenced(task_id: str) -> bool:
    for name in HISTORICAL_REPOS:
        completed = subprocess.run(
            [
                "rg", "-l", "-F", task_id, str(PROJECT_ROOT / name),
                "--glob", "!**/.git/**", "--glob", "!**/.cache/**",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in {0, 1}:
            raise RuntimeError(completed.stderr)
        if completed.stdout.strip():
            return False
    return True


def main() -> int:
    output_dir = REPO / "configs/alfworld_goal_relation_macro_v6"
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {output_dir}")
    candidates = sorted(
        str(path.relative_to(DATA_ROOT) / "game.tw-pddl")
        for path in DATA_ROOT.glob("pick_two_obj_and_place-*/trial_*")
    )
    unreferenced = sorted(
        (task_id for task_id in candidates if _historically_unreferenced(task_id)),
        key=_rank,
    )
    if len(unreferenced) != 7:
        raise SystemExit(
            f"expected seven all-repository-unreferenced multiplicity IDs, "
            f"found {len(unreferenced)}"
        )
    roles = {
        "qualification": unreferenced[:3],
        "formal": unreferenced[3:],
    }
    audit_body = {
        "namespace": NAMESPACE,
        "historical_repositories": list(HISTORICAL_REPOS),
        "valid_unseen_multiplicity_population": len(candidates),
        "all_repository_unreferenced_population": len(unreferenced),
        "selection_used_task_directory_identity_only": True,
        "selection_used_observation_or_outcome": False,
        "ranked_task_id_sha256": [_rank(task_id) for task_id in unreferenced],
        "roles": roles,
    }
    audit = audit_body | {
        "historical_identity_audit_sha256": stable_hash(audit_body),
    }
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
            "schema_version": f"alfworld-goal-relation-macro-{role}-config-v6",
            # Compatibility token consumed by the byte-frozen V3 evaluator;
            # the V6 role/version below carries the actual protocol status.
            "status": "FROZEN_CONSUMED_DEVELOPMENT_BEFORE_OUTCOMES",
            "v6_protocol_status": "FROZEN_BEFORE_ANY_V6_VALID_UNSEEN_RESET",
            "experiment_version": "VALID_UNSEEN_FAIL_CLOSED_V6",
            "role": role,
            "claim_boundary": (
                "OUTCOME-BLIND SHA256-SPLIT VALID_UNSEEN MULTIPLICITY "
                f"{role.upper()}; ONLY FOUR FORMAL IDENTITIES REMAIN AFTER "
                "CROSS-REPOSITORY HISTORICAL EXCLUSION; SMALL PROSPECTIVE "
                "REPLICATION, NOT A HIGH-POWER GENERALIZATION CLAIM"
            ),
            "historical_identity_audit_sha256": audit[
                "historical_identity_audit_sha256"
            ],
            "task_ids": task_ids,
            "alfworld_split": "eval_out_of_distribution",
            "seed": 618907,
            "v6_runner_file_sha256": _sha256(runner),
            "v5_target_runtime_file_sha256": _sha256(runtime),
            "output": f"runs/alfworld_goal_relation_macro_v6_{role}/report.json",
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

    output_dir.mkdir(parents=True)
    (output_dir / "identity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for role, config in configs.items():
        (output_dir / f"{role}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "output_dir": str(output_dir),
        "historical_identity_audit_sha256": audit[
            "historical_identity_audit_sha256"
        ],
        "roles": roles,
        "config_sha256": {
            role: config["config_sha256"] for role, config in configs.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
