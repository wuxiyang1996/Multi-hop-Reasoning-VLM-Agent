#!/usr/bin/env python3
"""Freeze a strict option-level audit of the consumed V15 preflight tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--broad-report", type=Path, required=True)
    parser.add_argument("--audit-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V15 strict audit: {args.output}")
    pool = _read(args.pool)
    broad = _read(args.broad_report)
    if stable_hash({
        key: value for key, value in pool.items() if key != "pool_sha256"
    }) != pool.get("pool_sha256"):
        raise SystemExit("V15 pool hash mismatch")
    if stable_hash({
        key: value for key, value in broad.items() if key != "report_sha256"
    }) != broad.get("report_sha256"):
        raise SystemExit("V15 broad report hash mismatch")
    if broad.get("status") != "OUTCOME_BLIND_SUBGOAL_CONTRAST_GATE_PASSED":
        raise SystemExit("V15 broad preflight did not complete")
    if (
        broad.get("reward_serialized")
        or broad.get("official_success_serialized")
        or broad.get("held_out_read_or_run")
    ):
        raise SystemExit("V15 broad preflight crossed the outcome boundary")
    task_ids = list(map(
        str, pool["splits"]["outcome_blind_subgoal_contrast_preflight"]
    ))
    body = {
        "schema_version": "strict-subgoal-option-audit-plan-v15",
        "status": "FROZEN_BEFORE_STRICT_AUDIT_REPLAY_OF_CONSUMED_V15_TASKS",
        "claim_boundary": (
            "CORRECTS_BROAD_ACTION_LEVEL_GATE_TO_OPTION_LEVEL_ON_ALREADY_"
            "CONSUMED_DEVELOPMENT_TASKS; SHADOW_REPLAY_ONLY; REWARD_AND_"
            "OFFICIAL_SUCCESS_MUST_NOT_BE READ_OR_SERIALIZED; NOT REAL_GAME_"
            "TRANSFER; CONFIRMATION_AND EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "broad_pool": _receipt(args.pool) | {"pool_sha256": pool["pool_sha256"]},
        "quarantined_broad_report": _receipt(args.broad_report) | {
            "report_sha256": broad["report_sha256"],
            "reason": (
                "Its primary contrast gate counted native-action disagreement, "
                "including within-option SEARCH-to-SEARCH choices."
            ),
        },
        "implementation": {
            "strict_audit_freezer": _receipt(Path(__file__)),
            "strict_audit_enumerator": _receipt(args.audit_code),
        },
        "task_authority": "CONSUMED_V15_DEVELOPMENT_IDENTITIES_ONLY",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "max_steps": int(pool["max_steps"]),
        "main_path_policy": "SOURCE_DISABLED_TARGET_NATIVE_CONTROL",
        "shadow_policies": [
            "AUTHENTIC_SOURCE_OPTION_CONTROLLER",
            "PHASE_PERMUTED_SOURCE_OPTION_CONTROL",
        ],
        "opportunity_definition": (
            "authentic abstract option differs from target-control abstract option"
        ),
        "source_specific_definition": (
            "authentic option differs from both target-control and phase-permuted options"
        ),
        "contrast_gate": {
            "minimum_tasks_with_option_contrast": 32,
            "minimum_tasks_with_source_specific_option_contrast": 16,
            "minimum_tasks_with_second_cycle_option_contrast": 16,
            "minimum_destination_groups_with_four_source_specific_tasks": 4,
        },
        "reward_serialized": False,
        "official_success_serialized": False,
        "confirmation_read": False,
        "existing_valid_unseen_heldout_read": False,
    }
    payload = body | {"plan_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "plan": str(args.output.resolve()),
        "plan_sha256": payload["plan_sha256"],
        "tasks": len(task_ids),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
