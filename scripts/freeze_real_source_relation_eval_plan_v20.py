#!/usr/bin/env python3
"""Freeze V20 development or sealed-confirmation policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


ROLES = (
    "utility_requalification",
    "development_gate",
    "sealed_confirmation",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def _validate_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--enumeration", type=Path, required=True)
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--runner-code", type=Path, required=True)
    parser.add_argument("--v13-branch-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 eval plan: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    enumeration = _read(args.enumeration)
    enumeration_hash = _validate_hash(enumeration, "report_sha256")
    role = str(enumeration.get("role"))
    if role not in ROLES:
        raise SystemExit(f"unexpected V20 eval role: {role}")
    if enumeration.get("status") != "OUTCOME_BLIND_EVAL_OPPORTUNITIES_COMPLETE":
        raise SystemExit("V20 eval enumeration is incomplete")
    if enumeration.get("outcomes_recorded") or enumeration.get("rewards_recorded"):
        raise SystemExit("V20 eval enumeration contains outcomes")
    if enumeration["manifest"]["manifest_sha256"] != manifest_hash:
        raise SystemExit("V20 eval enumeration references another manifest")
    if enumeration["candidate"]["candidate_sha256"] != candidate_hash:
        raise SystemExit("V20 eval enumeration references another candidate")
    if int(enumeration["task_count"]) != len(manifest["splits"][role]):
        raise SystemExit("V20 eval enumeration did not cover the frozen split")
    if len(enumeration["opportunities"]) < 12:
        raise SystemExit("V20 eval requires at least 12 action contrasts")
    if enumeration["policy_admission_counts"].get("v20_selective", 0) < 4:
        raise SystemExit("V20 eval requires at least four selective admissions")
    alpha = {
        "utility_requalification": 0.25,
        "development_gate": 0.10,
        "sealed_confirmation": 0.05,
    }[role]
    minimum_wins = {
        "utility_requalification": 2,
        "development_gate": 4,
        "sealed_confirmation": 5,
    }[role]
    frozen_status = {
        "utility_requalification": (
            "FROZEN_BEFORE_ANY_UTILITY_REQUALIFICATION_OUTCOME"
        ),
        "development_gate": "FROZEN_BEFORE_ANY_DEVELOPMENT_OUTCOME",
        "sealed_confirmation": (
            "FROZEN_BEFORE_ANY_SEALED_CONFIRMATION_OUTCOME"
        ),
    }[role]
    evaluation_task_ids = (
        [str(row["task_id"]) for row in enumeration["opportunities"]]
        if role == "utility_requalification"
        else list(map(str, manifest["splits"][role]))
    )
    body = {
        "schema_version": "real-source-relation-eval-plan-v20",
        "status": frozen_status,
        "claim_boundary": (
            "FIRST_PREACTION_AUTHENTIC_SOURCE_EDGE_CONTRAST_PER_TASK; ALL_"
            "POLICY_ADMISSIONS_AND_GATES_FROZEN_BEFORE_ANY_SPLIT_OUTCOME; "
            "EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "role": role,
        "manifest": _receipt(args.manifest) | {
            "manifest_sha256": manifest_hash,
        },
        "candidate": _receipt(args.candidate) | {
            "candidate_sha256": candidate_hash,
        },
        "enumeration": _receipt(args.enumeration) | {
            "report_sha256": enumeration_hash,
        },
        "parent_candidate": manifest["parent_candidate"],
        "implementation": {
            "plan_freezer": _receipt(Path(__file__)),
            "enumerator": _receipt(args.enumerator_code),
            "runner": _receipt(args.runner_code),
            "v13_branch_implementation": _receipt(args.v13_branch_code),
        },
        "max_steps": int(manifest["max_steps"]),
        "seed": int(enumeration["seed"]),
        "task_ids": evaluation_task_ids,
        "task_count": len(evaluation_task_ids),
        "evaluation_population": (
            "FROZEN_ACTION_CONTRAST_OPPORTUNITIES_ONLY"
            if role == "utility_requalification"
            else "FULL_FROZEN_SPLIT_WITH_NO_CONTRAST_TARGET_ONLY_TIES"
        ),
        "opportunities": enumeration["opportunities"],
        "opportunity_count": int(enumeration["opportunity_count"]),
        "policy_admission_counts": enumeration["policy_admission_counts"],
        "primary_policy": "v20_selective",
        "target_baseline_policy": "target_only_graph_erased",
        "negative_controls": [
            "always_source_edge",
            "causal_effect_only",
            "lexical_move_relation",
            "late_step_heuristic",
            "target_only_graph_erased",
        ],
        "gates": {
            "minimum_opportunities": 12,
            "minimum_primary_admissions": 4,
            "minimum_primary_success_wins": minimum_wins,
            "primary_success_delta_strictly_positive": True,
            "primary_one_sided_exact_sign_test_alpha": alpha,
            "primary_selected_utility_strictly_positive": True,
            "primary_loss_count_strictly_less_than_always_source": True,
            "primary_net_delta_strictly_greater_than_lexical_move_heuristic": True,
            "source_event_recall_at_least": 0.90,
            "all_exact_state_fork_invariants": True,
        },
        "selection_used_eval_outcomes": False,
        "existing_valid_unseen_read_or_run": False,
    }
    plan = body | {"plan_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "role": role,
        "task_count": plan["task_count"],
        "opportunity_count": plan["opportunity_count"],
        "policy_admission_counts": plan["policy_admission_counts"],
        "gates": plan["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
