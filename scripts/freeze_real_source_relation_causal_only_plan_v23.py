#!/usr/bin/env python3
"""Freeze V23 development or confirmation evaluation before split outcomes."""

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


ROLES = ("development_gate", "sealed_confirmation")


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
    parser.add_argument("--repair-code", type=Path, required=True)
    parser.add_argument("--runner-adapter-code", type=Path, required=True)
    parser.add_argument("--generic-runner-code", type=Path, required=True)
    parser.add_argument("--branch-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V23 plan: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    enumeration = _read(args.enumeration)
    enumeration_hash = _validate_hash(enumeration, "report_sha256")
    role = str(enumeration.get("role"))
    if role not in ROLES:
        raise SystemExit(f"unexpected V23 role: {role}")
    expected_candidate_status = {
        "development_gate": "CAUSAL_ONLY_DEVELOPMENT_GATE_AUTHORIZED",
        "sealed_confirmation": (
            "V23_DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
        ),
    }[role]
    if candidate.get("status") != expected_candidate_status:
        raise SystemExit("V23 candidate has no authority for this role")
    if enumeration.get("status") != (
        "OUTCOME_BLIND_V23_CAUSAL_ONLY_OPPORTUNITIES_COMPLETE"
    ):
        raise SystemExit("V23 enumeration is incomplete")
    if enumeration.get("outcomes_recorded") or enumeration.get("rewards_recorded"):
        raise SystemExit("V23 enumeration contains outcomes")
    if enumeration["manifest"]["manifest_sha256"] != manifest_hash:
        raise SystemExit("V23 enumeration references another manifest")
    if enumeration["candidate"]["candidate_sha256"] != candidate_hash:
        raise SystemExit("V23 enumeration references another candidate")
    if int(enumeration["task_count"]) != len(manifest["splits"][role]):
        raise SystemExit("V23 enumeration did not cover the full frozen split")
    if int(enumeration["opportunity_count"]) < 12:
        raise SystemExit("V23 requires at least 12 frozen opportunities")
    admissions = int(enumeration["policy_admission_counts"].get(
        "v23_causal_only", 0
    ))
    if admissions < 12:
        raise SystemExit("V23 causal-only neural gate admitted fewer than 12")
    alpha = 0.10 if role == "development_gate" else 0.05
    minimum_wins = 4 if role == "development_gate" else 5
    v23_gates = {
        "minimum_opportunities": 12,
        "minimum_primary_admissions": 12,
        "minimum_success_wins": minimum_wins,
        "one_sided_exact_sign_alpha": alpha,
        "success_delta_strictly_positive": True,
        "selected_incremental_utility_strictly_positive": True,
        "source_event_recall_at_least": 0.90,
        "all_exact_state_fork_invariants": True,
    }
    body = {
        "schema_version": "real-source-relation-causal-only-plan-v23",
        "status": (
            "FROZEN_BEFORE_ANY_DEVELOPMENT_OUTCOME"
            if role == "development_gate" else
            "FROZEN_BEFORE_ANY_SEALED_CONFIRMATION_OUTCOME"
        ),
        "claim_boundary": (
            "V23_CAUSAL_ONLY_POLICY_ACTIONS_ADMISSIONS_AND_GATES_FROZEN_"
            "BEFORE_CURRENT_SPLIT_OUTCOME; FULL_SPLIT_SUCCESS_ACCOUNTING; "
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
            "enumeration_receipt_repair": _receipt(args.repair_code),
            "runner": _receipt(args.generic_runner_code),
            "v23_runner_adapter": _receipt(args.runner_adapter_code),
            "v13_branch_implementation": _receipt(args.branch_code),
        },
        "max_steps": int(manifest["max_steps"]),
        "seed": int(enumeration["seed"]),
        "task_ids": list(map(str, manifest["splits"][role])),
        "task_count": int(enumeration["task_count"]),
        "evaluation_population": "FULL_FROZEN_SPLIT_WITH_NO_CONTRAST_TARGET_TIES",
        "opportunities": enumeration["opportunities"],
        "opportunity_count": int(enumeration["opportunity_count"]),
        "policy_admission_counts": enumeration["policy_admission_counts"],
        "primary_policy": "v23_causal_only",
        "target_baseline_policy": "target_only_graph_erased",
        "negative_controls": [
            "target_only_graph_erased", "v20_selective",
            "late_step_heuristic", "lexical_move_relation",
        ],
        "v23_gates": v23_gates,
        "gates": {
            "minimum_opportunities": 12,
            "minimum_primary_admissions": 12,
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
        "v23_admissions": admissions,
        "v23_gates": v23_gates,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
