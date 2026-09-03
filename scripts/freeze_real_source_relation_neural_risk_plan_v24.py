#!/usr/bin/env python3
"""Freeze the V24 sealed-confirmation plan before any outcome is read."""

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
    parser.add_argument("--trainer-code", type=Path, required=True)
    parser.add_argument("--model-code", type=Path, required=True)
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--runner-adapter-code", type=Path, required=True)
    parser.add_argument("--generic-runner-code", type=Path, required=True)
    parser.add_argument("--branch-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V24 plan: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    enumeration = _read(args.enumeration)
    enumeration_hash = _validate_hash(enumeration, "report_sha256")
    if (
        candidate.get("status") != "V24_SEALED_CONFIRMATION_AUTHORIZED"
        or not candidate.get("confirmation_authorized")
        or candidate.get("sealed_confirmation_read_or_run")
    ):
        raise SystemExit("V24 candidate has no clean confirmation authority")
    if enumeration.get("role") != "sealed_confirmation":
        raise SystemExit("V24 plan is only valid for sealed confirmation")
    if enumeration.get("status") != (
        "OUTCOME_BLIND_V24_NEURAL_RISK_OPPORTUNITIES_COMPLETE"
    ):
        raise SystemExit("V24 enumeration is incomplete")
    if enumeration.get("outcomes_recorded") or enumeration.get("rewards_recorded"):
        raise SystemExit("V24 enumeration contains outcomes")
    if enumeration["manifest"]["manifest_sha256"] != manifest_hash:
        raise SystemExit("V24 enumeration references another manifest")
    if enumeration["candidate"]["candidate_sha256"] != candidate_hash:
        raise SystemExit("V24 enumeration references another candidate")
    if int(enumeration["task_count"]) != len(manifest["splits"]["sealed_confirmation"]):
        raise SystemExit("V24 enumeration did not cover the full frozen split")
    gate_spec = dict(candidate["confirmation_gates"])
    opportunities = int(enumeration["opportunity_count"])
    admissions = int(enumeration["policy_admission_counts"].get(
        "v24_neural_risk", 0
    ))
    if opportunities < int(gate_spec["minimum_opportunities"]):
        raise SystemExit("V24 frozen confirmation has too few opportunities")
    if admissions < int(gate_spec["minimum_primary_admissions"]):
        raise SystemExit("V24 neural risk gate admitted too few opportunities")
    body = {
        "schema_version": "real-source-relation-neural-risk-plan-v24",
        "status": "FROZEN_BEFORE_ANY_SEALED_CONFIRMATION_OUTCOME",
        "claim_boundary": (
            "V24_NEURAL_POLICY_ACTIONS_ADMISSIONS_IMPLEMENTATIONS_AND_GATES_"
            "FROZEN_BEFORE_ANY_SEALED_CONFIRMATION_OUTCOME; FULL_SPLIT_"
            "SUCCESS_ACCOUNTING; EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "role": "sealed_confirmation",
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
            "trainer": _receipt(args.trainer_code),
            "neural_risk_model": _receipt(args.model_code),
            "enumerator": _receipt(args.enumerator_code),
            "runner": _receipt(args.generic_runner_code),
            "v24_runner_adapter": _receipt(args.runner_adapter_code),
            "v13_branch_implementation": _receipt(args.branch_code),
        },
        "max_steps": int(manifest["max_steps"]),
        "seed": int(enumeration["seed"]),
        "task_ids": list(map(str, manifest["splits"]["sealed_confirmation"])),
        "task_count": int(enumeration["task_count"]),
        "evaluation_population": "FULL_FROZEN_SPLIT_WITH_NO_CONTRAST_TARGET_TIES",
        "opportunities": enumeration["opportunities"],
        "opportunity_count": opportunities,
        "policy_admission_counts": enumeration["policy_admission_counts"],
        "primary_policy": "v24_neural_risk",
        "target_baseline_policy": "target_only_graph_erased",
        "negative_controls": [
            "always_source_edge", "causal_effect_only", "lexical_move_relation",
            "late_step_heuristic", "v20_selective", "target_only_graph_erased",
        ],
        "v24_gates": gate_spec,
        "gates": {
            "minimum_opportunities": int(gate_spec["minimum_opportunities"]),
            "minimum_primary_admissions": int(
                gate_spec["minimum_primary_admissions"]
            ),
            "minimum_primary_success_wins": int(
                gate_spec["minimum_success_wins"]
            ),
            "primary_success_delta_strictly_positive": True,
            "primary_one_sided_exact_sign_test_alpha": float(
                gate_spec["one_sided_exact_sign_alpha"]
            ),
            "primary_selected_utility_strictly_positive": True,
            "primary_loss_count_strictly_less_than_always_source": True,
            "primary_net_delta_strictly_greater_than_lexical_move_heuristic": True,
            "source_event_recall_at_least": float(
                gate_spec["source_event_recall_at_least"]
            ),
            "all_exact_state_fork_invariants": True,
        },
        "selection_used_confirmation_outcomes": False,
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
        "task_count": plan["task_count"],
        "opportunity_count": opportunities,
        "v24_admissions": admissions,
        "v24_gates": gate_spec,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
