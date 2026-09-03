#!/usr/bin/env python3
"""Outcome-blind V24 enumeration with frozen neural-risk admissions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_source_relation_neural_risk_v24 import (  # noqa: E402
    score_neural_risk,
)
from enumerate_real_source_relation_eval_v20 import main as enumerate_v20  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument(
        "--role", choices=("sealed_confirmation",), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V24 enumeration: {args.output}")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    if (
        candidate.get("status") != "V24_SEALED_CONFIRMATION_AUTHORIZED"
        or not candidate.get("confirmation_authorized")
        or candidate.get("sealed_confirmation_read_or_run")
    ):
        raise SystemExit("V24 candidate lacks clean sealed-confirmation authority")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if candidate["manifest"]["manifest_sha256"] != manifest_hash:
        raise SystemExit("V24 candidate references another V20 manifest")

    with tempfile.TemporaryDirectory(prefix="v24-enumeration-") as directory:
        temp = Path(directory)
        compatibility_body = dict(candidate)
        compatibility_body.pop("candidate_sha256")
        compatibility_body["status"] = (
            "DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
        )
        compatibility_body["confirmation_authorized"] = True
        compatibility = compatibility_body | {
            "candidate_sha256": stable_hash(compatibility_body)
        }
        compatibility_path = temp / "compatibility_candidate.json"
        compatibility_path.write_text(
            json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_path = temp / "raw_enumeration.json"
        original_argv = sys.argv
        try:
            sys.argv = [
                "enumerate_real_source_relation_eval_v20.py",
                "--manifest", str(args.manifest),
                "--candidate", str(compatibility_path),
                "--role", args.role,
                "--output", str(raw_path),
                "--alfworld-config", str(args.alfworld_config),
                "--alfworld-data", str(args.alfworld_data),
            ]
            result = enumerate_v20()
        finally:
            sys.argv = original_argv
        if result != 0:
            raise SystemExit(f"V24 compatibility enumeration failed: {result}")
        raw = _read(raw_path)

    tasks = []
    opportunities = []
    for raw_task in raw["tasks"]:
        task = dict(raw_task)
        task.pop("task_receipt_sha256", None)
        opportunity = task.get("first_action_contrast")
        if opportunity is not None:
            row = dict(opportunity)
            row.pop("fork_id", None)
            neural_score = score_neural_risk(
                candidate=candidate,
                base_score=row["candidate_score"],
            )
            if neural_score["outcome_fields_consumed"]:
                raise RuntimeError("V24 selector consumed an outcome field")
            admissions = dict(row["policy_admissions"])
            admissions["v24_neural_risk"] = bool(neural_score["admitted"])
            row["policy_admissions"] = admissions
            row["neural_risk_score"] = neural_score
            opportunity = row | {"fork_id": stable_hash(row)}
            task["first_action_contrast"] = opportunity
            opportunities.append(opportunity)
        tasks.append(task | {"task_receipt_sha256": stable_hash(task)})
    policy_counts = {
        policy: sum(bool(row["policy_admissions"][policy]) for row in opportunities)
        for policy in opportunities[0]["policy_admissions"]
    } if opportunities else {}
    body = dict(raw)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "real-source-relation-neural-risk-enumeration-v24",
        "status": "OUTCOME_BLIND_V24_NEURAL_RISK_OPPORTUNITIES_COMPLETE",
        "claim_boundary": (
            "V24_NEURAL_RISK_POLICY_ACTIONS_ADMISSIONS_AND_CONTRASTS_FROZEN_"
            "BEFORE_ANY_SEALED_CONFIRMATION_OUTCOME; REWARD_DISCARDED; "
            "VALID_UNSEEN_UNREAD"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "file_sha256": _sha256(args.manifest),
            "manifest_sha256": manifest_hash,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "file_sha256": _sha256(args.candidate),
            "candidate_sha256": candidate_hash,
        },
        "tasks": tasks,
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "opportunities_by_family": dict(Counter(
            str(row["task_family"]) for row in opportunities
        )),
        "policy_admission_counts": policy_counts,
        "compatibility_adapter": {
            "implementation": str(Path(__file__).resolve()),
            "v20_enumerator": str(
                (REPO / "scripts/enumerate_real_source_relation_eval_v20.py").resolve()
            ),
            "changed_actions_or_base_scores": False,
            "added_score": "frozen_v24_target_neural_utility_mlp",
            "added_policy": "v24_neural_risk",
            "repaired_generic_null_hash_receipts": [
                "manifest.manifest_sha256", "candidate.candidate_sha256"
            ],
        },
    })
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "task_count": report["task_count"],
        "opportunity_count": len(opportunities),
        "policy_admission_counts": policy_counts,
        "outcomes_recorded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
