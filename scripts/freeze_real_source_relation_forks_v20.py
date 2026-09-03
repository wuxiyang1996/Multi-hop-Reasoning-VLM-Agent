#!/usr/bin/env python3
"""Freeze matched causal fork cells after outcome-blind V20 selection."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.relation_edge_value_v13 import FEATURE_NAMES  # noqa: E402


ROLES = ("causal_adaptation", "causal_calibration")


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
    parser.add_argument("--contrast-report", type=Path, action="append", required=True)
    parser.add_argument("--fork-runner-code", type=Path, required=True)
    parser.add_argument("--v13-branch-code", type=Path, required=True)
    parser.add_argument("--value-model-code", type=Path, required=True)
    parser.add_argument("--trainer-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 fork plan: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_V20_SELECTED_TASK_RESET":
        raise SystemExit("V20 manifest has unexpected authority")
    reports = {}
    opportunities = []
    seen_tasks: set[str] = set()
    for path in args.contrast_report:
        report = _read(path)
        report_hash = _validate_hash(report, "report_sha256")
        role = str(report.get("role"))
        if role not in ROLES or role in reports:
            raise SystemExit(f"unexpected or duplicate V20 contrast role: {role}")
        if report.get("status") != "OUTCOME_BLIND_RELATION_CONTRASTS_COMPLETE":
            raise SystemExit("V20 contrast selection is incomplete")
        if report.get("outcomes_recorded") or report.get("rewards_recorded"):
            raise SystemExit("V20 contrast report contains outcomes")
        if report["manifest"]["manifest_sha256"] != manifest_hash:
            raise SystemExit("V20 contrast report references another manifest")
        role_tasks = set(map(str, manifest["splits"][role]))
        rows = list(report["opportunities"])
        if any(str(row["task_id"]) not in role_tasks for row in rows):
            raise SystemExit("V20 opportunity crossed its frozen role")
        for row in rows:
            task_id = str(row["task_id"])
            if task_id in seen_tasks:
                raise SystemExit("V20 matched fork task appears in two roles")
            seen_tasks.add(task_id)
            opportunities.append(row)
        reports[role] = _receipt(path) | {
            "report_sha256": report_hash,
            "task_count": int(report["task_count"]),
            "opportunity_count": len(rows),
            "seed": int(report["seed"]),
        }
    if set(reports) != set(ROLES):
        raise SystemExit("V20 requires adaptation and calibration contrast reports")
    role_counts = Counter(str(row["role"]) for row in opportunities)
    if role_counts["causal_adaptation"] < 24:
        raise SystemExit("V20 requires at least 24 adaptation action contrasts")
    if role_counts["causal_calibration"] < 12:
        raise SystemExit("V20 requires at least 12 calibration action contrasts")
    body = {
        "schema_version": "real-source-relation-causal-fork-plan-v20",
        "status": "FROZEN_BEFORE_ANY_V20_MATCHED_FORK_OUTCOME",
        "claim_boundary": (
            "OUTCOME_BLIND_FIRST_ACTION_CONTRASTS_FROM FROZEN_CAUSAL_"
            "ADAPTATION_AND_CALIBRATION_TASKS; SOURCE_EDGE_OR_TARGET_"
            "FALLBACK_ONCE_THEN_IDENTICAL_SOURCE_EDGE_DISABLED_CONTINUATION; "
            "DEVELOPMENT_CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "manifest": _receipt(args.manifest) | {
            "manifest_sha256": manifest_hash,
        },
        "parent_candidate": manifest["parent_candidate"],
        "contrast_reports": reports,
        "implementation": {
            "fork_plan_freezer": _receipt(Path(__file__)),
            "fork_runner": _receipt(args.fork_runner_code),
            "v13_branch_implementation": _receipt(args.v13_branch_code),
            "value_feature_model": _receipt(args.value_model_code),
            "target_causal_trainer": _receipt(args.trainer_code),
        },
        "max_steps": int(manifest["max_steps"]),
        "treatments": ["SOURCE_EDGE", "TARGET_ABSTAIN"],
        "continuation_policy": (
            "EDGE_PERMUTED_NODE_ONLY_FOR_BOTH_BRANCHES_AFTER_FIRST_ACTION"
        ),
        "feature_names": list(FEATURE_NAMES),
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "opportunity_counts_by_role": dict(role_counts),
        "selection_used_fork_outcomes": False,
        "development_or_confirmation_read_or_run": False,
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
        "opportunity_count": len(opportunities),
        "opportunity_counts_by_role": dict(role_counts),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
