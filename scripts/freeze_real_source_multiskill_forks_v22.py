#!/usr/bin/env python3
"""Freeze V22 matched typed-successor causal fork cells."""

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
    parser.add_argument("--branch-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V22 fork plan: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_V22_SELECTED_TASK_RESET":
        raise SystemExit("V22 manifest has unexpected authority")
    reports = {}
    opportunities = []
    seen_tasks: set[str] = set()
    for path in args.contrast_report:
        report = _read(path)
        report_hash = _validate_hash(report, "report_sha256")
        role = str(report.get("role"))
        if role not in ROLES or role in reports:
            raise SystemExit(f"unexpected or duplicate V22 contrast role: {role}")
        if report.get("status") != "OUTCOME_BLIND_TYPED_CONTRASTS_COMPLETE":
            raise SystemExit("V22 contrast selection is incomplete")
        if report.get("outcomes_recorded") or report.get("rewards_recorded"):
            raise SystemExit("V22 contrast selection contains outcomes")
        if report["manifest"]["manifest_sha256"] != manifest_hash:
            raise SystemExit("V22 contrast report references another manifest")
        rows = list(report["opportunities"])
        role_tasks = set(map(str, manifest["splits"][role]))
        if any(str(row["task_id"]) not in role_tasks for row in rows):
            raise SystemExit("V22 opportunity crossed its frozen role")
        for row in rows:
            task_id = str(row["task_id"])
            if task_id in seen_tasks:
                raise SystemExit("V22 task appears in two fork roles")
            seen_tasks.add(task_id)
            opportunities.append(row)
        reports[role] = _receipt(path) | {
            "report_sha256": report_hash,
            "seed": int(report["seed"]),
            "task_count": int(report["task_count"]),
            "opportunity_count": len(rows),
        }
    if set(reports) != set(ROLES):
        raise SystemExit("V22 requires adaptation and calibration reports")
    role_counts = Counter(str(row["role"]) for row in opportunities)
    effect_counts = Counter(str(row["requested_source_effect"]) for row in opportunities)
    if role_counts["causal_adaptation"] < 24:
        raise SystemExit("V22 preflight needs at least 24 adaptation contrasts")
    if role_counts["causal_calibration"] < 12:
        raise SystemExit("V22 preflight needs at least 12 calibration contrasts")
    if effect_counts["MUTATE"] < 8 or effect_counts["RELATE"] < 8:
        raise SystemExit("V22 preflight lacks causal breadth across MUTATE and RELATE")
    body = {
        "schema_version": "real-source-multiskill-fork-plan-v22",
        "status": "FROZEN_BEFORE_ANY_V22_MATCHED_FORK_OUTCOME",
        "claim_boundary": (
            "OUTCOME_BLIND_TYPED_ACTION_CONTRASTS; SOURCE_EDGE_OR_TARGET_"
            "FALLBACK_ONCE_THEN_IDENTICAL_SOURCE_EDGE_DISABLED_CONTINUATION; "
            "REQUALIFICATION_DEVELOPMENT_CONFIRMATION_VALID_UNSEEN_UNREAD"
        ),
        "manifest": _receipt(args.manifest) | {
            "manifest_sha256": manifest_hash,
        },
        "parent_candidate": manifest["parent_candidate"],
        "contrast_reports": reports,
        "implementation": {
            **manifest["implementation"],
            "branch_implementation": _receipt(args.branch_code),
            "fork_plan_freezer": _receipt(Path(__file__)),
        },
        "transfer_scope": {
            "allowed_source_effects": manifest["allowed_source_effects"],
            "active_required_properties": manifest["active_required_properties"],
        },
        "max_steps": int(manifest["max_steps"]),
        "treatments": ["SOURCE_EDGE", "TARGET_ABSTAIN"],
        "continuation_policy": "EDGE_PERMUTED_NODE_ONLY_AFTER_FIRST_ACTION",
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "opportunity_counts_by_role": dict(role_counts),
        "opportunity_counts_by_effect": dict(effect_counts),
        "selection_used_fork_outcomes": False,
        "prospective_requalification_read_or_run": False,
        "future_development_confirmation_read_or_run": False,
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
        "opportunity_counts_by_effect": dict(effect_counts),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
