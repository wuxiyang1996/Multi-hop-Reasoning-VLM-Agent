#!/usr/bin/env python3
"""Audit a selective source-edge rule on consumed V9--V11 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from motif_transfer.contracts import stable_hash


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(
    version: str,
    report: dict[str, Any],
    *,
    efficiency_weight: float,
) -> list[dict[str, Any]]:
    controls = {
        str(row["task_id"]): row
        for row in report["episodes"]["edge_permuted_ir"]
    }
    rows = []
    for episode in report["episodes"]["authentic_slot_ir"]:
        changed_edges = [
            row for row in episode["records"]
            if row["decision"]["changed_effect"]
            and isinstance(
                row["decision"].get("source_transition"), dict
            )
            and row["decision"]["source_transition"].get("kind")
            == "EDGE"
        ]
        if not changed_edges:
            continue
        task_id = str(episode["task_id"])
        control = controls[task_id]
        max_steps = int(report["max_steps"])
        success_delta = (
            int(bool(episode["official_success"]))
            - int(bool(control["official_success"]))
        )
        normalized_efficiency_delta = (
            int(control["steps"]) - int(episode["steps"])
        ) / max_steps
        first = changed_edges[0]
        utility = success_delta + (
            efficiency_weight * normalized_efficiency_delta
        )
        rows.append({
            "version": version,
            "task_id": task_id,
            "max_steps": max_steps,
            "first_changed_edge_step": int(first["step"]),
            "changed_edge_count": len(changed_edges),
            "success_delta": success_delta,
            "normalized_efficiency_delta": normalized_efficiency_delta,
            "utility": utility,
            "target_policy_ratio": float(
                first["decision"]["target_policy_ratio"]
            ),
            "realization_score": float(
                first["decision"]["best_realization_score"]
            ),
            "remaining_slots": int(
                first["decision"]["slot_state"]["remaining_slots"]
            ),
            "source_transition": dict(
                first["decision"]["source_transition"]
            ),
        })
    return rows


def _select_threshold(
    rows: list[dict[str, Any]], *, maximum: int
) -> dict[str, Any]:
    candidates = []
    for threshold in range(maximum + 1):
        admitted = [
            row for row in rows
            if row["first_changed_edge_step"] >= threshold
        ]
        candidates.append({
            "minimum_edge_step": threshold,
            "training_utility": sum(
                float(row["utility"]) for row in admitted
            ),
            "admitted_tasks": len(admitted),
        })
    # Conservative tie break: later intervention wins equal utility.
    return max(
        candidates,
        key=lambda row: (
            round(float(row["training_utility"]), 12),
            int(row["minimum_edge_step"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", nargs=2,
        metavar=("VERSION", "PATH"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--efficiency-weight", type=float, default=0.1)
    parser.add_argument("--maximum-threshold", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(
            f"refusing to overwrite V12 applicability audit: {args.output}"
        )
    if not 0.0 < args.efficiency_weight < 1.0:
        raise SystemExit("efficiency weight must keep success lexicographically primary")
    reports = {}
    receipts = {}
    rows: list[dict[str, Any]] = []
    for version, raw_path in args.report:
        if version in reports:
            raise SystemExit(f"duplicate version: {version}")
        path = Path(raw_path).resolve()
        report = _read(path)
        report_body = dict(report)
        report_hash = str(report_body.pop("report_sha256", ""))
        if stable_hash(report_body) != report_hash:
            raise SystemExit(f"invalid report hash: {path}")
        if report.get("phase") != "adaptation_gate":
            raise SystemExit("applicability audit accepts consumed gates only")
        if report.get("existing_valid_unseen_heldout_read"):
            raise SystemExit("report crossed existing heldout boundary")
        reports[version] = report
        receipts[version] = {
            "path": str(path),
            "report_sha256": report_hash,
            "status": str(report["status"]),
            "use_authority": "CONSUMED_DEVELOPMENT_ONLY",
        }
        rows.extend(_rows(
            version,
            report,
            efficiency_weight=args.efficiency_weight,
        ))
    versions = sorted(reports)
    if len(versions) < 3:
        raise SystemExit("V12 audit requires at least three consumed versions")
    folds = []
    for heldout in versions:
        training = [row for row in rows if row["version"] != heldout]
        testing = [row for row in rows if row["version"] == heldout]
        selection = _select_threshold(
            training, maximum=args.maximum_threshold
        )
        threshold = int(selection["minimum_edge_step"])
        admitted = [
            row for row in testing
            if row["first_changed_edge_step"] >= threshold
        ]
        rejected = [
            row for row in testing
            if row["first_changed_edge_step"] < threshold
        ]
        folds.append({
            "heldout_version": heldout,
            "training_versions": [
                version for version in versions if version != heldout
            ],
            "selected_minimum_edge_step": threshold,
            "training_utility": selection["training_utility"],
            "heldout_utility": sum(
                float(row["utility"]) for row in admitted
            ),
            "heldout_admitted_task_ids": sorted(
                str(row["task_id"]) for row in admitted
            ),
            "heldout_rejected_task_ids": sorted(
                str(row["task_id"]) for row in rejected
            ),
            "heldout_admitted_success_delta": sum(
                int(row["success_delta"]) for row in admitted
            ),
            "heldout_admitted_efficiency_delta": sum(
                float(row["normalized_efficiency_delta"])
                for row in admitted
            ),
        })
    thresholds = {
        int(row["selected_minimum_edge_step"]) for row in folds
    }
    selected_threshold = (
        next(iter(thresholds)) if len(thresholds) == 1 else None
    )
    admitted_all = [
        row for row in rows
        if selected_threshold is not None
        and row["first_changed_edge_step"] >= selected_threshold
    ]
    rejected_all = [
        row for row in rows
        if selected_threshold is not None
        and row["first_changed_edge_step"] < selected_threshold
    ]
    positive_success = [row for row in rows if row["success_delta"] > 0]
    negative_success = [row for row in rows if row["success_delta"] < 0]
    gates = {
        "three_version_grouped_cross_validation": len(versions) >= 3,
        "at_least_ten_changed_edge_tasks": len(rows) >= 10,
        "fold_threshold_consensus": selected_threshold is not None,
        "heldout_utility_nonnegative_each_fold": all(
            float(row["heldout_utility"]) >= 0.0 for row in folds
        ),
        "heldout_success_delta_nonnegative_each_fold": all(
            int(row["heldout_admitted_success_delta"]) >= 0
            for row in folds
        ),
        "retains_every_observed_success_rescue": all(
            row in admitted_all for row in positive_success
        ),
        "rejects_every_observed_success_loss": all(
            row in rejected_all for row in negative_success
        ),
        "selected_rule_admits_multiple_versions": len({
            str(row["version"]) for row in admitted_all
        }) >= 3,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "budgeted-relation-applicability-audit-v12",
        "status": (
            "CONSUMED_CROSS_VERSION_AUDIT_PASSED"
            if passed else "CONSUMED_CROSS_VERSION_AUDIT_FAILED_STOP"
        ),
        "claim_boundary": (
            "CONSUMED_V9_V10_V11_ADAPTATION_ONLY; NO_FRESH_TASK_RESET; "
            "SUCCESS_PRIMARY_AND_EFFICIENCY_SECONDARY; EXISTING_VALID_"
            "UNSEEN_AND_PRESERVED_CONFIRMATION_UNREAD"
        ),
        "reports": receipts,
        "efficiency_weight": args.efficiency_weight,
        "utility": (
            "PAIRED_OFFICIAL_SUCCESS_DELTA + efficiency_weight * "
            "PAIRED_NORMALIZED_STEP_SAVING"
        ),
        "threshold_feature": "TARGET_EPISODE_STEP_INDEX",
        "threshold_selection": (
            "MAXIMIZE_TRAINING_UTILITY; CONSERVATIVE_LATER_STEP_TIE_BREAK"
        ),
        "rows": rows,
        "changed_edge_task_count": len(rows),
        "positive_success_task_count": len(positive_success),
        "negative_success_task_count": len(negative_success),
        "folds": folds,
        "selected_minimum_edge_step": selected_threshold,
        "selected_rule_admitted_task_count": len(admitted_all),
        "selected_rule_rejected_task_count": len(rejected_all),
        "gates": gates,
        "passed": passed,
        "next_step": (
            "RUN_CONSUMED_CLOSED_LOOP_REPLAY_BEFORE_ANY_FRESH_V12_FREEZE"
            if passed else "STOP_WITHOUT_FRESH_V12_FREEZE"
        ),
        "existing_valid_unseen_heldout_read": False,
        "preserved_confirmation_read": False,
    }
    result = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "audit_sha256": result["audit_sha256"],
        "changed_edge_task_count": len(rows),
        "selected_minimum_edge_step": selected_threshold,
        "folds": folds,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
