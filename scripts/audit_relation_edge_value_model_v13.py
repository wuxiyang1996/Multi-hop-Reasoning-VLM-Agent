#!/usr/bin/env python3
"""Cross-validate a V13 target-native value head on consumed matched forks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.contracts import stable_hash
from motif_transfer.relation_edge_value_v13 import (
    ADMISSION_THRESHOLD,
    RIDGE_L2,
    fit_ridge_value_head,
    fork_utility,
    predict_relation_edge_value,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid V13 artifact hash: {field}")
    return claimed


def _training_row(fork: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    source = fork["branches"]["SOURCE_EDGE"]
    control = fork["branches"]["TARGET_ABSTAIN"]
    _validate(dict(source), "branch_sha256")
    _validate(dict(control), "branch_sha256")
    success_delta = (
        int(bool(source["official_success"]))
        - int(bool(control["official_success"]))
    )
    utility = fork_utility(
        source_success=bool(source["official_success"]),
        control_success=bool(control["official_success"]),
        source_steps=int(source["steps"]),
        control_steps=int(control["steps"]),
        source_completed_fraction=float(source["completed_fraction"]),
        control_completed_fraction=float(control["completed_fraction"]),
        max_steps=max_steps,
    )
    return {
        "fork_id": str(fork["fork_id"]),
        "version": str(fork["version"]),
        "task_id": str(fork["task_id"]),
        "fork_step": int(fork["fork_step"]),
        "features": dict(fork["features"]),
        "source_action": str(source["source_action"]),
        "control_action": str(control["control_action"]),
        "source_success": bool(source["official_success"]),
        "control_success": bool(control["official_success"]),
        "source_steps": int(source["steps"]),
        "control_steps": int(control["steps"]),
        "success_delta": success_delta,
        "efficiency_delta": (
            int(control["steps"]) - int(source["steps"])
        ) / max_steps,
        "progress_delta": (
            float(source["completed_fraction"])
            - float(control["completed_fraction"])
        ),
        "utility": utility,
        "source_relation_postcondition_observed": bool(
            source["fork_relation_postcondition_observed"]
        ),
        "informative_action_contrast": bool(
            fork["informative_action_contrast"]
        ),
    }


def _selection_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "selected_tasks": len(rows),
        "selected_versions": len({str(row["version"]) for row in rows}),
        "success_delta": sum(int(row["success_delta"]) for row in rows),
        "utility": sum(float(row["utility"]) for row in rows),
        "success_wins": sum(int(row["success_delta"]) > 0 for row in rows),
        "success_losses": sum(int(row["success_delta"]) < 0 for row in rows),
        "positive_utility_tasks": sum(
            float(row["utility"]) > 1e-12 for row in rows
        ),
        "negative_utility_tasks": sum(
            float(row["utility"]) < -1e-12 for row in rows
        ),
        "task_ids": sorted(str(row["task_id"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V13 value audit: {args.output}")
    report = _read(args.fork_report)
    report_hash = _validate(report, "report_sha256")
    if report.get("status") != "CONSUMED_MATCHED_FORKS_COMPLETE":
        raise SystemExit("V13 matched fork collection is incomplete")
    if not report.get("all_matched_invariants_passed"):
        raise SystemExit("V13 matched fork invariants failed")
    if report.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V13 fork report crossed heldout boundary")
    plan = _read(Path(str(report["plan"]["path"])))
    plan_hash = _validate(plan, "plan_sha256")
    if plan_hash != report["plan"]["plan_sha256"]:
        raise SystemExit("V13 report and plan hashes differ")
    max_steps = int(report["max_steps"])
    all_rows = [
        _training_row(row, max_steps) for row in report["forks"]
    ]
    informative = [
        row for row in all_rows if row["informative_action_contrast"]
    ]
    versions = sorted({str(row["version"]) for row in informative})
    folds = []
    crossfit_rows = []
    for heldout_version in versions:
        training = [
            row for row in informative
            if row["version"] != heldout_version
        ]
        testing = [
            row for row in informative
            if row["version"] == heldout_version
        ]
        model = fit_ridge_value_head(training, l2=RIDGE_L2)
        model_body = dict(model)
        model_hash = stable_hash(model_body)
        predictions = []
        for row in testing:
            prediction = predict_relation_edge_value(
                model, row["features"]
            )
            evaluated = dict(row) | {
                "predicted_utility": prediction,
                "admitted": prediction > ADMISSION_THRESHOLD,
            }
            predictions.append(evaluated)
            crossfit_rows.append(evaluated)
        selected = [row for row in predictions if row["admitted"]]
        folds.append({
            "heldout_version": heldout_version,
            "training_versions": [
                version for version in versions
                if version != heldout_version
            ],
            "training_tasks": len(training),
            "testing_tasks": len(testing),
            "value_head_sha256": model_hash,
            "value_head": model,
            "selection": _selection_summary(selected),
            "predictions": predictions,
        })
    selected = [row for row in crossfit_rows if row["admitted"]]
    admit_all = list(informative)
    step_nine = [row for row in informative if row["fork_step"] >= 9]
    aggregate = {
        "crossfit_value_head": _selection_summary(selected),
        "admit_all": _selection_summary(admit_all),
        "v12_step_nine": _selection_summary(step_nine),
    }
    positive = [row for row in informative if row["utility"] > 1e-12]
    negative = [row for row in informative if row["utility"] < -1e-12]
    requirements = plan["fresh_authorization_gates"]
    gates = {
        "four_version_grouped_cross_validation": len(versions) >= 4,
        "minimum_informative_task_forks": len(informative) >= int(
            requirements["minimum_informative_task_forks"]
        ),
        "minimum_positive_utility_tasks": len(positive) >= int(
            requirements["minimum_positive_utility_tasks"]
        ),
        "minimum_negative_utility_tasks": len(negative) >= int(
            requirements["minimum_negative_utility_tasks"]
        ),
        "minimum_selected_tasks": len(selected) >= int(
            requirements["minimum_selected_tasks"]
        ),
        "minimum_selected_versions": len({
            str(row["version"]) for row in selected
        }) >= int(requirements["minimum_selected_versions"]),
        "aggregate_selected_success_delta": (
            aggregate["crossfit_value_head"]["success_delta"]
            >= int(requirements[
                "minimum_aggregate_selected_success_delta"
            ])
        ),
        "heldout_selected_success_delta_nonnegative_each_fold": all(
            int(fold["selection"]["success_delta"]) >= 0
            for fold in folds
        ),
        "heldout_selected_utility_nonnegative_each_fold": all(
            float(fold["selection"]["utility"]) >= -1e-12
            for fold in folds
        ),
        "zero_selected_success_losses": (
            aggregate["crossfit_value_head"]["success_losses"] == 0
        ),
        "selected_utility_strictly_exceeds_admit_all": (
            aggregate["crossfit_value_head"]["utility"]
            > aggregate["admit_all"]["utility"] + 1e-12
        ),
        "selected_utility_strictly_exceeds_v12_step_nine": (
            aggregate["crossfit_value_head"]["utility"]
            > aggregate["v12_step_nine"]["utility"] + 1e-12
        ),
        "all_matched_fork_invariants": bool(
            report["all_matched_invariants_passed"]
        ),
    }
    passed = all(gates.values())
    full_model = fit_ridge_value_head(informative, l2=RIDGE_L2)
    full_model_body = dict(full_model)
    full_model_hash = stable_hash(full_model_body)
    body = {
        "schema_version": "relation-edge-value-audit-v13",
        "status": (
            "CONSUMED_GROUPED_VALUE_AUDIT_PASSED"
            if passed else "CONSUMED_GROUPED_VALUE_AUDIT_FAILED_STOP"
        ),
        "claim_boundary": (
            "CONSUMED_MATCHED_FORKS_ONLY; GROUPED_LEAVE_ONE_VERSION_OUT; "
            "FULL_VALUE_HEAD_HAS_NO_FRESH_AUTHORITY_UNLESS_ALL_GATES_PASS; "
            "CONFIRMATION_AND_EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "fork_report": {
            "path": str(args.fork_report.resolve()),
            "report_sha256": report_hash,
        },
        "plan_sha256": plan_hash,
        "max_steps": max_steps,
        "task_fork_count": len(all_rows),
        "informative_task_fork_count": len(informative),
        "positive_utility_task_count": len(positive),
        "negative_utility_task_count": len(negative),
        "neutral_utility_task_count": (
            len(informative) - len(positive) - len(negative)
        ),
        "value_model": {
            "kind": "STANDARDIZED_LINEAR_RIDGE_HEAD",
            "l2": RIDGE_L2,
            "admission_threshold": ADMISSION_THRESHOLD,
            "features_are_target_native_pre_action_only": True,
        },
        "folds": folds,
        "aggregate": aggregate,
        "gates": gates,
        "passed": passed,
        "full_value_head": full_model,
        "full_value_head_sha256": full_model_hash,
        "next_step": (
            "FREEZE_ONE_FRESH_V13_ADAPTATION_GATE"
            if passed else "STOP_WITHOUT_ANOTHER_FRESH_RELATION_EDGE_SPLIT"
        ),
        "confirmation_read": False,
        "existing_valid_unseen_heldout_read": False,
    }
    audit = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": audit["status"],
        "audit_sha256": audit["audit_sha256"],
        "task_fork_count": len(all_rows),
        "informative_task_fork_count": len(informative),
        "positive_utility_task_count": len(positive),
        "negative_utility_task_count": len(negative),
        "aggregate": aggregate,
        "gates": gates,
        "next_step": audit["next_step"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
