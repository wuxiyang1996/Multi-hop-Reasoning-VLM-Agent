#!/usr/bin/env python3
"""Gate V12 fresh freezing on consumed closed-loop replay dominance."""

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


def _validate(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid artifact hash: {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay", action="append", nargs=2,
        metavar=("VERSION", "PATH"), required=True,
    )
    parser.add_argument("--applicability-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(
            f"refusing to overwrite V12 development gate: {args.output}"
        )
    audit = _read(args.applicability_audit)
    _validate(audit, "audit_sha256")
    if audit.get("status") != "CONSUMED_CROSS_VERSION_AUDIT_PASSED":
        raise SystemExit("V12 static applicability audit did not pass")
    versions = {}
    for version, raw_path in args.replay:
        if version in versions:
            raise SystemExit(f"duplicate replay version: {version}")
        path = Path(raw_path).resolve()
        report = _read(path)
        _validate(report, "report_sha256")
        if report.get("phase") != "adaptation_gate":
            raise SystemExit("V12 development accepts gate-format replay only")
        if report.get("existing_valid_unseen_heldout_read"):
            raise SystemExit("V12 replay crossed heldout boundary")
        candidate = _read(Path(str(report["artifact_path"])))
        _validate(candidate, "candidate_sha256")
        if candidate.get("candidate_authority") != "CONSUMED_REPLAY":
            raise SystemExit("V12 replay candidate lacks consumed-only authority")
        if int(candidate["experiment_parameters"][
            "minimum_source_edge_step"
        ]) != int(audit["selected_minimum_edge_step"]):
            raise SystemExit("V12 replay rule differs from grouped audit")
        early_abstentions = sum(
            1
            for episode in report["episodes"]["authentic_slot_ir"]
            for record in episode["records"]
            if record["decision"].get("diagnostic")
            == "SOURCE_EDGE_EARLY_APPLICABILITY_ABSTENTION"
        )
        authentic = report["summaries"]["authentic_slot_ir"]
        node_only = report["summaries"]["edge_permuted_ir"]
        safety_only = report["summaries"]["property_permuted_router"]
        target = report["summaries"]["target_only"]
        versions[version] = {
            "path": str(path),
            "report_sha256": report["report_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "max_steps": int(report["max_steps"]),
            "authentic_successes": int(authentic["successes"]),
            "node_only_successes": int(node_only["successes"]),
            "safety_only_successes": int(safety_only["successes"]),
            "target_only_successes": int(target["successes"]),
            "authentic_mean_steps": float(authentic["mean_steps"]),
            "node_only_mean_steps": float(node_only["mean_steps"]),
            "paired_authentic_vs_node_only": report[
                "paired_official_success"
            ]["edge_permuted_ir"],
            "changed_source_edges": int(report[
                "source_transition_summaries"
            ]["authentic_slot_ir"]["source_edge_changed_count"]),
            "changed_tasks": int(authentic["changed_task_count"]),
            "early_applicability_abstentions": early_abstentions,
            "reopened_completed_slots": int(
                authentic["reopened_completed_slots"]
            ),
            "selected_postcondition_failures": int(
                authentic["selected_postcondition_failures"]
            ),
            "use_authority": "CONSUMED_CLOSED_LOOP_DEVELOPMENT_ONLY",
        }
    if len(versions) < 3:
        raise SystemExit("V12 replay gate requires three consumed versions")
    gates = {
        "three_consumed_closed_loop_versions": len(versions) >= 3,
        "authentic_success_noninferior_to_node_each_version": all(
            row["authentic_successes"] >= row["node_only_successes"]
            for row in versions.values()
        ),
        "authentic_success_noninferior_to_all_controls_each_version": all(
            row["authentic_successes"] >= max(
                row["node_only_successes"],
                row["safety_only_successes"],
                row["target_only_successes"],
            )
            for row in versions.values()
        ),
        "zero_paired_losses_to_node_each_version": all(
            int(row["paired_authentic_vs_node_only"]["losses"]) == 0
            for row in versions.values()
        ),
        "positive_paired_win_in_at_least_one_version": any(
            int(row["paired_authentic_vs_node_only"]["wins"]) > 0
            for row in versions.values()
        ),
        "authentic_mean_steps_noninferior_each_version": all(
            row["authentic_mean_steps"] <= row["node_only_mean_steps"]
            for row in versions.values()
        ),
        "changed_source_edges_each_version": all(
            row["changed_source_edges"] >= 2
            for row in versions.values()
        ),
        "early_applicability_abstention_observed": sum(
            row["early_applicability_abstentions"]
            for row in versions.values()
        ) >= 2,
        "zero_reopened_completed_slots": all(
            row["reopened_completed_slots"] == 0
            for row in versions.values()
        ),
        "zero_failed_selected_postconditions": all(
            row["selected_postcondition_failures"] == 0
            for row in versions.values()
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "selective-budgeted-relation-development-v12",
        "status": (
            "CONSUMED_CLOSED_LOOP_REPLAY_PASSED"
            if passed else "CONSUMED_CLOSED_LOOP_REPLAY_FAILED_STOP"
        ),
        "claim_boundary": (
            "CONSUMED_V9_V10_V11_TASKS_ONLY; CLOSED_LOOP_REPLAY_NOT_FRESH_"
            "EVIDENCE; FRESH_V12_FREEZE_ALLOWED_ONLY_IF_ALL_GATES_PASS; "
            "PRESERVED_CONFIRMATION_AND_EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "applicability_audit": {
            "path": str(args.applicability_audit.resolve()),
            "audit_sha256": audit["audit_sha256"],
            "selected_minimum_edge_step": audit[
                "selected_minimum_edge_step"
            ],
        },
        "versions": dict(sorted(versions.items())),
        "gates": gates,
        "passed": passed,
        "next_step": (
            "FREEZE_ONE_FRESH_V12_ADAPTATION_GATE"
            if passed else "STOP_WITHOUT_FRESH_V12_FREEZE"
        ),
        "existing_valid_unseen_heldout_read": False,
        "preserved_confirmation_read": False,
    }
    result = body | {"development_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "development_sha256": result["development_sha256"],
        "versions": result["versions"],
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
