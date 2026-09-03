#!/usr/bin/env python3
"""Freeze consumed-task matched forks before observing V13 fork outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.contracts import stable_hash
from motif_transfer.relation_edge_value_v13 import (
    ADMISSION_THRESHOLD,
    EFFICIENCY_WEIGHT,
    FEATURE_NAMES,
    PROGRESS_WEIGHT,
    RIDGE_L2,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen hash: {field}")
    return claimed


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def _candidate_transition(decision: Mapping[str, Any]) -> Mapping[str, Any]:
    transition = decision.get("source_transition")
    if not isinstance(transition, dict):
        transition = decision.get("candidate_source_transition")
    if not isinstance(transition, dict) or transition.get("kind") != "EDGE":
        raise SystemExit("V13 candidate opportunity lacks source EDGE receipt")
    return transition


def _first_opportunity(
    *, version: str, episode: Mapping[str, Any], max_steps: int
) -> tuple[dict[str, Any] | None, str | None]:
    records = list(episode["records"])
    selected_index = next((
        index
        for index, record in enumerate(records)
        if bool(record["decision"].get("source_applicability", {}).get(
            "source_edge_candidate"
        ))
        and "best_realization_score" in record["decision"]
        and "target_policy_ratio" in record["decision"]
    ), None)
    if selected_index is None:
        return None, "NO_ACTIONABLE_SOURCE_EDGE_CANDIDATE"
    record = records[selected_index]
    step = int(record["step"])
    if step != selected_index:
        raise SystemExit("V13 report step/prefix index mismatch")
    if step >= max_steps:
        return None, "FIRST_CANDIDATE_OUTSIDE_V13_ENDPOINT"
    decision = record["decision"]
    transition = _candidate_transition(decision)
    prefix = [str(row["decision"]["action"]) for row in records[:step]]
    state_body = {
        "task_id": str(episode["task_id"]),
        "step": step,
        "goal": str(record["goal"]),
        "before": record["before"],
        "native_actions": list(map(str, record["native_actions"])),
        "ledger_before": record["ledger_before"],
        "history": prefix,
        "property_probabilities": record["property_probabilities"],
    }
    row = {
        "version": version,
        "task_id": str(episode["task_id"]),
        "task_family": str(episode["task_family"]),
        "fork_step": step,
        "prefix_actions": prefix,
        "expected_fork_state_sha256": stable_hash(state_body),
        "expected_fallback_action": str(decision["fallback_action"]),
        "expected_source_graph_sha256": str(transition["graph_sha256"]),
        "expected_source_edge": {
            key: transition[key]
            for key in ("from", "to", "guard", "kind")
        },
        "selection_authority": (
            "FIRST_TARGET_ACTIONABLE_SOURCE_EDGE_CANDIDATE_IN_CONSUMED_"
            "V12_POLICY_TRACE"
        ),
    }
    return row | {"fork_id": stable_hash(row)}, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", nargs=2,
        metavar=("VERSION", "PATH"), required=True,
    )
    parser.add_argument("--fork-runner-code", type=Path, required=True)
    parser.add_argument("--value-model-code", type=Path, required=True)
    parser.add_argument("--value-audit-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V13 fork plan: {args.output}")
    if args.max_steps != 60:
        raise SystemExit("V13 fork endpoint is frozen at 60 total steps")
    reports = {}
    opportunities = []
    exclusions = []
    canonical_dependencies = None
    task_ids: set[str] = set()
    for version, raw_path in args.report:
        if version in reports:
            raise SystemExit(f"duplicate V13 report version: {version}")
        path = Path(raw_path).resolve()
        report = _read(path)
        report_hash = _validate(report, "report_sha256")
        if report.get("phase") != "adaptation_gate":
            raise SystemExit("V13 forks accept consumed adaptation reports only")
        if report.get("existing_valid_unseen_heldout_read"):
            raise SystemExit("V13 input crossed existing heldout boundary")
        candidate_path = Path(str(report["artifact_path"])).resolve()
        candidate = _read(candidate_path)
        candidate_hash = _validate(candidate, "candidate_sha256")
        if candidate.get("candidate_authority") not in {
            "CONSUMED_REPLAY", "FRESH_ADAPTATION"
        }:
            raise SystemExit("V13 input candidate lacks adaptation authority")
        dependencies = {
            "target_grounder": candidate["target_grounder"],
            "property_router_sha256": candidate["property_router"][
                "artifact_sha256"
            ],
            "slot_source_ir_sha256": candidate["slot_source_ir"][
                "ir_sha256"
            ],
            "thresholds": candidate["thresholds"],
            "allowed_source_effects": candidate["transfer_scope"][
                "allowed_source_effects"
            ],
            "active_required_properties": candidate["transfer_scope"][
                "active_required_properties"
            ],
        }
        dependency_hash = stable_hash(dependencies)
        if canonical_dependencies is None:
            canonical_dependencies = dependency_hash
        elif dependency_hash != canonical_dependencies:
            raise SystemExit("V13 reports do not share target/source dependencies")
        version_rows = 0
        version_exclusions = 0
        for episode in report["episodes"]["authentic_slot_ir"]:
            task_id = str(episode["task_id"])
            if task_id in task_ids:
                raise SystemExit(f"V13 task reused across versions: {task_id}")
            task_ids.add(task_id)
            row, reason = _first_opportunity(
                version=version,
                episode=episode,
                max_steps=args.max_steps,
            )
            if row is None:
                exclusions.append({
                    "version": version,
                    "task_id": task_id,
                    "reason": reason,
                })
                version_exclusions += 1
            else:
                opportunities.append(row)
                version_rows += 1
        reports[version] = _receipt(path) | {
            "report_sha256": report_hash,
            "candidate": _receipt(candidate_path) | {
                "candidate_sha256": candidate_hash,
            },
            "runner_seed": int(candidate["experiment_parameters"][
                "runner_seed"
            ]),
            "input_task_count": len(
                report["episodes"]["authentic_slot_ir"]
            ),
            "selected_fork_count": version_rows,
            "excluded_fork_count": version_exclusions,
            "use_authority": "CONSUMED_TASK_IDENTITIES_AND_PREFIXES_ONLY",
        }
    if len(reports) < 4:
        raise SystemExit("V13 requires four consumed version groups")
    if len(opportunities) < 48:
        raise SystemExit("V13 requires at least 48 matched-fork opportunities")
    body = {
        "schema_version": "relation-edge-intervention-fork-plan-v13",
        "status": "FROZEN_BEFORE_ANY_V13_FORK_OUTCOME",
        "claim_boundary": (
            "CONSUMED_V9_V10_V11_V12_ADAPTATION_TASKS_ONLY; FIRST_EDGE_"
            "OPPORTUNITY_WITH_TARGET_NATIVE_ACTION_GROUNDING_SELECTED_"
            "WITHOUT_FORK_OUTCOMES; TOTAL_ENDPOINT_60; CONFIRMATION_AND_"
            "EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "reports": dict(sorted(reports.items())),
        "canonical_dependency_sha256": canonical_dependencies,
        "implementation": {
            "fork_planner": _receipt(Path(__file__)),
            "fork_runner": _receipt(args.fork_runner_code),
            "value_model": _receipt(args.value_model_code),
            "value_audit": _receipt(args.value_audit_code),
        },
        "max_steps": args.max_steps,
        "continuation_policy": (
            "SOURCE_EDGE_OR_TARGET_FALLBACK_ONCE_THEN_EDGE_PERMUTED_NODE_"
            "ONLY_POLICY_FOR_BOTH_BRANCHES"
        ),
        "fork_treatments": ["SOURCE_EDGE", "TARGET_ABSTAIN"],
        "independent_unit": "TASK_FIRST_ACTIONABLE_SOURCE_EDGE_OPPORTUNITY",
        "outcome_blind_selection": True,
        "selection_used_episode_official_success": False,
        "selection_used_fork_outcomes": False,
        "feature_names": list(FEATURE_NAMES),
        "value_model": {
            "kind": "STANDARDIZED_LINEAR_RIDGE_HEAD",
            "l2": RIDGE_L2,
            "admission_threshold": ADMISSION_THRESHOLD,
            "efficiency_weight": EFFICIENCY_WEIGHT,
            "progress_weight": PROGRESS_WEIGHT,
            "grouped_validation": "LEAVE_ONE_SOURCE_VERSION_OUT",
        },
        "fresh_authorization_gates": {
            "minimum_informative_task_forks": 32,
            "minimum_positive_utility_tasks": 4,
            "minimum_negative_utility_tasks": 4,
            "minimum_selected_tasks": 8,
            "minimum_selected_versions": 3,
            "minimum_aggregate_selected_success_delta": 2,
            "heldout_selected_success_delta_nonnegative_each_fold": True,
            "heldout_selected_utility_nonnegative_each_fold": True,
            "zero_selected_success_losses": True,
            "selected_utility_strictly_exceeds_admit_all": True,
            "selected_utility_strictly_exceeds_v12_step_nine": True,
        },
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "excluded_opportunities": exclusions,
        "excluded_opportunity_count": len(exclusions),
        "confirmation_read": False,
        "existing_valid_unseen_heldout_read": False,
    }
    result = body | {"plan_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "plan_sha256": result["plan_sha256"],
        "opportunity_count": len(opportunities),
        "excluded_opportunity_count": len(exclusions),
        "versions": {
            key: value["selected_fork_count"]
            for key, value in sorted(reports.items())
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
