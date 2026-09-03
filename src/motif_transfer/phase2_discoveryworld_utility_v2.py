"""Selective matched-arm protocol after V1's outcome-blind coverage failure."""

from __future__ import annotations

from math import comb
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .direct_prospective_matrix_v1 import SOURCE_GAMES
from .phase2_discoveryworld_utility_v1 import CONDITIONS


SCHEMA = "phase2-discoveryworld-selective-utility-v2"
STATUS = "FROZEN_AFTER_TARGET_ACQUISITION_BEFORE_ANY_MATCHED_ARM"
RAW, AUTHENTIC, LEDGER_BLIND, WRONG, POSITION = CONDITIONS


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def validate_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def validate_manifest(manifest: Mapping[str, Any], *, repo: Path) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != STATUS:
        raise ValueError("wrong DiscoveryWorld selective V2 manifest")
    validate_self_hash(manifest, "manifest_sha256")
    if manifest.get("matched_outcomes_visible_at_freeze") is not False:
        raise ValueError("V2 was not frozen before matched outcomes")
    if manifest.get("eligibility_read_target_outcome") is not False:
        raise ValueError("V2 eligibility was not outcome blind")
    tasks = list(manifest.get("tasks") or ())
    if len(tasks) != 36 or sum(bool(row.get("applicable")) for row in tasks) != 35:
        raise ValueError("V2 must preserve exact 35 applicable + 1 abstention cohort")
    if sum(not bool(row.get("applicable")) for row in tasks) != 1:
        raise ValueError("V2 abstention count changed")
    counts = {game: 0 for game in SOURCE_GAMES}
    for row in tasks:
        counts[str(row["source_game"])] += 1
        if file_sha256(repo / str(row["target_episode"])) != row["target_episode_file_sha256"]:
            raise ValueError(f"target episode changed: {row['task_id']}")
        if row.get("applicable"):
            if file_sha256(repo / str(row["fork_config"])) != row["fork_config_file_sha256"]:
                raise ValueError(f"fork config changed: {row['task_id']}")
        elif row.get("abstention_rule") != "INHERIT_RECORDED_TARGET_ONLY_OUTCOME_FOR_ALL_ARMS":
            raise ValueError("inapplicable task does not fail closed")
    if set(counts.values()) != {6}:
        raise ValueError("six source lineages are not balanced")
    for relative, expected in manifest["runtime_file_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise ValueError(f"frozen V2 runtime changed: {relative}")


def sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    return min(
        1.0,
        2 * sum(comb(total, k) for k in range(min(wins, losses) + 1)) / 2 ** total,
    )


def make_cell(
    *, manifest_sha256: str, task: Mapping[str, Any], outcomes: Mapping[str, bool],
    recovery_steps: Mapping[str, int], routes: Sequence[Mapping[str, Any]],
    matched_result_file_sha256: str | None, all_matched_forks: bool,
    all_selection_receipts_valid: bool, runtime_error: str | None,
) -> dict[str, Any]:
    body = {
        "schema_version": "phase2-discoveryworld-selective-cell-v2",
        "manifest_sha256": manifest_sha256,
        "task_id": task["task_id"],
        "source_game": task["source_game"],
        "source_artifact_sha256": task["source_artifact_sha256"],
        "applicable": bool(task["applicable"]),
        "abstention_rule": task.get("abstention_rule"),
        "outcomes": dict(outcomes),
        "recovery_steps": dict(recovery_steps),
        "authentic_source_routes": list(routes),
        "matched_result_file_sha256": matched_result_file_sha256,
        "all_matched_forks": bool(all_matched_forks),
        "all_selection_receipts_valid": bool(all_selection_receipts_valid),
        "policy_runtime_saw_oracle_scorecard": False,
        "runtime_error": runtime_error,
    }
    return body | {"cell_sha256": stable_hash(body)}


def build_report(
    manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], *, repo: Path,
) -> dict[str, Any]:
    validate_manifest(manifest, repo=repo)
    expected = {row["task_id"]: row for row in manifest["tasks"]}
    observed = {row.get("task_id"): row for row in cells}
    counts = {condition: 0 for condition in CONDITIONS}
    wins = losses = ties = 0
    eligible_complete = True
    abstentions_valid = True
    receipts_valid = len(observed) == len(cells) == 36
    route_valid = True
    route_count = 0
    per_task = []
    for task_id, task in expected.items():
        cell = observed.get(task_id)
        if cell is None:
            receipts_valid = eligible_complete = abstentions_valid = False
            per_task.append({"task_id": task_id, "complete": False})
            continue
        body = dict(cell)
        claimed = body.pop("cell_sha256", None)
        cell_valid = claimed == stable_hash(body) and cell.get("manifest_sha256") == manifest["manifest_sha256"]
        outcomes = dict(cell.get("outcomes") or {})
        complete_conditions = tuple(outcomes) == CONDITIONS
        receipts_valid &= cell_valid and complete_conditions
        if complete_conditions:
            for condition in CONDITIONS:
                counts[condition] += int(bool(outcomes[condition]))
            if outcomes[AUTHENTIC] and not outcomes[RAW]:
                wins += 1
            elif outcomes[RAW] and not outcomes[AUTHENTIC]:
                losses += 1
            else:
                ties += 1
        routes = list(cell.get("authentic_source_routes") or ())
        if task["applicable"]:
            eligible_complete &= bool(
                cell.get("all_matched_forks") and cell.get("all_selection_receipts_valid")
                and cell.get("runtime_error") is None and routes
            )
            for route in routes:
                route_count += 1
                route_body = dict(route)
                route_claimed = route_body.pop("receipt_sha256", None)
                route_valid &= bool(
                    route_claimed == stable_hash(route_body)
                    and route.get("admitted") is True
                    and route.get("source_artifact_sha256") == task["source_artifact_sha256"]
                )
        else:
            abstentions_valid &= bool(
                not routes and cell.get("matched_result_file_sha256") is None
                and len(set(outcomes.values())) == 1
                and cell.get("abstention_rule")
                == "INHERIT_RECORDED_TARGET_ONLY_OUTCOME_FOR_ALL_ARMS"
                and cell.get("runtime_error") is None
            )
        per_task.append({
            "task_id": task_id, "source_game": task["source_game"],
            "applicable": task["applicable"], "outcomes": outcomes,
            "recovery_steps": cell.get("recovery_steps"), "cell_sha256": claimed,
            "strict_win": bool(complete_conditions and outcomes[AUTHENTIC] and not outcomes[RAW]),
            "strict_loss": bool(complete_conditions and outcomes[RAW] and not outcomes[AUTHENTIC]),
        })
    p_value = sign_p(wins, losses)
    discordant = wins + losses
    negative_rate = losses / discordant if discordant else 0.0
    gates = {
        "exact_36_cell_receipts_valid": receipts_valid,
        "exact_35_eligible_matched_arms_complete": eligible_complete,
        "one_inapplicable_task_failed_closed": abstentions_valid,
        "all_authentic_routes_valid_and_source_bound": route_valid and route_count > 0,
        "all_six_source_lineages_represented": {row["source_game"] for row in per_task} == set(SOURCE_GAMES),
        "authentic_success_strictly_improves_raw": counts[AUTHENTIC] > counts[RAW],
        "authentic_vs_raw_significant": p_value <= manifest["primary_endpoint"]["maximum_p"],
        "negative_transfer_rate_within_bound": negative_rate <= manifest["primary_endpoint"]["maximum_negative_rate"],
        "authentic_strictly_beats_ledger_blind": counts[AUTHENTIC] > counts[LEDGER_BLIND],
        "authentic_strictly_beats_wrong_controller": counts[AUTHENTIC] > counts[WRONG],
        "authentic_strictly_beats_position_control": counts[AUTHENTIC] > counts[POSITION],
    }
    body = {
        "schema_version": "phase2-discoveryworld-selective-utility-report-v2",
        "status": (
            "PHASE2_DISCOVERYWORLD_SELECTIVE_CAUSAL_UTILITY_VALIDATED"
            if all(gates.values()) else "PHASE2_DISCOVERYWORLD_SELECTIVE_CAUSAL_UTILITY_NOT_VALIDATED"
        ),
        "claim_boundary": manifest["claim_boundary"],
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": 36,
        "eligible_matched_tasks": 35,
        "fail_closed_abstentions": 1,
        "condition_successes": counts,
        "condition_success_rates": {key: value / 36 for key, value in counts.items()},
        "authentic_vs_raw": {
            "wins": wins, "losses": losses, "ties": ties,
            "success_gain": counts[AUTHENTIC] - counts[RAW],
            "percentage_point_gain": 100 * (counts[AUTHENTIC] - counts[RAW]) / 36,
            "exact_two_sided_sign_test_p": p_value,
            "discordant_negative_transfer_rate": negative_rate,
        },
        "source_route_count": route_count,
        "v1_coverage_status": "FAILED_35_OF_36_ELIGIBLE_NO_MATCHED_ARMS_RUN",
        "matched_outcomes_visible_at_v2_freeze": False,
        "gates": gates,
        "passed_gates": sum(bool(value) for value in gates.values()),
        "required_gates": len(gates),
        "per_task": per_task,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "AUTHENTIC", "CONDITIONS", "RAW", "SCHEMA", "STATUS", "build_report",
    "file_sha256", "make_cell", "read_object", "sign_p", "validate_manifest",
    "validate_self_hash",
]
