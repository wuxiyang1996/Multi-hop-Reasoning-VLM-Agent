"""Frozen contracts and aggregation for DiscoveryWorld Phase-2 utility V1.

The target-native policy first acquires a scientific decision state.  At the
first outcome-blind DROP/PUT proposal, five matched recovery arms share the
same neural binding, neural candidate grounder, native spatial realizer, and
official environment state.  Only the symbolic controller differs.
"""

from __future__ import annotations

from dataclasses import asdict
from math import comb
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .direct_prospective_matrix_v1 import SOURCE_GAMES


SCHEMA = "phase2-discoveryworld-causal-utility-v1"
STATUS = "FROZEN_BEFORE_ANY_PHASE2_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"
CONDITIONS = (
    "target_native_myopic",
    "authentic_sokoban_effect_plus_target",
    "commit_availability_control_plus_target",
    "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
)
RAW = CONDITIONS[0]
AUTHENTIC = CONDITIONS[1]
LEDGER_BLIND = CONDITIONS[2]
WRONG_CONTROLLER = CONDITIONS[3]
ALWAYS_POSITION = CONDITIONS[4]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_self_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def validate_manifest(manifest: Mapping[str, Any], *, repo: Path) -> None:
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("wrong DiscoveryWorld Phase-2 schema")
    if manifest.get("status") != STATUS:
        raise ValueError("DiscoveryWorld Phase-2 manifest is not frozen")
    validate_self_hash(manifest, "manifest_sha256")
    if tuple(manifest.get("conditions") or ()) != CONDITIONS:
        raise ValueError("matched condition order changed")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("reserve selection was not outcome blind")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("historical target outcomes are allowed")
    tasks = list(manifest.get("tasks") or ())
    if len(tasks) != 36:
        raise ValueError("Phase-2 DiscoveryWorld V1 requires exactly 36 tasks")
    ids = [str(row.get("task_id")) for row in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate target identity")
    if tuple(row.get("source_game") for row in tasks).count(None):
        raise ValueError("task omitted source lineage")
    counts = {game: 0 for game in SOURCE_GAMES}
    source_hashes = set()
    for row in tasks:
        game = str(row["source_game"])
        if game not in counts:
            raise ValueError(f"unknown source game: {game}")
        counts[game] += 1
        if row.get("scenario") != "Proteomics" or row.get("difficulty") != "Easy":
            raise ValueError("V1 target interface changed")
        if row.get("selected_target_previously_executed") is not False:
            raise ValueError("task does not attest freshness")
        source_path = repo / str(row["source_artifact"])
        if file_sha256(source_path) != str(row["source_artifact_file_sha256"]):
            raise ValueError(f"source artifact file changed: {game}")
        artifact = read_object(source_path)
        if artifact.get("artifact_sha256") != row.get("source_artifact_sha256"):
            raise ValueError(f"source artifact identity changed: {game}")
        if artifact.get("source_lineage", {}).get("game") != game:
            raise ValueError(f"source lineage mismatch: {game}")
        source_hashes.add(str(row["source_artifact_sha256"]))
    if set(counts.values()) != {6} or len(source_hashes) != 6:
        raise ValueError("six source lineages are not balanced and distinct")
    for relative, expected in (manifest.get("runtime_file_sha256") or {}).items():
        if file_sha256(repo / str(relative)) != str(expected):
            raise ValueError(f"frozen runtime changed: {relative}")


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    discordant = int(wins) + int(losses)
    if discordant == 0:
        return 1.0
    tail = min(int(wins), int(losses))
    probability = 2.0 * sum(comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, probability)


def _validate_route(route: Mapping[str, Any], expected_source: str) -> bool:
    body = dict(route)
    claimed = str(body.pop("receipt_sha256", ""))
    return bool(
        claimed
        and stable_hash(body) == claimed
        and route.get("admitted") is True
        and route.get("source_artifact_sha256") == expected_source
        and route.get("source_action") in {
            "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
        }
    )


def build_report(
    manifest: Mapping[str, Any], result_rows: Sequence[Mapping[str, Any]], *, repo: Path,
) -> dict[str, Any]:
    """Rebuild the primary report from immutable per-task receipts."""

    validate_manifest(manifest, repo=repo)
    expected = {str(row["task_id"]): row for row in manifest["tasks"]}
    observed = {str(row.get("task_id")): row for row in result_rows}
    duplicate_free = len(observed) == len(result_rows)
    per_task = []
    counts = {condition: 0 for condition in CONDITIONS}
    steps = {condition: [] for condition in CONDITIONS}
    wins = losses = ties = 0
    all_receipts_valid = duplicate_free
    all_matched = True
    all_runtime_complete = True
    all_routes_valid = True
    all_selection_receipts_valid = True
    zero_policy_oracle_use = True
    authentic_route_count = 0
    action_counts: dict[str, int] = {}
    for task_id, task in expected.items():
        row = observed.get(task_id)
        if row is None:
            per_task.append({"task_id": task_id, "complete": False, "reason": "MISSING"})
            all_receipts_valid = all_matched = all_runtime_complete = all_routes_valid = False
            continue
        body = dict(row)
        claimed = str(body.pop("cell_sha256", ""))
        receipt_valid = bool(claimed and stable_hash(body) == claimed)
        identity_ok = all(
            row.get(key) == task.get(key)
            for key in ("task_id", "source_game", "source_artifact_sha256")
        )
        outcomes = dict(row.get("outcomes") or {})
        recovery_steps = dict(row.get("recovery_steps") or {})
        conditions_complete = tuple(outcomes) == CONDITIONS and tuple(recovery_steps) == CONDITIONS
        matched = bool(row.get("all_matched_forks"))
        runtime_complete = row.get("runtime_error") is None and bool(row.get("mechanism_complete"))
        selections_valid = bool(row.get("all_selection_receipts_valid"))
        oracle_safe = row.get("policy_runtime_saw_oracle_scorecard") is False
        routes = list(row.get("authentic_source_routes") or ())
        routes_valid = bool(routes) and all(
            _validate_route(route, str(task["source_artifact_sha256"])) for route in routes
        )
        all_receipts_valid &= receipt_valid and identity_ok and conditions_complete
        all_matched &= matched
        all_runtime_complete &= runtime_complete
        all_routes_valid &= routes_valid
        all_selection_receipts_valid &= selections_valid
        zero_policy_oracle_use &= oracle_safe
        authentic_route_count += len(routes)
        for route in routes:
            action = str(route.get("source_action"))
            action_counts[action] = action_counts.get(action, 0) + 1
        if conditions_complete:
            for condition in CONDITIONS:
                success = bool(outcomes[condition])
                counts[condition] += int(success)
                steps[condition].append(int(recovery_steps[condition]))
            raw_success = bool(outcomes[RAW])
            authentic_success = bool(outcomes[AUTHENTIC])
            if authentic_success and not raw_success:
                wins += 1
            elif raw_success and not authentic_success:
                losses += 1
            else:
                ties += 1
        per_task.append({
            "task_id": task_id,
            "source_game": task["source_game"],
            "complete": bool(
                receipt_valid and identity_ok and conditions_complete and matched
                and runtime_complete and routes_valid and selections_valid and oracle_safe
            ),
            "outcomes": outcomes,
            "recovery_steps": recovery_steps,
            "authentic_strict_win": bool(
                conditions_complete and outcomes[AUTHENTIC] and not outcomes[RAW]
            ),
            "authentic_strict_loss": bool(
                conditions_complete and outcomes[RAW] and not outcomes[AUTHENTIC]
            ),
            "cell_sha256": row.get("cell_sha256"),
        })
    p_value = exact_two_sided_sign_test(wins, losses)
    discordant = wins + losses
    negative_rate = losses / discordant if discordant else 0.0
    complete_tasks = sum(bool(row["complete"]) for row in per_task)
    means = {
        condition: (sum(values) / len(values) if values else None)
        for condition, values in steps.items()
    }
    gates = {
        "exact_36_task_coverage": complete_tasks == 36 and len(result_rows) == 36,
        "all_cell_receipts_valid": all_receipts_valid,
        "all_runtime_and_mechanism_complete": all_runtime_complete,
        "all_policy_and_audit_forks_matched": all_matched,
        "all_selection_receipts_valid": all_selection_receipts_valid,
        "zero_policy_oracle_scorecard_use": zero_policy_oracle_use,
        "all_authentic_routes_source_bound_and_admitted": all_routes_valid,
        "all_six_source_lineages_represented": {
            row["source_game"] for row in per_task if row.get("complete")
        } == set(SOURCE_GAMES),
        "authentic_success_rate_strictly_improves_raw": counts[AUTHENTIC] > counts[RAW],
        "authentic_vs_raw_significant": p_value <= float(
            manifest["primary_endpoint"]["maximum_exact_two_sided_sign_p"]
        ),
        "negative_transfer_rate_within_frozen_bound": negative_rate <= float(
            manifest["primary_endpoint"]["maximum_discordant_negative_transfer_rate"]
        ),
        "authentic_strictly_beats_ledger_blind_control": counts[AUTHENTIC] > counts[LEDGER_BLIND],
        "authentic_strictly_beats_wrong_controller": counts[AUTHENTIC] > counts[WRONG_CONTROLLER],
        "authentic_strictly_beats_always_position_control": counts[AUTHENTIC] > counts[ALWAYS_POSITION],
        "nontrivial_source_route_exercised": authentic_route_count > 0 and bool(action_counts),
    }
    body = {
        "schema_version": "phase2-discoveryworld-causal-utility-report-v1",
        "status": (
            "PHASE2_DISCOVERYWORLD_CAUSAL_UTILITY_VALIDATED"
            if all(gates.values()) else "PHASE2_DISCOVERYWORLD_CAUSAL_UTILITY_NOT_VALIDATED"
        ),
        "claim_boundary": manifest["claim_boundary"],
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": len(expected),
        "complete_tasks": complete_tasks,
        "condition_successes": counts,
        "condition_success_rates": {
            condition: counts[condition] / len(expected) for condition in CONDITIONS
        },
        "condition_mean_recovery_steps": means,
        "authentic_vs_raw": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "success_gain": counts[AUTHENTIC] - counts[RAW],
            "percentage_point_gain": 100.0 * (counts[AUTHENTIC] - counts[RAW]) / len(expected),
            "exact_two_sided_sign_test_p": p_value,
            "discordant_negative_transfer_rate": negative_rate,
        },
        "authentic_source_route_count": authentic_route_count,
        "authentic_source_action_counts": action_counts,
        "historical_pilot_outcomes_included": False,
        "provider_or_scorecard_visible_to_source": False,
        "gates": gates,
        "passed_gates": sum(bool(value) for value in gates.values()),
        "required_gates": len(gates),
        "per_task": per_task,
    }
    return body | {"report_sha256": stable_hash(body)}


def make_cell_receipt(
    *, task: Mapping[str, Any], result: Mapping[str, Any], routes: Sequence[Mapping[str, Any]],
    matched_result_file_sha256: str, runtime_error: str | None,
) -> dict[str, Any]:
    outcomes = {
        condition: bool(result.get("conditions", {}).get(condition, {}).get("official_success"))
        for condition in CONDITIONS
    }
    recovery_steps = {
        condition: len(result.get("conditions", {}).get(condition, {}).get("recovery") or ())
        for condition in CONDITIONS
    }
    mechanism_complete = all(
        result.get("conditions", {}).get(condition, {}).get("runtime_error") is None
        for condition in CONDITIONS
    ) and all(condition in result.get("conditions", {}) for condition in CONDITIONS)
    body = {
        "schema_version": "phase2-discoveryworld-causal-utility-cell-v1",
        "task_id": task["task_id"],
        "source_game": task["source_game"],
        "source_artifact_sha256": task["source_artifact_sha256"],
        "matched_result_file_sha256": matched_result_file_sha256,
        "outcomes": outcomes,
        "recovery_steps": recovery_steps,
        "all_matched_forks": bool(result.get("all_matched_forks")),
        "all_selection_receipts_valid": bool(result.get("all_selection_receipts_valid")),
        "mechanism_complete": mechanism_complete,
        "policy_runtime_saw_oracle_scorecard": bool(
            result.get("policy_runtime_saw_oracle_scorecard")
        ),
        "authentic_source_routes": list(routes),
        "runtime_error": runtime_error,
    }
    return body | {"cell_sha256": stable_hash(body)}


__all__ = [
    "ALWAYS_POSITION", "AUTHENTIC", "CONDITIONS", "LEDGER_BLIND", "RAW",
    "SCHEMA", "STATUS", "WRONG_CONTROLLER", "build_report",
    "exact_two_sided_sign_test", "file_sha256", "make_cell_receipt",
    "read_object", "validate_manifest", "validate_self_hash",
]
