"""Frozen contracts and independent aggregation for Phase-2 WebShop utility."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .active_video_transfer import exact_binomial_two_sided
from .contracts import stable_hash
from .direct_prospective_matrix_v1 import SOURCE_GAMES
from .search_automaton_transfer_v16 import SourceSearchAutomaton
from .webshop_search_automaton_v16 import (
    AUTHENTIC,
    CEILING,
    CONDITIONS,
    LEDGER_BLIND,
    PERMUTED,
    RAW,
)


SCHEMA = "phase2-webshop-six-source-utility-v1"
STATUS = "FROZEN_BEFORE_ANY_PHASE2_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"
REPORT_SCHEMA = "phase2-webshop-six-source-utility-report-v1"
PASSED_STATUS = "PHASE2_WEBSHOP_CAUSAL_UTILITY_VALIDATED"
FAILED_STATUS = "PHASE2_WEBSHOP_CAUSAL_UTILITY_NOT_VALIDATED"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def validate_manifest(manifest: Mapping[str, Any], *, repo: Path) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != STATUS:
        raise ValueError("wrong Phase-2 WebShop manifest schema/status")
    validate_self_hash(manifest, "manifest_sha256")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("target selection was not outcome blind")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("historical target outcome reuse is allowed")
    if tuple(manifest.get("conditions") or ()) != CONDITIONS:
        raise ValueError("matched condition set/order changed")
    tasks = list(manifest.get("tasks") or ())
    if len(tasks) != 32:
        raise ValueError("Phase-2 WebShop requires exactly 32 target goals")
    identities = [str(row.get("target_identity")) for row in tasks]
    goal_hashes = [str(row.get("goal_sha256")) for row in tasks]
    asins = [str(row.get("asin")) for row in tasks]
    if any(len(set(rows)) != len(rows) for rows in (identities, goal_hashes, asins)):
        raise ValueError("Phase-2 targets are not semantically/product independent")
    games = [str(row.get("source_game")) for row in tasks]
    counts = Counter(games)
    if set(counts) != set(SOURCE_GAMES) or max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("six source lineages are not balanced across targets")
    for index, row in enumerate(tasks):
        if row.get("selected_target_previously_executed") is not False:
            raise ValueError("target task does not attest freshness")
        if row.get("source_game") != SOURCE_GAMES[index % len(SOURCE_GAMES)]:
            raise ValueError("outcome-blind round-robin source assignment changed")
        source_path = repo / str(row["source_artifact"])
        if file_sha256(source_path) != str(row["source_artifact_file_sha256"]):
            raise ValueError(f"source artifact file changed: {source_path}")
        artifact = json.loads(source_path.read_text(encoding="utf-8"))
        SourceSearchAutomaton(
            artifact, expected_sha256=str(row["source_artifact_sha256"]),
        )
        if str(artifact.get("source_lineage", {}).get("game")) != str(row["source_game"]):
            raise ValueError("source artifact binds a different game lineage")
    for relative, expected in (manifest.get("runtime_file_sha256") or {}).items():
        if file_sha256(repo / str(relative)) != str(expected):
            raise ValueError(f"frozen Phase-2 runtime changed: {relative}")
    bound_files = (
        ("target_grounder", "target_grounder_file_sha256"),
        ("prior_consumed_goal_manifest", "prior_consumed_goal_manifest_file_sha256"),
        (
            "prior_target_adapter_qualification",
            "prior_target_adapter_qualification_file_sha256",
        ),
        ("parent_phase1_manifest", "parent_phase1_manifest_file_sha256"),
    )
    for path_field, hash_field in bound_files:
        path = repo / str(manifest[path_field])
        if file_sha256(path) != str(manifest[hash_field]):
            raise ValueError(f"frozen Phase-2 input changed: {path_field}")
    audit_path = repo / str(manifest["parent_phase1_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_self_hash(audit, "audit_sha256")
    if audit.get("audit_sha256") != manifest.get("parent_phase1_audit_sha256"):
        raise ValueError("parent Phase-1 audit identity changed")
    for absolute, expected in (manifest.get("vendor_runtime_file_sha256") or {}).items():
        if file_sha256(Path(str(absolute))) != str(expected):
            raise ValueError(f"frozen WebShop vendor runtime changed: {absolute}")


def _selected_actions(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get("selected_action")) for row in receipt.get("steps") or ())


def _condition_summary(
    receipts: Sequence[Mapping[str, Any]], condition: str,
) -> dict[str, Any]:
    rows = [row for row in receipts if row.get("condition") == condition]
    rewards = [float(row.get("official_reward", 0.0)) for row in rows]
    steps = [int(row.get("step_count", 0)) for row in rows]
    actions = Counter()
    decisions = 0
    for row in rows:
        controller = row.get("v16_controller") or {}
        decisions += int(controller.get("source_decisions", 0))
        actions.update(controller.get("source_action_counts") or {})
    return {
        "tasks": len(rows),
        "strict_successes": sum(bool(row.get("strict_success")) for row in rows),
        "pass_successes": sum(bool(row.get("pass_success")) for row in rows),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "mean_steps": sum(steps) / len(steps) if steps else 0.0,
        "source_decisions": decisions,
        "source_action_counts": dict(sorted(actions.items())),
        "unsafe_commits": sum(len(row.get("unsafe_commits") or ()) for row in rows),
        "failures": sum(row.get("failure") is not None for row in rows),
    }


def _paired(
    authentic: Mapping[str, Mapping[str, Any]],
    comparator: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    task_ids = tuple(authentic)
    wins = losses = reward_wins = reward_losses = action_contrasts = 0
    for task_id in task_ids:
        left, right = authentic[task_id], comparator[task_id]
        a, b = bool(left.get("strict_success")), bool(right.get("strict_success"))
        wins += int(a and not b)
        losses += int(b and not a)
        ar, br = float(left.get("official_reward", 0.0)), float(
            right.get("official_reward", 0.0)
        )
        reward_wins += int(ar > br + 1e-12)
        reward_losses += int(br > ar + 1e-12)
        action_contrasts += int(_selected_actions(left) != _selected_actions(right))
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(task_ids) - wins - losses,
        "net_wins": wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        "reward_wins": reward_wins,
        "reward_losses": reward_losses,
        "reward_ties": len(task_ids) - reward_wins - reward_losses,
        "reward_net_wins": reward_wins - reward_losses,
        "reward_exact_two_sided_p": exact_binomial_two_sided(
            reward_wins, reward_losses,
        ),
        "action_contrast_tasks": action_contrasts,
    }


def build_report(
    manifest: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    cache_usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = {
        (str(task["target_identity"]), condition)
        for task in manifest["tasks"] for condition in CONDITIONS
    }
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    receipt_hashes_valid = True
    for row in receipts:
        key = (str(row.get("target_identity")), str(row.get("condition")))
        if key in indexed:
            raise ValueError(f"duplicate Phase-2 receipt: {key}")
        try:
            validate_self_hash(row, "receipt_sha256")
        except ValueError:
            receipt_hashes_valid = False
        indexed[key] = row
    observed = set(indexed)
    rows = [indexed[key] for key in sorted(expected)] if observed == expected else list(receipts)
    by_condition = {
        condition: {
            str(row["target_identity"]): row
            for row in rows if row.get("condition") == condition
        }
        for condition in CONDITIONS
    }
    summaries = {
        condition: _condition_summary(rows, condition) for condition in CONDITIONS
    }
    paired = {
        comparator: _paired(by_condition[AUTHENTIC], by_condition[comparator])
        for comparator in CONDITIONS if comparator != AUTHENTIC
    } if observed == expected else {}

    lineage: dict[str, Any] = {}
    for game in SOURCE_GAMES:
        assigned = [
            task for task in manifest["tasks"] if task["source_game"] == game
        ]
        task_ids = [str(task["target_identity"]) for task in assigned]
        auth_rows = [by_condition[AUTHENTIC][task_id] for task_id in task_ids]
        raw_rows = [by_condition[RAW][task_id] for task_id in task_ids]
        action_counts = Counter()
        for row in auth_rows:
            action_counts.update(
                (row.get("v16_controller") or {}).get("source_action_counts") or {}
            )
        lineage[game] = {
            "tasks": len(task_ids),
            "authentic_strict_successes": sum(
                bool(row.get("strict_success")) for row in auth_rows
            ),
            "raw_strict_successes": sum(bool(row.get("strict_success")) for row in raw_rows),
            "strict_wins": sum(
                bool(a.get("strict_success")) and not bool(b.get("strict_success"))
                for a, b in zip(auth_rows, raw_rows)
            ),
            "strict_losses": sum(
                bool(b.get("strict_success")) and not bool(a.get("strict_success"))
                for a, b in zip(auth_rows, raw_rows)
            ),
            "source_decisions": sum(
                int((row.get("v16_controller") or {}).get("source_decisions", 0))
                for row in auth_rows
            ),
            "source_action_counts": dict(sorted(action_counts.items())),
        }

    ceiling_exact = observed == expected and all(
        (
            bool(by_condition[AUTHENTIC][task_id].get("strict_success"))
            == bool(by_condition[CEILING][task_id].get("strict_success"))
            and abs(
                float(by_condition[AUTHENTIC][task_id].get("official_reward", 0.0))
                - float(by_condition[CEILING][task_id].get("official_reward", 0.0))
            ) <= 1e-12
            and _selected_actions(by_condition[AUTHENTIC][task_id])
            == _selected_actions(by_condition[CEILING][task_id])
        )
        for task_id in by_condition[AUTHENTIC]
    )
    gates = {
        "exact_32x5_receipt_matrix": observed == expected,
        "all_receipt_hashes_valid": receipt_hashes_valid and len(rows) == 160,
        "all_receipts_complete": len(rows) == 160 and all(
            row.get("failure") is None for row in rows
        ),
        "matched_initial_state_hashes": observed == expected and all(
            len({
                str(by_condition[condition][str(task["target_identity"])].get(
                    "initial_state_hash"
                )) for condition in CONDITIONS
            }) == 1
            for task in manifest["tasks"]
        ),
        "all_six_source_lineages_exercised": all(
            row["source_decisions"] > 0 for row in lineage.values()
        ),
        "all_three_symbolic_actions_exercised": set(
            summaries[AUTHENTIC]["source_action_counts"]
        ) == {"BACKTRACK_REPLAN", "COMMIT_VERIFY", "EXPLORE_UNTRIED"},
        "zero_authentic_unsafe_commits": summaries[AUTHENTIC]["unsafe_commits"] == 0,
        "authentic_strict_success_gain_over_raw": (
            summaries[AUTHENTIC]["strict_successes"] > summaries[RAW]["strict_successes"]
        ),
        "authentic_vs_raw_significant": bool(
            paired and paired[RAW]["wins"] > paired[RAW]["losses"]
            and paired[RAW]["exact_two_sided_p"] <= 0.05
        ),
        "zero_strict_negative_transfer_vs_raw": bool(
            paired and paired[RAW]["losses"] == 0
        ),
        "authentic_pass_success_not_below_raw": (
            summaries[AUTHENTIC]["pass_successes"] >= summaries[RAW]["pass_successes"]
        ),
        "authentic_mean_reward_not_below_raw": (
            summaries[AUTHENTIC]["mean_reward"] + 1e-12 >= summaries[RAW]["mean_reward"]
        ),
        "authentic_reward_pairing_net_nonnegative": bool(
            paired and paired[RAW]["reward_net_wins"] >= 0
        ),
        "authentic_significantly_beats_event_permuted": bool(
            paired and paired[PERMUTED]["wins"] > paired[PERMUTED]["losses"]
            and paired[PERMUTED]["exact_two_sided_p"] <= 0.05
        ),
        "authentic_significantly_beats_ledger_blind": bool(
            paired and paired[LEDGER_BLIND]["wins"] > paired[LEDGER_BLIND]["losses"]
            and paired[LEDGER_BLIND]["exact_two_sided_p"] <= 0.05
        ),
        "authentic_matches_target_native_ceiling_exactly": ceiling_exact,
        "every_lineage_has_zero_strict_losses_vs_raw": all(
            row["strict_losses"] == 0 for row in lineage.values()
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": REPORT_SCHEMA,
        "status": PASSED_STATUS if passed else FAILED_STATUS,
        "claim_boundary": (
            "Causal utility of the common, independently game-qualified search "
            "structure on 32 fresh WebShop goals. This is an aggregate shared-policy "
            "effect, not six powered per-game effect estimates and not an advantage "
            "over an isomorphic target-written ceiling."
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": len(manifest["tasks"]),
        "receipts": len(rows),
        "summaries": summaries,
        "paired": paired,
        "source_lineages": lineage,
        "cache_usage": dict(cache_usage or {}),
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "FAILED_STATUS",
    "PASSED_STATUS",
    "REPORT_SCHEMA",
    "SCHEMA",
    "STATUS",
    "build_report",
    "file_sha256",
    "validate_manifest",
    "validate_self_hash",
]
