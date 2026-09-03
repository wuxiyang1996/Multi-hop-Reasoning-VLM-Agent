"""Selective, arity-typed aggregation for fresh ALFWorld Phase-2 V3."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .direct_prospective_matrix_v1 import SOURCE_GAMES
from .phase2_webshop_utility_v1 import (
    _condition_summary, _paired, file_sha256, validate_self_hash,
)
from .search_automaton_transfer_v16 import SourceSearchAutomaton
from .webshop_search_automaton_v16 import AUTHENTIC, CEILING, CONDITIONS, LEDGER_BLIND, PERMUTED, RAW


SCHEMA = "phase2-alfworld-selective-utility-v3"
STATUS = "FROZEN_AFTER_V2_DEVELOPMENT_BEFORE_ANY_V3_TARGET_RESET_ACTION_OR_OUTCOME"
REPORT_SCHEMA = "phase2-alfworld-selective-utility-report-v3"
PASSED_STATUS = "PHASE2_ALFWORLD_SELECTIVE_CAUSAL_UTILITY_VALIDATED"
FAILED_STATUS = "PHASE2_ALFWORLD_SELECTIVE_CAUSAL_UTILITY_NOT_VALIDATED"
UNSUPPORTED_FAMILY = "pick_two_obj_and_place"


def validate_manifest(manifest: Mapping[str, Any], *, repo: Path) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != STATUS:
        raise ValueError("wrong ALFWorld V3 schema/status")
    validate_self_hash(manifest, "manifest_sha256")
    if str(Path(sys.executable).resolve()) != str(manifest["python_executable"]):
        raise ValueError("wrong frozen interpreter")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("selection was not outcome blind")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("historical outcome reuse is allowed")
    if manifest.get("target_split") != "eval_in_distribution":
        raise ValueError("wrong ALFWorld split")
    if manifest.get("environment_concurrency_policy") != "one_task_per_environment":
        raise ValueError("one-task environment policy required")
    selector = manifest.get("transfer_applicability") or {}
    if selector.get("abstain_family") != UNSUPPORTED_FAMILY or selector.get("criterion") != "target_task_arity_equals_one":
        raise ValueError("arity applicability rule changed")
    if tuple(manifest.get("conditions") or ()) != CONDITIONS:
        raise ValueError("condition matrix changed")
    tasks = list(manifest.get("tasks") or ())
    required = int(manifest["formal_task_count"])
    if len(tasks) != required or len({row["target_identity"] for row in tasks}) != required:
        raise ValueError("fresh V3 task matrix changed")
    consumed = set(manifest["excluded_prior_task_ids"])
    if consumed.intersection(row["target_identity"] for row in tasks):
        raise ValueError("V3 reused a prior task")
    if len(consumed) + len(tasks) != int(manifest["dataset_task_count"]):
        raise ValueError("V3 is not the complete remaining reserve")
    games = Counter(row["source_game"] for row in tasks)
    if set(games) != set(SOURCE_GAMES) or max(games.values()) - min(games.values()) > 1:
        raise ValueError("source assignment is not balanced")
    split_root = Path(manifest["alfworld_data_root"]) / "json_2.1.1" / "valid_seen"
    for index, row in enumerate(tasks):
        if row.get("selected_target_previously_executed") is not False:
            raise ValueError("V3 target does not attest freshness")
        if bool(row.get("transfer_applicable")) != (row["task_family"] != UNSUPPORTED_FAMILY):
            raise ValueError("task arity/applicability binding changed")
        if row["source_game"] != SOURCE_GAMES[index % len(SOURCE_GAMES)]:
            raise ValueError("source round-robin changed")
        if file_sha256(split_root / row["target_identity"]) != row["target_file_sha256"]:
            raise ValueError("target file changed")
        source_path = repo / row["source_artifact"]
        if file_sha256(source_path) != row["source_artifact_file_sha256"]:
            raise ValueError("source file changed")
        artifact = json.loads(source_path.read_text())
        SourceSearchAutomaton(artifact, expected_sha256=row["source_artifact_sha256"])
        if artifact.get("source_lineage", {}).get("game") != row["source_game"]:
            raise ValueError("source lineage changed")
    for relative, expected in manifest["runtime_file_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise ValueError(f"runtime changed: {relative}")
    for field, hash_field, relative in (
        ("target_grounder", "target_grounder_file_sha256", True),
        ("v2_manifest", "v2_manifest_file_sha256", True),
        ("v2_report", "v2_report_file_sha256", True),
        ("alfworld_config", "alfworld_config_file_sha256", False),
    ):
        path = repo / manifest[field] if relative else Path(manifest[field])
        if file_sha256(path) != manifest[hash_field]:
            raise ValueError(f"frozen input changed: {field}")


def build_report(manifest: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(row["target_identity"], condition) for row in manifest["tasks"] for condition in CONDITIONS}
    indexed = {}
    hashes_valid = True
    for row in receipts:
        key = (row.get("target_identity"), row.get("condition"))
        if key in indexed:
            raise ValueError(f"duplicate receipt: {key}")
        try:
            validate_self_hash(row, "receipt_sha256")
        except ValueError:
            hashes_valid = False
        indexed[key] = row
    complete = set(indexed) == expected
    rows = [indexed[key] for key in sorted(expected)] if complete else list(receipts)
    by = {condition: {row["target_identity"]: row for row in rows if row.get("condition") == condition} for condition in CONDITIONS}
    summaries = {condition: _condition_summary(rows, condition) for condition in CONDITIONS}
    paired = {condition: _paired(by[AUTHENTIC], by[condition]) for condition in CONDITIONS if condition != AUTHENTIC} if complete else {}
    lineages = {}
    for game in SOURCE_GAMES:
        ids = [row["target_identity"] for row in manifest["tasks"] if row["source_game"] == game]
        lineages[game] = {
            "tasks": len(ids),
            "authentic_successes": sum(by[AUTHENTIC][task_id]["strict_success"] for task_id in ids),
            "raw_successes": sum(by[RAW][task_id]["strict_success"] for task_id in ids),
            "source_decisions": sum(by[AUTHENTIC][task_id]["v16_controller"]["source_decisions"] for task_id in ids),
        }
    discordant = paired.get(RAW, {}).get("wins", 0) + paired.get(RAW, {}).get("losses", 0)
    negative_rate = paired.get(RAW, {}).get("losses", 0) / discordant if discordant else 0.0
    gates = {
        "exact_fresh_receipt_matrix": complete,
        "all_receipt_hashes_valid": hashes_valid and len(rows) == len(expected),
        "all_receipts_complete": len(rows) == len(expected) and all(row.get("failure") is None for row in rows),
        "matched_initial_states": complete and all(len({by[c][task["target_identity"]]["initial_state_hash"] for c in CONDITIONS}) == 1 for task in manifest["tasks"]),
        "all_six_lineages_exercised": all(row["source_decisions"] > 0 for row in lineages.values()),
        "all_three_symbolic_actions_exercised": set(summaries[AUTHENTIC]["source_action_counts"]) == {"BACKTRACK_REPLAN", "COMMIT_VERIFY", "EXPLORE_UNTRIED"},
        "authentic_success_gain": summaries[AUTHENTIC]["strict_successes"] > summaries[RAW]["strict_successes"],
        "authentic_vs_raw_significant": bool(paired and paired[RAW]["wins"] > paired[RAW]["losses"] and paired[RAW]["exact_two_sided_p"] <= 0.05),
        "bounded_negative_transfer": negative_rate <= float(manifest["gates"]["maximum_discordant_loss_rate"]),
        "beats_event_permuted_significantly": bool(paired and paired[PERMUTED]["wins"] > paired[PERMUTED]["losses"] and paired[PERMUTED]["exact_two_sided_p"] <= 0.05),
        "beats_ledger_blind_significantly": bool(paired and paired[LEDGER_BLIND]["wins"] > paired[LEDGER_BLIND]["losses"] and paired[LEDGER_BLIND]["exact_two_sided_p"] <= 0.05),
        "matches_target_native_ceiling_outcomes": bool(paired and paired[CEILING]["wins"] == 0 and paired[CEILING]["losses"] == 0),
        "zero_unsafe_commits": summaries[AUTHENTIC]["unsafe_commits"] == 0,
    }
    body = {
        "schema_version": REPORT_SCHEMA,
        "status": PASSED_STATUS if all(gates.values()) else FAILED_STATUS,
        "claim_boundary": "Selective aggregate utility on every remaining fresh valid_seen task; arity applicability learned only from consumed V2 development.",
        "manifest_sha256": manifest["manifest_sha256"], "tasks": len(manifest["tasks"]),
        "receipts": len(rows), "summaries": summaries, "paired": paired,
        "discordant_negative_transfer_rate": negative_rate, "source_lineages": lineages,
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = ["FAILED_STATUS", "PASSED_STATUS", "SCHEMA", "STATUS", "UNSUPPORTED_FAMILY", "build_report", "validate_manifest"]
