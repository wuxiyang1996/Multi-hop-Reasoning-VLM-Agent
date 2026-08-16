"""Contracts for the single-task-environment Phase-2 ALFWorld V2 run."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .direct_prospective_matrix_v1 import SOURCE_GAMES
from .phase2_alfworld_utility_v1 import build_report as build_v1_report
from .phase2_webshop_utility_v1 import file_sha256, validate_self_hash
from .search_automaton_transfer_v16 import SourceSearchAutomaton
from .webshop_search_automaton_v16 import CONDITIONS


SCHEMA = "phase2-alfworld-six-source-utility-v2"
STATUS = "FROZEN_AFTER_V1_RESET_ONLY_ORDER_FAILURE_BEFORE_ANY_V2_TARGET_RESET_ACTION_OR_OUTCOME"


def validate_manifest(manifest: Mapping[str, Any], *, repo: Path) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != STATUS:
        raise ValueError("wrong Phase-2 ALFWorld V2 manifest schema/status")
    validate_self_hash(manifest, "manifest_sha256")
    if str(Path(sys.executable).resolve()) != str(manifest.get("python_executable")):
        raise ValueError("Phase-2 ALFWorld V2 requires the frozen interpreter")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("target selection was not outcome blind")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("historical target outcome reuse is allowed")
    if manifest.get("target_split") != "eval_in_distribution":
        raise ValueError("wrong ALFWorld split")
    if manifest.get("environment_concurrency_policy") != "one_task_per_environment":
        raise ValueError("V2 must remove batch-order ambiguity")
    if tuple(manifest.get("conditions") or ()) != CONDITIONS:
        raise ValueError("matched condition set/order changed")
    tasks = list(manifest.get("tasks") or ())
    if len(tasks) != 32 or len({str(row.get("target_identity")) for row in tasks}) != 32:
        raise ValueError("V2 requires 32 unique targets")
    identities = {str(row["target_identity"]) for row in tasks}
    excluded = set(map(str, manifest.get("excluded_v1_reset_task_ids") or ()))
    historical = set(map(str, manifest.get("historical_outcome_task_ids") or ()))
    if identities.intersection(excluded | historical):
        raise ValueError("V2 reuses a consumed or historical target")
    families = Counter(str(row["task_family"]) for row in tasks)
    expected_families = Counter({str(k): int(v) for k, v in manifest["family_quotas"].items()})
    if families != expected_families:
        raise ValueError("family quotas changed")
    games = Counter(str(row["source_game"]) for row in tasks)
    if set(games) != set(SOURCE_GAMES) or max(games.values()) - min(games.values()) > 1:
        raise ValueError("source lineages are not balanced")
    split_root = Path(str(manifest["alfworld_data_root"])) / "json_2.1.1" / "valid_seen"
    for index, row in enumerate(tasks):
        if row.get("selected_target_previously_executed") is not False:
            raise ValueError("V2 target does not attest freshness")
        if row["source_game"] != SOURCE_GAMES[index % len(SOURCE_GAMES)]:
            raise ValueError("round-robin source assignment changed")
        if file_sha256(split_root / str(row["target_identity"])) != row["target_file_sha256"]:
            raise ValueError("target file changed")
        source_path = repo / str(row["source_artifact"])
        if file_sha256(source_path) != row["source_artifact_file_sha256"]:
            raise ValueError("source artifact file changed")
        artifact = json.loads(source_path.read_text(encoding="utf-8"))
        SourceSearchAutomaton(artifact, expected_sha256=str(row["source_artifact_sha256"]))
        if artifact.get("source_lineage", {}).get("game") != row["source_game"]:
            raise ValueError("source artifact lineage changed")
    for relative, expected in manifest["runtime_file_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise ValueError(f"frozen V2 runtime changed: {relative}")
    for path_field, hash_field, relative in (
        ("target_grounder", "target_grounder_file_sha256", True),
        ("parent_phase1_manifest", "parent_phase1_manifest_file_sha256", True),
        ("v1_manifest", "v1_manifest_file_sha256", True),
        ("v1_failed_preflight", "v1_failed_preflight_file_sha256", True),
        ("alfworld_config", "alfworld_config_file_sha256", False),
    ):
        path = repo / str(manifest[path_field]) if relative else Path(str(manifest[path_field]))
        if file_sha256(path) != str(manifest[hash_field]):
            raise ValueError(f"frozen V2 input changed: {path_field}")


def build_report(manifest: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return build_v1_report(manifest, receipts)


__all__ = ["SCHEMA", "STATUS", "build_report", "validate_manifest"]
