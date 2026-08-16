"""Frozen contracts for the fresh Phase-2 ALFWorld utility experiment."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .direct_prospective_matrix_v1 import SOURCE_GAMES
from .phase2_webshop_utility_v1 import (
    build_report as build_shared_utility_report,
    file_sha256,
    validate_self_hash,
)
from .search_automaton_transfer_v16 import SourceSearchAutomaton
from .webshop_search_automaton_v16 import CONDITIONS


SCHEMA = "phase2-alfworld-six-source-utility-v1"
STATUS = "FROZEN_BEFORE_ANY_PHASE2_ALFWORLD_TARGET_RESET_ACTION_OR_OUTCOME"
REPORT_SCHEMA = "phase2-alfworld-six-source-utility-report-v1"
PASSED_STATUS = "PHASE2_ALFWORLD_CAUSAL_UTILITY_VALIDATED"
FAILED_STATUS = "PHASE2_ALFWORLD_CAUSAL_UTILITY_NOT_VALIDATED"


def validate_manifest(manifest: Mapping[str, Any], *, repo: Path) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != STATUS:
        raise ValueError("wrong Phase-2 ALFWorld manifest schema/status")
    validate_self_hash(manifest, "manifest_sha256")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("target selection was not outcome blind")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("historical target outcome reuse is allowed")
    if manifest.get("target_split") != "eval_in_distribution":
        raise ValueError("fresh ALFWorld experiment must use frozen valid_seen split")
    if tuple(manifest.get("conditions") or ()) != CONDITIONS:
        raise ValueError("matched condition set/order changed")
    tasks = list(manifest.get("tasks") or ())
    if len(tasks) != 32:
        raise ValueError("Phase-2 ALFWorld requires exactly 32 tasks")
    identities = [str(row.get("target_identity")) for row in tasks]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate ALFWorld task identity")
    historical = set(map(str, manifest.get("historical_outcome_task_ids") or ()))
    if historical.intersection(identities):
        raise ValueError("selected task has a historical target outcome")
    families = Counter(str(row.get("task_family")) for row in tasks)
    if set(families) != set(manifest.get("family_quotas") or {}):
        raise ValueError("ALFWorld family coverage changed")
    if families != Counter({
        str(key): int(value)
        for key, value in (manifest.get("family_quotas") or {}).items()
    }):
        raise ValueError("ALFWorld family quotas changed")
    games = [str(row.get("source_game")) for row in tasks]
    counts = Counter(games)
    if set(counts) != set(SOURCE_GAMES) or max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("six source lineages are not balanced")
    data_root = Path(str(manifest["alfworld_data_root"])) / "json_2.1.1" / "valid_seen"
    for index, row in enumerate(tasks):
        if row.get("selected_target_previously_executed") is not False:
            raise ValueError("selected ALFWorld target does not attest freshness")
        if row.get("source_game") != SOURCE_GAMES[index % len(SOURCE_GAMES)]:
            raise ValueError("outcome-blind round-robin assignment changed")
        target_path = data_root / str(row["target_identity"])
        if file_sha256(target_path) != str(row["target_file_sha256"]):
            raise ValueError(f"ALFWorld task file changed: {target_path}")
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
            raise ValueError(f"frozen Phase-2 ALFWorld runtime changed: {relative}")
    grounder_path = repo / str(manifest["target_grounder"])
    if file_sha256(grounder_path) != str(manifest["target_grounder_file_sha256"]):
        raise ValueError("target-native neural grounder changed")
    grounder = json.loads(grounder_path.read_text(encoding="utf-8"))
    if grounder.get("target_grounder_gate", {}).get("passed") is not True:
        raise ValueError("target-native neural grounder gate did not pass")
    config_path = Path(str(manifest["alfworld_config"]))
    if file_sha256(config_path) != str(manifest["alfworld_config_file_sha256"]):
        raise ValueError("ALFWorld config changed")
    parent_path = repo / str(manifest["parent_phase1_manifest"])
    if file_sha256(parent_path) != str(manifest["parent_phase1_manifest_file_sha256"]):
        raise ValueError("parent Phase-1 manifest changed")


def build_report(
    manifest: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reuse the already frozen 32x5 causal utility aggregation contract."""

    shared = build_shared_utility_report(manifest, receipts)
    body = dict(shared)
    body.pop("report_sha256", None)
    passed = all((body.get("gates") or {}).values())
    body.update({
        "schema_version": REPORT_SCHEMA,
        "status": PASSED_STATUS if passed else FAILED_STATUS,
        "claim_boundary": (
            "Causal utility of the common, independently game-qualified search "
            "structure on 32 fresh ALFWorld valid_seen tasks. This is an aggregate "
            "shared-policy effect, not six powered per-game estimates and not an "
            "advantage over an isomorphic target-written ceiling. Target-native "
            "neural grounding remains required."
        ),
        "target_split": str(manifest["target_split"]),
    })
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "FAILED_STATUS", "PASSED_STATUS", "REPORT_SCHEMA", "SCHEMA", "STATUS",
    "build_report", "validate_manifest",
]
