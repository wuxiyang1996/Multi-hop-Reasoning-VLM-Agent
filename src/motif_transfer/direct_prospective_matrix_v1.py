"""Fail-closed contracts for the six-source/four-target direct matrix.

The unit of evidence is a *cell execution*, not a copied domain-level score.
Every cell binds exactly one independently qualified source lineage to one
previously unexecuted target task and records the source artifact hash in the
online routing receipts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .search_automaton_transfer_v16 import SourceSearchAutomaton


SCHEMA = "phase1-direct-prospective-matrix-v1"
STATUS = "FROZEN_BEFORE_ANY_SELECTED_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"
SOURCE_GAMES = (
    "tetris",
    "candy_crush",
    "gymv_columns",
    "gymv_streets_of_rage_2",
    "gymv_thunder_force_iii",
    "gymv_strider",
)
TARGET_DOMAINS = ("webshop", "alfworld", "discoveryworld", "tirbench")


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
        raise ValueError("wrong direct-matrix schema")
    if manifest.get("status") != STATUS:
        raise ValueError("direct-matrix manifest was not prospectively frozen")
    validate_self_hash(manifest, "manifest_sha256")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("target outcomes were visible during reserve selection")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("historical target outcome reuse is not fail-closed")

    sources = manifest.get("sources") or {}
    if tuple(sources) != SOURCE_GAMES:
        raise ValueError("source lineage order or coverage changed")
    source_hashes = set()
    for game, receipt in sources.items():
        path = repo / str(receipt["artifact"])
        if file_sha256(path) != str(receipt["artifact_file_sha256"]):
            raise ValueError(f"source artifact file changed: {game}")
        artifact = read_object(path)
        source = SourceSearchAutomaton(
            artifact, expected_sha256=str(receipt["artifact_sha256"])
        )
        if artifact.get("source_lineage", {}).get("game") != game:
            raise ValueError(f"source artifact lineage mismatch: {game}")
        source_hashes.add(source.artifact_sha256)
    if len(source_hashes) != len(SOURCE_GAMES):
        raise ValueError("source artifacts are not lineage-specific identities")

    cells = list(manifest.get("cells") or ())
    expected = {
        f"{game}__to__{domain}"
        for game in SOURCE_GAMES
        for domain in TARGET_DOMAINS
    }
    if {str(row.get("cell_id")) for row in cells} != expected:
        raise ValueError("direct matrix is not exactly the 6x4 Cartesian product")
    if len(cells) != 24:
        raise ValueError("direct matrix must contain exactly 24 cells")
    target_ids: dict[str, list[str]] = {domain: [] for domain in TARGET_DOMAINS}
    for cell in cells:
        game = str(cell["source_game"])
        domain = str(cell["target_domain"])
        if str(cell["source_artifact_sha256"]) != str(
            sources[game]["artifact_sha256"]
        ):
            raise ValueError(f"cell/source hash mismatch: {cell['cell_id']}")
        target_ids[domain].append(str(cell["target_task_id"]))
        if int(cell.get("target_task_multiplicity", 0)) != 1:
            raise ValueError("a cell must own exactly one target task")
        if cell.get("selected_target_previously_executed") is not False:
            raise ValueError("cell does not attest fresh target identity")
    if any(len(ids) != len(set(ids)) for ids in target_ids.values()):
        raise ValueError("a target task was assigned to multiple source cells")

    for relative, expected_hash in (manifest.get("runtime_file_sha256") or {}).items():
        path = repo / str(relative)
        if file_sha256(path) != str(expected_hash):
            raise ValueError(f"frozen runtime changed: {relative}")


def make_cell_execution_receipt(
    *,
    manifest_sha256: str,
    cell: Mapping[str, Any],
    source_artifact_sha256: str,
    conditions_executed: Sequence[str],
    expected_conditions: Sequence[str],
    target_initial_state_hashes: Sequence[str],
    authentic_source_decisions: Sequence[Mapping[str, Any]],
    target_native_grounding_used: bool,
    target_reset_or_sample_open_count: int,
    outcome_was_reused: bool = False,
    runtime_error: str | None = None,
) -> dict[str, Any]:
    """Create a standardized direct-execution receipt for one matrix cell."""

    source_hashes = {
        str(row.get("source_artifact_sha256"))
        for row in authentic_source_decisions
        if row.get("source_artifact_sha256")
    }
    admitted = [row for row in authentic_source_decisions if row.get("admitted")]
    action_set = sorted({str(row.get("source_action")) for row in admitted})
    gates = {
        "runtime_complete": runtime_error is None,
        "one_fresh_target_owned_by_cell": target_reset_or_sample_open_count == 1,
        "historical_outcome_not_reused": outcome_was_reused is False,
        "complete_matched_conditions": (
            tuple(conditions_executed) == tuple(expected_conditions)
        ),
        "matched_initial_target_state": (
            bool(target_initial_state_hashes)
            and len(set(map(str, target_initial_state_hashes))) == 1
        ),
        "target_native_grounding_used": bool(target_native_grounding_used),
        "source_routed_online": bool(authentic_source_decisions),
        "source_route_admitted": bool(admitted),
        "source_hash_in_every_routed_decision": (
            bool(authentic_source_decisions)
            and source_hashes == {source_artifact_sha256}
        ),
        "nontrivial_symbolic_action_exercised": bool(action_set),
    }
    body = {
        "schema_version": "phase1-direct-cell-execution-v1",
        "status": (
            "DIRECT_PROSPECTIVE_CELL_PASSED"
            if all(gates.values()) else "DIRECT_PROSPECTIVE_CELL_FAILED"
        ),
        "manifest_sha256": manifest_sha256,
        "cell_id": str(cell["cell_id"]),
        "source_game": str(cell["source_game"]),
        "target_domain": str(cell["target_domain"]),
        "target_task_id": str(cell["target_task_id"]),
        "source_artifact_sha256": source_artifact_sha256,
        "conditions_executed": list(conditions_executed),
        "expected_conditions": list(expected_conditions),
        "target_initial_state_hashes": list(map(str, target_initial_state_hashes)),
        "target_reset_or_sample_open_count": int(target_reset_or_sample_open_count),
        "outcome_was_reused": bool(outcome_was_reused),
        "authentic_source_decision_count": len(authentic_source_decisions),
        "authentic_admitted_source_decision_count": len(admitted),
        "authentic_source_actions": action_set,
        "authentic_source_decision_receipt_sha256": [
            str(row.get("receipt_sha256")) for row in authentic_source_decisions
        ],
        "runtime_error": runtime_error,
        "gates": gates,
    }
    return body | {"cell_receipt_sha256": stable_hash(body)}


def audit_cell_receipts(
    manifest: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Audit exact 24-cell coverage without allowing copied executions."""

    validate_self_hash(manifest, "manifest_sha256")
    by_expected = {str(row["cell_id"]): row for row in manifest["cells"]}
    by_observed = {str(row.get("cell_id")): row for row in receipts}
    if len(by_observed) != len(receipts):
        raise ValueError("duplicate cell receipt")
    per_cell = []
    execution_fingerprints = set()
    for cell_id in sorted(by_expected):
        receipt = by_observed.get(cell_id)
        if receipt is None:
            per_cell.append({"cell_id": cell_id, "passed": False, "reason": "MISSING"})
            continue
        validate_self_hash(receipt, "cell_receipt_sha256")
        expected = by_expected[cell_id]
        identity_ok = all(
            str(receipt.get(key)) == str(expected.get(key))
            for key in ("cell_id", "source_game", "target_domain", "target_task_id")
        ) and str(receipt.get("source_artifact_sha256")) == str(
            expected["source_artifact_sha256"]
        )
        gates_ok = bool(receipt.get("gates")) and all(receipt["gates"].values())
        fingerprint = stable_hash({
            "source": receipt.get("source_artifact_sha256"),
            "target": receipt.get("target_task_id"),
            "routes": receipt.get("authentic_source_decision_receipt_sha256"),
        })
        unique = fingerprint not in execution_fingerprints
        execution_fingerprints.add(fingerprint)
        per_cell.append({
            "cell_id": cell_id,
            "passed": identity_ok and gates_ok and unique,
            "identity_ok": identity_ok,
            "all_cell_gates_pass": gates_ok,
            "unique_execution_fingerprint": unique,
            "cell_receipt_sha256": receipt["cell_receipt_sha256"],
        })
    passed_count = sum(bool(row["passed"]) for row in per_cell)
    body = {
        "schema_version": "phase1-direct-prospective-matrix-audit-v1",
        "status": (
            "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED"
            if passed_count == 24 else "DIRECT_PROSPECTIVE_MATRIX_INCOMPLETE"
        ),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "passed_cells": passed_count,
        "required_cells": 24,
        "direct_new_joint_execution_cells": passed_count,
        "cells": per_cell,
    }
    return body | {"audit_sha256": stable_hash(body)}


__all__ = [
    "SCHEMA",
    "SOURCE_GAMES",
    "STATUS",
    "TARGET_DOMAINS",
    "audit_cell_receipts",
    "file_sha256",
    "make_cell_execution_receipt",
    "read_object",
    "validate_manifest",
    "validate_self_hash",
]
