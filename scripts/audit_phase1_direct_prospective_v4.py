#!/usr/bin/env python3
"""Audit the 18-cell V1 core plus six fresh DiscoveryWorld V4 cells."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    TARGET_DOMAINS,
    file_sha256,
    read_object,
    validate_manifest,
    validate_self_hash,
)
import scripts.run_phase1_direct_discoveryworld_v4 as discovery_v4  # noqa: E402


V1_MANIFEST = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
V4_MANIFEST = REPO / "configs/phase1_direct_prospective_v4/discoveryworld_manifest.json"
V1_RUN = REPO / "runs/phase1_direct_prospective_v1"
V4_RUN = REPO / "runs/phase1_direct_prospective_v4"
OUTPUT = REPO / "docs/results/phase1_direct_prospective_24_of_24_v4.json"
V4_PREPARATION = V4_RUN / "discoveryworld/preparation_receipt.json"
V4_FORK_RECEIPT = V4_RUN / "discoveryworld/frozen_forks/fork_freeze_receipt.json"


def _report_path(game: str, domain: str) -> Path:
    if domain == "webshop":
        return V1_RUN / domain / game / "direct_report.json"
    if domain in {"alfworld", "tirbench"}:
        return V1_RUN / domain / game / "report.json"
    if domain == "discoveryworld":
        return V4_RUN / domain / "cells" / game / "direct_report.json"
    raise ValueError(domain)


def _validate_report(report: Mapping[str, Any]) -> None:
    validate_self_hash(report, "report_sha256")
    receipt = report.get("cell_execution_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("missing cell execution receipt")
    validate_self_hash(receipt, "cell_receipt_sha256")


def _validate_v4_preparation(v4: Mapping[str, Any]) -> dict[str, Any]:
    preparation = read_object(V4_PREPARATION)
    validate_self_hash(preparation, "preparation_receipt_sha256")
    if preparation.get("status") != "SIX_TARGETS_COLLECTED_ONCE_AND_SIX_FORKS_FROZEN":
        raise ValueError("DiscoveryWorld V4 preparation is incomplete")
    if preparation.get("manifest_sha256") != v4["manifest_sha256"]:
        raise ValueError("DiscoveryWorld V4 preparation binds a different manifest")
    tasks = list(preparation.get("tasks") or ())
    expected_tasks = [str(row["target_task_id"]) for row in v4["cells"]]
    if [str(row.get("task_id")) for row in tasks] != expected_tasks:
        raise ValueError("DiscoveryWorld V4 preparation task order/identity changed")
    if any(int(row.get("target_process_count", 0)) != 1 for row in tasks):
        raise ValueError("DiscoveryWorld V4 target was collected more than once")

    fork_receipt = read_object(V4_FORK_RECEIPT)
    validate_self_hash(fork_receipt, "summary_sha256")
    if file_sha256(V4_FORK_RECEIPT) != preparation["fork_freeze_receipt_file_sha256"]:
        raise ValueError("DiscoveryWorld V4 frozen-fork receipt changed")
    receipts = list(fork_receipt.get("receipts") or ())
    if fork_receipt.get("status") != "FORMAL_RESERVE_FORKS_FROZEN":
        raise ValueError("DiscoveryWorld V4 forks are not formal reserve")
    if fork_receipt.get("outcome_fields_read_for_eligibility") is not False:
        raise ValueError("DiscoveryWorld V4 fork selection read outcome fields")
    if len(receipts) != len(expected_tasks) or not all(
        row.get("eligible") is True for row in receipts
    ):
        raise ValueError("DiscoveryWorld V4 does not have six eligible frozen forks")
    if [str(row.get("task_id")) for row in receipts] != expected_tasks:
        raise ValueError("DiscoveryWorld V4 fork task identity changed")
    if len(fork_receipt.get("generated_configs") or ()) != len(expected_tasks):
        raise ValueError("DiscoveryWorld V4 did not generate six fork configs")
    return {
        "status": preparation["status"],
        "preparation_receipt_sha256": preparation["preparation_receipt_sha256"],
        "target_process_count_per_task": [
            int(row["target_process_count"]) for row in tasks
        ],
        "fork_status": fork_receipt["status"],
        "fork_summary_sha256": fork_receipt["summary_sha256"],
        "eligible_forks": len(receipts),
        "outcome_fields_read_for_eligibility": False,
    }


def main() -> int:
    v1 = read_object(V1_MANIFEST)
    v4 = read_object(V4_MANIFEST)
    validate_manifest(v1, repo=REPO)
    discovery_v4.runner.validate_v2_manifest(v4)
    if v4["parent_v1_manifest_sha256"] != v1["manifest_sha256"]:
        raise ValueError("V4 does not bind the audited V1 manifest")
    if v4["parent_v1_manifest_file_sha256"] != file_sha256(V1_MANIFEST):
        raise ValueError("V4 parent V1 manifest file changed")
    v4_preparation = _validate_v4_preparation(v4)

    v1_cells = {str(row["cell_id"]): row for row in v1["cells"]}
    v4_cells = {str(row["cell_id"]): row for row in v4["cells"]}
    rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for domain in TARGET_DOMAINS:
        for game in SOURCE_GAMES:
            cell_id = f"{game}__to__{domain}"
            expected = v4_cells[cell_id] if domain == "discoveryworld" else v1_cells[cell_id]
            parent_manifest = v4 if domain == "discoveryworld" else v1
            report_path = _report_path(game, domain)
            reasons: list[str] = []
            if not report_path.is_file():
                rows.append({
                    "cell_id": cell_id,
                    "source_game": game,
                    "target_domain": domain,
                    "passed": False,
                    "reasons": ["MISSING_REPORT"],
                })
                continue
            report = read_object(report_path)
            try:
                _validate_report(report)
            except ValueError as exc:
                reasons.append(f"INVALID_REPORT_HASH:{exc}")
            receipt = report.get("cell_execution_receipt") or {}
            if receipt.get("manifest_sha256") != parent_manifest["manifest_sha256"]:
                reasons.append("WRONG_PARENT_MANIFEST")
            for key in ("cell_id", "source_game", "target_domain", "target_task_id"):
                if str(receipt.get(key)) != str(expected.get(key)):
                    reasons.append(f"IDENTITY_MISMATCH:{key}")
            if str(receipt.get("source_artifact_sha256")) != str(
                expected.get("source_artifact_sha256")
            ):
                reasons.append("SOURCE_ARTIFACT_MISMATCH")
            gates = receipt.get("gates") or {}
            reasons.extend(
                f"FAILED_GATE:{key}" for key, value in sorted(gates.items()) if not value
            )
            if receipt.get("status") != "DIRECT_PROSPECTIVE_CELL_PASSED":
                reasons.append("CELL_STATUS_NOT_PASSED")
            fingerprint = stable_hash({
                "source": receipt.get("source_artifact_sha256"),
                "target": receipt.get("target_task_id"),
                "routes": receipt.get("authentic_source_decision_receipt_sha256"),
            })
            if fingerprint in fingerprints:
                reasons.append("DUPLICATE_EXECUTION_FINGERPRINT")
            fingerprints.add(fingerprint)
            rows.append({
                "cell_id": cell_id,
                "source_game": game,
                "target_domain": domain,
                "target_task_id": expected["target_task_id"],
                "evidence_protocol": (
                    "DISCOVERYWORLD_V4" if domain == "discoveryworld" else "CORE_V1"
                ),
                "passed": not reasons,
                "reasons": reasons,
                "authentic_source_decision_count": receipt.get(
                    "authentic_source_decision_count", 0
                ),
                "authentic_source_actions": receipt.get("authentic_source_actions", []),
                "cell_receipt_sha256": receipt.get("cell_receipt_sha256"),
                "report": str(report_path.relative_to(REPO)),
                "report_file_sha256": file_sha256(report_path),
            })

    expected_ids = {
        f"{game}__to__{domain}" for domain in TARGET_DOMAINS for game in SOURCE_GAMES
    }
    observed_ids = {row["cell_id"] for row in rows}
    passed = sum(bool(row["passed"]) for row in rows)
    domain_counts = {
        domain: sum(row["passed"] and row["target_domain"] == domain for row in rows)
        for domain in TARGET_DOMAINS
    }
    exact_coverage = observed_ids == expected_ids
    unique_fingerprints = len(fingerprints) == len(rows)
    validated = passed == 24 and exact_coverage and unique_fingerprints
    body = {
        "schema_version": "phase1-direct-prospective-composite-audit-v4",
        "status": (
            "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED"
            if validated
            else "DIRECT_PROSPECTIVE_MATRIX_INCOMPLETE"
        ),
        "claim_boundary": (
            "24 direct source-lineage×target-domain mechanism executions. This is "
            "operational mechanism validation, not 24 powered success-rate estimates "
            "and not evidence that every source improves every task."
        ),
        "core_v1_manifest_sha256": v1["manifest_sha256"],
        "discoveryworld_v4_manifest_sha256": v4["manifest_sha256"],
        "discoveryworld_v1_v2_v3_counted": False,
        "historical_target_outcomes_counted": False,
        "discoveryworld_v4_preparation": v4_preparation,
        "direct_new_joint_execution_cells": passed,
        "passed_cells": passed,
        "required_cells": 24,
        "domain_passed_cells": domain_counts,
        "exact_cartesian_coverage": exact_coverage,
        "unique_execution_fingerprints": unique_fingerprints,
        "cells": rows,
    }
    result = body | {"audit_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "passed_cells": passed,
        "domain_passed_cells": domain_counts,
        "exact_cartesian_coverage": exact_coverage,
        "unique_execution_fingerprints": unique_fingerprints,
        "output": str(OUTPUT.relative_to(REPO)),
        "audit_sha256": result["audit_sha256"],
    }, indent=2))
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
