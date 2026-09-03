#!/usr/bin/env python3
"""Audit the 18-cell V1 core plus the uniform six-cell DiscoveryWorld V2."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    TARGET_DOMAINS,
    file_sha256,
    read_object,
    validate_manifest,
    validate_self_hash,
)
from scripts.run_phase1_direct_discoveryworld_v2 import (  # noqa: E402
    validate_v2_manifest,
)


V1_MANIFEST = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
V2_MANIFEST = REPO / "configs/phase1_direct_prospective_v2/discoveryworld_manifest.json"
V1_RUN = REPO / "runs/phase1_direct_prospective_v1"
V2_RUN = REPO / "runs/phase1_direct_prospective_v2"
OUTPUT = REPO / "docs/results/phase1_direct_prospective_24_of_24_v2.json"


def _report_path(game: str, domain: str) -> Path:
    if domain == "webshop":
        return V1_RUN / domain / game / "direct_report.json"
    if domain in {"alfworld", "tirbench"}:
        return V1_RUN / domain / game / "report.json"
    if domain == "discoveryworld":
        return V2_RUN / domain / "cells" / game / "direct_report.json"
    raise ValueError(domain)


def _validate_report(report: Mapping[str, Any]) -> None:
    validate_self_hash(report, "report_sha256")
    receipt = report.get("cell_execution_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("missing cell execution receipt")
    validate_self_hash(receipt, "cell_receipt_sha256")


def main() -> int:
    v1 = read_object(V1_MANIFEST)
    v2 = read_object(V2_MANIFEST)
    validate_manifest(v1, repo=REPO)
    validate_v2_manifest(v2)
    if v2["parent_v1_manifest_sha256"] != v1["manifest_sha256"]:
        raise ValueError("V2 does not bind the audited V1 manifest")
    if v2["parent_v1_manifest_file_sha256"] != file_sha256(V1_MANIFEST):
        raise ValueError("V2 parent V1 manifest file changed")

    v1_cells = {str(row["cell_id"]): row for row in v1["cells"]}
    v2_cells = {str(row["cell_id"]): row for row in v2["cells"]}
    rows = []
    fingerprints: set[str] = set()
    for domain in TARGET_DOMAINS:
        for game in SOURCE_GAMES:
            cell_id = f"{game}__to__{domain}"
            expected = v2_cells[cell_id] if domain == "discoveryworld" else v1_cells[cell_id]
            parent_manifest = v2 if domain == "discoveryworld" else v1
            report_path = _report_path(game, domain)
            reasons = []
            if not report_path.is_file():
                rows.append({
                    "cell_id": cell_id, "source_game": game,
                    "target_domain": domain, "passed": False,
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
            failed_gates = sorted(key for key, value in gates.items() if not value)
            reasons.extend(f"FAILED_GATE:{key}" for key in failed_gates)
            if receipt.get("status") != "DIRECT_PROSPECTIVE_CELL_PASSED":
                reasons.append("CELL_STATUS_NOT_PASSED")
            fingerprint = stable_hash({
                "source": receipt.get("source_artifact_sha256"),
                "target": receipt.get("target_task_id"),
                "routes": receipt.get("authentic_source_decision_receipt_sha256"),
            })
            unique = fingerprint not in fingerprints
            fingerprints.add(fingerprint)
            if not unique:
                reasons.append("DUPLICATE_EXECUTION_FINGERPRINT")
            rows.append({
                "cell_id": cell_id,
                "source_game": game,
                "target_domain": domain,
                "target_task_id": expected["target_task_id"],
                "evidence_protocol": (
                    "DISCOVERYWORLD_V2" if domain == "discoveryworld" else "CORE_V1"
                ),
                "passed": not reasons,
                "reasons": reasons,
                "authentic_source_decision_count": receipt.get(
                    "authentic_source_decision_count", 0,
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
        domain: sum(
            row["passed"] and row["target_domain"] == domain for row in rows
        )
        for domain in TARGET_DOMAINS
    }
    body = {
        "schema_version": "phase1-direct-prospective-composite-audit-v2",
        "status": (
            "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED"
            if passed == 24 and observed_ids == expected_ids
            else "DIRECT_PROSPECTIVE_MATRIX_INCOMPLETE"
        ),
        "claim_boundary": (
            "24 direct source-lineage×target-domain mechanism executions. This "
            "is operational mechanism validation, not 24 powered success-rate "
            "estimates and not evidence that every source improves every task."
        ),
        "core_v1_manifest_sha256": v1["manifest_sha256"],
        "discoveryworld_v2_manifest_sha256": v2["manifest_sha256"],
        "discoveryworld_v1_counted": False,
        "direct_new_joint_execution_cells": passed,
        "passed_cells": passed,
        "required_cells": 24,
        "domain_passed_cells": domain_counts,
        "exact_cartesian_coverage": observed_ids == expected_ids,
        "unique_execution_fingerprints": len(fingerprints) == len(rows),
        "cells": rows,
    }
    result = body | {"audit_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"], "passed_cells": passed,
        "domain_passed_cells": domain_counts,
        "output": str(OUTPUT.relative_to(REPO)),
        "audit_sha256": result["audit_sha256"],
    }, indent=2))
    return 0 if result["status"] == "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
