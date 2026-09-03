#!/usr/bin/env python3
"""Fail-closed audit of all 24 direct prospective transfer cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    TARGET_DOMAINS,
    audit_cell_receipts,
    file_sha256,
    read_object,
    validate_manifest,
    validate_self_hash,
)


REPORT_NAMES = {
    "webshop": "direct_report.json",
    "alfworld": "report.json",
    "discoveryworld": "direct_report.json",
    "tirbench": "report.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_direct_prospective_v1/manifest.json",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO / "docs/results/"
            "phase1_direct_prospective_24cell_audit_v1.json"
        ),
    )
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    receipts = []
    report_files = {}
    report_errors = {}
    for domain in TARGET_DOMAINS:
        for game in SOURCE_GAMES:
            base = args.run_root / domain
            if domain == "discoveryworld":
                path = base / "cells" / game / REPORT_NAMES[domain]
            else:
                path = base / game / REPORT_NAMES[domain]
            cell_id = f"{game}__to__{domain}"
            if not path.is_file():
                report_errors[cell_id] = "MISSING_REPORT"
                continue
            try:
                report = read_object(path)
                validate_self_hash(report, "report_sha256")
                receipt = dict(report["cell_execution_receipt"])
                validate_self_hash(receipt, "cell_receipt_sha256")
            except Exception as exc:
                report_errors[cell_id] = f"{type(exc).__name__}: {exc}"
                continue
            receipts.append(receipt)
            report_files[cell_id] = {
                "path": str(path.relative_to(REPO)),
                "file_sha256": file_sha256(path),
                "report_sha256": report["report_sha256"],
            }
    matrix = audit_cell_receipts(manifest, receipts)
    domain_counts = {
        domain: sum(
            bool(row["passed"])
            for row in matrix["cells"] if row["cell_id"].endswith(f"__to__{domain}")
        )
        for domain in TARGET_DOMAINS
    }
    gates = {
        "manifest_fail_closed": True,
        "all_24_reports_present_and_valid": not report_errors and len(receipts) == 24,
        "six_distinct_source_artifact_hashes": len({
            row["source_artifact_sha256"] for row in manifest["cells"]
        }) == 6,
        "24_unique_target_identities_within_domain": all(
            len({
                row["target_task_id"] for row in manifest["cells"]
                if row["target_domain"] == domain
            }) == 6
            for domain in TARGET_DOMAINS
        ),
        "six_direct_cells_pass_each_target_domain": all(
            domain_counts[domain] == 6 for domain in TARGET_DOMAINS
        ),
        "direct_24_of_24": matrix["passed_cells"] == 24,
    }
    body = {
        "schema_version": "phase1-direct-prospective-24cell-final-audit-v1",
        "status": (
            "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED"
            if all(gates.values()) else "DIRECT_PROSPECTIVE_MATRIX_INCOMPLETE"
        ),
        "claim_boundary": manifest["claim_boundary"],
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "mechanism_cells_previously_validated": 24,
        "direct_new_joint_execution_cells": matrix["passed_cells"],
        "required_direct_cells": 24,
        "domain_pass_counts": domain_counts,
        "report_errors": report_errors,
        "report_files": report_files,
        "gates": gates,
        "matrix_audit": matrix,
    }
    report = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "direct_new_joint_execution_cells": report[
            "direct_new_joint_execution_cells"
        ],
        "domain_pass_counts": domain_counts,
        "gates": gates,
        "report_errors": report_errors,
        "output": str(args.output),
    }, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
