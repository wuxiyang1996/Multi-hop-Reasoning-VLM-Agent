#!/usr/bin/env python3
"""Write the compact acquisition dependency for V38 method selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report_path = REPO_ROOT / (
        "runs/agqa2_aggregate_temporal_v38_development/report.json"
    )
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or not all(
        report["qualification_gates"].values()
    ):
        raise ValueError("V38 development method report is not qualified")
    config = json.loads((REPO_ROOT / (
        "configs/agqa2_aggregate_temporal_v38_development.json"
    )).read_text())
    core = {
        "schema_version": "agqa2-aggregate-temporal-v38-development-summary-v1",
        "status": report["status"],
        "grounder_qualified": True,
        "grounder_sha256": config["expected_grounder_sha256"],
        "postground_target_grounder_sha256": report["target_grounder_sha256"],
        "source_program_sha256": report["source_program_sha256"],
        "rows": report["rows"],
        "source_executor_authorizations": report[
            "source_executor_authorizations"
        ],
        "source_vs_target_native": report["source_vs_target_native"],
        "qualification_gates": report["qualification_gates"],
        "development_report_sha256": report["report_sha256"],
        "development_report_file_sha256": _sha256(report_path),
        "method_selected_after_v37_development_outcome_access": True,
        "confirmatory_claim": False,
    }
    summary = core | {"summary_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_aggregate_temporal_v38_development_summary.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
