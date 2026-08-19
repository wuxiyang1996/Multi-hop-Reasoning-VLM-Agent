#!/usr/bin/env python3
"""Complete V46 qualification after adding one legacy prereg alias."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.evaluate_agqa2_interval_reliability_v46 import (  # noqa: E402
    evaluate_calibrated,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_interval_reliability_v47_qualification.json",
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=REPO_ROOT / "runs/agqa2_interval_reliability_v46_qualification/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_interval_reliability_v47_qualification/report.json",
    )
    args = parser.parse_args()
    result = evaluate_calibrated(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(), formal=False,
    )
    body = deepcopy(result)
    body.pop("report_sha256", None)
    qualified = all(body["qualification_gates"].values())
    body.update({
        "schema_version": "agqa2-interval-reliability-v47-qualification-report-v1",
        "status": (
            "AGQA2_INTERVAL_RELIABILITY_V47_QUALIFICATION_QUALIFIED"
            if qualified else
            "AGQA2_INTERVAL_RELIABILITY_V47_QUALIFICATION_NOT_QUALIFIED"
        ),
        "confirmatory_claim": False,
        "v46_reclassified_as_success": False,
        "v46_completion_change": (
            "ADD_LEGACY_EVIDENCE_ALIAS_EQUAL_TO_FROZEN_V45_ARTIFACT_HASH_ONLY"
        ),
        "new_provider_calls": 0,
    })
    final = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: final[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
