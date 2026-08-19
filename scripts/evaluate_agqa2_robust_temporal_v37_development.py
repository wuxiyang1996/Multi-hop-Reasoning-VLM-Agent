#!/usr/bin/env python3
"""Score the unchanged V36 base receipts as V37 development evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.collect_agqa2_robust_temporal_v34_formal import (  # noqa: E402
    evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_robust_temporal_v37_development.json",
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=(
            REPO_ROOT
            / "runs/agqa2_robust_temporal_v36_development/base_report.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_robust_temporal_v37_development/report.json",
    )
    args = parser.parse_args()
    result = evaluate(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
    )
    body = deepcopy(result)
    body.pop("report_sha256", None)
    qualified = all(body["qualification_gates"].values())
    body.update({
        "schema_version": "agqa2-robust-temporal-v37-development-report-v1",
        "status": (
            "AGQA2_ROBUST_TEMPORAL_V37_DEVELOPMENT_QUALIFIED"
            if qualified else
            "AGQA2_ROBUST_TEMPORAL_V37_DEVELOPMENT_NOT_QUALIFIED"
        ),
        "split": "consumed_development",
        "confirmatory_claim": False,
        "claim_boundary": (
            "UNCHANGED_HASHED_V36_BASE_RECEIPTS;V37_ONLY_RESTORES_THE_"
            "PREEXISTING_V33_EVIDENCE_LINEAGE_KEY;NOT_CONFIRMATORY"
        ),
        "new_provider_calls": 0,
        "v34_v35_v36_reclassified_as_success": False,
    })
    final = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: final[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "new_provider_calls", "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
