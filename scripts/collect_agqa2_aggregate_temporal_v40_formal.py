#!/usr/bin/env python3
"""Collect and evaluate the corrected frozen V40 formal run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.collect_agqa2_robust_temporal_v36_development import (  # noqa: E402
    collect_development,
)
from scripts.evaluate_agqa2_aggregate_temporal_v38 import (  # noqa: E402
    evaluate_aggregate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_aggregate_temporal_v40_formal.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=REPO_ROOT / "runs/agqa2_aggregate_temporal_v40_formal/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_aggregate_temporal_v40_formal/report.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    collect_development(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.base_report.resolve(), workers=args.workers,
        limit=None,
    )
    result = evaluate_aggregate(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
        formal=True,
    )
    body = deepcopy(result)
    body.pop("report_sha256", None)
    qualified = all(body["qualification_gates"].values())
    body.update({
        "schema_version": "agqa2-aggregate-temporal-v40-formal-report-v1",
        "status": (
            "AGQA2_AGGREGATE_TEMPORAL_V40_FORMAL_QUALIFIED"
            if qualified else
            "AGQA2_AGGREGATE_TEMPORAL_V40_FORMAL_NOT_QUALIFIED"
        ),
        "confirmatory_claim": qualified,
        "v39_preflight_reclassified_as_result": False,
    })
    final = body | {"report_sha256": stable_hash(body)}
    args.output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: final[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "provider_calls", "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
