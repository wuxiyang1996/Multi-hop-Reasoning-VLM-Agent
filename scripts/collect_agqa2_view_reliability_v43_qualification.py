#!/usr/bin/env python3
"""Collect the fresh V43 train-split reliability qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from scripts.collect_agqa2_robust_temporal_v36_development import (  # noqa: E402
    collect_development,
)
from scripts.evaluate_agqa2_view_reliability_v43 import (  # noqa: E402
    evaluate_calibrated,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_view_reliability_v43_qualification.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=REPO_ROOT / "runs/agqa2_view_reliability_v43_qualification/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_view_reliability_v43_qualification/report.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    collect_development(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.base_report.resolve(), workers=args.workers,
        limit=None,
    )
    result = evaluate_calibrated(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
        formal=False,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "provider_calls", "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
