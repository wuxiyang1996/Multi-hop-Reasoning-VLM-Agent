#!/usr/bin/env python3
"""Verify and summarize the frozen DiscoveryWorld Phase-2 utility run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.phase2_discoveryworld_utility_v1 import (  # noqa: E402
    build_report, read_object,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase2_discoveryworld_utility_v1/manifest.json",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO / "runs/phase2_discoveryworld_utility_v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    cells = []
    for task in manifest["tasks"]:
        path = args.run_root / "cells" / str(task["task_id"]) / "cell.json"
        if path.is_file():
            cells.append(read_object(path))
    report = build_report(manifest, cells, repo=REPO)
    output = args.output or args.run_root / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "successes": report["condition_successes"],
        "authentic_vs_raw": report["authentic_vs_raw"],
        "gates": f"{report['passed_gates']}/{report['required_gates']}",
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if report["status"].endswith("_VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
