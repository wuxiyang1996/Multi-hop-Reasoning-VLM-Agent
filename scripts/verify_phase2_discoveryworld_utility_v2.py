#!/usr/bin/env python3
"""Aggregate the selective DiscoveryWorld V2 causal utility result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.phase2_discoveryworld_utility_v2 import build_report, read_object  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "configs/phase2_discoveryworld_utility_v2/manifest.json")
    parser.add_argument("--run-root", type=Path, default=REPO / "runs/phase2_discoveryworld_utility_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    cells = [read_object(args.run_root / "cells" / row["task_id"] / "cell.json") for row in manifest["tasks"] if (args.run_root / "cells" / row["task_id"] / "cell.json").is_file()]
    report = build_report(manifest, cells, repo=REPO)
    output = args.output or args.run_root / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "successes": report["condition_successes"], "paired": report["authentic_vs_raw"], "gates": f"{report['passed_gates']}/{report['required_gates']}"}, indent=2))
    return 0 if report["status"].endswith("_VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
