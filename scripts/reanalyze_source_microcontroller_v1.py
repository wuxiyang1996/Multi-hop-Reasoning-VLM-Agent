#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.multihorizon_replay import file_hash
from motif_transfer.phase1_assets import read_jsonl
from motif_transfer.source_microcontroller import analyze_microcontroller_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add scale-robust diagnostics without changing the v1 primary gate."
    )
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    report = analyze_microcontroller_rows(read_jsonl(args.rows))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report.update({
        "analysis_status": "POSTHOC_SCALE_ROBUSTNESS_DIAGNOSTIC",
        "primary_gate_changed": False,
        "plan_content_sha256": plan["plan_sha256"],
        "plan_file_sha256": file_hash(args.plan),
        "rows_sha256": file_hash(args.rows),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "gates": report["gates"],
        "output": str(args.output.resolve()),
        "primary_gate_changed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
