#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.source_multihorizon import build_multihorizon_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an outcome-blind source multi-horizon replay plan."
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-per-split", type=int, default=1)
    parser.add_argument(
        "--include-split",
        action="append",
        choices=("discovery", "qualification", "held_out"),
        dest="included_splits",
    )
    args = parser.parse_args()
    plan = build_multihorizon_plan(
        args.evidence,
        config_path=args.config,
        maximum_per_split=args.maximum_per_split,
        included_splits=tuple(
            args.included_splits or ("qualification", "held_out")
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen plan: {args.output}")
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "game": plan["game"],
        "plan_sha256": plan["plan_sha256"],
        "selected_counts": plan["selected_counts"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
