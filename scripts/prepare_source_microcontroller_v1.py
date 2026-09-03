#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.source_microcontroller import build_source_microcontroller_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a discovery-induced event micro-controller replay plan."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument("--observational-output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    evidence = []
    for row in config.get("evidence", ()):
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = root / path
        evidence.append((str(row["game"]), path))
    if not evidence:
        raise SystemExit("config contains no source evidence")
    for path in (args.plan_output, args.observational_output):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    plan, observational = build_source_microcontroller_plan(
        evidence, config_path=args.config,
    )
    args.plan_output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.observational_output.write_text(
        json.dumps(observational, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "branch_map": plan["branch_map"],
        "plan": str(args.plan_output.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "point_count": observational["point_count"],
        "selected_counts": plan["selected_counts"],
        "snapshots": len(plan["snapshots"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
