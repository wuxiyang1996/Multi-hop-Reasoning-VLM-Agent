#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.real_source_interventions import build_frozen_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = build_frozen_plan(
        {name: Path(path) for name, path in config["evidence"].items()},
        game=str(config["game"]),
        namespace=str(config["namespace"]),
        snapshots_per_episode=int(config["snapshots_per_episode"]),
        actions_per_snapshot=int(config["actions_per_snapshot"]),
        minimum_step=int(config["minimum_step"]),
    )
    output = Path(config["plan_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in plan["snapshots"]:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(json.dumps({"plan_path": str(output), "plan_sha256": plan["plan_sha256"], "snapshots": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
