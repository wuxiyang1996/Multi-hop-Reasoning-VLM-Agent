#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from motif_transfer.real_source_interventions import build_live_frozen_plan


def _load_adapter(source_script: Path):
    spec = importlib.util.spec_from_file_location("fresh_real_source_runtime", source_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source runtime: {source_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._SourceReplayAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runtime_path = Path(config["source_runtime_script"]).resolve()
    adapter_class = _load_adapter(runtime_path)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            plan = build_live_frozen_plan(
                adapter_class,
                game=str(config["game"]),
                seeds=[int(value) for value in config["fresh_seeds"]],
                namespace=str(config["namespace"]),
                max_steps=int(config["max_steps"]),
                rollout_steps=int(config["rollout_steps"]),
                snapshots_per_episode=int(config["snapshots_per_episode"]),
                actions_per_snapshot=int(config["actions_per_snapshot"]),
                minimum_step=int(config["minimum_step"]),
                runtime_receipt={
                    "source_runtime_script": str(runtime_path),
                    "source_runtime_script_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
                    "python": sys.executable,
                },
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
