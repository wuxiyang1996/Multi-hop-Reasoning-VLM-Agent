#!/usr/bin/env python3
"""Run the frozen causal-effect candidate on untouched source episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.effect_option_qualification import collect_source_qualification
from motif_transfer.visual_intervention_receipts import load_runtime_env_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("qualification", "held_out"), default="qualification",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    manifest = collect_source_qualification(
        plan, artifact, split=args.split, output_dir=args.output_dir,
        env_factory=load_runtime_env_factory(args.runtime_root), workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["summary"].get("next_step") == "STOP_PROTOCOL_FAILURE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
