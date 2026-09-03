#!/usr/bin/env python3
"""Freeze discovery-only anonymous causal visual-effect options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.causal_effect_options import (
    build_causal_effect_option_artifact,
    validate_causal_effect_option_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--discovery-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stable-effect-min-rate", type=float, default=0.75)
    parser.add_argument("--null-effect-max-rate", type=float, default=0.0)
    parser.add_argument("--minimum-snapshots", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    artifact = build_causal_effect_option_artifact(
        plan,
        args.discovery_dir,
        stable_effect_min_rate=args.stable_effect_min_rate,
        null_effect_max_rate=args.null_effect_max_rate,
        minimum_snapshots=args.minimum_snapshots,
    )
    validate_causal_effect_option_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "artifact": str(args.output.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "class_members": artifact["source_grounding"]["class_members"],
        "effect_rates": artifact["source_grounding"]["effect_rates"],
        "lifecycle": artifact["lifecycle"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
