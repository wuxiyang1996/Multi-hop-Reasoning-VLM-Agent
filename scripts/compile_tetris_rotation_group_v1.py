#!/usr/bin/env python3
"""Compile and source-confirm the anonymous Tetris rotation group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.tetris_rotation_transfer import (  # noqa: E402
    compile_source_rotation_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/real_game_multitarget_v5_manifest.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "runs/tetris_rotation_group_v1/source_artifact.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite source artifact: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    artifact = compile_source_rotation_artifact(
        manifest=manifest, roles=("discovery", "qualification", "held_out"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": artifact["status"],
        "summaries": artifact["summaries"],
        "gates": artifact["gates"],
        "artifact_sha256": artifact["artifact_sha256"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    if not all(artifact["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
