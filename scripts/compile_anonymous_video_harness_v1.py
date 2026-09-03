#!/usr/bin/env python3
"""Compile the source-only anonymous video Harness controller artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.anonymous_video_harness import compile_anonymous_source_controller


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--lineages", type=Path,
        default=Path("runs/phase3_source_induction_v1_development/lineages"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/anonymous_video_harness_v1/controller.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    artifact = compile_anonymous_source_controller(
        root=root, lineage_directory=args.lineages,
    )
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": artifact["status"],
        "operators": len(artifact["operators"]),
        "transitions": len(artifact["transitions"]),
        "artifact_sha256": artifact["artifact_sha256"],
        "output": str(output),
    }, indent=2))
    return 0 if artifact["status"] == "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
