#!/usr/bin/env python3
"""Induce the target-independent video-transfer algebra from a frozen catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.source_video_operator_algebra import induce_source_video_algebra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    catalog = json.loads((root / args.catalog).read_text(encoding="utf-8"))
    result = induce_source_video_algebra(root=root, catalog=catalog)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "primitive_count": len(result["primitives"]),
        "abstaining_sources": len(result["source_abstentions"]),
        "artifact_sha256": result["artifact_sha256"],
        "output": str(output),
    }, indent=2))
    return 0 if result["status"] == "SOURCE_VIDEO_OPERATOR_ALGEBRA_QUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
