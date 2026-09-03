#!/usr/bin/env python3
"""Build matched semantic and frozen-random raw trajectory control artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    MemoryControl,
    build_trajectory_memory_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=73021)
    parser.add_argument("--maximum-trajectory-tokens", type=int, default=2400)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise SystemExit("source input must be one JSON object")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in MemoryControl:
        artifact = build_trajectory_memory_artifact(
            method, source, random_seed=args.random_seed,
            maximum_trajectory_tokens=args.maximum_trajectory_tokens,
        )
        output = args.output_dir / f"{method.value}.json"
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({
            "method": method.value,
            "retrieval_strategy": artifact["retrieval_strategy"],
            "trajectories": len(artifact["items"]),
            "artifact_sha256": artifact["artifact_sha256"],
            "output": str(output),
        })
    print(json.dumps({"controls": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
