#!/usr/bin/env python3
"""Freeze Ours candidates into the same raw/gated memory runtime schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_domain_memory_baselines import build_external_memory_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(payload, dict):
        raise SystemExit("source and candidates must each be one JSON object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("candidate payload needs a candidates list")
    artifact = build_external_memory_artifact(
        "ours", source, candidates,
        producer_identity=payload.get("producer_identity") or {"producer": "ours"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "method": "ours", "items": len(artifact["items"]),
        "artifact_sha256": artifact["artifact_sha256"], "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
