#!/usr/bin/env python3
"""Build a target-blind typed-operator capability artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.source_operator_capability_induction import induce_from_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-artifact", type=Path, required=True)
    parser.add_argument("--acquisition-confirmation", type=Path, required=True)
    parser.add_argument("--temporal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = induce_from_paths(
        acquisition_artifact_path=args.acquisition_artifact,
        acquisition_confirmation_path=args.acquisition_confirmation,
        temporal_manifest_path=args.temporal_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "authorized_operators": result["authorized_operators"],
        "artifact_sha256": result["artifact_sha256"],
        "output": str(args.output),
    }, indent=2))
    return 0 if result["status"] == "SOURCE_CAPABILITIES_INDUCED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
