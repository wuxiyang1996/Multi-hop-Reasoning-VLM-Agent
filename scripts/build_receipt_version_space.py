#!/usr/bin/env python3
"""Freeze a registered multi-example exact binding version space."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.multistep_binding import multistep_artifact_from_dict  # noqa: E402
from harness.receipt_version_space import build_receipt_version_space  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-set-id", required=True)
    parser.add_argument("--expected-example-count", type=int, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = tuple(multistep_artifact_from_dict(json.loads(
        path.read_text(encoding="utf-8")
    )) for path in args.artifact)
    version_space = build_receipt_version_space(
        adaptation_set_id=args.adaptation_set_id,
        artifacts=artifacts,
        expected_example_count=args.expected_example_count,
    )
    payload = version_space.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "artifact_hash": version_space.artifact_hash,
        "status": version_space.status.value,
        "observed_examples": len(version_space.examples),
        "expected_examples": version_space.expected_example_count,
        "n_versions": len(version_space.versions),
        "n_viable_versions": len(version_space.viable_schema_hashes),
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
