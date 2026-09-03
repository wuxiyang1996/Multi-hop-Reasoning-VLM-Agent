#!/usr/bin/env python3
"""Verify every path/hash frozen by the formal protocol before acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("protocol verification is immutable")
    protocol = json.loads(args.protocol.read_text())
    checks = {}
    for path, expected in protocol["implementation_file_sha256s"].items():
        checks[path] = _sha256(Path(path)) == expected
    direct = {
        protocol["qualified_grounder"]["qualification_file"]:
            protocol["qualified_grounder"]["qualification_file_sha256"],
        protocol["qualified_grounder"]["public_ontology"]:
            protocol["qualified_grounder"]["public_ontology_sha256"],
        protocol["source_harness"]["source_capability_file"]:
            protocol["source_harness"]["source_capability_file_sha256"],
        protocol["source_harness"]["anonymous_controller_file"]:
            protocol["source_harness"]["anonymous_controller_file_sha256"],
    }
    for path, expected in direct.items():
        checks[path] = _sha256(Path(path)) == expected
    body = {
        "schema_version": "agqa-query-grounder-v2-formal-protocol-verification-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_file_sha256": _sha256(args.protocol),
        "checks": checks,
        "all_frozen_hashes_match": all(checks.values()),
        "verified_before_video_acquisition": True,
        "target_outcomes_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
