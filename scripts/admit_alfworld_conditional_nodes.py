#!/usr/bin/env python3
"""Freeze receipt-grounded multi-example conditional node programs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.conditional_node_program import (  # noqa: E402
    admit_conditional_programs, proposal_from_dict,
)
from harness.skill_admission import target_demo_receipt_from_dict  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-set-id", required=True)
    parser.add_argument("--proposal-artifact", type=Path, required=True)
    parser.add_argument("--demo", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal_payload = json.loads(args.proposal_artifact.read_text(encoding="utf-8"))
    unsigned = dict(proposal_payload)
    claimed = unsigned.pop("artifact_sha256", None)
    import hashlib
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if claimed != hashlib.sha256(raw.encode()).hexdigest():
        raise SystemExit("proposal artifact hash mismatch")
    demos = tuple(target_demo_receipt_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    ) for path in args.demo)
    proposals = tuple(proposal_from_dict(row) for row in proposal_payload["candidates"])
    known_receipts = tuple(row["receipt_sha256"] for row in proposal_payload["rows"])
    artifact = admit_conditional_programs(
        adaptation_set_id=args.adaptation_set_id, proposals=proposals, demos=demos,
        source_graphs=proposal_payload["source_graphs"],
        known_proposal_receipt_hashes=known_receipts,
        source_treatment=proposal_payload["source_treatment"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "artifact_hash": artifact.artifact_hash,
        "status": artifact.status.value,
        "n_proposed": len(proposals), "n_admitted": len(artifact.candidates),
        "n_rejected": len(artifact.rejected_candidates), "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
