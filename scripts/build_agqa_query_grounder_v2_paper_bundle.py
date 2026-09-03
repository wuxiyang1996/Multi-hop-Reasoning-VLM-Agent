#!/usr/bin/env python3
"""Build the compact paper-facing audit bundle for AGQA Query Grounder V2."""

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


def build_bundle(paths: dict[str, Path]) -> dict:
    values = {key: json.loads(path.read_text()) for key, path in paths.items()}
    qualification = values["qualification"]
    protocol = values["protocol"]
    manifest = values["manifest"]
    preoutcome = values["preoutcome"]
    formal = values["formal"]
    cost = values["cost"]
    if qualification.get("status") != "QUERY_GROUNDER_V2_POWERED_QUALIFIED":
        raise ValueError("paper bundle requires a passed grounder qualification")
    if manifest.get("status") != "AGQA_QUERY_GROUNDER_V2_FRESH_FORMAL_FROZEN":
        raise ValueError("paper bundle requires a frozen fresh formal reserve")
    if preoutcome.get("status") != "ALL_FIVE_ARM_DECISIONS_FROZEN_BEFORE_FORMAL_OUTCOMES":
        raise ValueError("paper bundle requires a passed pre-outcome freeze")
    if formal.get("status") not in {
        "AGQA_QUERY_GROUNDER_V2_FRESH_FORMAL_TRANSFER_VALIDATED",
        "AGQA_QUERY_GROUNDER_V2_FRESH_FORMAL_GATES_FAILED",
    }:
        raise ValueError("paper bundle received an unknown formal status")
    if formal["protocol_file_sha256"] != _sha256(paths["protocol"]):
        raise ValueError("formal result is not bound to the supplied protocol")
    if formal["manifest_file_sha256"] != _sha256(paths["manifest"]):
        raise ValueError("formal result is not bound to the supplied manifest")
    if formal["preoutcome_file_sha256"] != _sha256(paths["preoutcome"]):
        raise ValueError("formal result is not bound to the supplied pre-outcome receipt")
    table = [{
        "arm": arm,
        "correct": formal["summaries"][arm]["correct"],
        "total": formal["summaries"][arm]["total"],
        "accuracy": formal["summaries"][arm]["accuracy"],
        "symbolic_commits": formal["summaries"][arm]["symbolic_commits"],
    } for arm in protocol["arms"]]
    body = {
        "schema_version": "agqa-query-grounder-v2-paper-bundle-v1",
        "status": formal["status"],
        "claim": protocol["claim"],
        "claim_scope": formal["claim_scope"],
        "qualification": {
            "tasks": qualification["qualification_rows"],
            "metrics": qualification["metrics"],
            "gates": qualification["gates"],
            "report_sha256": qualification["report_sha256"],
        },
        "formal": {
            "tasks": len(formal["rows"]),
            "main_table": table,
            "comparisons": formal["comparisons"],
            "gates": formal["gates"],
            "secondary_target": formal["secondary_target"],
            "failure_taxonomy": formal["failure_taxonomy"],
            "ablations": formal["ablations"],
            "report_sha256": formal["report_sha256"],
        },
        "cost": cost,
        "shared_arm_contract": protocol["shared_arm_contract"],
        "claim_boundary": protocol["claim_boundary"],
        "artifact_file_sha256s": {
            key: _sha256(path) for key, path in sorted(paths.items())
        },
    }
    body["bundle_sha256"] = stable_hash(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("qualification", "protocol", "manifest", "preoutcome", "formal", "cost"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("paper bundle is immutable")
    paths = {name: getattr(args, name) for name in (
        "qualification", "protocol", "manifest", "preoutcome", "formal", "cost",
    )}
    body = build_bundle(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"],
        "formal_tasks": body["formal"]["tasks"],
        "bundle_sha256": body["bundle_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
