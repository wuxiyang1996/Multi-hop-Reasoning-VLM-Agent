#!/usr/bin/env python3
"""Evaluate one frozen TIR V3 stage and emit a content-bound authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_tir_nonmaze import (  # noqa: E402
    evaluate_matched_receipts,
    validate_grounder_artifact,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("development_holdout", "qualification", "formal"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config_body = dict(config)
    config_claimed = str(config_body.pop("config_sha256", ""))
    if not config_claimed or stable_hash(config_body) != config_claimed:
        raise SystemExit("TIR V3 manifest hash mismatch")
    artifact_path = REPO / config["grounder"]["path"]
    if file_sha256(artifact_path) != config["grounder"]["file_sha256"]:
        raise SystemExit("TIR grounder file drift")
    artifact = json.loads(artifact_path.read_text())
    validate_grounder_artifact(artifact)
    if artifact["artifact_sha256"] != config["grounder"]["artifact_sha256"]:
        raise SystemExit("TIR grounder artifact drift")
    sources = []
    for row in config["source_programs"]:
        path = REPO / row["path"]
        if file_sha256(path) != row["file_sha256"]:
            raise SystemExit(f"source artifact file drift: {row['path']}")
        source = json.loads(path.read_text())
        if source["artifact_sha256"] != row["artifact_sha256"]:
            raise SystemExit(f"source artifact hash drift: {row['path']}")
        sources.append(source)
    receipts = json.loads(args.receipts.read_text())
    expected_ids = list(map(str, config["splits"][args.stage]))
    observed_ids = [str(row["sample_id"]) for row in receipts]
    if observed_ids != expected_ids:
        raise SystemExit("TIR receipts do not match frozen stage/order")
    for row in receipts:
        receipt_body = dict(row)
        claimed = str(receipt_body.pop("receipt_sha256", ""))
        if not claimed or stable_hash(receipt_body) != claimed:
            raise SystemExit(f"receipt hash mismatch: {row.get('sample_id')}")
        if row.get("formal_outcome_exposed_to_neural_calls") is not False:
            raise SystemExit("receipt lacks target-outcome isolation attestation")
        if row.get("source_program_or_identity_exposed_to_neural_calls") is not False:
            raise SystemExit("receipt exposed source information to neural calls")
        if row.get("evaluation_role") != args.stage:
            raise SystemExit("receipt evaluation role mismatch")
    gates = config[f"{args.stage}_gates"]
    report = evaluate_matched_receipts(
        receipts, grounder_artifact=artifact,
        source_artifacts=sources, gates=gates, role=args.stage,
    )
    report_body = dict(report)
    report_body.pop("report_sha256", None)
    report_body.update({
        "phase3_tir_manifest_sha256": config["config_sha256"],
        "phase3_tir_manifest_file_sha256": file_sha256(args.config),
        "receipts_file_sha256": file_sha256(args.receipts),
        "grounder_thresholds_frozen": True,
        "formal_results_used_to_change_protocol": False,
        "source_identity_used_as_runtime_feature": False,
        "source_program_updated_for_target": False,
        "target_outcome_read_by_neural_grounder_or_source_runtime": False,
    })
    report = report_body | {"report_sha256": stable_hash(report_body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "successes": report["successes"],
        "paired": report["paired"],
        "behavior": report["behavior"],
        "required_gates": {
            name: report["gates"][name]
            for name in report["required_gate_names"]
        },
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
