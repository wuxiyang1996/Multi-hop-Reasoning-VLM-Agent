#!/usr/bin/env python3
"""Reanalyze already-consumed wrapper adaptation receipts without new calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import stable_hash  # noqa: E402
from motif_transfer.candidate_transfer_experiment import (  # noqa: E402
    evaluate_candidate_adaptation,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipt_paths = list(args.receipts)
    receipts = [
        row
        for path in receipt_paths
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]
    expected = list(config["splits"]["adaptation"])
    by_id = {str(row["sample_id"]): row for row in receipts}
    if len(by_id) != len(receipts):
        raise SystemExit("duplicate receipt identities across input files")
    if set(by_id) != set(expected):
        raise SystemExit("receipt identities do not equal the frozen adaptation IDs")
    receipts = [by_id[sample_id] for sample_id in expected]
    controlled_path = Path(config["source"]["controlled_v3_config"])
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"]["controlled_v3_config_content_sha256"]:
        raise SystemExit("controlled source config content hash mismatch")
    report, artifact = evaluate_candidate_adaptation(
        receipts, config=config, controlled_config=controlled,
    )
    report["claim_boundary"] = (
        "Reanalysis of already-consumed adaptation wrapper receipts; no new "
        "target outcome, qualification ID, or held-out ID was read."
    )
    report["source_receipts"] = {
        "files": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in receipt_paths
        ],
        "original_collection_contract_sha256_values": sorted({
            str(row.get("collection_contract_sha256") or "") for row in receipts
        }),
    }
    report["analysis_code"] = {
        "candidate_transfer_experiment_sha256": file_sha256(
            REPO / "src/motif_transfer/candidate_transfer_experiment.py"
        ),
        "active_video_transfer_sha256": file_sha256(
            REPO / "src/motif_transfer/active_video_transfer.py"
        ),
        "reanalyzer_sha256": file_sha256(Path(__file__).resolve()),
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = stable_hash(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "target_grounder_candidate.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["target_grounder_candidate"] = {
        "path": str(artifact_path.resolve()),
        "sha256": file_sha256(artifact_path),
        "content_sha256": artifact["artifact_sha256"],
    }
    report_path = args.output_dir / "adaptation_reanalysis_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "baseline_accuracy": report["baseline_accuracy"],
        "conditions": report["conditions_cross_fitted"],
        "gates": report["gates"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
