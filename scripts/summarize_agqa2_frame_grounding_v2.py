#!/usr/bin/env python3
"""Create a compact, reviewable AGQA frame-grounding result artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(report_path: Path, config_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("AGQA V2 report hash mismatch")
    config = json.loads(config_path.read_text())
    if report["config_sha256"] != _sha256(config_path):
        raise ValueError("AGQA V2 config/report mismatch")
    rows = []
    for row in report["rows"]:
        rows.append({
            "task_id": row["task_id"],
            "video_id": row["video_id"],
            "oracle_route": row["oracle_route_evaluator_only"],
            "predicted_route": row["grounding_receipt"]["obligation_kind"],
            "comparison": row["grounding_receipt"]["comparison"],
            "coverage": row["grounding_receipt"]["coverage"],
            "canonicalizations": row["grounding_receipt"]["canonicalizations"],
            "grounding_receipt_sha256": row["grounding_receipt"]["receipt_sha256"],
            "typed_decision": row["target_native_execution"]["decision"],
            "gold_answer_evaluator_only": row["gold_answer_evaluator_only"],
            "direct_response": row["direct_response"],
            "decisive_correct": row["decisive_correct"],
            "direct_correct": row["direct_correct"],
            "typed_fallback_correct": row["typed_fallback_correct"],
            "unified_harness_executor_authorized": row[
                "unified_harness_executor_authorized"
            ],
            "unified_harness_correct": row["unified_harness_correct"],
        })
    core = {
        "schema_version": "agqa2-frame-grounding-summary-v2",
        "status": report["status"],
        "claim_boundary": report["claim_boundary"],
        "accepted_report_sha256": claimed,
        "accepted_report_path": str(report_path.relative_to(REPO_ROOT)),
        "config_sha256": report["config_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "grounder_sha256": report["grounder_sha256"],
        "model": report["model"],
        "sample_count": report["sample_count"],
        "unique_video_count": report["unique_video_count"],
        "new_video_downloads": report["new_video_downloads"],
        "accepted_receipt_provider_calls": report["provider_calls"],
        "accepted_receipt_reported_provider_cost_usd": report[
            "reported_provider_cost_usd"
        ],
        "cumulative_development_spend_fully_reconstructible": False,
        "why_cumulative_spend_is_not_exact": (
            "PREQUALIFICATION_SCHEMA_AND_ABORTED_8B_CALLS_DID_NOT_ALL_RETURN_"
            "PERSISTED_USAGE;CONFIG_PRESERVES_THEIR_FAILURE_SEQUENCE"
        ),
        "metrics": report["metrics"],
        "qualification_gates": report["qualification_gates"],
        "grounder_qualified": report["grounder_qualified"],
        "source_portfolio_caveat": report["source_portfolio_caveat"],
        "rows": rows,
        "decision": (
            "DO_NOT_AUTHORIZE_AGQA_TYPED_EXECUTOR;KEEP_UNIFIED_HARNESS_ON_"
            "TARGET_NATIVE_DIRECT_FALLBACK"
        ),
        "untouched_formal_claim": False,
        "source_provenance_claim": False,
    }
    return core | {"summary_sha256": stable_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "runs/agqa2_frame_grounding_v2_development/report.json",
    )
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_frame_grounding_v2_development.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "docs/results/agqa2_frame_grounding_v2_summary.json",
    )
    args = parser.parse_args()
    summary = summarize(args.report.resolve(), args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "metrics": summary["metrics"],
        "summary_sha256": summary["summary_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
