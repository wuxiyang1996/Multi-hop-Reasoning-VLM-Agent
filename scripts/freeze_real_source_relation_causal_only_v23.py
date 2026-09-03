#!/usr/bin/env python3
"""Freeze the V21 secondary causal-only policy as a new V23 hypothesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v21-candidate", type=Path, required=True)
    parser.add_argument("--v21-report", type=Path, required=True)
    parser.add_argument("--v20-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V23 candidate: {args.output}")
    candidate = _read(args.v21_candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    report = _read(args.v21_report)
    report_hash = _validate_hash(report, "report_sha256")
    manifest = _read(args.v20_manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if report.get("status") != "UTILITY_REQUALIFICATION_FAILED_STOP":
        raise SystemExit("V23 requires the completed fail-closed V21 report")
    if report.get("development_authorized"):
        raise SystemExit("V21 unexpectedly authorized development")
    secondary = report["policy_metrics"]["causal_effect_only"]
    primary = report["policy_metrics"]["v20_selective"]
    if not (
        secondary["success_delta"] > 0
        and secondary["success_wins"] >= 5
        and secondary["source_event_recall"] >= 0.95
    ):
        raise SystemExit("V21 causal-only secondary evidence is not positive")
    body = {
        "schema_version": "real-source-relation-causal-only-candidate-v23",
        "status": "CAUSAL_ONLY_DEVELOPMENT_GATE_AUTHORIZED",
        "claim_boundary": (
            "NEW_POST_V21_HYPOTHESIS; V21_PRIMARY_SELECTIVE_POLICY_FAILED; "
            "SECONDARY_CAUSAL_ONLY_POLICY_FROZEN BEFORE_V20_DEVELOPMENT; "
            "NO_CLAIM_FROM_V21_P_VALUE; DEVELOPMENT_AND_CONFIRMATION_UNREAD"
        ),
        "v21_candidate": {
            "path": str(args.v21_candidate.resolve()),
            "file_sha256": _sha256(args.v21_candidate),
            "candidate_sha256": candidate_hash,
        },
        "v21_report": {
            "path": str(args.v21_report.resolve()),
            "file_sha256": _sha256(args.v21_report),
            "report_sha256": report_hash,
            "primary_failure_preserved": {
                "policy": "v20_selective",
                "success_wins": primary["success_wins"],
                "success_losses": primary["success_losses"],
                "success_delta": primary["success_delta"],
                "one_sided_exact_sign_p": primary["one_sided_exact_sign_p"],
            },
            "secondary_motivating_evidence_not_confirmation": {
                "policy": "causal_effect_only",
                "success_wins": secondary["success_wins"],
                "success_losses": secondary["success_losses"],
                "success_delta": secondary["success_delta"],
                "one_sided_exact_sign_p": secondary["one_sided_exact_sign_p"],
            },
        },
        "manifest": {
            "path": str(args.v20_manifest.resolve()),
            "file_sha256": _sha256(args.v20_manifest),
            "manifest_sha256": manifest_hash,
        },
        "parent_candidate": candidate["parent_candidate"],
        "source_summary": candidate["source_summary"],
        "target_causal_effect_head": candidate["target_causal_effect_head"],
        "target_causal_effect_metrics": candidate["target_causal_effect_metrics"],
        "target_incremental_utility_head": candidate[
            "target_incremental_utility_head"
        ],
        "selective_risk_calibration": candidate["selective_risk_calibration"],
        "primary_policy": {
            "name": "v23_causal_only",
            "admit_when": (
                "source_causal_effect_probability>=0.5 AND causal_effect_margin>0"
            ),
            "uses_target_outcome_at_inference": False,
            "uses_source_action_or_coordinate": False,
        },
        "development_gates": {
            "minimum_opportunities": 12,
            "minimum_success_wins": 4,
            "one_sided_exact_sign_alpha": 0.10,
            "success_delta_strictly_positive": True,
            "selected_incremental_utility_strictly_positive": True,
            "source_event_recall_at_least": 0.90,
            "all_exact_state_fork_invariants": True,
        },
        "development_authorized": True,
        "confirmation_authorized": False,
        "development_read_or_run": False,
        "confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    output = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": output["candidate_sha256"],
        "status": output["status"],
        "primary_policy": output["primary_policy"],
        "development_gates": output["development_gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
