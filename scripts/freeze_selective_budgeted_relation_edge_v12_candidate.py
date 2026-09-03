#!/usr/bin/env python3
"""Freeze a V12 consumed replay or fresh-gate candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.contracts import stable_hash
from motif_transfer.slot_aware_alfworld_harness_v12 import (
    CONDITION_SEMANTICS,
    MINIMUM_SOURCE_EDGE_STEP,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen artifact hash: {field}")


def _receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-candidate", type=Path, required=True)
    parser.add_argument("--applicability-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--authority",
        choices=("consumed_replay", "fresh_adaptation"),
        required=True,
    )
    parser.add_argument("--development-report", type=Path, action="append")
    parser.add_argument("--development-summary", type=Path)
    parser.add_argument("--v12-harness-code", type=Path, required=True)
    parser.add_argument("--v10-harness-code", type=Path, required=True)
    parser.add_argument("--v9-graph-code", type=Path, required=True)
    parser.add_argument("--slot-ledger-code", type=Path, required=True)
    parser.add_argument("--v12-runner-code", type=Path, required=True)
    parser.add_argument("--shared-v9-runner-code", type=Path, required=True)
    parser.add_argument("--shared-v8-runner-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--runner-seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V12 candidate: {args.output}")
    parent = _read(args.v11_candidate)
    _validate_hash(parent, "candidate_sha256")
    if parent.get("schema_version") != (
        "budgeted-relation-edge-alfworld-candidate-v11"
    ):
        raise SystemExit("wrong V11 parent candidate")
    audit = _read(args.applicability_audit)
    audit_body = dict(audit)
    audit_hash = str(audit_body.pop("audit_sha256", ""))
    if stable_hash(audit_body) != audit_hash:
        raise SystemExit("invalid V12 applicability audit hash")
    if audit.get("status") != "CONSUMED_CROSS_VERSION_AUDIT_PASSED":
        raise SystemExit("V12 applicability audit did not pass")
    if int(audit["selected_minimum_edge_step"]) != (
        MINIMUM_SOURCE_EDGE_STEP
    ):
        raise SystemExit("V12 code differs from audited applicability rule")
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V12 manifest selection used target outcomes")
    development_reports = []
    for path in args.development_report or []:
        report = _read(path)
        _validate_hash(report, "report_sha256")
        if report.get("existing_valid_unseen_heldout_read"):
            raise SystemExit("development report crossed heldout boundary")
        development_reports.append(_receipt(path) | {
            "report_sha256": report["report_sha256"],
            "status": report["status"],
            "use_authority": "CONSUMED_DEVELOPMENT_ONLY",
        })
    development_summary = None
    if args.development_summary:
        summary = _read(args.development_summary)
        summary_body = dict(summary)
        summary_hash = str(summary_body.pop("development_sha256", ""))
        if stable_hash(summary_body) != summary_hash:
            raise SystemExit("invalid V12 development summary hash")
        if summary.get("status") != "CONSUMED_CLOSED_LOOP_REPLAY_PASSED":
            raise SystemExit("V12 consumed closed-loop replay did not pass")
        development_summary = _receipt(args.development_summary) | {
            "development_sha256": summary_hash,
            "use_authority": "AUTHORIZE_ONE_FRESH_V12_ADAPTATION_FREEZE",
        }
    if args.authority == "fresh_adaptation" and not development_summary:
        raise SystemExit("fresh V12 candidate requires passing replay summary")
    body = dict(parent)
    parent_hash = str(body.pop("candidate_sha256"))
    body["schema_version"] = (
        "selective-budgeted-relation-edge-alfworld-candidate-v12"
    )
    body["experiment_version"] = "v12"
    body["status"] = "ADAPTATION_GATE_ONLY"
    body["claim_boundary"] = (
        "SELECTIVE_EDGE_RULE_FROM_CONSUMED_GROUPED_CROSS_VERSION_AUDIT; "
        + (
            "CONSUMED_TASK_REPLAY_ONLY; NO_FRESH_EVIDENCE"
            if args.authority == "consumed_replay"
            else "FROZEN_BEFORE_FRESH_V12_ADAPTATION_RESET; CONFIRMATION_"
                 "FORBIDDEN_UNTIL_GATE_PASSES"
        )
        + "; EXISTING_VALID_UNSEEN_FORBIDDEN"
    )
    body["v11_parent_candidate"] = _receipt(args.v11_candidate) | {
        "candidate_sha256": parent_hash
    }
    body["applicability_audit"] = _receipt(
        args.applicability_audit
    ) | {
        "audit_sha256": audit_hash,
        "selected_minimum_edge_step": MINIMUM_SOURCE_EDGE_STEP,
        "use_authority": "CONSUMED_DEVELOPMENT_RULE_SELECTION",
    }
    body["development_reports"] = development_reports
    body["development_summary"] = development_summary
    body["candidate_authority"] = args.authority.upper()
    body["manifest"] = _receipt(args.manifest) | {
        "manifest_sha256": manifest["manifest_sha256"]
    }
    body["manifest_schema"] = str(manifest["schema_version"])
    body["manifest_status"] = str(manifest["status"])
    body["implementation"] = {
        "v12_harness": _receipt(args.v12_harness_code),
        "v10_control_harness": _receipt(args.v10_harness_code),
        "v9_graph_executor": _receipt(args.v9_graph_code),
        "slot_ledger": _receipt(args.slot_ledger_code),
        "v12_runner": _receipt(args.v12_runner_code),
        "shared_v9_runner": _receipt(args.shared_v9_runner_code),
        "shared_v8_runner_helpers": _receipt(
            args.shared_v8_runner_code
        ),
    }
    body["experiment_parameters"] = dict(body["experiment_parameters"])
    body["experiment_parameters"].update({
        "max_steps": args.max_steps,
        "runner_seed": args.runner_seed,
        "minimum_source_edge_step": MINIMUM_SOURCE_EDGE_STEP,
        "applicability_feature": "TARGET_EPISODE_STEP_INDEX",
        "applicability_selection_authority": (
            "CONSUMED_V9_V10_V11_GROUPED_CROSS_VERSION_AUDIT"
        ),
    })
    body["condition_semantics"] = CONDITION_SEMANTICS
    body["transfer_scope"] = dict(body["transfer_scope"])
    body["transfer_scope"]["name"] = (
        "SELECTIVE_BUDGETED_EXECUTED_BIND_TO_RELATE_SOURCE_EDGE"
    )
    body["permissions"] = dict(body["permissions"])
    body["permissions"]["forbidden"] = list(
        body["permissions"]["forbidden"]
    ) + [
        "CHANGE_MINIMUM_SOURCE_EDGE_STEP_AFTER_FREEZE",
        "USE_CONSUMED_REPLAY_AS_FRESH_EVIDENCE",
    ]
    result = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": result["candidate_sha256"],
        "authority": args.authority,
        "manifest_sha256": manifest["manifest_sha256"],
        "max_steps": args.max_steps,
        "runner_seed": args.runner_seed,
        "minimum_source_edge_step": MINIMUM_SOURCE_EDGE_STEP,
        "confirmation_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
