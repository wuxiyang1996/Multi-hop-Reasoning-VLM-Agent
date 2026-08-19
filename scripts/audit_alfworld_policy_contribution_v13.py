#!/usr/bin/env python3
"""Independently audit program-driven policy contribution in ALFWorld V13."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_policy_contribution import (  # noqa: E402
    audit_policy_contribution,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/alfworld_unified_goal_acquisition_v13_formal.json"
DEFAULT_REPORT = REPO / "runs/alfworld_unified_goal_acquisition_v13_formal/report.json"
DEFAULT_OUTPUT = REPO / "docs/results/alfworld_policy_contribution_v13_audit.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if claimed != stable_hash(body):
        raise ValueError(f"invalid {field}: {claimed}")


def _sha(path: Path) -> str:
    return hashlib.sha256(_bytes(path)).hexdigest()


def _bytes(path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    archive = Path(str(path) + ".gz")
    if not archive.is_file():
        raise FileNotFoundError(path)
    return gzip.decompress(archive.read_bytes())


def build_audit(config_path: Path, report_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    report = _read(report_path)
    _self_hash(config, "config_sha256")
    _self_hash(report, "report_sha256")
    if report["config_sha256"] != config["config_sha256"]:
        raise ValueError("ALFWorld V13 report/config lineage mismatch")
    if set(report["task_ids"]) != set(config["task_ids"]):
        raise ValueError("ALFWorld V13 task identities differ from preregistration")
    for rows in report["episodes"].values():
        for episode in rows:
            _self_hash(episode, "episode_sha256")
            for record in episode["records"]:
                _self_hash(record, "record_sha256")
    for task_id, receipts in report["authority_receipts"].items():
        phase7_sha = report["phase7_authorizations"][task_id][
            "authorization_sha256"
        ]
        for receipt in receipts:
            _self_hash(receipt, "receipt_sha256")
            if receipt["phase7_authorization_sha256"] != phase7_sha:
                raise ValueError("authority receipt escaped its task authorization")

    contribution = audit_policy_contribution(report)
    gates = {
        "v13_original_gates_pass": all(report["gates"].values()),
        "all_policy_contribution_gates_pass": all(
            contribution["gates"].values()
        ),
        "all_seven_v13_rescues_have_causal_bridge": (
            contribution["rescues"] == 7
            and all(
                row["acquisition_divergence_before_terminal"]
                and row["terminal_source_transition_reaches_success"]
                for row in contribution["rescued_task_audits"]
            )
        ),
    }
    body = {
        "schema_version": "alfworld-policy-contribution-audit-v13",
        "status": (
            "ALFWORLD_V13_PROGRAM_DRIVEN_POLICY_CONTRIBUTION_VALIDATED"
            if all(gates.values()) else
            "ALFWORLD_V13_PROGRAM_DRIVEN_POLICY_CONTRIBUTION_FAILED"
        ),
        "claim_boundary": (
            "Retrospective causal audit of a prospectively frozen V13 run. "
            "The source-induced IR selects anonymous acquisition/relation "
            "options and changes the policy trajectory; the target-native "
            "grounder and executor alone bind and emit ALFWorld actions. This "
            "audit does not turn the V13 outcomes into a new prospective test."
        ),
        "config_file_sha256": _sha(config_path),
        "config_sha256": config["config_sha256"],
        "report_file_sha256": _sha(report_path),
        "report_sha256": report["report_sha256"],
        "contribution": contribution,
        "gates": gates,
    }
    return body | {"audit_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit(args.config, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": audit["status"],
        "contribution": audit["contribution"],
        "gates": audit["gates"],
        "audit_sha256": audit["audit_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(audit["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
