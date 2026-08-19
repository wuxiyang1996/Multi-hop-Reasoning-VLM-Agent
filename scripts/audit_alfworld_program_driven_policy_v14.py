#!/usr/bin/env python3
"""Independently recompute the prospective ALFWorld V14 policy claim."""

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

from motif_transfer.alfworld_goal_acquisition_v10 import (  # noqa: E402
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
)
from motif_transfer.alfworld_policy_contribution import (  # noqa: E402
    audit_policy_contribution,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/alfworld_program_driven_policy_v14_formal.json"
DEFAULT_REPORT = REPO / "runs/alfworld_program_driven_policy_v14_formal/report.json"
DEFAULT_OUTPUT = REPO / "docs/results/alfworld_program_driven_policy_v14_summary.json"


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
        raise ValueError("V14 report/config lineage mismatch")
    if set(report["task_ids"]) != set(config["task_ids"]):
        raise ValueError("V14 report task identities differ from preregistration")

    episode_count = 0
    record_count = 0
    for rows in report["episodes"].values():
        if len(rows) != int(config["task_count"]):
            raise ValueError("V14 has an incomplete matched arm")
        for episode in rows:
            _self_hash(episode, "episode_sha256")
            episode_count += 1
            for record in episode["records"]:
                _self_hash(record, "record_sha256")
                record_count += 1
    receipt_count = 0
    for task_id, receipts in report["authority_receipts"].items():
        phase7_sha = report["phase7_authorizations"][task_id][
            "authorization_sha256"
        ]
        for receipt in receipts:
            _self_hash(receipt, "receipt_sha256")
            if receipt["phase7_authorization_sha256"] != phase7_sha:
                raise ValueError("authority receipt escaped task authorization")
            receipt_count += 1

    contribution = audit_policy_contribution(report)
    if contribution != report["policy_contribution"]:
        raise ValueError("stored V14 policy contribution is not reproducible")
    successes = {
        "neural_only": report["summaries"][RAW]["successes"],
        "source_induced": report["summaries"][AUTHENTIC]["successes"],
        "source_cardinality_control": report["summaries"][
            CARDINALITY_CONTROL
        ]["successes"],
        "source_effect_permuted": report["summaries"][EFFECT_CONTROL][
            "successes"
        ],
        "generic_scaffold": report["summaries"][GENERIC]["successes"],
        "target_native_ceiling": report["summaries"][CEILING]["successes"],
    }
    paired = report["paired"][RAW]
    gates = {
        "all_frozen_report_gates_pass": all(report["gates"].values()),
        "all_recomputed_policy_gates_pass": all(contribution["gates"].values()),
        "all_21_remaining_tasks_executed": (
            config["selection"]["selected_all_remaining_candidates"] is True
            and int(config["task_count"]) == 21
            and contribution["tasks"] == 21
        ),
        "source_strictly_beats_all_noncausal_controls": all(
            successes["source_induced"] > successes[name]
            for name in (
                "neural_only", "source_cardinality_control",
                "source_effect_permuted", "generic_scaffold",
            )
        ),
        "source_matches_target_native_ceiling": (
            successes["source_induced"] == successes["target_native_ceiling"]
        ),
        "prospective_effect_is_significant_and_nonnegative": (
            int(paired["wins"]) == 7
            and int(paired["losses"]) == 0
            and float(paired["exact_two_sided_p"]) <= 0.05
        ),
        "every_rescue_has_recomputed_causal_bridge": (
            contribution["rescues"] == 7
            and all(
                row["acquisition_divergence_before_terminal"]
                and row["terminal_source_transition_reaches_success"]
                and row["target_native_authority_receipts_align"]
                for row in contribution["rescued_task_audits"]
            )
        ),
    }
    compact_contribution = {
        key: value for key, value in contribution.items()
        if key != "rescued_task_audits"
    } | {
        "rescued_tasks": [
            {
                "task_id": row["task_id"],
                "first_source_divergence_step": row[
                    "first_source_divergence_step"
                ],
                "terminal_transition_step": row["terminal_transition_step"],
                "source_divergent_actions": row["source_divergent_actions"],
            }
            for row in contribution["rescued_task_audits"]
        ],
    }
    body = {
        "schema_version": "alfworld-program-driven-policy-independent-audit-v14",
        "status": (
            "ALFWORLD_PROGRAM_DRIVEN_POLICY_V14_INDEPENDENTLY_VALIDATED"
            if all(gates.values()) else
            "ALFWORLD_PROGRAM_DRIVEN_POLICY_V14_INDEPENDENT_AUDIT_FAILED"
        ),
        "claim_boundary": (
            "Prospective replication on all 21 execution-untouched tasks left "
            "after V13. Source IR controls anonymous option selection; the "
            "frozen target-native grounder/executor controls concrete actions."
        ),
        "config_file_sha256": _sha(config_path),
        "config_sha256": config["config_sha256"],
        "report_file_sha256": _sha(report_path),
        "report_sha256": report["report_sha256"],
        "verified_hashes": {
            "episodes": episode_count,
            "records": record_count,
            "authority_receipts": receipt_count,
        },
        "successes": successes,
        "paired_vs_neural_only": dict(paired),
        "policy_contribution": compact_contribution,
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
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if all(audit["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
