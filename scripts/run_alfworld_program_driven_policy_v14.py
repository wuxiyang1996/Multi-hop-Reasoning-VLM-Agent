#!/usr/bin/env python3
"""Run the prospective ALFWorld program-driven policy V14 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_unified_goal_acquisition_v11 as frozen  # noqa: E402
import run_alfworld_unified_goal_acquisition_v13 as v13  # noqa: E402
from motif_transfer.alfworld_policy_contribution import (  # noqa: E402
    audit_policy_contribution,
)


V14_STATUS = "FROZEN_BEFORE_ANY_ALFWORLD_V14_RESERVE_RESET_OR_OUTCOME"


def _normalize_and_audit(report: dict, config: dict) -> dict:
    normalized = v13._normalize_report(report, config)
    body = dict(normalized)
    body.pop("report_sha256", None)
    contribution = audit_policy_contribution(normalized)
    minimum_divergences = int(config["gates"][
        "minimum_source_divergent_actions"
    ])
    contribution_gates = {
        "program_driven_policy_contribution_gates_pass": all(
            contribution["gates"].values()
        ),
        "minimum_source_divergent_actions_observed": (
            int(contribution["source_divergent_actions"])
            >= minimum_divergences
        ),
        "every_rescue_has_causal_policy_bridge": (
            bool(contribution["rescued_task_audits"])
            and all(
                row["acquisition_divergence_before_terminal"]
                and row["terminal_source_transition_reaches_success"]
                and row["target_native_authority_receipts_align"]
                for row in contribution["rescued_task_audits"]
            )
        ),
    }
    body["schema_version"] = "alfworld-program-driven-policy-report-v14"
    body["role"] = "prospective_untouched_policy_contribution_replication"
    body["v14_all_remaining_untouched_tasks"] = True
    body["v14_action_runtime"] = "V11_FROZEN_ACTION_RUNTIME_UNCHANGED"
    body["v14_transport_runtime"] = "V13_VALID_TRAIN_TRANSPORT_UNCHANGED"
    body["policy_contribution"] = contribution
    body["gates"] = dict(body["gates"]) | contribution_gates
    body["status"] = (
        "ALFWORLD_PROGRAM_DRIVEN_POLICY_V14_FORMAL_VALIDATED"
        if all(body["gates"].values()) else
        "ALFWORLD_PROGRAM_DRIVEN_POLICY_V14_FORMAL_FAILED"
    )
    return body | {"report_sha256": frozen.stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = frozen._read(args.config)
    frozen._self_hash(config, "config_sha256")
    if config.get("status") != V14_STATUS:
        raise ValueError("expected a frozen ALFWorld V14 replication config")

    checks = {
        "v14_launcher_file_sha256": Path(__file__).resolve(),
        "policy_contribution_file_sha256": REPO / (
            "src/motif_transfer/alfworld_policy_contribution.py"
        ),
        "v13_launcher_file_sha256": REPO / (
            "scripts/run_alfworld_unified_goal_acquisition_v13.py"
        ),
        "valid_train_transport_config_file_sha256": REPO / str(
            config["valid_train_transport_config"]
        ),
    }
    for field, path in checks.items():
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if config.get(field) != observed:
            raise ValueError(f"V14 frozen dependency changed: {path}")

    frozen.FORMAL_STATUS = V14_STATUS
    frozen.ALFWorldTextBatchEnvironment = v13._ValidTrainBatchEnvironment
    report = _normalize_and_audit(frozen.run(args.config), config)
    output = REPO / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": report["summaries"],
        "paired": report["paired"],
        "policy_contribution": report["policy_contribution"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
