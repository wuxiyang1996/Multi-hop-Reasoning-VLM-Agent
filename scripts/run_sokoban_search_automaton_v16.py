#!/usr/bin/env python3
"""Induce and fresh-confirm the source-only Sokoban search automaton V16."""

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
from motif_transfer.sokoban_commit_skill import (  # noqa: E402
    build_fresh_confirmation_plan,
)
from motif_transfer.sokoban_search_automaton_v16 import (  # noqa: E402
    matched_decision_rows,
    source_state_receipts,
    summarize_source_gate,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_plan(config: dict[str, Any], key: str) -> dict[str, Any]:
    spec = config[key]
    return build_fresh_confirmation_plan(
        seeds=range(int(spec["start"]), int(spec["stop_exclusive"])),
        snapshots_per_episode=int(spec["snapshots_per_episode"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/sokoban_search_automaton_v16.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/sokoban_search_automaton_v16",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=REPO / "docs/results/sokoban_search_automaton_v16_summary.json",
    )
    args = parser.parse_args()
    config = _read(args.config)
    if config.get("status") != (
        "FROZEN_BEFORE_FRESH_CONFIRMATION_GENERATION_OR_READ"
    ):
        raise SystemExit("source V16 config is not frozen at the required boundary")
    if config.get("target_authorized_before_source_gate") is not False:
        raise SystemExit("source V16 config improperly authorizes target execution")

    discovery_path = REPO / str(config["discovery_plan"])
    discovery_plan = _read(discovery_path)
    calibration_plan = _seed_plan(config, "calibration_seeds")
    fresh_plan = _seed_plan(config, "fresh_confirmation_seeds")
    discovery = source_state_receipts(discovery_plan)
    calibration = source_state_receipts(calibration_plan)
    fresh = source_state_receipts(fresh_plan)
    gate = summarize_source_gate(
        discovery_receipts=discovery,
        calibration_receipts=calibration,
        fresh_receipts=fresh,
        requirements=config["source_gate"],
    )

    artifact_body = {
        "schema_version": "sokoban-search-automaton-artifact-v16",
        "status": (
            "SOURCE_SEARCH_AUTOMATON_FROZEN"
            if gate["source_gate_passed"] else "SOURCE_SEARCH_AUTOMATON_REJECTED"
        ),
        "claim_boundary": config["claim_boundary"],
        "learned_policy": gate["learned_policy"],
        "transfer_contract": {
            "state": "ABSTRACT_EVENT_PLUS_ATTEMPT_LEDGER",
            "advance_only_after_observed_effect": True,
            "unknown_event": "ABSTAIN",
            "target_permission": (
                "TARGET_BINDS_EVENTS_CANDIDATES_AND_NATIVE_ACTIONS_FROM_ITS_OWN_"
                "ADAPTATION_RECEIPTS; SOURCE_PATHS_TOKENS_AND_ORDER_FORBIDDEN"
            ),
        },
        "source_lineage": {
            "config_file_sha256": _sha256(args.config),
            "discovery_plan_file_sha256": _sha256(discovery_path),
            "discovery_plan_sha256": discovery_plan["plan_sha256"],
            "calibration_plan_sha256": calibration_plan["plan_sha256"],
            "fresh_confirmation_plan_sha256": fresh_plan["plan_sha256"],
        },
        "target_authorized": gate["source_gate_passed"],
    }
    artifact = artifact_body | {"artifact_sha256": stable_hash(artifact_body)}
    report_body = {
        "schema_version": "sokoban-search-automaton-report-v16",
        "status": (
            "FRESH_SOURCE_SEARCH_AUTOMATON_CONFIRMED"
            if gate["source_gate_passed"] else "FRESH_SOURCE_GATE_FAILED_STOP"
        ),
        "claim_boundary": config["claim_boundary"],
        "artifact_sha256": artifact["artifact_sha256"],
        **gate,
        "source_receipt_counts": {
            "discovery_matched_action_rows": len(matched_decision_rows(discovery)),
            "calibration_matched_action_rows": len(matched_decision_rows(calibration)),
            "fresh_matched_action_rows": len(matched_decision_rows(fresh)),
        },
        "fresh_receipt_examples": fresh[:6],
        "target_data_read_or_run": False,
        "provider_calls": 0,
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    compact = {
        key: value for key, value in report.items()
        if key != "fresh_receipt_examples"
    }
    compact["artifact"] = artifact
    compact["fresh_confirmation_plan_sha256"] = fresh_plan["plan_sha256"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        (args.output_dir / "artifact.json", artifact),
        (args.output_dir / "fresh_confirmation_plan.json", fresh_plan),
        (args.output_dir / "fresh_confirmation_report.json", report),
        (args.summary, compact),
    )
    for path, payload in outputs:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": report["status"],
        "source_gate_passed": report["source_gate_passed"],
        "split_counts": report["split_counts"],
        "fresh_branch_advantages": report["fresh_branch_advantages"],
        "fresh_policy_metrics": report["fresh_policy_metrics"],
        "gates": report["gates"],
        "artifact_sha256": artifact["artifact_sha256"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    if not gate["source_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
