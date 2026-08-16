#!/usr/bin/env python3
"""Freeze selective V2 after V1 coverage failure and before matched arms."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_discoveryworld_utility_v2 import (  # noqa: E402
    SCHEMA, STATUS, file_sha256, read_object,
)


OUTPUT = REPO / "configs/phase2_discoveryworld_utility_v2/manifest.json"
V1_ROOT = REPO / "runs/phase2_discoveryworld_utility_v1"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {OUTPUT}")
    if (V1_ROOT / "cells").exists() and any((V1_ROOT / "cells").glob("*/matched_result.json")):
        raise SystemExit("matched outcomes already exist; V2 freeze is no longer prospective")
    v1_manifest_path = REPO / "configs/phase2_discoveryworld_utility_v1/manifest.json"
    v1 = read_object(v1_manifest_path)
    fork_path = V1_ROOT / "frozen_forks/fork_freeze_receipt.json"
    fork_receipt = read_object(fork_path)
    if fork_receipt.get("outcome_fields_read_for_eligibility") is not False:
        raise SystemExit("fork eligibility was not outcome blind")
    eligible_ids = {
        row["task_id"] for row in fork_receipt["receipts"] if row["eligible"]
    }
    if len(eligible_ids) != 35:
        raise SystemExit(f"expected exact V1 coverage failure 35/36, got {len(eligible_ids)}")
    tasks = []
    for row in v1["tasks"]:
        task_id = row["task_id"]
        episode = V1_ROOT / "target_only" / f"{task_id}.json"
        applicable = task_id in eligible_ids
        body = dict(row)
        body.update({
            "applicable": applicable,
            "target_episode": str(episode.relative_to(REPO)),
            "target_episode_file_sha256": file_sha256(episode),
        })
        if applicable:
            fork = V1_ROOT / "frozen_forks" / f"{task_id}.json"
            body.update({
                "fork_config": str(fork.relative_to(REPO)),
                "fork_config_file_sha256": file_sha256(fork),
                "abstention_rule": None,
            })
        else:
            body.update({
                "fork_config": None,
                "fork_config_file_sha256": None,
                "abstention_rule": "INHERIT_RECORDED_TARGET_ONLY_OUTCOME_FOR_ALL_ARMS",
            })
        tasks.append(body)
    runtime_files = (
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/discoveryworld_env.py",
        "src/motif_transfer/discoveryworld_policy.py",
        "src/motif_transfer/discoveryworld_sokoban_transfer.py",
        "src/motif_transfer/discoveryworld_applicability_grounder_v4.py",
        "src/motif_transfer/search_automaton_transfer_v16.py",
        "src/motif_transfer/phase2_discoveryworld_utility_v2.py",
        "scripts/run_discoveryworld_commit_recovery_v1.py",
        "scripts/run_phase1_direct_discoveryworld_v1.py",
        "scripts/freeze_phase2_discoveryworld_utility_v2.py",
        "scripts/run_phase2_discoveryworld_utility_v2.py",
        "scripts/verify_phase2_discoveryworld_utility_v2.py",
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": (
            "Selective causal utility on all 35 outcome-blind eligible first-commit forks "
            "from the fixed 36-task V1 cohort, with the sole inapplicable task failing closed."
        ),
        "parent_v1_manifest_sha256": v1["manifest_sha256"],
        "parent_v1_manifest_file_sha256": file_sha256(v1_manifest_path),
        "v1_coverage_result": "FAILED_35_OF_36_ELIGIBLE",
        "v1_matched_arms_executed_before_v2_freeze": 0,
        "matched_outcomes_visible_at_freeze": False,
        "eligibility_read_target_outcome": False,
        "target_acquisition_outcomes_already_consumed": True,
        "fork_freeze_receipt_file_sha256": file_sha256(fork_path),
        "fork_freeze_summary_sha256": fork_receipt["summary_sha256"],
        "conditions": list(v1["conditions"]),
        "primary_endpoint": {
            "metric": "official_success_after_matched_recovery_with_fail_closed_abstention",
            "test": "exact_two_sided_paired_sign_test",
            "maximum_p": 0.05,
            "maximum_negative_rate": 0.25,
        },
        "tasks": tasks,
        "runtime_file_sha256": {path: file_sha256(REPO / path) for path in runtime_files},
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "tasks": len(tasks),
        "applicable": sum(row["applicable"] for row in tasks),
        "abstentions": sum(not row["applicable"] for row in tasks),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
