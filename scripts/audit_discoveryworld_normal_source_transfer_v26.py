#!/usr/bin/env python3
"""Independent receipt audit for the V26 DiscoveryWorld Normal formal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_normal_transfer import (  # noqa: E402
    source_role_operator_ids,
    trace_conforms,
)


FORMAL_ROOT = REPO / "runs/discoveryworld_normal_source_transfer_v26_formal"
FORMAL_CONFIG = REPO / "configs/discoveryworld_normal_source_transfer_v26_formal.json"
FORMAL_REPORT = FORMAL_ROOT / "formal_report.json"
QUALIFICATION = REPO / "runs/discoveryworld_normal_source_value_v26_qualification_artifacts/qualification_report.json"
SOURCE = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> bool:
    body = dict(value)
    claimed = body.pop(field, None)
    return bool(claimed) and claimed == stable_hash(body)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_audit() -> dict[str, Any]:
    config = _read(FORMAL_CONFIG)
    report = _read(FORMAL_REPORT)
    qualification = _read(QUALIFICATION)
    source = _read(SOURCE)
    conditions = list(map(str, config["conditions"]))
    seeds = [int(task["seed"]) for task in config["tasks"]]
    roles = source_role_operator_ids(source)
    receipts: dict[str, dict[int, dict[str, Any]]] = {condition: {} for condition in conditions}
    monitors: dict[str, dict[int, dict[str, Any]]] = {condition: {} for condition in conditions}
    episode_hashes_valid = True
    monitor_hashes_valid = True
    runtime_lineage_valid = True
    task_identities_valid = True
    for condition in conditions:
        for seed in seeds:
            episode = _read(FORMAL_ROOT / condition / f"proteomics.normal.seed{seed}.json")
            monitor = _read(FORMAL_ROOT / condition / f"proteomics.normal.seed{seed}.monitor.json")
            episode_hashes_valid &= _self_hash(episode, "episode_sha256")
            monitor_hashes_valid &= _self_hash(monitor, "monitor_sha256")
            runtime_lineage_valid &= episode["runtime_hashes"] == {
                **report["runtime_hashes"], "condition": condition,
            }
            task_identities_valid &= episode["task"] == {
                "scenario": "Proteomics", "difficulty": "Normal", "seed": seed,
            }
            receipts[condition][seed] = episode
            monitors[condition][seed] = monitor

    common_random_numbers = all(
        len({receipts[condition][seed]["initial_policy_state_sha256"] for condition in conditions}) == 1
        and len({receipts[condition][seed]["initial_audit_world_sha256"] for condition in conditions}) == 1
        for seed in seeds
    )
    authentic_neural_actions_match = all(
        [step["action"] for step in receipts["authentic_source"][seed]["steps"]]
        == [step["action"] for step in receipts["neural_only"][seed]["steps"]]
        for seed in seeds
    )
    authentic_program_conformance = True
    authentic_monitor_clean = True
    permuted_first_denial_is_binding = True
    permuted_fails_closed = True
    for seed in seeds:
        authentic_monitor = monitors["authentic_source"][seed]
        sequence = tuple(
            str(roles[row["grounded_role"]])
            for row in authentic_monitor["decisions"]
            if roles[row["grounded_role"]] is not None
        )
        authentic_program_conformance &= trace_conforms(sequence, source)
        authentic_monitor_clean &= (
            authentic_monitor["abstentions"] == 0
            and authentic_monitor["final_phase"] == "DONE"
            and all(row["allowed"] is True for row in authentic_monitor["decisions"])
        )
        permuted_monitor = monitors["source_permuted"][seed]
        denied = [row for row in permuted_monitor["decisions"] if row["allowed"] is False]
        permuted_first_denial_is_binding &= bool(denied) and denied[0]["grounded_role"] == "BINDING"
        permuted_fails_closed &= (
            permuted_monitor["abstentions"] > 0
            and receipts["source_permuted"][seed]["evaluation"]["official_success"] is False
        )
    authentic_successes = sum(
        receipts["authentic_source"][seed]["evaluation"]["official_success"] for seed in seeds
    )
    neural_successes = sum(
        receipts["neural_only"][seed]["evaluation"]["official_success"] for seed in seeds
    )
    permuted_successes = sum(
        receipts["source_permuted"][seed]["evaluation"]["official_success"] for seed in seeds
    )
    all_zero_oracle = all(
        receipt["policy_runtime_saw_oracle_scorecard"] is False
        for by_seed in receipts.values() for receipt in by_seed.values()
    )
    gates = {
        "formal_config_frozen": config["status"] == "FROZEN_BEFORE_V26_FORMAL_RESET_OR_OUTCOME",
        "formal_report_self_hash": _self_hash(report, "report_sha256"),
        "qualification_report_self_hash": _self_hash(qualification, "report_sha256"),
        "source_artifact_self_hash": _self_hash(source, "artifact_sha256"),
        "qualification_authorized_formal": qualification["all_qualification_gates_passed"] is True,
        "episode_self_hashes": episode_hashes_valid,
        "monitor_self_hashes": monitor_hashes_valid,
        "runtime_lineage": runtime_lineage_valid,
        "task_identities": task_identities_valid,
        "common_random_numbers": common_random_numbers,
        "authentic_and_neural_only_actions_match": authentic_neural_actions_match,
        "authentic_program_conformance": authentic_program_conformance,
        "authentic_monitor_has_no_abstention": authentic_monitor_clean,
        "permuted_first_denial_is_binding": permuted_first_denial_is_binding,
        "permuted_fails_closed": permuted_fails_closed,
        "authentic_full_success": authentic_successes == len(seeds),
        "authentic_nonnegative_vs_neural_only": authentic_successes >= neural_successes,
        "authentic_strictly_beats_permuted": authentic_successes > permuted_successes,
        "zero_oracle_exposure": all_zero_oracle,
        "runner_reported_all_gates": report["all_formal_gates_passed"] is True and all(report["gates"].values()),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "discoveryworld-normal-source-transfer-v26-independent-audit",
        "status": "DISCOVERYWORLD_NORMAL_V26_INDEPENDENT_AUDIT_PASSED" if passed else "DISCOVERYWORLD_NORMAL_V26_INDEPENDENT_AUDIT_FAILED",
        "claim_boundary": (
            "The audit validates prospective lineage, online typed-program conformance, common random numbers, nonnegative transfer, and destructive-control separation. "
            "Because authentic and neural-only action traces and success are identical, it explicitly rejects an incremental success-rate claim."
        ),
        "metrics": {
            "tasks_per_condition": len(seeds),
            "authentic_source_successes": authentic_successes,
            "neural_only_successes": neural_successes,
            "source_permuted_successes": permuted_successes,
            "authentic_minus_neural_only": authentic_successes - neural_successes,
            "authentic_minus_source_permuted": authentic_successes - permuted_successes,
            "paired_one_sided_sign_test_p_authentic_vs_permuted": 2.0 ** -len(seeds),
            "complete_target_trajectories_replaced": qualification["metrics"]["complete_target_trajectories_replaced"],
            "qualification_grounder_exact_accuracy": qualification["metrics"]["grounding"]["exact_accuracy"],
        },
        "interpretation": {
            "program_transfer_validated": passed,
            "incremental_success_rate_gain_validated": False,
            "source_provenance_identifiable_from_behavior_alone": False,
            "source_information_value": "ONE_COMPLETE_ORDERED_SUCCESSFUL_TARGET_TRAJECTORY",
            "what_transferred": "SOURCE_INDUCED_ACQUISITION_BINDING_RELATION_PROGRAM",
            "what_remained_target_native": "PROTEOMICS_PERCEPTION_ACTION_CANDIDATES_AND_NEURAL_GROUNDING",
        },
        "gates": gates,
        "all_audit_gates_passed": passed,
        "integrity": {
            "formal_config_file_sha256": _sha(FORMAL_CONFIG),
            "formal_report_sha256": report["report_sha256"],
            "qualification_report_sha256": qualification["report_sha256"],
            "source_artifact_sha256": source["artifact_sha256"],
        },
    }
    return body | {"audit_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"], "metrics": audit["metrics"],
        "failed_gates": sorted(key for key, value in audit["gates"].items() if not value),
        "audit_sha256": audit["audit_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if audit["all_audit_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
