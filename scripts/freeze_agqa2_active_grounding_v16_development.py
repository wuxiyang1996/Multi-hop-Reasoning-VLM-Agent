#!/usr/bin/env python3
"""Freeze V16 development on the consumed V15 replication receipts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


def _compact_v15_result() -> Path:
    report_path = REPO_ROOT / "runs/agqa2_active_grounding_v15_replication/report.json"
    report = json.loads(report_path.read_text())
    expected_failed_gate = {
        key for key, passed in report["qualification_gates"].items() if not passed
    }
    if report.get("grounder_qualified") or expected_failed_gate != {
        "no_typed_vs_direct_losses"
    }:
        raise ValueError(f"unexpected V15 result state: {expected_failed_gate}")
    negative_rows = [
        {
            "task_id": row["task_id"],
            "video_id": row["video_id"],
            "route": row["oracle_route_evaluator_only"],
            "comparison": row["query_plan"]["comparison"],
            "authorization_reason": row["calibrated_target_native_execution"]["reason"],
            "typed_prediction": row["typed_fallback_prediction"],
            "direct_response": row["direct_response"],
            "gold_answer": row["gold_answer_evaluator_only"],
        }
        for row in report["rows"]
        if not row["typed_fallback_correct"] and row["direct_correct"]
    ]
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "accepted_runtime_provider_calls",
        "accepted_runtime_reported_provider_cost_usd", "provider_calls",
        "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-active-grounding-v15-replication-result",
        "report_file_sha256": _sha256(report_path),
        "failed_gates": sorted(expected_failed_gate),
        "negative_transfer_rows": negative_rows,
        "interpretation": (
            "AVERAGE_GAIN_PRESENT_BUT_ZERO_NEGATIVE_TRANSFER_GATE_FAILED;"
            "V15_IS_DEVELOPMENT_ONLY_FOR_V16_SELECTIVE_OVERRIDE_RULES"
        ),
    })
    result = core | {"result_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_active_grounding_v15_replication_result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v16_development"
    if (
        (run_root / "report.json").is_file()
        or any((run_root / "runtime_receipts").glob("*.json"))
    ):
        raise RuntimeError("V16 development is already consumed")
    v15_result_path = _compact_v15_result()

    parent_manifest_path = (
        REPO_ROOT / "configs/agqa2_active_grounding_v15_replication_manifest.json"
    )
    parent = _verified_json(parent_manifest_path, "manifest_sha256")
    manifest_core = {
        key: deepcopy(value) for key, value in parent.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v16-development",
        "status": "FROZEN_V16_CONSUMED_V15_SELECTIVE_OVERRIDE_DEVELOPMENT",
        "split": "development",
        "claim_boundary": (
            "EXACT_CONSUMED_V15_36_CANDIDATE_POOL;ONLY_SELECTIVE_OVERRIDE_"
            "SOUNDNESS_RULES_MAY_CHANGE;NO_NEW_NEURAL_ACQUISITION"
        ),
        "selection_rule": "REUSE_EXACT_CONSUMED_V15_CANDIDATES_AND_RECEIPTS",
        "parent_v15_manifest_sha256": parent["manifest_sha256"],
        "v15_result_file_sha256": _sha256(v15_result_path),
        "new_video_downloads": 0,
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base_config_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_replication.json"
    base_config = json.loads(base_config_path.read_text())
    config = deepcopy(base_config)
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v16",
        "status": "FROZEN_V16_SELECTIVE_OVERRIDE_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V16_DEVELOPMENT_RUNTIME_RECEIPT"
        ),
        "report_version": "V16",
    })
    config.pop("development_qualification_report", None)
    config.pop("development_qualification_file_sha256", None)
    config["execution_calibration"].update({
        "minimum_exists_override_confidence": 0.8,
        "require_globally_separated_order_override": True,
        "new_v16_soundness_conditions": [
            "EXISTS_OVERRIDE_MINIMUM_SUPPORTED_EVENT_CONFIDENCE",
            "ORDER_OVERRIDE_GLOBAL_INTERVAL_SEPARATION",
        ],
    })
    for label in ("module", "collector", "executor"):
        config["grounder"][f"{label}_sha256"] = _sha256(
            REPO_ROOT / config["grounder"][label]
        )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    if config["acquisition"] != base_config["acquisition"]:
        raise AssertionError("V16 changed neural acquisition")
    if config["runtime_selection"] != base_config["runtime_selection"]:
        raise AssertionError("V16 changed the development selector")
    if config["qualification_gates"] != base_config["qualification_gates"]:
        raise AssertionError("V16 weakened a qualification gate")

    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v16-development",
        "status": "FROZEN_BEFORE_ANY_V16_DEVELOPMENT_RUNTIME_RECEIPT",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "v15_failed_replication_result": str(v15_result_path.relative_to(REPO_ROOT)),
        "v15_failed_replication_result_file_sha256": _sha256(v15_result_path),
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "development_gates": deepcopy(config["qualification_gates"]),
        "accepted_call_replay": {
            "source": "runs/agqa2_active_grounding_v15_replication/call_cache",
            "policy": "REUSE_ONLY_EXACT_INPUT_HASH_MATCHES;NO_NEW_PROVIDER_CALL_EXPECTED",
        },
        "failure_policy": {
            "development": "MUST_PASS_THE_UNWEAKENED_V15_GATES",
            "reserve": "FREEZE_NEW_VIDEO_DISJOINT_POOL_ONLY_AFTER_QUALIFICATION",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_development_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "candidate_count": manifest["sample_count"],
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "config_file_sha256": _sha256(config_path),
        "next": "copy exact V15 input-addressed call cache and run V16 development",
    }, indent=2))


if __name__ == "__main__":
    main()
