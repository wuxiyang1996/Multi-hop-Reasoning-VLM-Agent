#!/usr/bin/env python3
"""Freeze adapted V8 development and reuse the uncalled V7 fresh pool."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


DEVELOPMENT_TASK_IDS = (
    "MXATD-1952", "9VF2C-8026", "G2JR9-2207", "D1WYU-22569",
    "ATV2F-105958", "LBRYS-1587", "AVSPS-7277", "OQ54Y-39706",
    "35ZZP-23529", "PTQE0-9191", "JZZH6-4381", "LD9EC-1390",
)


def main() -> None:
    v7_reserve_run = REPO_ROOT / "runs/agqa2_active_grounding_v7_reserve"
    if v7_reserve_run.exists() and any(v7_reserve_run.rglob("*.json")):
        raise RuntimeError("V7 reserve pool was called and cannot become V8 reserve")
    v8_reserve_run = REPO_ROOT / "runs/agqa2_active_grounding_v8_reserve"
    if v8_reserve_run.exists() and any(v8_reserve_run.rglob("*.json")):
        raise RuntimeError("V8 reserve is already consumed")
    v7_result_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v7_development_result.json"
    if json.loads(v7_result_path.read_text())["v7_reserve_provider_calls_started"]:
        raise ValueError("V7 fresh pool is no longer raw-video-unseen")

    manifest_paths = [
        REPO_ROOT / "configs/agqa2_active_grounding_v3_development_manifest.json",
        REPO_ROOT / "configs/agqa2_active_grounding_v5_reserve_manifest.json",
        REPO_ROOT / "configs/agqa2_active_grounding_v6_reserve_manifest.json",
    ]
    parent_manifests = [_verified_json(path, "manifest_sha256") for path in manifest_paths]
    by_task = {
        str(row["task_id"]): dict(row)
        for manifest in parent_manifests for row in manifest["samples"]
    }
    if not set(DEVELOPMENT_TASK_IDS) <= set(by_task):
        raise ValueError("V8 adaptation task is missing from parent manifests")
    samples = [by_task[task_id] for task_id in DEVELOPMENT_TASK_IDS]
    base = parent_manifests[0]
    development_core = {
        "schema_version": "agqa2-active-grounding-manifest-v8",
        "status": "FROZEN_V8_CONSUMED_ADAPTATION_DEVELOPMENT",
        "split": "development",
        "claim_boundary": (
            "CONSUMED_OUTCOME_SELECTED_ADAPTATION_DEVELOPMENT;FOUR_CANDIDATES_"
            "PER_ROUTE;OUTCOME_BLIND_RUNTIME_SELECTOR_EVALUATES_THREE_PER_ROUTE"
        ),
        "selection_rule": (
            "USE_KNOWN_V3_V5_V6_ROWS_TO_COVER_RECURRENT_CONSENSUS_POSITIVE_"
            "NEGATIVE_AND_CONFLICT_CASES;DEVELOPMENT_ONLY_NOT_CONFIRMATORY"
        ),
        "archive_path": base["archive_path"],
        "archive_sha256": base["archive_sha256"],
        "entry": base["entry"],
        "video_root": base["video_root"],
        "per_route_candidates": 4,
        "per_route_evaluated": 3,
        "route_counts": {
            route: sum(row["oracle_route"] == route for row in samples)
            for route in (
                "RELATION_RECURRENT", "TEMPORAL_PAIR_RECURRENT",
                "TEMPORAL_SINGLE_NONRECURRENT",
            )
        },
        "samples": samples,
        "sample_count": 12,
        "unique_video_count": 12,
        "development_outcomes_used_for_candidate_construction": True,
        "runtime_selector_reads_outcomes": False,
        "parent_manifest_sha256": [row["manifest_sha256"] for row in parent_manifests],
    }
    development = development_core | {
        "manifest_sha256": stable_hash(development_core)
    }
    development_path = REPO_ROOT / "configs/agqa2_active_grounding_v8_development_manifest.json"
    development_path.write_text(json.dumps(development, indent=2, sort_keys=True) + "\n")

    v7_reserve_path = REPO_ROOT / "configs/agqa2_active_grounding_v7_reserve_manifest.json"
    v7_reserve = _verified_json(v7_reserve_path, "manifest_sha256")
    reserve_core = {
        key: deepcopy(value) for key, value in v7_reserve.items()
        if key != "manifest_sha256"
    }
    reserve_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v8",
        "status": "FROZEN_V8_RAW_VIDEO_UNSEEN_BEFORE_V8_NEURAL_CALLS",
        "claim_boundary": (
            "DOWNLOADED_AND_FROZEN_AS_V7_POOL_BUT_NEVER_MODEL_CALLED;RAW_VIDEO_"
            "UNSEEN_FROM_V3_TO_V7;V8_SINGLE_RUN;NOT_UNTOUCHED_METADATA"
        ),
        "parent_v7_reserve_manifest_sha256": v7_reserve["manifest_sha256"],
        "prior_v8_raw_video_exposure": False,
    })
    reserve = reserve_core | {"manifest_sha256": stable_hash(reserve_core)}
    reserve_path = REPO_ROOT / "configs/agqa2_active_grounding_v8_reserve_manifest.json"
    reserve_path.write_text(json.dumps(reserve, indent=2, sort_keys=True) + "\n")

    base_config_path = REPO_ROOT / "configs/agqa2_active_grounding_v7_development.json"
    config = json.loads(base_config_path.read_text())
    gates = deepcopy(config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v8",
        "status": "FROZEN_BEFORE_ANY_V8_NEURAL_CALL",
        "claim_boundary": reserve["claim_boundary"],
        "development_manifest": str(development_path.relative_to(REPO_ROOT)),
        "reserve_manifest": str(reserve_path.relative_to(REPO_ROOT)),
        "v7_development_result": str(v7_result_path.relative_to(REPO_ROOT)),
        "v7_development_result_file_sha256": _sha256(v7_result_path),
        "runtime_selection": config["runtime_selection"],
        "development_gates": gates,
        "reserve_gates": deepcopy(gates),
        "failure_policy": {
            "development": "MUST_QUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": "RUN_ONCE_ON_UNCALLED_V7_POOL_AS_V8;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v8_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v8",
        "status": "FROZEN_V8_ADAPTED_DEVELOPMENT_CANDIDATE",
        "split": "development",
        "claim_boundary": development["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(development_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(development_path),
        "expected_manifest_status": development["status"],
        "expected_preregistration_status": preregistration["status"],
        "report_version": "V8",
    })
    config["grounder"].update({
        "module_sha256": _sha256(REPO_ROOT / config["grounder"]["module"]),
        "collector_sha256": _sha256(REPO_ROOT / config["grounder"]["collector"]),
        "executor": "src/motif_transfer/agqa_frame_grounder.py",
        "executor_sha256": _sha256(REPO_ROOT / "src/motif_transfer/agqa_frame_grounder.py"),
    })
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v8_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "development_manifest_sha256": development["manifest_sha256"],
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "reserve_video_count": len(reserve["samples"]),
        "development_config_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
