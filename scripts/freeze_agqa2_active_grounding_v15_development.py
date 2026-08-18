#!/usr/bin/env python3
"""Freeze AGQA V15 after separating grounder and evaluation identities."""

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


def _write_v14_abort() -> Path:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v14_replication"
    artifacts = sorted(
        str(path.relative_to(REPO_ROOT)) for path in run_root.rglob("*")
        if path.is_file()
    ) if run_root.exists() else []
    if artifacts:
        raise RuntimeError(f"V14 has runtime artifacts and cannot be a zero-call abort: {artifacts}")

    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication.json"
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_manifest.json"
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_preregistration.json"
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_selection.json"
    config = json.loads(config_path.read_text())
    manifest = _verified_json(manifest_path, "manifest_sha256")
    dependency_path = REPO_ROOT / config["development_qualification_report"]
    dependency = json.loads(dependency_path.read_text())
    sources, _ = _load_sources(config)
    legacy_grounder_core = _grounder_semantic_core(config, sources) | {
        "runtime_selection": config.get("runtime_selection"),
    }
    legacy_grounder_sha256 = stable_hash(legacy_grounder_core)
    if legacy_grounder_sha256 == dependency["grounder_sha256"]:
        raise AssertionError("V14 should reproduce the preflight lineage mismatch")
    for sample in manifest["samples"]:
        video_path = Path(sample["video_path"])
        if not video_path.is_file() or _sha256(video_path) != sample["video_sha256"]:
            raise ValueError(f"V14 frozen video is missing or changed: {video_path}")

    core = {
        "schema_version": "agqa2-active-grounding-v14-preflight-abort",
        "status": "V14_ABORTED_BEFORE_ANY_RUNTIME_OR_PROVIDER_CALL",
        "stage": "DEVELOPMENT_GROUNDER_IDENTITY_PREFLIGHT",
        "reason": (
            "DATASET_LEVEL_RUNTIME_SELECTION_COUNTS_WERE_INCORRECTLY_INCLUDED_"
            "IN_THE_PER_SAMPLE_GROUNDER_IDENTITY"
        ),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_file_sha256": _sha256(config_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest_file_sha256": _sha256(manifest_path),
        "selection_file_sha256": _sha256(selection_path),
        "dependency_grounder_sha256": dependency["grounder_sha256"],
        "legacy_v14_grounder_sha256": legacy_grounder_sha256,
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "raw_videos_downloaded_and_content_sealed": True,
        "raw_video_decode_or_grounder_inspection_started": False,
        "runtime_receipts_created": 0,
        "provider_calls_started": False,
        "reported_provider_cost_usd": 0.0,
        "frozen_pool_reuse_policy": (
            "MAY_REUSE_EXACT_UNCALLED_V14_POOL_AFTER_V15_DEVELOPMENT_"
            "REQUALIFIES_THE_HASH_BOUNDARY_CORRECTION"
        ),
    }
    payload = core | {"abort_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_active_grounding_v14_preflight_abort.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v15_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V15 development is already consumed")
    abort_path = _write_v14_abort()

    parent_manifest_path = (
        REPO_ROOT / "configs/agqa2_active_grounding_v13_development_manifest.json"
    )
    parent = _verified_json(parent_manifest_path, "manifest_sha256")
    manifest_core = {
        key: deepcopy(value) for key, value in parent.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v15-development",
        "status": "FROZEN_V15_CONSUMED_HASH_BOUNDARY_REQUALIFICATION_DEVELOPMENT",
        "claim_boundary": (
            "EXACT_V13_CONSUMED_DEVELOPMENT_POOL;PER_SAMPLE_GROUNDER_BEHAVIOR_"
            "UNCHANGED;ONLY_GROUNDER_VS_EVALUATION_IDENTITY_BOUNDARY_CORRECTED"
        ),
        "selection_rule": "REUSE_EXACT_V13_DEVELOPMENT_CANDIDATES",
        "parent_v13_manifest_sha256": parent["manifest_sha256"],
        "v14_zero_call_abort_file_sha256": _sha256(abort_path),
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base_config_path = REPO_ROOT / "configs/agqa2_active_grounding_v13_development.json"
    base_config = json.loads(base_config_path.read_text())
    config = deepcopy(base_config)
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v15",
        "status": "FROZEN_V15_HASH_BOUNDARY_REQUALIFICATION_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V15_DEVELOPMENT_RUNTIME_RECEIPT"
        ),
        "report_version": "V15",
    })
    for label in ("module", "collector", "executor"):
        config["grounder"][f"{label}_sha256"] = _sha256(
            REPO_ROOT / config["grounder"][label]
        )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    if config["grounder"]["module_sha256"] != base_config["grounder"]["module_sha256"]:
        raise AssertionError("V15 changed the neural grounder module")
    if config["grounder"]["executor_sha256"] != base_config["grounder"]["executor_sha256"]:
        raise AssertionError("V15 changed the target-native symbolic executor")
    if config["acquisition"] != base_config["acquisition"]:
        raise AssertionError("V15 changed acquisition semantics")
    if config["execution_calibration"] != base_config["execution_calibration"]:
        raise AssertionError("V15 changed execution calibration")

    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v15-development",
        "status": "FROZEN_BEFORE_ANY_V15_DEVELOPMENT_RUNTIME_RECEIPT",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "v14_zero_call_abort": str(abort_path.relative_to(REPO_ROOT)),
        "v14_zero_call_abort_file_sha256": _sha256(abort_path),
        "identity_boundary_change": {
            "grounder_semantic_sha256": expected_grounder_sha256,
            "evaluation_protocol_sha256": expected_evaluation_sha256,
            "excluded_from_grounder_identity": [
                "runtime_selection.candidate_count",
                "runtime_selection.per_predicted_route",
                "runtime_selection.mode",
                "qualification_gates",
            ],
            "included_in_evaluation_protocol_identity": [
                "runtime_selection", "qualification_gates",
            ],
            "per_sample_grounder_behavior_changed": False,
        },
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "acquisition": deepcopy(config["acquisition"]),
        "development_gates": deepcopy(config["qualification_gates"]),
        "accepted_call_replay": {
            "source": "runs/agqa2_active_grounding_v13_development/call_cache",
            "policy": "REUSE_ONLY_EXACT_INPUT_HASH_MATCHES;NO_NEW_PROVIDER_CALL_EXPECTED",
        },
        "failure_policy": {
            "development": "MUST_REQUALIFY_WITHOUT_LOWERING_GATES",
            "replication": "DO_NOT_RUN_UNCALLED_V14_POOL_UNTIL_REQUALIFIED",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_development_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "config_file_sha256": _sha256(config_path),
        "next": "copy exact V13 input-addressed call cache and run V15 development",
    }, indent=2))


if __name__ == "__main__":
    main()
