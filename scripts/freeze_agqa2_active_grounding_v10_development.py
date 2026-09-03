#!/usr/bin/env python3
"""Freeze V10 adaptation on the consumed V9 reserve pool."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v10_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V10 development is already consumed")
    v9_report_path = REPO_ROOT / "runs/agqa2_active_grounding_v9_reserve/report.json"
    v9_report = json.loads(v9_report_path.read_text())
    if v9_report["status"] != "AGQA2_ACTIVE_GROUNDER_V9_RESERVE_NOT_QUALIFIED":
        raise ValueError("V10 adaptation requires the preserved V9 failure")

    parent_path = REPO_ROOT / "configs/agqa2_active_grounding_v9_reserve_manifest.json"
    parent = _verified_json(parent_path, "manifest_sha256")
    manifest_core = {
        key: deepcopy(value) for key, value in parent.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v10",
        "status": "FROZEN_V10_CONSUMED_V9_RESERVE_ADAPTATION_DEVELOPMENT",
        "split": "development",
        "claim_boundary": (
            "CONSUMED_V9_RESERVE_OUTCOMES_USED_FOR_V10_ADAPTATION;NO_"
            "CONFIRMATORY_CLAIM;EXACT_V9_POOL_WITH_CALIBRATED_SELECTOR"
        ),
        "selection_rule": "REUSE_ALL_12_V9_RESERVE_CANDIDATES_WITHOUT_RESELECTION",
        "development_outcomes_used_for_v10_rule_design": True,
        "parent_v9_reserve_manifest_sha256": parent["manifest_sha256"],
        "parent_v9_report_sha256": v9_report["report_sha256"],
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v10_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v9_development.json"
    ).read_text())
    calibration = deepcopy(base_config["execution_calibration"])
    calibration.update({
        "minimum_single_interval_nesting_margin_frames": 6,
        "minimum_repeated_interval_dominance_margin_frames": 2,
        "new_v10_topologies": [
            "BOUNDARY_ALIGNED_SINGLE_INTERVAL_NESTING",
            "ALIGNED_REPEATED_INTERVAL_DOMINANCE",
        ],
    })
    runtime_selection = {
        "mode": "OUTCOME_BLIND_CALIBRATED_EVIDENCE_RANK_V1",
        "candidate_count": 12,
        "per_predicted_route": 3,
        "reads_direct_response_only_inside_frozen_calibration": True,
        "forbidden": ["answer", "program", "scene_graph", "source_identity"],
    }
    acquisition = deepcopy(base_config["acquisition"])
    acquisition.update({
        "maximum_source_ir_rescans_per_recurrent_operand": 2,
        "consensus_tiebreak_on_conflict": True,
        "receipt_selection": "THREE_VIEW_MAJORITY_ELSE_ABSTAIN",
    })
    gates = deepcopy(base_config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v10-development",
        "status": "FROZEN_BEFORE_ANY_V10_DEVELOPMENT_RUNTIME_RECEIPT",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "parent_v9_report_file_sha256": _sha256(v9_report_path),
        "execution_calibration": calibration,
        "runtime_selection": runtime_selection,
        "acquisition": acquisition,
        "development_gates": gates,
        "failure_policy": {
            "development": "MUST_QUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": "DO_NOT_FREEZE_A_NEW_RESERVE_UNTIL_DEVELOPMENT_QUALIFIES",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v10_development_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")

    config = deepcopy(base_config)
    config.pop("development_qualification_report", None)
    config.pop("development_qualification_file_sha256", None)
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v10",
        "status": "FROZEN_V10_CONSUMED_ADAPTATION_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": preregistration["status"],
        "report_version": "V10",
        "execution_calibration": calibration,
        "runtime_selection": runtime_selection,
        "acquisition": acquisition,
    })
    config["media"].update({
        "tiebreak_frame_count": 24,
        "tiebreak_frame_max_side": 512,
        "tiebreak_frames_per_panel": 4,
        "tiebreak_panel_frame_width": 224,
    })
    tiebreak_model = deepcopy(config["rescan_model"])
    tiebreak_model["id"] = "google/gemini-2.5-flash-lite"
    tiebreak_model.pop("reasoning", None)
    config["tiebreak_model"] = tiebreak_model
    for label in ("module", "collector", "executor"):
        config["grounder"][f"{label}_sha256"] = _sha256(
            REPO_ROOT / config["grounder"][label]
        )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v10_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": manifest["sample_count"],
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
