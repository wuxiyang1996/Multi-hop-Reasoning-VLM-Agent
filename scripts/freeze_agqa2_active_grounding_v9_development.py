#!/usr/bin/env python3
"""Freeze V9 adaptation on consumed V8 development rows only."""

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
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v9_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V9 development is already consumed")

    parent_manifest_path = (
        REPO_ROOT / "configs/agqa2_active_grounding_v8_development_manifest.json"
    )
    parent_manifest = _verified_json(parent_manifest_path, "manifest_sha256")
    manifest_core = {
        key: deepcopy(value) for key, value in parent_manifest.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v9",
        "status": "FROZEN_V9_CONSUMED_ADAPTATION_DEVELOPMENT",
        "claim_boundary": (
            "CONSUMED_V8_ADAPTATION_ROWS_ONLY;V9_FIXES_PUBLIC_QUESTION_ANSWER_"
            "SPACE_AND_CALIBRATES_EXECUTION_WITHOUT_ANSWER_PROGRAM_OR_SCENE_GRAPH"
        ),
        "selection_rule": (
            "REUSE_EXACT_V8_DEVELOPMENT_CANDIDATES_AND_OUTCOME_BLIND_RUNTIME_"
            "SELECTOR;NO_NEW_DEVELOPMENT_CANDIDATE_SELECTION"
        ),
        "development_outcomes_used_for_v9_rule_design": True,
        "parent_manifest_sha256": parent_manifest["manifest_sha256"],
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = (
        REPO_ROOT / "configs/agqa2_active_grounding_v9_development_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v8_development.json"
    ).read_text())
    calibration = {
        "mode": "INDEPENDENT_TARGET_EVIDENCE_V1",
        "minimum_duration_margin_frames": 3,
        "agreement_authorizes_without_claiming_gain": True,
        "direct_corroboration_cannot_change_direct_prediction": True,
        "typed_override_requires_independent_symbolic_topology": True,
        "forbidden_fields": [
            "official_answer", "functional_program", "scene_graph_grounding",
            "source_identity",
        ],
    }
    gates = deepcopy(base_config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v9-development",
        "status": "FROZEN_BEFORE_ANY_V9_DEVELOPMENT_RUNTIME_RECEIPT",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "execution_calibration": calibration,
        "runtime_selection": deepcopy(base_config["runtime_selection"]),
        "development_gates": gates,
        "failure_policy": {
            "development": "MUST_QUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": (
                "DO_NOT_FREEZE_OR_CALL_A_NEW_RESERVE_UNTIL_DEVELOPMENT_QUALIFIES"
            ),
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = (
        REPO_ROOT / "configs/agqa2_active_grounding_v9_development_preregistration.json"
    )
    prereg_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n"
    )

    config = deepcopy(base_config)
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v9",
        "status": "FROZEN_V9_ADAPTATION_DEVELOPMENT_CANDIDATE",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": preregistration["status"],
        "report_version": "V9",
        "execution_calibration": calibration,
    })
    for label in ("module", "collector", "executor"):
        config["grounder"][f"{label}_sha256"] = _sha256(
            REPO_ROOT / config["grounder"][label]
        )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v9_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "config_file_sha256": _sha256(config_path),
        "candidate_count": manifest["sample_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
