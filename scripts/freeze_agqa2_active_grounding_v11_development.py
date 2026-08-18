#!/usr/bin/env python3
"""Freeze the V11 model-routing correction on consumed V9/V10 development."""

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
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v11_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V11 development is already consumed")
    v10_report_path = REPO_ROOT / "runs/agqa2_active_grounding_v10_development/report.json"
    v10_report = json.loads(v10_report_path.read_text())
    if v10_report["qualification_gates"]["provider_cost_within_cap"]:
        raise ValueError("V11 is only justified by the preserved V10 cost failure")
    passing_noncost = {
        key: value for key, value in v10_report["qualification_gates"].items()
        if key != "provider_cost_within_cap"
    }
    if not all(passing_noncost.values()):
        raise ValueError("V10 had a non-cost failure; V11 cannot be a routing-only fix")

    parent_path = REPO_ROOT / "configs/agqa2_active_grounding_v10_development_manifest.json"
    parent = _verified_json(parent_path, "manifest_sha256")
    manifest_core = {
        key: deepcopy(value) for key, value in parent.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v11",
        "status": "FROZEN_V11_CONSUMED_MODEL_ROUTING_DEVELOPMENT",
        "claim_boundary": (
            "EXACT_CONSUMED_V10_DEVELOPMENT_POOL;ONLY_FIX_RESCAN_AND_TIEBREAK_"
            "MODEL_ROUTING;NO_NEW_DECISION_RULE_OR_GATE_CHANGE"
        ),
        "selection_rule": "REUSE_EXACT_V10_DEVELOPMENT_CANDIDATES",
        "parent_v10_manifest_sha256": parent["manifest_sha256"],
        "parent_v10_report_sha256": v10_report["report_sha256"],
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v11_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v10_development.json"
    ).read_text())
    gates = deepcopy(base_config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v11-development",
        "status": "FROZEN_BEFORE_ANY_V11_DEVELOPMENT_RUNTIME_RECEIPT",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "v10_failed_report_file_sha256": _sha256(v10_report_path),
        "only_code_change": (
            "PRIMARY_RESCAN_USES_RESCAN_MODEL_AND_CONSENSUS_TIEBREAK_USES_"
            "TIEBREAK_MODEL"
        ),
        "execution_calibration": deepcopy(base_config["execution_calibration"]),
        "runtime_selection": deepcopy(base_config["runtime_selection"]),
        "acquisition": deepcopy(base_config["acquisition"]),
        "development_gates": gates,
        "failure_policy": {
            "development": "MUST_QUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": "DO_NOT_FREEZE_A_NEW_RESERVE_UNTIL_DEVELOPMENT_QUALIFIES",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v11_development_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")

    config = deepcopy(base_config)
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v11",
        "status": "FROZEN_V11_MODEL_ROUTING_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": preregistration["status"],
        "report_version": "V11",
    })
    for label in ("module", "collector", "executor"):
        config["grounder"][f"{label}_sha256"] = _sha256(
            REPO_ROOT / config["grounder"][label]
        )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v11_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
