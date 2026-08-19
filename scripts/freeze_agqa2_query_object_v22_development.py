#!/usr/bin/env python3
"""Freeze the cheaper V22 cross-model QUERY_OBJECT development run."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_query_object_v20 import _evaluation_core, _semantic_core  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v22_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V22 QUERY_OBJECT development is already consumed")
    v21_report_path = REPO_ROOT / "runs/agqa2_query_object_v21_development/report.json"
    v21_report = json.loads(v21_report_path.read_text())
    gates = v21_report["qualification_gates"]
    failed = [key for key, value in gates.items() if not value]
    if failed != ["provider_cost_within_cap"]:
        raise ValueError(f"V22 expects only the V21 cost failure, got {failed}")

    parent_manifest = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v21_development_manifest.json"
    ).read_text())
    core = deepcopy(parent_manifest)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-development-manifest-v22",
        "status": "FROZEN_V22_QUERY_OBJECT_DEVELOPMENT_BEFORE_GEMINI3_CALLS",
        "claim_boundary": (
            "EXACT_V20_V21_AGQA_TRAIN_DEVELOPMENT_SPLIT;UNCHANGED_TWO_OF_THREE_"
            "CONSENSUS;CHEAPER_THIRD_MODEL;NOT_CONFIRMATORY"
        ),
        "parent_v21_manifest_sha256": parent_manifest["manifest_sha256"],
        "parent_v21_report_sha256": v21_report["report_sha256"],
        "v21_outcome_used_for_development": True,
    })
    manifest = core | {"manifest_sha256": stable_hash(core)}
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v22_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v21_development.json"
    ).read_text())
    collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v22.py"
    parent_collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v21.py"
    config.update({
        "schema_version": "agqa2-query-object-development-config-v22",
        "status": "FROZEN_V22_QUERY_OBJECT_DEVELOPMENT",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V22_GEMINI3_CALL",
        "report_version": "V22_QUERY_OBJECT",
    })
    config["grounder"].update({
        "collector": str(collector.relative_to(REPO_ROOT)),
        "collector_sha256": _sha256(collector),
        "protocol": "SOURCE_RECURRENT_GEMINI_CROSS_MODEL_CONSENSUS_QUERY_OBJECT_V22",
    })
    config["query_object_grounder"].update({
        "parent_consensus_collector": str(parent_collector.relative_to(REPO_ROOT)),
        "parent_consensus_collector_sha256": _sha256(parent_collector),
        "secondary_model": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_name": "OPENROUTER_API_KEY",
            "id": "google/gemini-3-flash-preview",
            "temperature": 0,
            "timeout_seconds": 240,
            "max_retries": 2,
            "schema_retries": 2,
            "max_ontology_tokens": 300,
            "reasoning": {"effort": "minimal", "exclude": True},
        },
    })
    config["query_object_calibration"]["views"] = [
        "source_controlled_qwen_isolated_relation_grounder",
        "gemini_2_5_flash_lite_fixed_ontology_grounder",
        "gemini_3_flash_fixed_ontology_grounder",
    ]
    config["preregistration"] = "configs/agqa2_query_object_v22_development_preregistration.json"
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    prereg = {
        "schema_version": "agqa2-query-object-development-preregistration-v22",
        "status": "FROZEN_BEFORE_ANY_V22_GEMINI3_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "qualification_gates": config["qualification_gates"],
        "v21_report_file_sha256": _sha256(v21_report_path),
        "v21_report_sha256": v21_report["report_sha256"],
        "all_non_cost_gates_preserved": True,
        "cost_cap_changed": False,
        "consensus_rule_changed": False,
        "new_calls": "GEMINI3_FIXED_ONTOLOGY_VIEW_ONLY",
        "confirmatory_claim_allowed": False,
        "atomic_route_artifacts_modified": False,
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v22_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "unchanged_gates": config["qualification_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
