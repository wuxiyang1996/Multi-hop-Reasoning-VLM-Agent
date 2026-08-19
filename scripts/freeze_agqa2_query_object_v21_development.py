#!/usr/bin/env python3
"""Freeze V21 cross-model QUERY_OBJECT consensus on the V20 dev split."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_query_object_v20 import (  # noqa: E402
    _evaluation_core, _semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v21_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V21 QUERY_OBJECT development is already consumed")
    v20_report_path = REPO_ROOT / "runs/agqa2_query_object_v20_development/report.json"
    v20_report = json.loads(v20_report_path.read_text())
    if v20_report["status"] != "AGQA2_QUERY_OBJECT_V20_DEVELOPMENT_NOT_QUALIFIED":
        raise ValueError("V21 expects the frozen V20 coverage failure")
    if v20_report["metrics"]["typed_vs_direct_losses"] != 0:
        raise ValueError("V21 cannot preserve a V20 rule with negative transfer")

    old_manifest = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v20_development_manifest.json"
    ).read_text())
    core = deepcopy(old_manifest)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-development-manifest-v21",
        "status": "FROZEN_V21_QUERY_OBJECT_DEVELOPMENT_BEFORE_SECONDARY_CALLS",
        "claim_boundary": (
            "EXACT_V20_AGQA_TRAIN_DEVELOPMENT_SPLIT;CROSS_MODEL_TWO_OF_THREE_"
            "NEURAL_OBJECT_CONSENSUS;NOT_CONFIRMATORY"
        ),
        "parent_v20_manifest_sha256": old_manifest["manifest_sha256"],
        "parent_v20_report_sha256": v20_report["report_sha256"],
        "v20_outcome_used_for_development": True,
    })
    manifest = core | {"manifest_sha256": stable_hash(core)}
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v21_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v20_development.json"
    ).read_text())
    collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v21.py"
    consensus = REPO_ROOT / "src/motif_transfer/agqa_query_object_consensus.py"
    parent_collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v20.py"
    config.update({
        "schema_version": "agqa2-query-object-development-config-v21",
        "status": "FROZEN_V21_QUERY_OBJECT_DEVELOPMENT",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V21_SECONDARY_CALL",
        "report_version": "V21_QUERY_OBJECT",
    })
    config["grounder"].update({
        "collector": str(collector.relative_to(REPO_ROOT)),
        "collector_sha256": _sha256(collector),
        "protocol": "SOURCE_RECURRENT_CROSS_MODEL_TWO_OF_THREE_QUERY_OBJECT_V21",
    })
    config["query_object_grounder"].update({
        "consensus_module": str(consensus.relative_to(REPO_ROOT)),
        "consensus_module_sha256": _sha256(consensus),
        "parent_query_object_collector": str(parent_collector.relative_to(REPO_ROOT)),
        "parent_query_object_collector_sha256": _sha256(parent_collector),
        "secondary_model": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_name": "OPENROUTER_API_KEY",
            "id": "anthropic/claude-sonnet-4.6",
            "temperature": 0,
            "timeout_seconds": 240,
            "max_retries": 2,
            "schema_retries": 2,
            "max_ontology_tokens": 300,
        },
    })
    config["query_object_calibration"] = {
        "mode": "CROSS_MODEL_TWO_OF_THREE_NEURAL_CONSENSUS_V1",
        "minimum_ontology_confidences": [0.8, 0.8],
        "minimum_neural_votes": 2,
        "direct_response_is_not_a_consensus_vote": True,
        "views": [
            "source_controlled_isolated_relation_grounder",
            "gemini_fixed_ontology_grounder",
            "claude_fixed_ontology_grounder",
        ],
    }
    config["preregistration"] = "configs/agqa2_query_object_v21_development_preregistration.json"
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    prereg = {
        "schema_version": "agqa2-query-object-development-preregistration-v21",
        "status": "FROZEN_BEFORE_ANY_V21_SECONDARY_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "qualification_gates": config["qualification_gates"],
        "v20_report_file_sha256": _sha256(v20_report_path),
        "v20_report_sha256": v20_report["report_sha256"],
        "v20_calls_reused_without_new_cost": True,
        "new_calls": "CLAUDE_FIXED_ONTOLOGY_VIEW_ONLY",
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
    config_path = REPO_ROOT / "configs/agqa2_query_object_v21_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "v20_report_sha256": v20_report["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
