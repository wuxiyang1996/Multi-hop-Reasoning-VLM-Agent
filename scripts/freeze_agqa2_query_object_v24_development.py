#!/usr/bin/env python3
"""Freeze V24 development requalification after a runtime-only V23 abort."""

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
    run_root = REPO_ROOT / "runs/agqa2_query_object_v24_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V24 QUERY_OBJECT development is already consumed")
    v22_report_path = REPO_ROOT / "runs/agqa2_query_object_v22_development/report.json"
    v22_report = json.loads(v22_report_path.read_text())
    if not v22_report.get("grounder_qualified"):
        raise ValueError("V24 requires the qualified V22 development grounder")
    v23_errors_path = REPO_ROOT / "runs/agqa2_query_object_v23_reserve/worker_errors.json"
    v23_errors = json.loads(v23_errors_path.read_text())
    if list(v23_errors.get("errors", {}).values()) != [
        "ValueError: operand schema retries exhausted: evidence frames must lie inside the claimed interval"
    ]:
        raise ValueError("V24 is only authorized for the frozen interval-envelope failure")

    parent_manifest = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v22_development_manifest.json"
    ).read_text())
    core = deepcopy(parent_manifest)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-development-manifest-v24",
        "status": "FROZEN_V24_QUERY_OBJECT_DEVELOPMENT_BEFORE_REQUALIFICATION",
        "claim_boundary": (
            "EXACT_V22_TRAIN_DEVELOPMENT_SPLIT;DETERMINISTIC_INTERVAL_EVIDENCE_"
            "ENVELOPE_NORMALIZATION;NO_NEW_PROVIDER_CALLS"
        ),
        "parent_v22_manifest_sha256": parent_manifest["manifest_sha256"],
        "v23_runtime_outcomes_read": False,
        "v23_gold_answers_read": False,
    })
    manifest = core | {"manifest_sha256": stable_hash(core)}
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v24_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v22_development.json"
    ).read_text())
    collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v24.py"
    normalization = REPO_ROOT / "src/motif_transfer/agqa_operand_normalization.py"
    config.update({
        "schema_version": "agqa2-query-object-development-config-v24",
        "status": "FROZEN_V24_QUERY_OBJECT_DEVELOPMENT",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_V24_CACHE_ONLY_REQUALIFICATION",
        "report_version": "V24_QUERY_OBJECT",
    })
    config["grounder"].update({
        "collector": str(collector.relative_to(REPO_ROOT)),
        "collector_sha256": _sha256(collector),
        "protocol": "V22_CONSENSUS_PLUS_DETERMINISTIC_INTERVAL_ENVELOPE_V24",
    })
    config["query_object_grounder"].update({
        "normalization_module": str(normalization.relative_to(REPO_ROOT)),
        "normalization_module_sha256": _sha256(normalization),
        "normalization_rule": (
            "OBSERVED_START_MIN_WITH_EXISTING_EVIDENCE_AND_END_MAX_WITH_EXISTING_"
            "EVIDENCE;NO_NEW_EVIDENCE_OR_LABEL"
        ),
    })
    config["preregistration"] = "configs/agqa2_query_object_v24_development_preregistration.json"
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    prereg = {
        "schema_version": "agqa2-query-object-development-preregistration-v24",
        "status": "FROZEN_BEFORE_V24_CACHE_ONLY_REQUALIFICATION",
        "claim_boundary": manifest["claim_boundary"],
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "qualification_gates": config["qualification_gates"],
        "v22_report_file_sha256": _sha256(v22_report_path),
        "v22_report_sha256": v22_report["report_sha256"],
        "v23_worker_errors_file_sha256": _sha256(v23_errors_path),
        "new_provider_calls_allowed": 0,
        "outcome_or_label_dependent_repair": False,
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
    config_path = REPO_ROOT / "configs/agqa2_query_object_v24_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
