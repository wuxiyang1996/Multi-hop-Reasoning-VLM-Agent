#!/usr/bin/env python3
"""Freeze the bounded-schema V28 candidate on development tasks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_bounded_ontology_protocol import (  # noqa: E402
    MAX_FREE_TEXT_CHARACTERS, PROTOCOL_VERSION,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_query_object_v20 import _evaluation_core, _semantic_core  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v28_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V28 development is already consumed")
    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v27_development.json"
    ).read_text())
    module = REPO_ROOT / "src/motif_transfer/agqa_bounded_ontology_protocol.py"
    collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v28.py"
    config["query_object_grounder"]["bounded_ontology_protocol"] = {
        "version": PROTOCOL_VERSION,
        "maximum_free_text_characters": MAX_FREE_TEXT_CHARACTERS,
        "bounded_fields": ["visual_description", "uncertainty"],
        "unchanged_fields": [
            "decision", "relation_observed", "confidence", "evidence_frames",
        ],
        "module": str(module.relative_to(REPO_ROOT)),
        "module_sha256": _sha256(module),
        "collector": str(collector.relative_to(REPO_ROOT)),
        "collector_sha256": _sha256(collector),
    }
    config.update({
        "schema_version": "agqa2-query-object-development-config-v28",
        "status": "FROZEN_V28_BOUNDED_SCHEMA_DEVELOPMENT",
        "claim_boundary": (
            "DEVELOPMENT_ONLY_BOUNDED_FREE_TEXT_REPAIR;160_CHARACTER_LIMIT_"
            "ON_EXPLANATORY_FIELDS;DECISION_CONFIDENCE_EVIDENCE_UNCHANGED"
        ),
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V28_DEVELOPMENT_CALL"
        ),
        "preregistration": (
            "configs/agqa2_query_object_v28_development_preregistration.json"
        ),
        "report_version": "V28_BOUNDED_ONTOLOGY",
    })
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    prereg = {
        "schema_version": "agqa2-query-object-development-preregistration-v28",
        "status": "FROZEN_BEFORE_ANY_V28_DEVELOPMENT_CALL",
        "claim_boundary": config["claim_boundary"],
        "parent_v27_grounder_sha256": (
            "bdca5c4a06aeb2c3e877d9fa6c2d0794cb01ce3faa7d6e3c009434172c0189c7"
        ),
        "candidate_grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "bounded_ontology_protocol": deepcopy(
            config["query_object_grounder"]["bounded_ontology_protocol"]
        ),
        "development_manifest": config["manifest"],
        "development_manifest_file_sha256": config["manifest_file_sha256"],
        "qualification_gates": deepcopy(config["qualification_gates"]),
        "v26_or_v27_formal_outcomes_inspected": False,
        "failure_policy": (
            "QUALIFY_ON_EXISTING_DEVELOPMENT_TASKS_ONLY;FREEZE_A_NEW_VIDEO_"
            "DISJOINT_V28_POOL_ONLY_IF_ALL_EXISTING_GATES_PASS"
        ),
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v28_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "candidate_grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "bounded_free_text_characters": MAX_FREE_TEXT_CHARACTERS,
    }, indent=2))


if __name__ == "__main__":
    main()
