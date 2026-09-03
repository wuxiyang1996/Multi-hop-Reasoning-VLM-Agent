#!/usr/bin/env python3
"""Freeze the V27 token-cap repair on the existing development split."""

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
    run_root = REPO_ROOT / "runs/agqa2_query_object_v27_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V27 development is already consumed")
    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v24_development.json"
    ).read_text())
    primary = config["query_object_grounder"]["model"]
    if primary["id"] != "google/gemini-2.5-flash-lite":
        raise ValueError("unexpected V24 primary ontology model")
    old_cap = int(primary["max_ontology_tokens"])
    if old_cap != 300:
        raise ValueError("V24 ontology cap is not the audited 300-token limit")
    primary["max_ontology_tokens"] = 500
    config.update({
        "schema_version": "agqa2-query-object-development-config-v27",
        "status": "FROZEN_V27_TOKEN_CAP_REPAIR_DEVELOPMENT",
        "claim_boundary": (
            "DEVELOPMENT_ONLY_TOKEN_TRUNCATION_REPAIR;PRIMARY_ONTOLOGY_MAX_"
            "TOKENS_300_TO_500;NO_PROMPT_MODEL_FRAME_THRESHOLD_SOURCE_OR_"
            "EVALUATOR_CHANGE"
        ),
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V27_DEVELOPMENT_CALL"
        ),
        "preregistration": (
            "configs/agqa2_query_object_v27_development_preregistration.json"
        ),
        "report_version": "V27_QUERY_OBJECT_TOKEN_CAP_REPAIR",
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
        "schema_version": "agqa2-query-object-development-preregistration-v27",
        "status": "FROZEN_BEFORE_ANY_V27_DEVELOPMENT_CALL",
        "claim_boundary": config["claim_boundary"],
        "parent_v24_grounder_sha256": (
            "2a84c1c9f170206bd216e171072c70d8faf73bd6d7c89d562fff808bf7141a3e"
        ),
        "candidate_grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "only_semantic_change": {
            "field": "query_object_grounder.model.max_ontology_tokens",
            "before": old_cap,
            "after": 500,
        },
        "development_manifest": config["manifest"],
        "development_manifest_file_sha256": config["manifest_file_sha256"],
        "qualification_gates": deepcopy(config["qualification_gates"]),
        "v26_formal_outcomes_inspected": False,
        "v26_formal_report_created": False,
        "failure_policy": (
            "QUALIFY_ON_EXISTING_DEVELOPMENT_TASKS_ONLY;DO_NOT_READ_V26_GOLD;"
            "FREEZE_V27_REPLAY_ONLY_IF_ALL_EXISTING_GATES_PASS"
        ),
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v27_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "candidate_grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "only_semantic_change": prereg["only_semantic_change"],
    }, indent=2))


if __name__ == "__main__":
    main()
