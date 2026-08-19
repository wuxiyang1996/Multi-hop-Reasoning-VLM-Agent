#!/usr/bin/env python3
"""Freeze V36 closed-syntax development after outcome-unread V35 abort."""

from __future__ import annotations

from copy import deepcopy
import hashlib
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


PARENT_CONFIG = "configs/agqa2_robust_temporal_v35_development.json"
PARENT_ABORT = "docs/results/agqa2_robust_temporal_v35_runtime_abort.json"
MANIFEST = "configs/agqa2_robust_temporal_v36_development_manifest.json"
PREREG = "configs/agqa2_robust_temporal_v36_development_preregistration.json"
CONFIG = "configs/agqa2_robust_temporal_v36_development.json"
COLLECTOR = "scripts/collect_agqa2_robust_temporal_v36_development.py"
CORE_EVALUATOR = "scripts/collect_agqa2_robust_temporal_v34_formal.py"
NORMALIZATION = "src/motif_transfer/agqa_operand_normalization_v2.py"
ADAPTER = "src/motif_transfer/agqa_robust_temporal_transfer.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    abort_path = REPO_ROOT / PARENT_ABORT
    abort = json.loads(abort_path.read_text())
    abort_body = dict(abort)
    claimed = abort_body.pop("result_sha256")
    if stable_hash(abort_body) != claimed:
        raise ValueError("V35 abort hash mismatch")
    if (
        abort["status"]
        != "AGQA2_ROBUST_TEMPORAL_V35_DEVELOPMENT_RUNTIME_INCOMPLETE"
        or abort["evaluator_loop_entered"]
        or abort["official_answer_field_accessed"]
        or abort["completed_runtime_receipts"] != 99
    ):
        raise ValueError("V35 is not eligible for syntax-only completion")
    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    old_manifest_path = REPO_ROOT / parent["manifest"]
    old_manifest = json.loads(old_manifest_path.read_text())
    manifest_core = dict(old_manifest)
    old_manifest_sha = manifest_core.pop("manifest_sha256")
    if stable_hash(manifest_core) != old_manifest_sha:
        raise ValueError("V35 manifest hash mismatch")
    manifest_core.update({
        "schema_version": "agqa2-robust-temporal-v36-development-manifest-v1",
        "status": "FROZEN_V36_OUTCOME_UNREAD_SYNTAX_COMPLETION_DEVELOPMENT",
        "claim_boundary": (
            "EXACT_V34_V35_OUTCOME_UNREAD_ONE_HUNDRED_VIDEO_POOL;STABLE_SORT_"
            "DEDUP_EXISTING_VALID_EVIDENCE_IDS_THEN_INTERVAL_ENVELOPE;"
            "DEVELOPMENT_ONLY"
        ),
        "parent_v35_manifest_sha256": old_manifest_sha,
        "parent_v35_runtime_abort_result_sha256": abort["result_sha256"],
        "v35_evaluator_loop_entered": False,
        "v35_official_answer_field_accessed": False,
    })
    manifest = manifest_core | {
        "manifest_sha256": stable_hash(manifest_core)
    }
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    config = deepcopy(parent)
    collector_path = REPO_ROOT / COLLECTOR
    normalization_path = REPO_ROOT / NORMALIZATION
    adapter_path = REPO_ROOT / ADAPTER
    core_evaluator_path = REPO_ROOT / CORE_EVALUATOR
    config["grounder"].update({
        "collector": COLLECTOR,
        "collector_sha256": _sha256(collector_path),
        "protocol": (
            "V19_ACQUISITION_PLUS_CLOSED_DETERMINISTIC_OPERAND_SYNTAX_"
            "NORMALIZATION_V36"
        ),
    })
    config["syntax_transport_normalization"] = {
        "normalization_module": NORMALIZATION,
        "normalization_module_sha256": _sha256(normalization_path),
        "evidence_rule": (
            "STABLE_SORT_AND_DEDUP_ALREADY_EMITTED_VALID_FRAME_IDS"
        ),
        "interval_rule": (
            "OBSERVED_START_MIN_WITH_EXISTING_EVIDENCE_AND_END_MAX_WITH_"
            "EXISTING_EVIDENCE;NO_NEW_EVIDENCE_LABEL_CONFIDENCE_OR_OBJECT"
        ),
        "json_transport_retry_attempts": 3,
        "json_transport_retry_rule": (
            "RETRY_IDENTICAL_REQUEST_ONLY_ON_JSON_DECODE_ERROR"
        ),
        "outcome_or_label_dependent_repair": False,
    }
    config.update({
        "schema_version": "agqa2-robust-temporal-v36-development-config-v1",
        "status": "FROZEN_V36_ROBUST_TEMPORAL_DEVELOPMENT",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        # Compatibility status consumed by the unchanged core evaluator.
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "report_version": "V36_BASE",
    })
    sources, _ = _load_sources(config)
    parent_grounder_sha256 = stable_hash(
        _grounder_semantic_core(config, sources)
    )
    base_evaluation_sha256 = stable_hash(
        _evaluation_protocol_core(config)
    )
    grounder_core = {
        "schema_version": "agqa2-robust-temporal-grounder-v36",
        "parent_grounder_semantic_core": _grounder_semantic_core(
            config, sources,
        ),
        "postground_adapter_module_sha256": _sha256(adapter_path),
        "normalization_module_sha256": _sha256(normalization_path),
        "collector_module_sha256": _sha256(collector_path),
        "binding_rule": (
            "AT_LEAST_TWO_UNIQUE_OBSERVED_INTERVAL_HYPOTHESES_PER_OPERAND;"
            "ALL_CROSS_VIEW_INTERVAL_PAIRS_STRICTLY_SEPARATED_AND_SAME_RELATION"
        ),
        "minimum_confidence": 0.5,
        "current_outcome_read": False,
    }
    postground_sha256 = stable_hash(grounder_core)
    gates = deepcopy(config["postground"]["formal_gates"])
    protocol_core = {
        "schema_version": "agqa2-robust-temporal-v36-development-protocol-v1",
        "sample_count": 100,
        "source_program_sha256": config["postground"][
            "source_program_sha256"
        ],
        "target_grounder_sha256": postground_sha256,
        "target_executor_sha256": config["postground"][
            "target_executor_sha256"
        ],
        "adapter_module_sha256": _sha256(adapter_path),
        "normalization_module_sha256": _sha256(normalization_path),
        "collector_module_sha256": _sha256(collector_path),
        "core_evaluator_module_sha256": _sha256(core_evaluator_path),
        "fallback": "PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION",
        "gates": gates,
        "confirmatory_claim": False,
    }
    protocol_sha256 = stable_hash(protocol_core)
    prereg = {
        "schema_version": "agqa2-robust-temporal-v36-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v36_status": "FROZEN_BEFORE_V36_SYNTAX_COMPLETION_OR_OUTCOME_ACCESS",
        "claim_boundary": manifest["claim_boundary"],
        "v35_abort_summary": PARENT_ABORT,
        "v35_abort_summary_file_sha256": _sha256(abort_path),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "parent_grounder_sha256": parent_grounder_sha256,
        "postground_target_grounder_sha256": postground_sha256,
        "base_evaluation_protocol_sha256": base_evaluation_sha256,
        "postground_evaluation_protocol": protocol_core,
        "postground_evaluation_protocol_sha256": protocol_sha256,
        "development_gates": gates,
        "new_provider_calls_allowed": (
            "ONLY_UNCACHED_PRIOR_FAILED_BRANCHES_OR_IDENTICAL_JSON_RETRY"
        ),
        "confirmatory_claim_allowed": False,
        "future_policy": (
            "V36_MUST_QUALIFY_BEFORE_ONE_NEW_VIDEO_DISJOINT_FORMAL"
        ),
    }
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": parent_grounder_sha256,
        "expected_evaluation_protocol_sha256": base_evaluation_sha256,
    })
    config["postground"].update({
        "adapter_module": ADAPTER,
        "adapter_module_sha256": _sha256(adapter_path),
        "evaluator_module": COLLECTOR,
        "evaluator_module_sha256": _sha256(collector_path),
        "core_evaluator_module": CORE_EVALUATOR,
        "core_evaluator_module_sha256": _sha256(core_evaluator_path),
        "target_grounder_sha256": postground_sha256,
        "evaluation_protocol_sha256": protocol_sha256,
        "formal_gates": gates,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v36_status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_grounder_sha256": parent_grounder_sha256,
        "postground_target_grounder_sha256": postground_sha256,
        "postground_evaluation_protocol_sha256": protocol_sha256,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
