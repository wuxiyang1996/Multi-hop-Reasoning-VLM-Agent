#!/usr/bin/env python3
"""Freeze V35 development completion after the outcome-unread V34 abort."""

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


V34_CONFIG = "configs/agqa2_robust_temporal_v34_formal.json"
V34_ABORT = "docs/results/agqa2_robust_temporal_v34_runtime_abort.json"
V35_MANIFEST = "configs/agqa2_robust_temporal_v35_development_manifest.json"
V35_PREREG = (
    "configs/agqa2_robust_temporal_v35_development_preregistration.json"
)
V35_CONFIG = "configs/agqa2_robust_temporal_v35_development.json"
COLLECTOR = "scripts/collect_agqa2_robust_temporal_v35_development.py"
CORE_EVALUATOR = "scripts/collect_agqa2_robust_temporal_v34_formal.py"
NORMALIZATION = "src/motif_transfer/agqa_operand_normalization.py"
ADAPTER = "src/motif_transfer/agqa_robust_temporal_transfer.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    abort_path = REPO_ROOT / V34_ABORT
    abort = json.loads(abort_path.read_text())
    abort_body = dict(abort)
    claimed = abort_body.pop("result_sha256")
    if stable_hash(abort_body) != claimed:
        raise ValueError("V34 abort summary hash mismatch")
    if (
        abort["status"] != "AGQA2_ROBUST_TEMPORAL_V34_RUNTIME_INCOMPLETE"
        or abort["evaluator_loop_entered"]
        or abort["official_answer_field_accessed"]
        or abort["completed_runtime_receipts"] != 98
    ):
        raise ValueError("V34 is not eligible for outcome-unread development")

    v34 = json.loads((REPO_ROOT / V34_CONFIG).read_text())
    source_manifest_path = REPO_ROOT / v34["manifest"]
    source_manifest = json.loads(source_manifest_path.read_text())
    source_body = dict(source_manifest)
    source_claimed = source_body.pop("manifest_sha256")
    if stable_hash(source_body) != source_claimed:
        raise ValueError("V34 manifest hash mismatch")
    manifest_core = deepcopy(source_body)
    manifest_core.update({
        "schema_version": "agqa2-robust-temporal-v35-development-manifest-v1",
        "status": "FROZEN_V35_OUTCOME_UNREAD_RUNTIME_COMPLETION_DEVELOPMENT",
        "split": "development",
        "claim_boundary": (
            "EXACT_V34_OUTCOME_UNREAD_ONE_HUNDRED_VIDEO_POOL;DETERMINISTIC_"
            "INTERVAL_EVIDENCE_ENVELOPE_NORMALIZATION;IDENTICAL_REQUEST_JSON_"
            "TRANSPORT_RETRY;DEVELOPMENT_ONLY"
        ),
        "parent_v34_manifest_sha256": source_manifest["manifest_sha256"],
        "parent_v34_runtime_abort_result_sha256": abort["result_sha256"],
        "prior_neural_grounder_or_model_video_exposure": True,
        "v34_evaluator_loop_entered": False,
        "v34_official_answer_field_accessed": False,
    })
    manifest = manifest_core | {
        "manifest_sha256": stable_hash(manifest_core)
    }
    manifest_path = REPO_ROOT / V35_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    config = deepcopy(v34)
    collector_path = REPO_ROOT / COLLECTOR
    normalization_path = REPO_ROOT / NORMALIZATION
    adapter_path = REPO_ROOT / ADAPTER
    core_evaluator_path = REPO_ROOT / CORE_EVALUATOR
    config["grounder"].update({
        "collector": COLLECTOR,
        "collector_sha256": _sha256(collector_path),
        "protocol": (
            "V19_ACQUISITION_PLUS_DETERMINISTIC_INTERVAL_ENVELOPE_AND_"
            "IDENTICAL_REQUEST_JSON_TRANSPORT_RETRY_V35"
        ),
    })
    config["syntax_transport_normalization"] = {
        "normalization_module": NORMALIZATION,
        "normalization_module_sha256": _sha256(normalization_path),
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
        "schema_version": "agqa2-robust-temporal-v35-development-config-v1",
        "status": "FROZEN_V35_ROBUST_TEMPORAL_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": V35_MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "report_version": "V35_BASE",
    })
    for key in (
        "development_qualification_report",
        "development_qualification_file_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    parent_grounder_sha256 = stable_hash(
        _grounder_semantic_core(config, sources)
    )
    evaluation_protocol_sha256 = stable_hash(
        _evaluation_protocol_core(config)
    )
    grounder_core = {
        "schema_version": "agqa2-robust-temporal-grounder-v35",
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
    postground_target_grounder_sha256 = stable_hash(grounder_core)
    gates = {
        "required_valid_rows": 100,
        "required_unique_videos": 100,
        "minimum_source_authorizations": 20,
        "minimum_source_wins": 7,
        "maximum_source_losses": 1,
        "minimum_source_minus_target_correct": 5,
        "maximum_exact_one_sided_pvalue": 0.05,
        "required_effect_shuffled_abstentions": 100,
        "required_wrong_source_abstentions": 100,
        "required_generic_scaffold_matches": 100,
        "required_target_written_equivalent_matches": 100,
        "maximum_reported_provider_cost_usd": 0.9,
    }
    protocol_core = {
        "schema_version": "agqa2-robust-temporal-v35-development-protocol-v1",
        "sample_count": 100,
        "source_program_sha256": config["postground"][
            "source_program_sha256"
        ],
        "target_grounder_sha256": postground_target_grounder_sha256,
        "target_executor_sha256": config["postground"][
            "target_executor_sha256"
        ],
        "adapter_module_sha256": _sha256(adapter_path),
        "collector_module_sha256": _sha256(collector_path),
        "core_evaluator_module_sha256": _sha256(core_evaluator_path),
        "fallback": "PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION",
        "gates": gates,
        "confirmatory_claim": False,
    }
    postground_protocol_sha256 = stable_hash(protocol_core)
    prereg = {
        "schema_version": "agqa2-robust-temporal-v35-preregistration-v1",
        # Compatibility status consumed by the unchanged V34 core evaluator;
        # the schema and explicit lineage fields below remain V35-specific.
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v35_status": "FROZEN_BEFORE_V35_RUNTIME_COMPLETION_OR_OUTCOME_ACCESS",
        "claim_boundary": manifest["claim_boundary"],
        "v34_abort_summary": V34_ABORT,
        "v34_abort_summary_file_sha256": _sha256(abort_path),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "parent_grounder_sha256": parent_grounder_sha256,
        "postground_target_grounder_sha256": (
            postground_target_grounder_sha256
        ),
        "base_evaluation_protocol_sha256": evaluation_protocol_sha256,
        "postground_evaluation_protocol": protocol_core,
        "postground_evaluation_protocol_sha256": (
            postground_protocol_sha256
        ),
        "development_gates": gates,
        "new_provider_calls_allowed": (
            "ONLY_UNCACHED_V34_FAILED_ROWS_OR_IDENTICAL_JSON_TRANSPORT_RETRY"
        ),
        "confirmatory_claim_allowed": False,
        "future_policy": (
            "V35_MUST_QUALIFY_BEFORE_ONE_NEW_VIDEO_DISJOINT_V36_FORMAL"
        ),
    }
    prereg_path = REPO_ROOT / V35_PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": V35_PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": parent_grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_protocol_sha256,
    })
    config["postground"].update({
        "adapter_module": ADAPTER,
        "adapter_module_sha256": _sha256(adapter_path),
        "evaluator_module": COLLECTOR,
        "evaluator_module_sha256": _sha256(collector_path),
        "core_evaluator_module": CORE_EVALUATOR,
        "core_evaluator_module_sha256": _sha256(core_evaluator_path),
        "target_grounder_sha256": postground_target_grounder_sha256,
        "evaluation_protocol_sha256": postground_protocol_sha256,
        "formal_gates": gates,
    })
    config_path = REPO_ROOT / V35_CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v35_status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_grounder_sha256": parent_grounder_sha256,
        "postground_target_grounder_sha256": (
            postground_target_grounder_sha256
        ),
        "postground_evaluation_protocol_sha256": (
            postground_protocol_sha256
        ),
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
