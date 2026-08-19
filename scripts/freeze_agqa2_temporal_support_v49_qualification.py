#!/usr/bin/env python3
"""Freeze a 200-row fresh AGQA train qualification for the V48 rule."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_temporal_support_calibrator import (  # noqa: E402
    temporal_support_target_grounder_sha256,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)
import scripts.freeze_agqa2_interval_reliability_v46_qualification as v46  # noqa: E402


NONCE = "agqa2-v49-temporal-support-train-qualification-200"
SAMPLE_COUNT = 200
TRAINING_ARTIFACT = "configs/agqa2_temporal_support_v48/training_artifact.json"
PARENT_CONFIG = "configs/agqa2_interval_reliability_v47_qualification.json"
PARENT_MANIFEST = "configs/agqa2_temporal_selective_v19_development_manifest.json"
SELECTION = "configs/agqa2_temporal_support_v49_qualification_selection.json"
MANIFEST = "configs/agqa2_temporal_support_v49_qualification_manifest.json"
PREREG = "configs/agqa2_temporal_support_v49_qualification_preregistration.json"
CONFIG = "configs/agqa2_temporal_support_v49_qualification.json"
DOWNLOAD_RECEIPT = "runs/agqa2_temporal_support_v49_download/receipt.json"
AGGREGATE_ADAPTER = "src/motif_transfer/agqa_aggregate_temporal_transfer.py"
CALIBRATOR = "src/motif_transfer/agqa_temporal_support_calibrator.py"
EVALUATOR = "scripts/evaluate_agqa2_temporal_support_v49.py"


_sha256 = v46._sha256
_verified = v46._verified


def _new_selection(parent_manifest: Mapping[str, Any]) -> dict[str, Any]:
    old_nonce, old_count = v46.NONCE, v46.SAMPLE_COUNT
    try:
        v46.NONCE, v46.SAMPLE_COUNT = NONCE, SAMPLE_COUNT
        inherited = v46._new_selection(parent_manifest)
    finally:
        v46.NONCE, v46.SAMPLE_COUNT = old_nonce, old_count
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-temporal-support-selection-v49",
        "status": "FROZEN_V49_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V49_CALLS",
        "split": "development",
        "claim_boundary": (
            "TWO_HUNDRED_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_ATOMIC_BEFORE_"
            "AFTER_ROWS_FROM_OFFICIAL_TRAIN_METADATA;V48_TEMPORAL_SUPPORT_"
            "RULE_QUALIFICATION_ONLY"
        ),
        "selection_nonce": NONCE,
        "selection_metadata_split": "official_train_balanced",
        "prior_v49_neural_grounder_exposure": False,
        "answer_read_during_freeze": False,
    })
    core.pop("prior_v46_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: Mapping[str, Any]) -> dict[str, Any]:
    inherited = v46._seal(selection)
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-temporal-support-manifest-v49",
        "status": "FROZEN_V49_TRAIN_QUALIFICATION_VIDEOS_UNSEEN",
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run = REPO_ROOT / "runs/agqa2_temporal_support_v49_qualification"
    if run.exists() and any(run.rglob("*.json")):
        raise RuntimeError("V49 qualification already has runtime artifacts")
    artifact_path = REPO_ROOT / TRAINING_ARTIFACT
    artifact = _verified(artifact_path, "artifact_sha256")
    if artifact["status"] != (
        "V48_TRAINED_ON_500_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION"
    ):
        raise ValueError("V48 training artifact is not frozen")
    parent_manifest = _verified(REPO_ROOT / PARENT_MANIFEST, "manifest_sha256")
    selection_path = REPO_ROOT / SELECTION
    selection = (
        _verified(selection_path, "manifest_sha256")
        if selection_path.exists() else _new_selection(parent_manifest)
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = [
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    ]
    if missing:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "sample_count": selection["sample_count"],
            "missing_video_count": len(missing),
            "missing_video_ids": missing,
            "next": "download the exact frozen train videos, then rerun",
        }, indent=2, sort_keys=True))
        return

    receipt_path = REPO_ROOT / DOWNLOAD_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]
        or len(receipt.get("videos") or ()) != SAMPLE_COUNT
    ):
        raise ValueError("V49 download receipt is incomplete or mismatched")
    manifest = _seal(selection)
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    config = deepcopy(parent)
    config.pop("interval_reliability_calibration", None)
    config.update({
        "schema_version": "agqa2-temporal-support-v49-qualification-config-v1",
        "status": "FROZEN_V49_TEMPORAL_SUPPORT_QUALIFICATION",
        "split": "development", "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST, "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "report_version": "V49_QUALIFICATION_BASE",
    })
    config["qualification_gates"] = {
        "required_valid_runtime_rows": SAMPLE_COUNT,
        "minimum_route_correct": SAMPLE_COUNT,
        "minimum_decisive_executions": SAMPLE_COUNT + 1,
        "minimum_decisive_accuracy": 0.0,
        "minimum_typed_vs_direct_wins": 0,
        "maximum_typed_vs_direct_losses": SAMPLE_COUNT,
        "required_source_permuted_abstentions": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 1.80,
    }
    sources, _ = _load_sources(config)
    parent_grounder = stable_hash(_grounder_semantic_core(config, sources))
    if parent_grounder != parent["expected_grounder_sha256"]:
        raise AssertionError("V49 changed the frozen target-native acquisition")
    base_evaluation = stable_hash(_evaluation_protocol_core(config))
    aggregate_path, calibrator_path = (
        REPO_ROOT / AGGREGATE_ADAPTER, REPO_ROOT / CALIBRATOR,
    )
    evaluator_path = REPO_ROOT / EVALUATOR
    target_grounder = temporal_support_target_grounder_sha256(
        parent_grounder_sha256=parent_grounder,
        aggregate_adapter_sha256=_sha256(aggregate_path),
        normalization_module_sha256=config[
            "syntax_transport_normalization"
        ]["normalization_module_sha256"],
        acquisition_collector_sha256=config["grounder"]["collector_sha256"],
        calibrator_module_sha256=_sha256(calibrator_path),
        calibration_artifact_sha256=artifact["artifact_sha256"],
    )
    training = artifact["rule"]
    route_calibration = {
        "wins": training["training_wins"], "losses": training["training_losses"],
        "ties": training["training_ties"], "decision": "SELECT_SKILL",
        "reason": "V48_RISK_FIRST_FINITE_CLASS_TEMPORAL_SUPPORT_INDUCTION",
    }
    gates = {
        "required_valid_rows": SAMPLE_COUNT,
        "required_unique_videos": SAMPLE_COUNT,
        "minimum_source_authorizations": 35,
        "minimum_source_wins": 8,
        "maximum_source_losses": 1,
        "minimum_source_minus_target_correct": 7,
        "maximum_exact_one_sided_pvalue": 0.05,
        "required_effect_shuffled_abstentions": SAMPLE_COUNT,
        "required_wrong_source_abstentions": SAMPLE_COUNT,
        "required_generic_scaffold_matches": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 1.80,
    }
    protocol = {
        "schema_version": "agqa2-temporal-support-v49-qualification-protocol-v1",
        "sample_count": SAMPLE_COUNT,
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "target_grounder_sha256": target_grounder,
        "target_executor_sha256": config["postground"]["target_executor_sha256"],
        "aggregate_adapter_sha256": _sha256(aggregate_path),
        "calibrator_module_sha256": _sha256(calibrator_path),
        "calibration_artifact_sha256": artifact["artifact_sha256"],
        "calibration_rule_sha256": artifact["rule"]["rule_sha256"],
        "evaluator_module_sha256": _sha256(evaluator_path),
        "runtime_calibrator_authority": (
            "ABSTENTION_ONLY;NO_INTERVAL_RELATION_OR_BINDING_CREATION_OR_EDIT"
        ),
        "runtime_features": artifact["feature_space"],
        "fallback": "PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION",
        "gates": gates, "confirmatory_claim": False,
    }
    protocol_sha = stable_hash(protocol)
    prereg = {
        "schema_version": "agqa2-temporal-support-v49-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v49_status": "FROZEN_BEFORE_ANY_V49_QUALIFICATION_PROVIDER_OR_OUTCOME_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        # Required compatibility name in the unchanged V34 execution core.
        "qualified_v33_development_report_sha256": artifact["artifact_sha256"],
        "qualified_development_artifact_sha256": artifact["artifact_sha256"],
        "v48_training_artifact": TRAINING_ARTIFACT,
        "v48_training_artifact_file_sha256": _sha256(artifact_path),
        "v48_training_artifact_sha256": artifact["artifact_sha256"],
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "development_calibration": route_calibration,
        "base_evaluation_protocol_sha256": base_evaluation,
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "qualification_gates": gates,
        "cost_projection": {"projected_200_row_cost_usd": 1.40,
                            "frozen_cap_usd": 1.80},
        "failure_policy": {
            "qualification": "RUN_ONCE;NO_POST_OUTCOME_THRESHOLD_CHANGE",
            "failed_gate": "STOP_BEFORE_NEW_TEST_FORMAL",
            "passed": "FREEZE_ONE_NEW_VIDEO_DISJOINT_TEST_FORMAL",
        },
        "confirmatory_claim_allowed": False,
    }
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": parent_grounder,
        "expected_evaluation_protocol_sha256": base_evaluation,
        "temporal_support_calibration": {
            "module": CALIBRATOR, "module_sha256": _sha256(calibrator_path),
            "artifact": TRAINING_ARTIFACT,
            "artifact_file_sha256": _sha256(artifact_path),
            "artifact_sha256": artifact["artifact_sha256"],
            "rule_sha256": artifact["rule"]["rule_sha256"],
        },
    })
    config["postground"].update({
        "adapter_module": AGGREGATE_ADAPTER,
        "adapter_module_sha256": _sha256(aggregate_path),
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "target_grounder_sha256": target_grounder,
        "development_calibration": route_calibration,
        "evaluation_protocol_sha256": protocol_sha,
        "formal_gates": gates,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v49_status"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": SAMPLE_COUNT, "parent_grounder_sha256": parent_grounder,
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "provider_cost_cap_usd": 1.80,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
