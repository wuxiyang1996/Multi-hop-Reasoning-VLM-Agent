#!/usr/bin/env python3
"""Freeze a fresh AGQA train qualification for the frozen V45 rule."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_interval_reliability_calibrator import (  # noqa: E402
    interval_calibrated_target_grounder_sha256,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)
import scripts.freeze_agqa2_view_reliability_v43_qualification as v43  # noqa: E402


NONCE = "agqa2-v46-interval-reliability-train-qualification-150"
SAMPLE_COUNT = 150
TRAINING_ARTIFACT = "configs/agqa2_interval_reliability_v45/training_artifact.json"
PARENT_CONFIG = "configs/agqa2_view_reliability_v43_qualification.json"
PARENT_MANIFEST = "configs/agqa2_temporal_selective_v19_development_manifest.json"
SELECTION = "configs/agqa2_interval_reliability_v46_qualification_selection.json"
MANIFEST = "configs/agqa2_interval_reliability_v46_qualification_manifest.json"
PREREG = "configs/agqa2_interval_reliability_v46_qualification_preregistration.json"
CONFIG = "configs/agqa2_interval_reliability_v46_qualification.json"
DOWNLOAD_RECEIPT = "runs/agqa2_interval_reliability_v46_download/receipt.json"
AGGREGATE_ADAPTER = "src/motif_transfer/agqa_aggregate_temporal_transfer.py"
CALIBRATOR = "src/motif_transfer/agqa_interval_reliability_calibrator.py"
EVALUATOR = "scripts/evaluate_agqa2_interval_reliability_v46.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"hash mismatch: {path}")
    return value


def _new_selection(parent_manifest: Mapping[str, Any]) -> dict[str, Any]:
    old_nonce, old_count = v43.NONCE, v43.SAMPLE_COUNT
    try:
        v43.NONCE, v43.SAMPLE_COUNT = NONCE, SAMPLE_COUNT
        inherited = v43._new_selection(parent_manifest)
    finally:
        v43.NONCE, v43.SAMPLE_COUNT = old_nonce, old_count
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-interval-reliability-selection-v46",
        "status": "FROZEN_V46_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V46_CALLS",
        "split": "development",
        "claim_boundary": (
            "ONE_HUNDRED_FIFTY_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_ATOMIC_"
            "BEFORE_AFTER_ROWS_FROM_OFFICIAL_TRAIN_METADATA;V45_INTERVAL_"
            "RELIABILITY_RULE_QUALIFICATION_ONLY"
        ),
        "selection_nonce": NONCE,
        "selection_metadata_split": "official_train_balanced",
        "prior_v46_neural_grounder_exposure": False,
        "answer_read_during_freeze": False,
    })
    core.pop("prior_v43_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: Mapping[str, Any]) -> dict[str, Any]:
    inherited = v43._seal(selection)
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-interval-reliability-manifest-v46",
        "status": "FROZEN_V46_TRAIN_QUALIFICATION_VIDEOS_UNSEEN",
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run = REPO_ROOT / "runs/agqa2_interval_reliability_v46_qualification"
    if run.exists() and any(run.rglob("*.json")):
        raise RuntimeError("V46 qualification already has runtime artifacts")

    artifact_path = REPO_ROOT / TRAINING_ARTIFACT
    artifact = _verified(artifact_path, "artifact_sha256")
    if artifact["status"] != (
        "V45_TRAINED_ON_350_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION"
    ):
        raise ValueError("V45 training artifact is not frozen")
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
        or receipt.get("selection_manifest_sha256")
        != selection["manifest_sha256"]
        or len(receipt.get("videos") or ()) != SAMPLE_COUNT
    ):
        raise ValueError("V46 download receipt is incomplete or mismatched")
    manifest = _seal(selection)
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    config = deepcopy(parent)
    config.pop("view_reliability_calibration", None)
    config.update({
        "schema_version": "agqa2-interval-reliability-v46-qualification-config-v1",
        "status": "FROZEN_V46_INTERVAL_RELIABILITY_QUALIFICATION",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "report_version": "V46_QUALIFICATION_BASE",
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
        "maximum_reported_provider_cost_usd": 1.35,
    }
    sources, _ = _load_sources(config)
    parent_grounder = stable_hash(_grounder_semantic_core(config, sources))
    if parent_grounder != parent["expected_grounder_sha256"]:
        raise AssertionError("V46 changed the frozen target-native acquisition")
    base_evaluation = stable_hash(_evaluation_protocol_core(config))
    aggregate_path = REPO_ROOT / AGGREGATE_ADAPTER
    calibrator_path = REPO_ROOT / CALIBRATOR
    evaluator_path = REPO_ROOT / EVALUATOR
    target_grounder = interval_calibrated_target_grounder_sha256(
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
        "wins": training["training_wins"],
        "losses": training["training_losses"],
        "ties": training["training_ties"],
        "decision": "SELECT_SKILL",
        "reason": "V45_FINITE_CLASS_INTERVAL_RELIABILITY_INDUCTION",
    }
    gates = {
        "required_valid_rows": SAMPLE_COUNT,
        "required_unique_videos": SAMPLE_COUNT,
        "minimum_source_authorizations": 25,
        "minimum_source_wins": 7,
        "maximum_source_losses": 1,
        "minimum_source_minus_target_correct": 6,
        "maximum_exact_one_sided_pvalue": 0.05,
        "required_effect_shuffled_abstentions": SAMPLE_COUNT,
        "required_wrong_source_abstentions": SAMPLE_COUNT,
        "required_generic_scaffold_matches": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 1.35,
    }
    protocol = {
        "schema_version": "agqa2-interval-reliability-v46-qualification-protocol-v1",
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
        "gates": gates,
        "confirmatory_claim": False,
    }
    protocol_sha = stable_hash(protocol)
    prereg = {
        "schema_version": "agqa2-interval-reliability-v46-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v46_status": (
            "FROZEN_BEFORE_ANY_V46_QUALIFICATION_PROVIDER_OR_OUTCOME_CALL"
        ),
        "claim_boundary": manifest["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "qualified_development_artifact_sha256": artifact["artifact_sha256"],
        "v45_training_artifact": TRAINING_ARTIFACT,
        "v45_training_artifact_file_sha256": _sha256(artifact_path),
        "v45_training_artifact_sha256": artifact["artifact_sha256"],
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "development_calibration": route_calibration,
        "base_evaluation_protocol_sha256": base_evaluation,
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "qualification_gates": gates,
        "cost_projection": {
            "projected_150_row_cost_usd": 1.05,
            "frozen_cap_usd": 1.35,
        },
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
        "interval_reliability_calibration": {
            "module": CALIBRATOR,
            "module_sha256": _sha256(calibrator_path),
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
        "status": prereg["v46_status"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": SAMPLE_COUNT,
        "parent_grounder_sha256": parent_grounder,
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "provider_cost_cap_usd": 1.35,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
