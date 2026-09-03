#!/usr/bin/env python3
"""Freeze one untouched, video-disjoint V56 formal test for the V54 rule."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_asymmetric_support_calibrator import (  # noqa: E402
    asymmetric_target_grounder_sha256,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)
import scripts.freeze_agqa2_asymmetric_support_v55_qualification as v55  # noqa: E402
import scripts.freeze_agqa2_robust_temporal_v34_formal as v34  # noqa: E402


NONCE = "agqa2-v56-asymmetric-support-untouched-test-formal-300"
SAMPLE_COUNT = 300
QUALIFICATION_REPORT = (
    "runs/agqa2_asymmetric_support_v55_qualification/report.json"
)
QUALIFICATION_SUMMARY = (
    "docs/results/agqa2_asymmetric_support_v55_qualification_summary.json"
)
TRAINING_ARTIFACT = "configs/agqa2_asymmetric_support_v54/training_artifact.json"
PARENT_CONFIG = "configs/agqa2_asymmetric_support_v55_qualification.json"
TEST_METADATA_MANIFEST = "configs/agqa2_temporal_selective_v19_development_manifest.json"
SELECTION = "configs/agqa2_asymmetric_support_v56_formal_selection.json"
MANIFEST = "configs/agqa2_asymmetric_support_v56_formal_manifest.json"
PREREG = "configs/agqa2_asymmetric_support_v56_formal_preregistration.json"
CONFIG = "configs/agqa2_asymmetric_support_v56_formal.json"
DOWNLOAD_RECEIPT = "runs/agqa2_asymmetric_support_v56_download/receipt.json"
AGGREGATE_ADAPTER = "src/motif_transfer/agqa_aggregate_temporal_transfer.py"
CALIBRATOR = "src/motif_transfer/agqa_asymmetric_support_calibrator.py"
EVALUATOR = "scripts/evaluate_agqa2_asymmetric_support_v55.py"
FORMAL_COLLECTOR = "scripts/collect_agqa2_asymmetric_support_v56_formal.py"


_sha256 = v55._sha256
_verified = v55._verified


def _write_qualification_summary(
    report: Mapping[str, Any], *, parent_grounder_sha256: str,
) -> Path:
    path = REPO_ROOT / QUALIFICATION_SUMMARY
    core = {
        "schema_version": "agqa2-asymmetric-support-v55-qualification-summary-v1",
        "status": report["status"],
        "grounder_qualified": (
            report["status"]
            == "AGQA2_ASYMMETRIC_SUPPORT_V55_QUALIFICATION_QUALIFIED"
            and all(report["qualification_gates"].values())
        ),
        # The base collector verifies the unchanged neural acquisition identity.
        # The post-ground calibrator identity is frozen independently below.
        "grounder_sha256": parent_grounder_sha256,
        "postground_target_grounder_sha256": report["target_grounder_sha256"],
        "source_program_sha256": report["source_program_sha256"],
        "rows": report["rows"],
        "source_executor_authorizations": report["source_executor_authorizations"],
        "source_vs_target_native": report["source_vs_target_native"],
        "qualification_gates": report["qualification_gates"],
        "qualification_report_sha256": report["report_sha256"],
        # Compatibility alias consumed by the unchanged reserve collector.
        "report_sha256": report["report_sha256"],
        "qualification_report_file_sha256": _sha256(REPO_ROOT / QUALIFICATION_REPORT),
        "confirmatory_claim": False,
    }
    summary = core | {"summary_sha256": stable_hash(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def _new_selection(metadata: Mapping[str, Any]) -> dict[str, Any]:
    # V34's selector reads only public questions and program structure.  It does
    # not read answers, scene graphs, direct predictions, or any video pixels.
    source = deepcopy(metadata)
    source["entry"] = "AGQA_balanced/test_balanced.txt"
    excluded = _configured_video_ids()
    excluded.update(
        path.stem for path in Path(source["video_root"]).glob("*.mp4")
    )
    old_nonce, old_count = v34.NONCE, v34.SAMPLE_COUNT
    try:
        v34.NONCE, v34.SAMPLE_COUNT = NONCE, SAMPLE_COUNT
        inherited = v34._selection(source, excluded)
    finally:
        v34.NONCE, v34.SAMPLE_COUNT = old_nonce, old_count
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.pop("prior_v34_neural_grounder_exposure", None)
    core.update({
        "schema_version": "agqa2-asymmetric-support-selection-v56-formal",
        "status": "FROZEN_V56_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V56_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "THREE_HUNDRED_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_ATOMIC_"
            "BEFORE_AFTER_ROWS_FROM_OFFICIAL_TEST_METADATA;EXACT_V54_"
            "ASYMMETRIC_SUPPORT_RULE;ONE_CONFIRMATORY_FORMAL_RUN"
        ),
        "selection_nonce": NONCE,
        "selection_metadata_split": "official_test_balanced",
        "prior_v56_neural_grounder_exposure": False,
        "answer_read_during_freeze": False,
        "v55_qualification_videos_excluded": True,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: Mapping[str, Any]) -> dict[str, Any]:
    inherited = v34._seal(selection)
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-asymmetric-support-manifest-v56-formal",
        "status": "FROZEN_V56_TEST_FORMAL_VIDEOS_UNSEEN",
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run = REPO_ROOT / "runs/agqa2_asymmetric_support_v56_formal"
    if run.exists() and any(run.rglob("*.json")):
        raise RuntimeError("V56 formal already has runtime artifacts")

    report_path = REPO_ROOT / QUALIFICATION_REPORT
    report = _verified(report_path, "report_sha256")
    if (
        report["status"]
        != "AGQA2_ASYMMETRIC_SUPPORT_V55_QUALIFICATION_QUALIFIED"
        or not all(report["qualification_gates"].values())
        or report.get("confirmatory_claim")
    ):
        raise ValueError("V55 fresh qualification did not pass every frozen gate")
    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    qualified_parent_grounder = parent["expected_grounder_sha256"]
    summary_path = _write_qualification_summary(
        report, parent_grounder_sha256=qualified_parent_grounder,
    )

    artifact_path = REPO_ROOT / TRAINING_ARTIFACT
    artifact = _verified(artifact_path, "artifact_sha256")
    if artifact["artifact_sha256"] != report["calibration_artifact_sha256"]:
        raise ValueError("V54 artifact differs from the qualified V55 artifact")
    if artifact["rule"]["rule_sha256"] != report["calibration_rule_sha256"]:
        raise ValueError("V54 rule differs from the qualified V55 rule")

    metadata = _verified(REPO_ROOT / TEST_METADATA_MANIFEST, "manifest_sha256")
    selection_path = REPO_ROOT / SELECTION
    selection = (
        _verified(selection_path, "manifest_sha256")
        if selection_path.exists() else _new_selection(metadata)
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
            "sample_count": SAMPLE_COUNT,
            "missing_video_count": len(missing),
            "missing_video_ids": missing,
            "next": "download the exact frozen test videos, then rerun",
        }, indent=2, sort_keys=True))
        return

    receipt_path = REPO_ROOT / DOWNLOAD_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]
        or len(receipt.get("videos") or ()) != SAMPLE_COUNT
    ):
        raise ValueError("V56 download receipt is incomplete or mismatched")
    manifest = _seal(selection)
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-asymmetric-support-v56-formal-config-v1",
        "status": "FROZEN_V56_ASYMMETRIC_SUPPORT_FORMAL",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "development_qualification_report": QUALIFICATION_SUMMARY,
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V56_FORMAL_BASE",
    })
    # Keep the V55 base-receipt gates byte-for-byte equivalent apart from the
    # inherited JSON copy; formal transfer gates are frozen separately below.
    sources, _ = _load_sources(config)
    parent_grounder = stable_hash(_grounder_semantic_core(config, sources))
    if parent_grounder != qualified_parent_grounder:
        raise AssertionError("V56 parent acquisition identity drift")
    base_evaluation = stable_hash(_evaluation_protocol_core(config))
    if base_evaluation != parent["expected_evaluation_protocol_sha256"]:
        raise AssertionError("V56 changed the qualified base evaluation protocol")

    aggregate_path = REPO_ROOT / AGGREGATE_ADAPTER
    calibrator_path = REPO_ROOT / CALIBRATOR
    evaluator_path = REPO_ROOT / EVALUATOR
    collector_path = REPO_ROOT / FORMAL_COLLECTOR
    target_grounder = asymmetric_target_grounder_sha256(
        parent_grounder_sha256=parent_grounder,
        aggregate_adapter_sha256=_sha256(aggregate_path),
        normalization_module_sha256=config[
            "syntax_transport_normalization"
        ]["normalization_module_sha256"],
        acquisition_collector_sha256=config["grounder"]["collector_sha256"],
        calibrator_module_sha256=_sha256(calibrator_path),
        calibration_artifact_sha256=artifact["artifact_sha256"],
    )
    if target_grounder != report["target_grounder_sha256"]:
        raise AssertionError("V56 changed the V55-qualified target grounder")
    if config["postground"]["source_program_sha256"] != report["source_program_sha256"]:
        raise AssertionError("V56 changed the source-induced symbolic program")

    gates = dict(parent["postground"]["formal_gates"])
    expected_gates = {
        "required_valid_rows": SAMPLE_COUNT,
        "required_unique_videos": SAMPLE_COUNT,
        "minimum_source_authorizations": 35,
        "minimum_source_wins": 9,
        "maximum_source_losses": 1,
        "minimum_source_minus_target_correct": 8,
        "maximum_exact_one_sided_pvalue": 0.05,
        "required_effect_shuffled_abstentions": SAMPLE_COUNT,
        "required_wrong_source_abstentions": SAMPLE_COUNT,
        "required_generic_scaffold_matches": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 2.70,
    }
    if gates != expected_gates:
        raise AssertionError("V56 formal gates differ from frozen V55 gates")
    route_calibration = dict(parent["postground"]["development_calibration"])
    protocol = {
        "schema_version": "agqa2-asymmetric-support-v56-formal-protocol-v1",
        "sample_count": SAMPLE_COUNT,
        "primary_endpoint": "SOURCE_INDUCED_VS_TARGET_NATIVE_PAIRED_ACCURACY",
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "target_grounder_sha256": target_grounder,
        "target_executor_sha256": config["postground"]["target_executor_sha256"],
        "aggregate_adapter_sha256": _sha256(aggregate_path),
        "calibrator_module_sha256": _sha256(calibrator_path),
        "calibration_artifact_sha256": artifact["artifact_sha256"],
        "calibration_rule_sha256": artifact["rule"]["rule_sha256"],
        "evaluator_module_sha256": _sha256(evaluator_path),
        "formal_collector_module_sha256": _sha256(collector_path),
        "development_calibration": route_calibration,
        "runtime_calibrator_authority": (
            "ABSTENTION_ONLY;NO_INTERVAL_RELATION_OR_BINDING_CREATION_OR_EDIT"
        ),
        "runtime_features": artifact["feature_space"],
        "fallback": "PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION",
        "controls": [
            "SOURCE_EFFECT_SHUFFLED", "WRONG_SOURCE_TEMPORAL_SINGLE",
            "HANDWRITTEN_GENERIC_EQUIVALENT", "TARGET_WRITTEN_EQUIVALENT",
        ],
        "formal_gates": gates,
        "current_outcome_authorization": False,
    }
    protocol_sha = stable_hash(protocol)
    prereg = {
        "schema_version": "agqa2-asymmetric-support-v56-preregistration-v1",
        # Compatibility status consumed by the unchanged acquisition core.
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v56_status": "FROZEN_BEFORE_ANY_V56_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "qualified_v55_report_sha256": report["report_sha256"],
        "qualified_v55_report_file_sha256": _sha256(report_path),
        # Compatibility alias consumed only as route evidence identity by the
        # unchanged V34 post-ground evaluator.
        "qualified_v33_development_report_sha256": artifact["artifact_sha256"],
        "qualified_parent_grounder_sha256": parent_grounder,
        "qualified_target_grounder_sha256": target_grounder,
        "v54_training_artifact_sha256": artifact["artifact_sha256"],
        "v54_calibration_rule_sha256": artifact["rule"]["rule_sha256"],
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "target_executor_sha256": config["postground"]["target_executor_sha256"],
        "source_program_induced_from_game_interventions": True,
        "source_program_target_data_read": False,
        "development_calibration": route_calibration,
        "base_evaluation_protocol_sha256": base_evaluation,
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "formal_gates": gates,
        "cost_projection": {
            "v55_observed_300_row_cost_usd": report["reported_provider_cost_usd"],
            "projected_300_row_cost_usd": report["reported_provider_cost_usd"],
            "frozen_cap_usd": 2.70,
        },
        "failure_policy": {
            "formal": "RUN_ONCE_ON_FROZEN_POOL;NO_POST_OUTCOME_ADAPTATION",
            "transport_failure": "RETRY_ONLY_MISSING_RECEIPTS;KEEP_FIXED_POOL",
            "failed_gate": "REPORT_NOT_QUALIFIED;DO_NOT_RESAMPLE",
        },
        "confirmatory_claim_allowed": True,
    }
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": parent_grounder,
        "expected_evaluation_protocol_sha256": base_evaluation,
    })
    config["postground"].update({
        "target_grounder_sha256": target_grounder,
        "development_calibration": route_calibration,
        "evaluation_protocol_sha256": protocol_sha,
        "formal_gates": gates,
        "formal_collector_module": FORMAL_COLLECTOR,
        "formal_collector_module_sha256": _sha256(collector_path),
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v56_status"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": SAMPLE_COUNT,
        "parent_grounder_sha256": parent_grounder,
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "provider_cost_cap_usd": 2.70,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
