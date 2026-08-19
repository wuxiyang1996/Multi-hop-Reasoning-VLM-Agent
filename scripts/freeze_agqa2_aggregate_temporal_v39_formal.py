#!/usr/bin/env python3
"""Freeze a fresh video-disjoint V39 aggregate-temporal formal test."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_aggregate_temporal_transfer import (  # noqa: E402
    aggregate_target_grounder_sha256,
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
import scripts.freeze_agqa2_robust_temporal_v34_formal as v34  # noqa: E402


NONCE = "agqa2-v39-aggregate-recurrent-temporal-formal-100"
SAMPLE_COUNT = 100
DEVELOPMENT_REPORT = "runs/agqa2_aggregate_temporal_v38_development/report.json"
DEVELOPMENT_MANIFEST = "configs/agqa2_temporal_selective_v19_development_manifest.json"
PARENT_CONFIG = "configs/agqa2_robust_temporal_v36_development.json"
SELECTION = "configs/agqa2_aggregate_temporal_v39_formal_selection.json"
MANIFEST = "configs/agqa2_aggregate_temporal_v39_formal_manifest.json"
PREREG = "configs/agqa2_aggregate_temporal_v39_formal_preregistration.json"
CONFIG = "configs/agqa2_aggregate_temporal_v39_formal.json"
DOWNLOAD_RECEIPT = "runs/agqa2_aggregate_temporal_v39_download/receipt.json"
ADAPTER = "src/motif_transfer/agqa_aggregate_temporal_transfer.py"
EVALUATOR = "scripts/collect_agqa2_aggregate_temporal_v39_formal.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {path}")
    return value


def _verified_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"manifest hash mismatch: {path}")
    return value


def _new_selection(development: Mapping[str, Any]) -> dict[str, Any]:
    excluded = _configured_video_ids()
    excluded.update(
        path.stem for path in Path(development["video_root"]).glob("*.mp4")
    )
    old_nonce, old_count = v34.NONCE, v34.SAMPLE_COUNT
    try:
        v34.NONCE, v34.SAMPLE_COUNT = NONCE, SAMPLE_COUNT
        inherited = v34._selection(development, excluded)
    finally:
        v34.NONCE, v34.SAMPLE_COUNT = old_nonce, old_count
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.pop("prior_v34_neural_grounder_exposure", None)
    core.update({
        "schema_version": "agqa2-aggregate-temporal-selection-v39-formal",
        "status": "FROZEN_V39_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V39_CALLS",
        "claim_boundary": (
            "ONE_HUNDRED_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_ATOMIC_"
            "BEFORE_AFTER_ROWS;EXACT_V38_OPERATOR_RECURRENCE_RULE;FORMAL"
        ),
        "selection_nonce": NONCE,
        "prior_v39_neural_grounder_exposure": False,
        "v38_development_video_ids_excluded": True,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: Mapping[str, Any]) -> dict[str, Any]:
    samples = []
    for row in selection["samples"]:
        path = Path(row["video_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append(dict(row) | {
            "video_sha256": _sha256(path),
            "video_bytes": path.stat().st_size,
        })
    core = {
        key: deepcopy(value) for key, value in selection.items()
        if key not in {"manifest_sha256", "raw_video_archive"}
    }
    core.update({
        "schema_version": "agqa2-aggregate-temporal-manifest-v39-formal",
        "status": "FROZEN_V39_RAW_VIDEO_UNSEEN_BEFORE_FORMAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": len(samples),
        "local_integrity_decode_probe_completed": True,
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_aggregate_temporal_v39_formal"
    if (run_root / "report.json").exists():
        raise RuntimeError("V39 formal report already exists")
    development_report_path = REPO_ROOT / DEVELOPMENT_REPORT
    development_report = _verified_report(development_report_path)
    if (
        development_report["status"]
        != "AGQA2_AGGREGATE_TEMPORAL_V38_DEVELOPMENT_METHOD_SELECTED"
        or not all(development_report["qualification_gates"].values())
        or not development_report[
            "method_selected_after_v37_development_outcome_access"
        ]
    ):
        raise ValueError("V38 did not select the aggregate recurrence method")
    development = _verified_manifest(REPO_ROOT / DEVELOPMENT_MANIFEST)
    selection_path = REPO_ROOT / SELECTION
    selection = (
        _verified_manifest(selection_path)
        if selection_path.exists() else _new_selection(development)
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
            "next": "download the exact frozen videos, then rerun this freezer",
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
        raise ValueError("V39 download receipt is incomplete or mismatched")
    manifest = _seal(selection)
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-aggregate-temporal-v39-formal-config-v1",
        "status": "FROZEN_V39_AGGREGATE_TEMPORAL_FORMAL",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "report_version": "V39_BASE",
    })
    sources, _ = _load_sources(config)
    parent_grounder = stable_hash(_grounder_semantic_core(config, sources))
    base_evaluation = stable_hash(_evaluation_protocol_core(config))
    adapter_path = REPO_ROOT / ADAPTER
    evaluator_path = REPO_ROOT / EVALUATOR
    normalization = config["syntax_transport_normalization"]
    target_grounder = aggregate_target_grounder_sha256(
        parent_grounder_sha256=parent_grounder,
        adapter_module_sha256=_sha256(adapter_path),
        normalization_module_sha256=normalization[
            "normalization_module_sha256"
        ],
        acquisition_collector_sha256=config["grounder"]["collector_sha256"],
    )
    if target_grounder != development_report["target_grounder_sha256"]:
        raise AssertionError("V39 changed the V38 aggregate target grounder")
    calibration = development_report["source_vs_target_native"]
    route_calibration = {
        "wins": calibration["wins"],
        "losses": calibration["losses"],
        "ties": calibration["ties"],
        "decision": "SELECT_SKILL",
        "reason": "V38_OPERATOR_LEVEL_RECURRENCE_METHOD_SELECTION",
    }
    gates = {
        "required_valid_rows": SAMPLE_COUNT,
        "required_unique_videos": SAMPLE_COUNT,
        "minimum_source_authorizations": 20,
        "minimum_source_wins": 7,
        "maximum_source_losses": 1,
        "minimum_source_minus_target_correct": 5,
        "maximum_exact_one_sided_pvalue": 0.05,
        "required_effect_shuffled_abstentions": SAMPLE_COUNT,
        "required_wrong_source_abstentions": SAMPLE_COUNT,
        "required_generic_scaffold_matches": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 0.9,
    }
    protocol = {
        "schema_version": "agqa2-aggregate-temporal-v39-formal-protocol-v1",
        "sample_count": SAMPLE_COUNT,
        "primary_endpoint": "SOURCE_INDUCED_VS_TARGET_NATIVE_PAIRED_ACCURACY",
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "target_grounder_sha256": target_grounder,
        "target_executor_sha256": config["postground"]["target_executor_sha256"],
        "adapter_module_sha256": _sha256(adapter_path),
        "evaluator_module_sha256": _sha256(evaluator_path),
        "development_calibration": route_calibration,
        "binding_rule": (
            "BINARY_ARITY_GROUNDED;OPERATOR_LEVEL_RECURRENCE_MINIMUM_THREE_"
            "TOTAL_VIEWS;ALL_CROSS_VIEW_PAIRS_STRICT_AND_CONSISTENT"
        ),
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
        "schema_version": "agqa2-aggregate-temporal-v39-preregistration-v1",
        # Compatibility status consumed by the unchanged outcome-blind core.
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v39_status": "FROZEN_BEFORE_ANY_V39_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        # Compatibility key; its value is the actual V38 route calibration.
        "qualified_v33_development_report_sha256": development_report[
            "report_sha256"
        ],
        "v38_method_selection_report_sha256": development_report[
            "report_sha256"
        ],
        "v38_method_selection_report_file_sha256": _sha256(
            development_report_path
        ),
        "source_program_sha256": config["postground"]["source_program_sha256"],
        "source_program_induced_from_interventions": True,
        "source_program_target_data_read": False,
        "development_calibration": route_calibration,
        "base_evaluation_protocol_sha256": base_evaluation,
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "formal_gates": gates,
        "cost_projection": {
            "v36_mean_cost_per_row_usd": (
                development_report["reported_provider_cost_usd"] / 100.0
            ),
            "projected_100_row_cost_usd": development_report[
                "reported_provider_cost_usd"
            ],
            "frozen_cap_usd": 0.9,
        },
        "failure_policy": {
            "formal": "RUN_ONCE_ON_FROZEN_POOL;NO_POST_OUTCOME_ADAPTATION",
            "failed_gate": "REPORT_NOT_QUALIFIED;DO_NOT_RESAMPLE",
        },
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
        "adapter_module": ADAPTER,
        "adapter_module_sha256": _sha256(adapter_path),
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
        "status": prereg["v39_status"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": SAMPLE_COUNT,
        "parent_grounder_sha256": parent_grounder,
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "provider_cost_cap_usd": 0.9,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
