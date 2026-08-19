#!/usr/bin/env python3
"""Freeze a fresh V25 reserve for the qualified V24 QUERY_OBJECT grounder."""

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
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import _configured_video_ids  # noqa: E402
import scripts.freeze_agqa2_query_object_v23_reserve as v23  # noqa: E402


NONCE = "agqa2-query-object-v25-v24-grounder-final-30-video-confirmation"


def _development_summary() -> tuple[Path, dict]:
    report_path = REPO_ROOT / "runs/agqa2_query_object_v24_development/report.json"
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or not report.get("grounder_qualified"):
        raise ValueError("V24 development qualification is invalid")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-query-object-v24-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "claim_scope": "ATOMIC_QUERY_OBJECT_WITH_INTERVAL_ENVELOPE_NORMALIZATION",
        "provider_attempts_replayed_without_new_calls": 83,
        "confirmatory": False,
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_query_object_v24_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def _selection(development_manifest: dict, excluded: set[str]) -> dict:
    v23.NONCE = NONCE
    inherited = v23._select(development_manifest, excluded)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-reserve-selection-v25",
        "status": "FROZEN_V25_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V25_CALLS",
        "claim_boundary": (
            "UNCHANGED_V24_QUERY_OBJECT_GROUNDER;30_NEW_CROSS_EXPERIMENT_AND_"
            "V23_VIDEO_DISJOINT_TEST_ROWS;ONE_OUTCOME_BLIND_CONFIRMATION;"
            "NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "prior_v25_neural_grounder_exposure": False,
    })
    core.pop("prior_v23_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _manifest(selection: dict) -> dict:
    inherited = v23._seal(selection)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-reserve-manifest-v25",
        "status": "FROZEN_V25_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v25_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V25 QUERY_OBJECT reserve is already consumed")
    summary_path, development = _development_summary()
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_query_object_v24_development_manifest.json",
        "manifest_sha256",
    )
    excluded = _configured_video_ids()
    video_root = Path(development_manifest["video_root"])
    excluded.update(path.stem for path in video_root.glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_query_object_v25_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file() else _selection(development_manifest, excluded)
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
            "missing_video_ids": missing,
            "next": "download exact frozen videos and rerun",
        }, indent=2))
        return

    receipt_path = REPO_ROOT / "runs/agqa2_query_object_v25_download/receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("V25 download receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]
    ):
        raise ValueError("V25 download receipt is incomplete or mismatched")
    manifest = _manifest(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v25_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v24_development.json"
    ).read_text())
    config.update({
        "schema_version": "agqa2-query-object-reserve-config-v25",
        "status": "FROZEN_V25_QUERY_OBJECT_CONFIRMATION",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V25_RESERVE_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V24_QUERY_OBJECT",
    })
    config["qualification_gates"] = {
        "required_valid_runtime_rows": 30,
        "minimum_route_correct": 30,
        "minimum_decisive_executions": 15,
        "minimum_decisive_accuracy": 0.75,
        "maximum_typed_vs_direct_losses": 0,
        "minimum_typed_vs_direct_wins": 2,
        "required_source_permuted_abstentions": 30,
        "required_target_written_equivalent_matches": 30,
        "maximum_reported_provider_cost_usd": 0.35,
    }
    config["preregistration"] = "configs/agqa2_query_object_v25_reserve_preregistration.json"
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    if grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V25 reserve changed the qualified V24 grounder")
    prereg = {
        "schema_version": "agqa2-query-object-reserve-preregistration-v25",
        "status": "FROZEN_BEFORE_ANY_V25_RESERVE_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": grounder_sha256,
        "reserve_evaluation_protocol_sha256": evaluation_sha256,
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "selection_manifest_sha256": selection["manifest_sha256"],
        "sealed_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "reserve_gates": deepcopy(config["qualification_gates"]),
        "grounder_changed_after_development": False,
        "v23_video_ids_excluded": True,
        "failure_policy": (
            "RUN_ONCE;FAIL_CLOSED_TO_MATCHED_DIRECT;NO_POST_RESERVE_TUNING_OR_"
            "ADDITIONAL_FRESH_SEED"
        ),
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v25_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "sample_count": manifest["sample_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
