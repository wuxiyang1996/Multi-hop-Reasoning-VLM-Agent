#!/usr/bin/env python3
"""Freeze a powered AGQA V17 reserve without changing the V16 grounder."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core, _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
import scripts.freeze_agqa2_active_grounding_v16_reserve as v16  # noqa: E402


NONCE = "agqa2-v17-unchanged-v16-grounder-powered-45-row-replication"
CANDIDATES_PER_ROUTE = 18
EVALUATED_PER_ROUTE = 15


def _compact_v16_result() -> tuple[Path, dict]:
    report_path = REPO_ROOT / "runs/agqa2_active_grounding_v16_reserve/report.json"
    report = json.loads(report_path.read_text())
    failed = {
        key for key, passed in report["qualification_gates"].items() if not passed
    }
    if report.get("grounder_qualified") or failed != {
        "minimum_decisive_executions", "minimum_typed_vs_direct_wins",
    }:
        raise ValueError(f"unexpected V16 reserve state: {failed}")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "accepted_runtime_provider_calls",
        "accepted_runtime_reported_provider_cost_usd", "provider_calls",
        "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-active-grounding-v16-reserve-result",
        "report_file_sha256": _sha256(report_path),
        "failed_gates": sorted(failed),
        "interpretation": (
            "ZERO_NEGATIVE_TRANSFER_AND_POSITIVE_ONE_ROW_GAIN;FORMAL_GATE_"
            "FAILED_FOR_SPARSE_EFFECT_AND_ONE_ROW_DECISIVE_SHORTFALL"
        ),
    })
    result = core | {"result_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_active_grounding_v16_reserve_result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path, result


def _selection(development: dict, excluded: set[str]) -> dict:
    v16.NONCE = NONCE
    v16.CANDIDATES_PER_ROUTE = CANDIDATES_PER_ROUTE
    v16.EVALUATED_PER_ROUTE = EVALUATED_PER_ROUTE
    inherited = v16._select(development, excluded)
    core = dict(inherited)
    core.pop("manifest_sha256")
    samples = []
    for row in core["samples"]:
        samples.append(dict(row) | {
            "applicability_rule": (
                "V17_ATOMIC_TYPED_ARITY_PLUS_PROGRAM_ROOT_ANSWER_"
                "SPACE_COMPATIBILITY"
            ),
        })
    core.update({
        "schema_version": "agqa2-active-grounding-selection-v17",
        "status": "FROZEN_V17_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V17_CALLS",
        "claim_boundary": (
            "UNCHANGED_V16_SELECTIVE_OVERRIDE_GROUNDER;54_NEW_CROSS_"
            "EXPERIMENT_VIDEO_DISJOINT_CANDIDATES;45_ROW_OUTCOME_BLIND_"
            "POWERED_REPLICATION;NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_VIDEO_IDS_REFERENCED_BY_PRIOR_CONFIGS_AND_ALL_MP4S_"
            "PRESENT_IN_SHARED_CHARADES_ROOT;REQUIRE_ATOMIC_TYPED_ARITY_AND_"
            "PROGRAM_ROOT_ANSWER_SPACE_COMPATIBILITY;EIGHTEEN_FIXED_HASH_"
            "CANDIDATES_PER_ROUTE;NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "per_route_candidates": CANDIDATES_PER_ROUTE,
        "per_route_evaluated": EVALUATED_PER_ROUTE,
        "samples": samples,
        "prior_v17_neural_grounder_exposure": False,
    })
    core.pop("prior_v16_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: dict) -> dict:
    inherited = v16._seal(selection)
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-active-grounding-manifest-v17-reserve",
        "status": "FROZEN_V17_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
        "claim_boundary": selection["claim_boundary"],
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v17_powered_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V17 powered reserve is already consumed")
    v16_result_path, v16_result = _compact_v16_result()
    development_summary_path = (
        REPO_ROOT / "docs/results/agqa2_active_grounding_v16_development_summary.json"
    )
    development = json.loads(development_summary_path.read_text())
    if not development.get("grounder_qualified"):
        raise ValueError("V16 development grounder is not qualified")

    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v16_development_manifest.json",
        "manifest_sha256",
    )
    excluded = v16._configured_video_ids()
    excluded.update(path.stem for path in Path(development_manifest["video_root"]).glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve_selection.json"
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
            "candidate_count": selection["sample_count"],
            "selected_video_ids": [row["video_id"] for row in selection["samples"]],
            "missing_video_ids": missing,
            "next": "download exact frozen videos and rerun",
        }, indent=2))
        return

    manifest = _seal(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v16_reserve.json"
    ).read_text())
    config = deepcopy(base_config)
    config["runtime_selection"].update({
        "candidate_count": 3 * CANDIDATES_PER_ROUTE,
        "per_predicted_route": EVALUATED_PER_ROUTE,
    })
    config["qualification_gates"].update({
        "required_valid_runtime_rows": 3 * EVALUATED_PER_ROUTE,
        "minimum_route_correct": 3 * EVALUATED_PER_ROUTE,
        "minimum_decisive_executions": 30,
        "minimum_typed_vs_direct_wins": 3,
        "maximum_typed_vs_direct_losses": 0,
        "required_source_permuted_abstentions": 3 * EVALUATED_PER_ROUTE,
        "required_target_written_equivalent_matches": 3 * EVALUATED_PER_ROUTE,
        "maximum_reported_provider_cost_usd": 0.45,
    })
    config.update({
        "schema_version": "agqa2-active-grounding-powered-reserve-config-v17",
        "status": "FROZEN_V17_UNCHANGED_V16_GROUNDER_POWERED_RESERVE",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V17_POWERED_RESERVE_CALL",
        "development_qualification_report": str(development_summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(development_summary_path),
        "report_version": "V17",
    })
    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    if expected_grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V17 changed the qualified V16 grounder")
    if expected_evaluation_sha256 == development["evaluation_protocol_sha256"]:
        raise AssertionError("V17 powered protocol should have a new identity")
    receipt_path = REPO_ROOT / "runs/agqa2_active_grounding_v17_download/receipt.json"
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v17-powered-reserve",
        "status": "FROZEN_BEFORE_ANY_V17_POWERED_RESERVE_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_unchanged_v16_grounder_sha256": expected_grounder_sha256,
        "powered_evaluation_protocol_sha256": expected_evaluation_sha256,
        "development_qualification_summary": str(development_summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(development_summary_path),
        "v16_reserve_result": str(v16_result_path.relative_to(REPO_ROOT)),
        "v16_reserve_result_file_sha256": _sha256(v16_result_path),
        "power_planning": {
            "v16_development_rows": 30,
            "v16_development_decisive": development["metrics"]["decisive_executions"],
            "v16_development_wins": development["metrics"]["typed_vs_direct_wins"],
            "v16_reserve_rows": 30,
            "v16_reserve_decisive": v16_result["metrics"]["decisive_executions"],
            "v16_reserve_wins": v16_result["metrics"]["typed_vs_direct_wins"],
            "pooled_rows": 60,
            "pooled_decisive": (
                development["metrics"]["decisive_executions"]
                + v16_result["metrics"]["decisive_executions"]
            ),
            "pooled_wins": (
                development["metrics"]["typed_vs_direct_wins"]
                + v16_result["metrics"]["typed_vs_direct_wins"]
            ),
            "pooled_losses": 0,
            "v17_candidate_rows": 3 * CANDIDATES_PER_ROUTE,
            "v17_evaluated_rows": 3 * EVALUATED_PER_ROUTE,
            "grounder_or_selector_semantics_changed": False,
        },
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "reserve_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "reserve_gates": deepcopy(config["qualification_gates"]),
        "cost_projection": {
            "v16_candidate_count": 36,
            "v16_cost_usd": v16_result["reported_provider_cost_usd"],
            "v17_candidate_count": 54,
            "linear_projection_usd": 1.5 * v16_result["reported_provider_cost_usd"],
            "frozen_cap_usd": 0.45,
        },
        "failure_policy": {
            "reserve": "RUN_ONCE_ON_FROZEN_V17_POOL;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": manifest["sample_count"],
        "evaluated_count": config["qualification_gates"]["required_valid_runtime_rows"],
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
