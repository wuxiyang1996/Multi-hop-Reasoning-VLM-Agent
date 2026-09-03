#!/usr/bin/env python3
"""Freeze final fresh AGQA V19 temporal-operator selective confirmation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core, _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
import scripts.freeze_agqa2_active_grounding_v16_reserve as v16  # noqa: E402


NONCE = "agqa2-v19-temporal-operator-selective-final-60-row-confirmation"
CANDIDATES_PER_ROUTE = 24
EVALUATED_PER_ROUTE = 20


def _provider_cache_flags(value: Any) -> Iterator[bool]:
    if isinstance(value, Mapping):
        usage = value.get("usage")
        if (
            "cache_reused" in value
            and isinstance(usage, Mapping)
            and not usage.get("local_non_provider_call", False)
        ):
            yield bool(value["cache_reused"])
        for child in value.values():
            yield from _provider_cache_flags(child)
    elif isinstance(value, list):
        for child in value:
            yield from _provider_cache_flags(child)


def _development_summary() -> tuple[Path, dict]:
    report_path = REPO_ROOT / "runs/agqa2_temporal_selective_v19_development/report.json"
    report = json.loads(report_path.read_text())
    if not report.get("grounder_qualified"):
        raise ValueError("V19 temporal-selective development did not qualify")
    receipt_paths = sorted((report_path.parent / "runtime_receipts").glob("*.json"))
    flags: list[bool] = []
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        if not receipt.get("direct_cache_reused"):
            raise ValueError(f"V19 development direct call was not replayed: {path}")
        flags.extend(_provider_cache_flags(receipt))
    if not flags or not all(flags):
        raise ValueError("V19 development made a non-replayed provider call")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-temporal-selective-v19-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "accepted_provider_receipts_replayed_from_v17": len(flags),
        "new_provider_calls_during_v19_requalification": 0,
        "runtime_receipt_count": len(receipt_paths),
        "claim_scope": "TEMPORAL_OPERATOR_TRANSFER_ONLY",
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_temporal_selective_v19_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def _selection(development: dict, excluded: set[str]) -> dict:
    v16.NONCE = NONCE
    v16.CANDIDATES_PER_ROUTE = CANDIDATES_PER_ROUTE
    v16.EVALUATED_PER_ROUTE = EVALUATED_PER_ROUTE
    inherited = v16._select(development, excluded)
    core = dict(inherited)
    core.pop("manifest_sha256")
    samples = [
        dict(row) | {
            "applicability_rule": (
                "V19_ATOMIC_TYPED_ARITY_PLUS_PROGRAM_ROOT_ANSWER_"
                "SPACE_COMPATIBILITY"
            ),
        }
        for row in core["samples"]
    ]
    core.update({
        "schema_version": "agqa2-temporal-selective-selection-v19",
        "status": "FROZEN_V19_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V19_CALLS",
        "claim_boundary": (
            "QUALIFIED_V19_TEMPORAL_OPERATOR_SELECTIVE_GROUNDER;72_NEW_"
            "CROSS_EXPERIMENT_VIDEO_DISJOINT_CANDIDATES;60_ROW_OUTCOME_"
            "BLIND_FINAL_CONFIRMATION;RELATION_SOURCE_OVERRIDE_ABSTAINS;"
            "NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_VIDEO_IDS_REFERENCED_BY_PRIOR_CONFIGS_AND_ALL_MP4S_"
            "PRESENT_IN_SHARED_CHARADES_ROOT;REQUIRE_ATOMIC_TYPED_ARITY_AND_"
            "PROGRAM_ROOT_ANSWER_SPACE_COMPATIBILITY;TWENTY_FOUR_FIXED_HASH_"
            "CANDIDATES_PER_ROUTE;NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "per_route_candidates": CANDIDATES_PER_ROUTE,
        "per_route_evaluated": EVALUATED_PER_ROUTE,
        "samples": samples,
        "prior_v19_neural_grounder_exposure": False,
    })
    core.pop("prior_v16_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: dict) -> dict:
    inherited = v16._seal(selection)
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-temporal-selective-manifest-v19-reserve",
        "status": "FROZEN_V19_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
        "claim_boundary": selection["claim_boundary"],
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_temporal_selective_v19_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V19 temporal-selective reserve is already consumed")
    summary_path, development = _development_summary()
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_temporal_selective_v19_development_manifest.json",
        "manifest_sha256",
    )
    excluded = v16._configured_video_ids()
    excluded.update(path.stem for path in Path(development_manifest["video_root"]).glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_reserve_selection.json"
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
    manifest_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_temporal_selective_v19_development.json"
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
        "minimum_typed_vs_direct_wins": 2,
        "maximum_typed_vs_direct_losses": 0,
        "required_source_permuted_abstentions": 3 * EVALUATED_PER_ROUTE,
        "required_target_written_equivalent_matches": 3 * EVALUATED_PER_ROUTE,
        "maximum_reported_provider_cost_usd": 0.60,
    })
    config.update({
        "schema_version": "agqa2-temporal-selective-reserve-config-v19",
        "status": "FROZEN_V19_FINAL_TEMPORAL_SELECTIVE_RESERVE",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V19_FINAL_RESERVE_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V19",
    })
    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    if expected_grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V19 reserve changed the qualified development grounder")
    if expected_evaluation_sha256 == development["evaluation_protocol_sha256"]:
        raise AssertionError("V19 reserve should have a powered protocol identity")
    receipt_path = REPO_ROOT / "runs/agqa2_temporal_selective_v19_download/receipt.json"
    preregistration = {
        "schema_version": "agqa2-temporal-selective-preregistration-v19-reserve",
        "status": "FROZEN_BEFORE_ANY_V19_FINAL_RESERVE_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": expected_grounder_sha256,
        "powered_evaluation_protocol_sha256": expected_evaluation_sha256,
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "power_planning": {
            "development_evaluated_rows": development["metrics"]["valid_runtime_rows"],
            "development_decisive": development["metrics"]["decisive_executions"],
            "development_wins": development["metrics"]["typed_vs_direct_wins"],
            "development_losses": development["metrics"]["typed_vs_direct_losses"],
            "reserve_candidate_rows": 3 * CANDIDATES_PER_ROUTE,
            "reserve_evaluated_rows": 3 * EVALUATED_PER_ROUTE,
            "minimum_reserve_decisive": config["qualification_gates"]["minimum_decisive_executions"],
            "minimum_reserve_wins": config["qualification_gates"]["minimum_typed_vs_direct_wins"],
            "grounder_semantics_changed_after_development": False,
        },
        "claim_scope": (
            "TEMPORAL_PAIR_SINGLE_OCCURRENCE_AND_TEMPORAL_SINGLE_DURATION_"
            "TOPOLOGY_TRANSFER;RELATION_SOURCE_OVERRIDE_EXCLUDED"
        ),
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "reserve_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "reserve_gates": deepcopy(config["qualification_gates"]),
        "cost_projection": {
            "development_candidate_count": 54,
            "development_cost_usd": development["reported_provider_cost_usd"],
            "reserve_candidate_count": 72,
            "linear_projection_usd": (
                development["reported_provider_cost_usd"] * 72 / 54
            ),
            "frozen_cap_usd": 0.60,
        },
        "failure_policy": {
            "reserve": "RUN_ONCE_ON_FROZEN_V19_POOL;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_reserve_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_reserve.json"
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
