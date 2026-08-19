#!/usr/bin/env python3
"""Freeze the powered V26 source-vs-target-only QUERY_OBJECT confirmation."""

from __future__ import annotations

from copy import deepcopy
from math import comb
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_query_object_v20 import (  # noqa: E402
    _evaluation_core as _base_evaluation_core,
    _semantic_core,
)
from scripts.collect_agqa2_query_object_v26 import (  # noqa: E402
    _evaluation_core as _source_specific_evaluation_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)
import scripts.freeze_agqa2_query_object_v23_reserve as v23  # noqa: E402


NONCE = "agqa2-query-object-v26-powered-source-specific-120-video-confirmation"
PER_GROUP = 40
TOTAL_ROWS = 3 * PER_GROUP
PILOT_SOURCE_WIN_RATE = 2 / 30


def _at_least_five_win_probability() -> float:
    return sum(
        comb(TOTAL_ROWS, wins)
        * PILOT_SOURCE_WIN_RATE ** wins
        * (1 - PILOT_SOURCE_WIN_RATE) ** (TOTAL_ROWS - wins)
        for wins in range(5, TOTAL_ROWS + 1)
    )


def _development_summary() -> tuple[Path, dict]:
    path = REPO_ROOT / "docs/results/agqa2_query_object_v24_development_summary.json"
    summary = _verified_json(path, "summary_sha256")
    report_path = REPO_ROOT / "runs/agqa2_query_object_v24_development/report.json"
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or not report.get("grounder_qualified"):
        raise ValueError("V24 development report is not a valid qualification")
    if (
        _sha256(report_path) != summary["development_report_file_sha256"]
        or report["grounder_sha256"] != summary["grounder_sha256"]
    ):
        raise ValueError("V24 development summary/report mismatch")
    return path, summary


def _selection(development_manifest: dict, excluded: set[str]) -> dict:
    v23.NONCE = NONCE
    v23.PER_GROUP = PER_GROUP
    inherited = v23._select(development_manifest, excluded)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-reserve-selection-v26",
        "status": "FROZEN_V26_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V26_CALLS",
        "claim_boundary": (
            "UNCHANGED_V24_QUERY_OBJECT_GROUNDER;PREREGISTERED_TARGET_ONLY_"
            "TWO_ONTOLOGY_COMPARATOR;120_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_"
            "TEST_ROWS;ONE_POWERED_OUTCOME_BLIND_CONFIRMATION"
        ),
        "selection_nonce": NONCE,
        "sample_size_rationale": {
            "fixed_total_rows": TOTAL_ROWS,
            "rows_per_relation_group": PER_GROUP,
            "pilot_v25_source_only_wins": 2,
            "pilot_v25_rows": 30,
            "pilot_source_win_rate": PILOT_SOURCE_WIN_RATE,
            "probability_of_at_least_five_wins_under_pilot_rate": (
                _at_least_five_win_probability()
            ),
            "selection_made_before_v26_video_download_or_provider_calls": True,
        },
        "prior_v26_neural_grounder_exposure": False,
    })
    core.pop("prior_v23_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _manifest(selection: dict) -> dict:
    inherited = v23._seal(selection)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-reserve-manifest-v26",
        "status": "FROZEN_V26_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v26_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V26 QUERY_OBJECT reserve is already consumed")
    summary_path, development = _development_summary()
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_query_object_v24_development_manifest.json",
        "manifest_sha256",
    )
    excluded = _configured_video_ids()
    video_root = Path(development_manifest["video_root"])
    excluded.update(path.stem for path in video_root.glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_query_object_v26_reserve_selection.json"
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
            "relation_group_counts": selection["relation_group_counts"],
            "missing_video_ids": missing,
            "next": "download the exact frozen videos and rerun",
        }, indent=2))
        return

    receipt_path = REPO_ROOT / "runs/agqa2_query_object_v26_download/receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("V26 download receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]
        or len(receipt.get("videos", [])) != TOTAL_ROWS
    ):
        raise ValueError("V26 download receipt is incomplete or mismatched")
    manifest = _manifest(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v26_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v24_development.json"
    ).read_text())
    config.update({
        "schema_version": "agqa2-query-object-reserve-config-v26",
        "status": "FROZEN_V26_QUERY_OBJECT_SOURCE_SPECIFIC_CONFIRMATION",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V26_RESERVE_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V26_SOURCE_SPECIFIC_QUERY_OBJECT",
    })
    config["qualification_gates"] = {
        "required_valid_runtime_rows": TOTAL_ROWS,
        "minimum_route_correct": TOTAL_ROWS,
        "minimum_decisive_executions": 60,
        "minimum_decisive_accuracy": 0.75,
        "maximum_typed_vs_direct_losses": 2,
        "minimum_typed_vs_direct_wins": 10,
        "required_source_permuted_abstentions": TOTAL_ROWS,
        "required_target_written_equivalent_matches": TOTAL_ROWS,
        "maximum_reported_provider_cost_usd": 1.20,
    }
    evaluator_path = REPO_ROOT / "src/motif_transfer/agqa_query_object_source_specific.py"
    collector_path = REPO_ROOT / "scripts/collect_agqa2_query_object_v26.py"
    config["source_specific_evaluation"] = {
        "policy": "TARGET_ONLY_TWO_ONTOLOGY_AGREEMENT_ELSE_MATCHED_DIRECT_V1",
        "module": str(evaluator_path.relative_to(REPO_ROOT)),
        "module_sha256": _sha256(evaluator_path),
        "collector": str(collector_path.relative_to(REPO_ROOT)),
        "collector_sha256": _sha256(collector_path),
        "minimum_ontology_confidences": [0.8, 0.8],
        "direct_response_is_fallback_not_vote": True,
        "source_view_read": False,
        "gold_answer_read_during_prediction": False,
        "primary_endpoint": (
            "ONE_SIDED_EXACT_PAIRED_SIGN_TEST_ON_SOURCE_VS_TARGET_ONLY_"
            "DISCORDANT_CORRECTNESS"
        ),
        "qualification_gates": {
            "required_valid_paired_rows": TOTAL_ROWS,
            "minimum_target_only_decisive": 36,
            "minimum_source_vs_target_only_wins": 5,
            "maximum_source_vs_target_only_losses": 2,
            "minimum_source_minus_target_only_correct": 5,
            "maximum_exact_one_sided_pvalue": 0.05,
        },
    }
    config["preregistration"] = (
        "configs/agqa2_query_object_v26_reserve_preregistration.json"
    )
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
        "expected_source_specific_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    base_evaluation_sha256 = stable_hash(_base_evaluation_core(config))
    config["expected_evaluation_protocol_sha256"] = base_evaluation_sha256
    source_evaluation_sha256 = stable_hash(_source_specific_evaluation_core(config))
    if grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V26 changed the qualified V24 neural grounder")

    prereg = {
        "schema_version": "agqa2-query-object-reserve-preregistration-v26",
        "status": "FROZEN_BEFORE_ANY_V26_RESERVE_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": grounder_sha256,
        "base_evaluation_protocol_sha256": base_evaluation_sha256,
        "source_specific_evaluation_protocol_sha256": source_evaluation_sha256,
        "source_specific_evaluation": deepcopy(config["source_specific_evaluation"]),
        "base_mechanism_gates": deepcopy(config["qualification_gates"]),
        "selection_manifest_sha256": selection["manifest_sha256"],
        "sealed_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "grounder_changed_after_development": False,
        "sample_size_rationale": deepcopy(selection["sample_size_rationale"]),
        "multiplicity_policy": "ONE_PRIMARY_PAIRED_ENDPOINT;NO_GATE_SELECTION",
        "failure_policy": (
            "RUN_FIXED_120_ROWS_ONCE;REPORT_FAILURE_IF_ANY_PRIMARY_GATE_FAILS;"
            "NO_POST_RESERVE_TUNING_OR_ADDITIONAL_FRESH_SEED"
        ),
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_source_specific_evaluation_protocol_sha256": (
            source_evaluation_sha256
        ),
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v26_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "base_evaluation_protocol_sha256": base_evaluation_sha256,
        "source_specific_evaluation_protocol_sha256": source_evaluation_sha256,
        "sample_count": manifest["sample_count"],
        "base_gates": config["qualification_gates"],
        "source_specific_gates": config["source_specific_evaluation"][
            "qualification_gates"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
