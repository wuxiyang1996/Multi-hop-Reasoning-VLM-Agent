#!/usr/bin/env python3
"""Freeze a fresh powered reserve for the qualified bounded V28 grounder."""

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
    _evaluation_core as _source_evaluation_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)
import scripts.freeze_agqa2_query_object_v23_reserve as v23  # noqa: E402


NONCE = "agqa2-query-object-v28-bounded-source-specific-fresh-120-confirmation"
PER_GROUP = 40
TOTAL_ROWS = 3 * PER_GROUP
PILOT_SOURCE_WIN_RATE = 2 / 30


def _power_probability() -> float:
    return sum(
        comb(TOTAL_ROWS, wins)
        * PILOT_SOURCE_WIN_RATE ** wins
        * (1 - PILOT_SOURCE_WIN_RATE) ** (TOTAL_ROWS - wins)
        for wins in range(5, TOTAL_ROWS + 1)
    )


def _development_summary() -> tuple[Path, dict]:
    report_path = REPO_ROOT / "runs/agqa2_query_object_v28_development/report.json"
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or not report.get("grounder_qualified"):
        raise ValueError("V28 bounded development grounder did not qualify")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-query-object-v28-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "claim_scope": "BOUNDED_ONTOLOGY_FREE_TEXT_WITH_UNCHANGED_DECISION_SCHEMA",
        "confirmatory": False,
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_query_object_v28_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def _selection(development_manifest: dict, excluded: set[str]) -> dict:
    v23.NONCE = NONCE
    v23.PER_GROUP = PER_GROUP
    inherited = v23._select(development_manifest, excluded)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-reserve-selection-v28",
        "status": "FROZEN_V28_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V28_CALLS",
        "claim_boundary": (
            "QUALIFIED_BOUNDED_V28_QUERY_OBJECT_GROUNDER;PREREGISTERED_"
            "TARGET_ONLY_TWO_ONTOLOGY_COMPARATOR;120_NEW_VIDEO_DISJOINT_TEST_"
            "ROWS_EXCLUDING_V26_V27;ONE_POWERED_OUTCOME_BLIND_CONFIRMATION"
        ),
        "selection_nonce": NONCE,
        "sample_size_rationale": {
            "fixed_total_rows": TOTAL_ROWS,
            "rows_per_relation_group": PER_GROUP,
            "pilot_v25_source_only_wins": 2,
            "pilot_v25_rows": 30,
            "pilot_source_win_rate": PILOT_SOURCE_WIN_RATE,
            "probability_of_at_least_five_wins_under_pilot_rate": (
                _power_probability()
            ),
        },
        "v26_v27_video_ids_excluded": True,
        "prior_v28_neural_grounder_exposure": False,
    })
    core.pop("prior_v23_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _manifest(selection: dict) -> dict:
    inherited = v23._seal(selection)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-query-object-reserve-manifest-v28",
        "status": "FROZEN_V28_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v28_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V28 reserve is already consumed")
    summary_path, development = _development_summary()
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_query_object_v24_development_manifest.json",
        "manifest_sha256",
    )
    excluded = _configured_video_ids()
    video_root = Path(development_manifest["video_root"])
    excluded.update(path.stem for path in video_root.glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_query_object_v28_reserve_selection.json"
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

    receipt_path = REPO_ROOT / "runs/agqa2_query_object_v28_download/receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("V28 download receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]
        or len(receipt.get("videos", [])) != TOTAL_ROWS
    ):
        raise ValueError("V28 download receipt is incomplete or mismatched")
    manifest = _manifest(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v28_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v28_development.json"
    ).read_text())
    v26 = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v26_reserve.json"
    ).read_text())
    config.update({
        "schema_version": "agqa2-query-object-reserve-config-v28",
        "status": "FROZEN_V28_SOURCE_SPECIFIC_CONFIRMATION",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V28_RESERVE_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V28_SOURCE_SPECIFIC_QUERY_OBJECT",
        "qualification_gates": deepcopy(v26["qualification_gates"]),
        "source_specific_evaluation": deepcopy(v26["source_specific_evaluation"]),
        "preregistration": (
            "configs/agqa2_query_object_v28_reserve_preregistration.json"
        ),
    })
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
    source_evaluation_sha256 = stable_hash(_source_evaluation_core(config))
    if grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V28 reserve changed the qualified development grounder")
    if (
        config["qualification_gates"] != v26["qualification_gates"]
        or config["source_specific_evaluation"]["qualification_gates"]
        != v26["source_specific_evaluation"]["qualification_gates"]
    ):
        raise AssertionError("V28 changed a V26 source-specific gate")
    v26_abort = _verified_json(
        REPO_ROOT / "docs/results/agqa2_query_object_v26_runtime_abort.json",
        "abort_sha256",
    )
    v27_abort = _verified_json(
        REPO_ROOT / "docs/results/agqa2_query_object_v27_runtime_abort.json",
        "abort_sha256",
    )
    prereg = {
        "schema_version": "agqa2-query-object-reserve-preregistration-v28",
        "status": "FROZEN_BEFORE_ANY_V28_RESERVE_CALL",
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
        "v26_abort_sha256": v26_abort["abort_sha256"],
        "v27_abort_sha256": v27_abort["abort_sha256"],
        "grounder_changed_after_v28_development": False,
        "source_specific_gates_changed_after_v26_freeze": False,
        "v26_v27_video_ids_excluded": True,
        "v26_v27_formal_outcomes_inspected": False,
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
    config_path = REPO_ROOT / "configs/agqa2_query_object_v28_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "base_evaluation_protocol_sha256": base_evaluation_sha256,
        "source_specific_evaluation_protocol_sha256": source_evaluation_sha256,
        "sample_count": manifest["sample_count"],
        "v26_v27_video_ids_excluded": True,
        "gates_changed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
