#!/usr/bin/env python3
"""Freeze the outcome-blind V27 replay of the incomplete V26 acquisition."""

from __future__ import annotations

from copy import deepcopy
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


def _development_summary() -> tuple[Path, dict]:
    report_path = REPO_ROOT / "runs/agqa2_query_object_v27_development/report.json"
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or not report.get("grounder_qualified"):
        raise ValueError("V27 token-cap repair did not qualify on development")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-query-object-v27-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "claim_scope": "TOKEN_CAP_REPAIR_300_TO_500_ONLY",
        "confirmatory": False,
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_query_object_v27_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def _replay_manifest(abort: dict) -> dict:
    v26_path = REPO_ROOT / "configs/agqa2_query_object_v26_reserve_manifest.json"
    v26 = _verified_json(v26_path, "manifest_sha256")
    core = {
        "schema_version": "agqa2-query-object-replay-manifest-v27",
        "status": "FROZEN_V27_OUTCOME_BLIND_TRANSPORT_REPAIR_REPLAY",
        "split": "reserve",
        "claim_boundary": (
            "EXACT_V26_PRESELECTED_120_VIDEO_POOL;NO_V26_FORMAL_GOLD_"
            "EVALUATION;PRIMARY_ONTOLOGY_MAX_TOKENS_300_TO_500_REQUALIFIED_"
            "ON_DEVELOPMENT;IDENTICAL_SOURCE_SPECIFIC_GATES;NOT_A_NEW_SEED"
        ),
        "samples": deepcopy(v26["samples"]),
        "sample_count": v26["sample_count"],
        "unique_video_count": v26["unique_video_count"],
        "relation_group_counts": deepcopy(v26["relation_group_counts"]),
        "archive_path": v26["archive_path"],
        "archive_sha256": v26["archive_sha256"],
        "entry": v26["entry"],
        "video_root": v26["video_root"],
        "v26_selection_manifest_sha256": v26["selection_manifest_sha256"],
        "v26_sealed_manifest_sha256": v26["manifest_sha256"],
        "v26_manifest_file_sha256": _sha256(v26_path),
        "v26_abort_sha256": abort["abort_sha256"],
        "prior_v26_neural_grounder_video_exposure": True,
        "prior_v26_formal_gold_evaluation": False,
        "prior_v26_official_answers_inspected_for_repair": False,
        "v26_runtime_predictions_used_to_change_gates": False,
        "new_seed_selected_after_v26_abort": False,
        "all_v26_accepted_provider_calls_reused_by_input_hash_when_compatible": True,
        "only_grounder_change": {
            "field": "query_object_grounder.model.max_ontology_tokens",
            "before": 300,
            "after": 500,
        },
    }
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v27_replay"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V27 replay is already consumed")
    summary_path, development = _development_summary()
    abort_path = REPO_ROOT / "docs/results/agqa2_query_object_v26_runtime_abort.json"
    abort = _verified_json(abort_path, "abort_sha256")
    if (
        abort["formal_report_created"]
        or abort["formal_gold_evaluation_started"]
        or abort["completed_runtime_receipts"] != 119
    ):
        raise ValueError("V26 abort is not an outcome-blind runtime failure")
    manifest = _replay_manifest(abort)
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v27_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v27_development.json"
    ).read_text())
    v26_config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v26_reserve.json"
    ).read_text())
    config.update({
        "schema_version": "agqa2-query-object-replay-config-v27",
        "status": "FROZEN_V27_SOURCE_SPECIFIC_REPLAY",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V27_REPLAY_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V27_SOURCE_SPECIFIC_QUERY_OBJECT_REPLAY",
        "qualification_gates": deepcopy(v26_config["qualification_gates"]),
        "source_specific_evaluation": deepcopy(
            v26_config["source_specific_evaluation"]
        ),
        "preregistration": (
            "configs/agqa2_query_object_v27_replay_preregistration.json"
        ),
    })
    wrapper_path = REPO_ROOT / "scripts/collect_agqa2_query_object_v27.py"
    config["source_specific_evaluation"]["report_wrapper"] = {
        "path": str(wrapper_path.relative_to(REPO_ROOT)),
        "sha256": _sha256(wrapper_path),
        "outcome_calculation_changed": False,
    }
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
        raise AssertionError("V27 replay changed the qualified V27 development grounder")
    if (
        config["qualification_gates"] != v26_config["qualification_gates"]
        or config["source_specific_evaluation"]["qualification_gates"]
        != v26_config["source_specific_evaluation"]["qualification_gates"]
    ):
        raise AssertionError("V27 changed a V26 preregistered gate")

    prereg = {
        "schema_version": "agqa2-query-object-replay-preregistration-v27",
        "status": "FROZEN_BEFORE_ANY_V27_REPLAY_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": grounder_sha256,
        "base_evaluation_protocol_sha256": base_evaluation_sha256,
        "source_specific_evaluation_protocol_sha256": source_evaluation_sha256,
        "source_specific_evaluation": deepcopy(config["source_specific_evaluation"]),
        "base_mechanism_gates": deepcopy(config["qualification_gates"]),
        "replay_manifest_sha256": manifest["manifest_sha256"],
        "v26_abort_sha256": abort["abort_sha256"],
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "grounder_changed_after_v27_development": False,
        "source_specific_gates_changed_after_v26_freeze": False,
        "sample_pool_changed_after_v26_abort": False,
        "v26_formal_outcomes_inspected": False,
        "multiplicity_policy": "ONE_PRIMARY_PAIRED_ENDPOINT;NO_GATE_SELECTION",
        "failure_policy": (
            "REPLAY_EXACT_FIXED_120_ROWS_ONCE;REUSE_COMPATIBLE_CALLS_BY_"
            "INPUT_HASH;REPORT_FAILURE_IF_ANY_PRIMARY_GATE_FAILS;NO_NEW_SEED"
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
    config_path = REPO_ROOT / "configs/agqa2_query_object_v27_replay.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "base_evaluation_protocol_sha256": base_evaluation_sha256,
        "source_specific_evaluation_protocol_sha256": source_evaluation_sha256,
        "sample_count": manifest["sample_count"],
        "sample_pool_changed": False,
        "gates_changed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
