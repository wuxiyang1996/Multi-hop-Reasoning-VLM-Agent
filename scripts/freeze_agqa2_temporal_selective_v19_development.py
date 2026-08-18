#!/usr/bin/env python3
"""Freeze V19 typed operator-applicability development on V17 receipts."""

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


def _compact_parent_results() -> tuple[Path, Path]:
    v17_report_path = REPO_ROOT / "runs/agqa2_active_grounding_v17_powered_reserve/report.json"
    v17 = json.loads(v17_report_path.read_text())
    v17_core = {
        key: deepcopy(v17[key]) for key in (
            "status", "grounder_qualified", "grounder_sha256",
            "evaluation_protocol_sha256", "metrics", "controls",
            "qualification_gates", "reported_provider_cost_usd", "report_sha256",
        )
    }
    v17_core.update({
        "schema_version": "agqa2-active-grounding-v17-powered-result",
        "report_file_sha256": _sha256(v17_report_path),
        "failed_gates": sorted(
            key for key, value in v17["qualification_gates"].items() if not value
        ),
        "interpretation": (
            "POWERED_REPLICATION_FOUND_FOUR_WINS_AND_THREE_LOSSES;AVERAGE_"
            "GAIN_POSITIVE_BUT_SELECTIVE_TRANSFER_NOT_QUALIFIED"
        ),
    })
    v17_result = v17_core | {"result_sha256": stable_hash(v17_core)}
    v17_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v17_powered_result.json"
    v17_path.write_text(json.dumps(v17_result, indent=2, sort_keys=True) + "\n")

    v18_report_path = REPO_ROOT / "runs/agqa2_override_adjudicator_v18_development/report.json"
    v18 = json.loads(v18_report_path.read_text())
    v18_core = {
        key: deepcopy(v18[key]) for key in (
            "status", "grounder_qualified", "model", "sample_count", "metrics",
            "reported_provider_cost_usd", "qualification_gates", "report_sha256",
        )
    }
    v18_core.update({
        "schema_version": "agqa2-override-adjudicator-v18-development-result",
        "report_file_sha256": _sha256(v18_report_path),
        "interpretation": (
            "INDEPENDENT_STRONG_VLM_DID_NOT_ALIGN_WITH_AGQA_TARGET_ONTOLOGY;"
            "DO_NOT_INTEGRATE_OR_USE_FOR_FRESH"
        ),
    })
    v18_result = v18_core | {"result_sha256": stable_hash(v18_core)}
    v18_path = REPO_ROOT / "docs/results/agqa2_override_adjudicator_v18_development_result.json"
    v18_path.write_text(json.dumps(v18_result, indent=2, sort_keys=True) + "\n")
    return v17_path, v18_path


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_temporal_selective_v19_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V19 development is already consumed")
    v17_result_path, v18_result_path = _compact_parent_results()
    parent_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve_manifest.json",
        "manifest_sha256",
    )
    manifest_core = {
        key: deepcopy(value) for key, value in parent_manifest.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v19-development",
        "status": "FROZEN_V19_CONSUMED_TYPED_APPLICABILITY_DEVELOPMENT",
        "split": "development",
        "claim_boundary": (
            "EXACT_CONSUMED_V17_54_CANDIDATE_POOL;LEARN_OPERATOR_LEVEL_"
            "ABSTENTION_FOR_TARGET_ONTOLOGY_AND_RECURRENT_ORDER_AMBIGUITY;"
            "NO_NEW_NEURAL_ACQUISITION"
        ),
        "selection_rule": "REUSE_EXACT_CONSUMED_V17_CANDIDATES_AND_RECEIPTS",
        "parent_v17_manifest_sha256": parent_manifest["manifest_sha256"],
        "v17_result_file_sha256": _sha256(v17_result_path),
        "v18_result_file_sha256": _sha256(v18_result_path),
        "new_video_downloads": 0,
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve.json"
    ).read_text())
    config = deepcopy(base_config)
    config["execution_calibration"].update({
        "allow_exists_source_override": False,
        "maximum_order_override_events_per_operand": 1,
        "new_v19_typed_applicability_rules": [
            "RELATION_EXISTS_TARGET_ONTOLOGY_UNQUALIFIED_ABSTAIN",
            "TEMPORAL_PAIR_REQUIRES_SINGLE_EVENT_PER_OPERAND",
            "TEMPORAL_SINGLE_REUSES_QUALIFIED_DURATION_TOPOLOGY",
        ],
    })
    config["qualification_gates"].update({
        "required_valid_runtime_rows": 45,
        "minimum_route_correct": 45,
        "minimum_decisive_executions": 22,
        "minimum_typed_vs_direct_wins": 2,
        "maximum_typed_vs_direct_losses": 0,
        "required_source_permuted_abstentions": 45,
        "required_target_written_equivalent_matches": 45,
        "maximum_reported_provider_cost_usd": 0.45,
    })
    config.update({
        "schema_version": "agqa2-temporal-selective-development-config-v19",
        "status": "FROZEN_V19_TYPED_APPLICABILITY_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V19_DEVELOPMENT_RECEIPT",
        "report_version": "V19",
    })
    config.pop("development_qualification_report", None)
    config.pop("development_qualification_file_sha256", None)
    for label in ("module", "collector", "executor"):
        config["grounder"][f"{label}_sha256"] = _sha256(
            REPO_ROOT / config["grounder"][label]
        )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    if config["acquisition"] != base_config["acquisition"]:
        raise AssertionError("V19 changed neural acquisition")
    if config["runtime_selection"] != base_config["runtime_selection"]:
        raise AssertionError("V19 changed the outcome-blind selector")
    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    preregistration = {
        "schema_version": "agqa2-temporal-selective-preregistration-v19-development",
        "status": "FROZEN_BEFORE_ANY_V19_DEVELOPMENT_RECEIPT",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "development_manifest_sha256": manifest["manifest_sha256"],
        "v17_failed_powered_result": str(v17_result_path.relative_to(REPO_ROOT)),
        "v17_failed_powered_result_file_sha256": _sha256(v17_result_path),
        "v18_failed_adjudicator_result": str(v18_result_path.relative_to(REPO_ROOT)),
        "v18_failed_adjudicator_result_file_sha256": _sha256(v18_result_path),
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "development_gates": deepcopy(config["qualification_gates"]),
        "accepted_call_replay": {
            "source": "runs/agqa2_active_grounding_v17_powered_reserve/call_cache",
            "policy": "REUSE_ONLY_EXACT_INPUT_HASH_MATCHES;NO_NEW_PROVIDER_CALL_EXPECTED",
        },
        "claim_scope_if_qualified": (
            "TEMPORAL_OPERATOR_TRANSFER_ONLY;RELATION_ACQUISITION_MAY_RUN_BUT_"
            "RELATION_SOURCE_OVERRIDE_IS_OUT_OF_SCOPE_AND_MUST_ABSTAIN"
        ),
        "failure_policy": {
            "development": "MUST_PASS_BEFORE_ANY_NEW_VIDEO_SELECTION",
            "fresh": "ONE_FINAL_VIDEO_DISJOINT_TEMPORAL_SELECTIVE_CONFIRMATION",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_development_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_temporal_selective_v19_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "candidate_count": manifest["sample_count"],
        "evaluated_count": config["qualification_gates"]["required_valid_runtime_rows"],
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
