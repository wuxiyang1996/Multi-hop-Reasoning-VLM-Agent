from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa2_multiclass_router_v65_formal_v2 import evaluate


def _sealed(value: dict, key: str) -> dict:
    return value | {key: stable_hash(value)}


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    selection_path = Path("configs/agqa2_multiclass_router_formal_v2_selection.json")
    selection = _sealed({
        "status": "FROZEN_V78_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS",
        "answer_read_during_selection": False,
        "program_read_during_selection": False,
        "router_model_file_sha256": "router-model",
        "router_qualification_file_sha256": "router-report-file",
        "route_counts": {"RELATION_RECURRENT": 1},
        "samples": [{
            "task_id": "new-1", "video_id": "new",
            "predicted_route": "RELATION_RECURRENT",
        }],
    }, "manifest_sha256")
    prior = _sealed({
        "samples": [{"task_id": "old-1", "video_id": "old"}],
    }, "manifest_sha256")
    manifest = _sealed({"samples": selection["samples"]}, "manifest_sha256")
    config = {
        "preregistration": str(selection_path),
        "frozen_runtime": {
            "git_commit": "commit",
            "dependency_overlay_sha256": {"dep": "sha"},
        },
        "grounder": {"collector_sha256": "collector", "module_sha256": "module"},
        "target_native_program_router": {
            "model_file_sha256": "router-model",
            "qualification_file_sha256": "router-report-file",
        },
    }
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    row = {
        "task_id": "new-1", "video_id": "new",
        "unified_harness_correct": True, "direct_correct": False,
        "query_plan": {"obligation_kind": "RELATION_RECURRENT"},
        "oracle_route_evaluator_only": "RELATION_RECURRENT",
        "unified_harness_executor_authorized": True,
        "predicted_route_correct": True,
        "source_permuted_wrong_type_abstained": True,
        "target_written_equivalent_dynamics_match": True,
        "runtime_answer_read": False, "runtime_functional_program_read": False,
        "runtime_scene_graph_read": False, "runtime_source_identity_read": False,
        "operand_grounder_question_read": False,
        "operand_grounder_competing_operand_read": False,
        "official_answer_first_read_after_all_runtime_rows_froze": True,
    }
    report = _sealed({
        "rows": [row], "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"],
        "preregistration_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "grounder_sha256": "grounder", "reported_provider_cost_usd": 0.01,
    }, "report_sha256")
    protocol = _sealed({
        "status": "FROZEN_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS",
        "claim_boundary": "test",
        "cohort": {
            "sample_count": 1,
            "selection_status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "prior_v1_selection_manifest_sha256": prior["manifest_sha256"],
        },
        "lineage": {
            "expected_grounder_sha256": "grounder", "frozen_runtime_git_commit": "commit",
            "v65_collector_sha256": "collector", "v65_grounder_module_sha256": "module",
            "dependency_overlay_sha256": {"dep": "sha"},
            "program_router_model_sha256": "router-model",
            "program_router_qualification_file_sha256": "router-report-file",
        },
        "gates": {
            "minimum_source_authorizations": 1, "maximum_losses": 0,
            "minimum_net_gain": 1, "minimum_wins": 1,
            "maximum_one_sided_exact_pvalue": 0.5,
            "maximum_reported_provider_cost_usd": 1.0,
        },
    }, "protocol_sha256")
    return protocol, config, config_path, selection, prior, manifest, report


def test_v2_evaluator_accepts_complete_blind_disjoint_four_arm_run(tmp_path: Path):
    args = _fixture(tmp_path)
    result = evaluate(
        protocol=args[0], config=args[1], config_path=args[2], selection=args[3],
        prior_selection=args[4], manifest=args[5], report=args[6],
    )
    assert result["status"] == "PASSED"
    assert all(result["gates"].values())


def test_v2_evaluator_rejects_overlap_with_v1(tmp_path: Path):
    protocol, config, config_path, selection, prior, manifest, report = _fixture(tmp_path)
    prior_body = {"samples": [{"task_id": "old-1", "video_id": "new"}]}
    prior = _sealed(prior_body, "manifest_sha256")
    protocol_body = dict(protocol)
    protocol_body.pop("protocol_sha256")
    protocol_body["cohort"] = dict(protocol_body["cohort"])
    protocol_body["cohort"]["prior_v1_selection_manifest_sha256"] = prior["manifest_sha256"]
    protocol = _sealed(protocol_body, "protocol_sha256")
    result = evaluate(
        protocol=protocol, config=config, config_path=config_path,
        selection=selection, prior_selection=prior, manifest=manifest, report=report,
    )
    assert result["status"] == "FAILED"
    assert result["gates"]["fresh_video_heldout_cohort"] is False
