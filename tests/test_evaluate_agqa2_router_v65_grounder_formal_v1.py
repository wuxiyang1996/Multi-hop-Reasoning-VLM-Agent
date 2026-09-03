import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa2_router_v65_grounder_formal_v1 import evaluate


GROUND_SHA = "grounder"
COMMIT = "ded7448839183851aa10c3cd3e12d253f04e1ceb"
COLLECTOR_SHA = "collector"
MODULE_SHA = "module"
DEPENDENCY_OVERLAY = {"dependency.py": "dependency-sha"}


def _sealed(body, key):
    return body | {key: stable_hash(body)}


def test_four_arm_formal_evaluator_passes_only_real_paired_gain(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config = {
        "_config_path": str(config_path),
        "frozen_runtime": {
            "git_commit": COMMIT,
            "dependency_overlay_sha256": DEPENDENCY_OVERLAY,
        },
        "grounder": {
            "collector_sha256": COLLECTOR_SHA,
            "module_sha256": MODULE_SHA,
        },
    }
    # _config_path is evaluator-local metadata and is not in the serialized config.
    serialized_config = {k: v for k, v in config.items() if k != "_config_path"}
    config_path.write_text(json.dumps(serialized_config))
    selection = _sealed({
        "status": "FROZEN_V66_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS",
        "answer_read_during_selection": False,
        "program_read_during_selection": False,
        "samples": [{"task_id": f"t{i}"} for i in range(80)],
    }, "manifest_sha256")
    manifest = _sealed({"samples": [{"task_id": f"t{i}"} for i in range(80)]}, "manifest_sha256")
    rows = []
    for index in range(80):
        neural = index >= 6
        source = True if index < 6 else neural
        rows.append({
            "task_id": f"t{index}", "video_id": f"v{index}",
            "direct_correct": neural,
            "unified_harness_correct": source,
            "unified_harness_executor_authorized": index < 20,
            "predicted_route_correct": True,
            "source_permuted_wrong_type_abstained": True,
            "target_written_equivalent_dynamics_match": True,
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_scene_graph_read": False,
            "runtime_source_identity_read": False,
            "operand_grounder_question_read": False,
            "operand_grounder_competing_operand_read": False,
            "official_answer_first_read_after_all_runtime_rows_froze": True,
        })
    report = _sealed({
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "grounder_sha256": GROUND_SHA,
        "reported_provider_cost_usd": 0.2,
        "rows": rows,
    }, "report_sha256")
    protocol = _sealed({
        "schema_version": "test", "status": "FROZEN_BEFORE_ANY_FORMAL_PROVIDER_OR_OUTCOME_ACCESS",
        "claim_boundary": "test", "cohort": {"sample_count": 80},
        "lineage": {
            "expected_grounder_sha256": GROUND_SHA,
            "frozen_runtime_git_commit": COMMIT,
            "v65_collector_sha256": COLLECTOR_SHA,
            "v65_grounder_module_sha256": MODULE_SHA,
            "dependency_overlay_sha256": DEPENDENCY_OVERLAY,
        },
        "gates": {
            "minimum_source_authorizations": 20, "maximum_losses": 4,
            "minimum_net_gain": 5, "minimum_wins": 6,
            "maximum_one_sided_exact_pvalue": 0.05,
            "maximum_reported_provider_cost_usd": 0.75,
        },
    }, "protocol_sha256")
    result = evaluate(
        protocol=protocol, config=config, selection=selection,
        manifest=manifest, report=report,
    )
    assert result["status"] == "PASSED"
    assert result["arm_correct"] == {
        "neural_only": 74, "source_induced": 80,
        "source_permuted": 74, "target_written_equivalent": 80,
    }
    assert result["source_vs_neural_only"]["wins"] == 6
    assert result["source_vs_neural_only"]["losses"] == 0


def test_success_gate_rejects_equal_source_and_neural(tmp_path: Path):
    # The exact paired test itself must not mistake all ties for transfer.
    from scripts.evaluate_agqa2_router_v65_grounder_formal_v1 import _paired
    rows = [{"source": True, "neural": True} for _ in range(80)]
    paired = _paired(rows, "source", "neural")
    assert paired["wins"] == paired["losses"] == 0
    assert paired["one_sided_exact_binomial_pvalue"] == 1.0
