from copy import deepcopy

from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa2_source_triggered_reanswer_v9 import evaluate


def _seal(body):
    return body | {"report_sha256": stable_hash(body)}


def test_evaluator_applies_triggered_answer_and_checks_controls():
    config = {
        "claim_boundary": "test",
        "qualification_gate": {
            "minimum_wins": 1, "maximum_losses": 0, "minimum_net_gain": 1,
            "minimum_route_accuracy": 1.0,
            "minimum_source_permuted_abstention_rate": 1.0,
            "minimum_target_written_equivalent_rate": 1.0,
            "maximum_combined_cost_usd": 1.0,
        },
    }
    base_body = {
        "sample_count": 2, "reported_provider_cost_usd": 0.2,
        "metrics": {"route_accuracy": 1.0},
        "controls": {"source_permuted_abstentions": 2,
                     "target_written_equivalent_matches": 2},
        "rows": [
            {"task_id": "a", "direct_response": "no",
             "gold_answer_evaluator_only": "yes"},
            {"task_id": "b", "direct_response": "no",
             "gold_answer_evaluator_only": "no"},
        ],
    }
    base = _seal(base_body)
    runtime_body = {
        "config_sha256": stable_hash(config),
        "base_report_sha256": base["report_sha256"],
        "triggered_count": 1, "reported_provider_cost_usd": 0.1,
        "rows": [
            {"task_id": "a", "source_triggered": True, "response": "Yes.",
             "answer_read": False, "program_read": False,
             "scene_graph_read": False, "source_identity_read": False},
            {"task_id": "b", "source_triggered": False, "response": None},
        ],
    }
    result = evaluate(config, base, _seal(runtime_body))
    assert result["status"] == "PASSED"
    assert result["metrics"]["wins"] == 1
    assert result["metrics"]["losses"] == 0


def test_evaluator_detects_runtime_tampering():
    config = {"claim_boundary": "test", "qualification_gate": {}}
    base = _seal({"sample_count": 0, "reported_provider_cost_usd": 0,
                  "metrics": {}, "controls": {}, "rows": []})
    runtime = _seal({"config_sha256": stable_hash(config),
                     "base_report_sha256": base["report_sha256"],
                     "triggered_count": 0, "reported_provider_cost_usd": 0,
                     "rows": []})
    tampered = deepcopy(runtime)
    tampered["triggered_count"] = 1
    try:
        evaluate(config, base, tampered)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampering was not detected")
