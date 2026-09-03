from scripts.evaluate_agqa_query_grounder_v2_fresh_formal import _formal_gates


REQUIRED = {
    "source_vs_neural_exact_two_sided_p_maximum": 0.05,
    "source_vs_permuted_exact_two_sided_p_maximum": 0.05,
    "negative_transfer_loss_fraction_maximum": 0.05,
}


def _summaries():
    return {
        "neural_only": {"correct": 40},
        "generic_scaffold": {"correct": 60},
        "source_permuted": {"correct": 41},
        "source_induced": {"correct": 55},
        "target_written_isomorphic": {"correct": 55},
    }


def _comparisons():
    return {
        "neural_only": {"wins": 16, "losses": 1, "exact_two_sided_p": 0.001},
        "generic_scaffold": {"wins": 2, "losses": 7, "exact_two_sided_p": 0.18},
        "source_permuted": {"wins": 15, "losses": 1, "exact_two_sided_p": 0.001},
    }


def _rows(count=100):
    return [{
        "predictions": {
            "source_induced": "book",
            "target_written_isomorphic": "book",
        },
        "query_grounding_v2_receipt_sha256": "a" * 64,
        "grounding_receipt_sha256": "b" * 64,
    } for _ in range(count)]


def test_formal_gates_match_preregistered_transfer_requirements():
    gates = _formal_gates(_summaries(), _comparisons(), _rows(), REQUIRED)
    assert gates and all(gates.values())


def test_generic_ceiling_is_reported_but_is_not_a_pass_gate():
    summaries = _summaries()
    assert summaries["generic_scaffold"]["correct"] > summaries["source_induced"]["correct"]
    assert all(_formal_gates(summaries, _comparisons(), _rows(), REQUIRED).values())


def test_negative_transfer_and_isomorphic_fail_closed():
    comparisons = _comparisons()
    comparisons["neural_only"] = {
        "wins": 20, "losses": 6, "exact_two_sided_p": 0.01,
    }
    rows = _rows()
    rows[0]["predictions"]["target_written_isomorphic"] = "table"
    gates = _formal_gates(_summaries(), comparisons, rows, REQUIRED)
    assert not gates["negative_transfer_losses_at_most_five_percent"]
    assert not gates["target_written_isomorphic_prediction_equivalence"]
