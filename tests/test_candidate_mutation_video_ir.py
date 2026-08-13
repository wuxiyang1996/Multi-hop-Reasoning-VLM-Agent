from motif_transfer.candidate_mutation_video_ir import evaluate_candidate_mutation_program


def _candidate(slot, bind, unbound, bound, wrong):
    return {
        "slot": slot,
        "identity_verification": {"identity_match_probability": bind},
        "unbound_mutation": {"support_probability": unbound},
        "bound_mutation": {"support_probability": bound},
        "wrong_guard_mutation": {"support_probability": wrong},
    }


def test_authentic_bind_mutate_can_change_commit():
    row = evaluate_candidate_mutation_program(
        sample_id="s", gold_answer="B", baseline_answer="A",
        fork={"complete": True, "candidates": [
            _candidate("A", 0.1, 0.8, 0.2, 0.9),
            _candidate("B", 0.9, 0.7, 0.95, 0.1),
        ]},
    )
    assert row["conditions"]["target_unbound_mutation_verification"]["committed_answer"] == "A"
    assert row["conditions"]["authentic_bound_mutation_program"]["committed_answer"] == "B"
    assert row["authentic_action_contrast"]


def test_failed_bind_falls_back_to_matched_unbound_measurement():
    row = evaluate_candidate_mutation_program(
        sample_id="s", gold_answer="A", baseline_answer="B",
        fork={"complete": True, "candidates": [
            _candidate("A", 0.2, 0.8, 0.1, 0.1),
            _candidate("B", 0.2, 0.3, 0.9, 0.9),
        ]},
    )
    assert row["conditions"]["authentic_bound_mutation_program"]["scores"] == [0.8, 0.3]
    assert not row["authentic_action_contrast"]
