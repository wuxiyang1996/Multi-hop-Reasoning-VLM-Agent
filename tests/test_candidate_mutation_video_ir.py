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


def test_v8_mutation_uses_decoy_guard_and_noop_fallback():
    candidates = [
        _candidate("A", .9, .2, .9, .1),
        _candidate("B", .9, .8, .2, .9),
    ]
    for index, candidate in enumerate(candidates):
        candidate.update({
            "bind_entity_visual_description": f"carrier {index}",
            "decoy_entity_visual_description": f"decoy {index}",
            "decoy_identity_verification": {"identity_match_probability": .1},
        })
        candidate["bound_mutation"]["panel_sha256"] = f"bound-{index}"
        candidate["wrong_guard_mutation"]["panel_sha256"] = f"wrong-{index}"
    row = evaluate_candidate_mutation_program(
        sample_id="v8", gold_answer="A", baseline_answer="A",
        fork={
            "complete": True, "answer_contract": "single_choice",
            "candidates": candidates,
        },
    )
    assert row["conditions"]["authentic_bound_mutation_program"]["correct"]
    assert row["conditions"]["wrong_guard_bound_mutation"]["correct"]
    assert row["guard_failure_transition"] == "NOOP_TO_BASELINE"
    assert row["distinct_wrong_control_candidates"] == 2
    assert row["distinct_shuffled_control_candidates"] == 2


def test_v8_mutation_preserves_binary_vector_contract():
    candidates = [
        _candidate("0", .9, .4, .8, .2),
        _candidate("1", .9, .7, .2, .8),
    ]
    row = evaluate_candidate_mutation_program(
        sample_id="binary", gold_answer="10", baseline_answer="00",
        fork={
            "complete": True, "answer_contract": "binary_vector",
            "candidates": candidates,
        },
    )
    assert row["conditions"]["authentic_bound_mutation_program"]["committed_answer"] == "10"
