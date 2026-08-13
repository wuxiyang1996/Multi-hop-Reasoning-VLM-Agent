from motif_transfer.candidate_claim_video_ir import evaluate_candidate_claim_program


def _candidate(slot, bind, unbound, bound, wrong):
    return {
        "slot": slot,
        "track": {"observed_true": bind >= .5, "sensor_reliability": max(bind, 1-bind)},
        "unbound_relation": {"support_probability": unbound},
        "bound_relation": {"support_probability": bound},
        "wrong_guard_relation": {"support_probability": wrong},
    }


def test_independent_identity_audit_overrides_tracker_self_report():
    candidate = _candidate("A", .9, .2, .8, .2)
    candidate["identity_verification"] = {"identity_match_probability": .1}
    fork = {"complete": True, "answer_contract": "single_choice", "candidates": [
        candidate, _candidate("B", .7, .6, .7, .2),
    ]}
    report = evaluate_candidate_claim_program(
        sample_id="audit", gold_answer="B", baseline_answer="A", fork=fork,
    )
    assert report["bind_probabilities"][0] == .1
    assert report["conditions"]["authentic_bound_claim_program"]["correct"]


def test_candidate_claim_program_mcq_attributes_gain_to_bound_guard():
    fork = {
        "complete": True,
        "answer_contract": "single_choice",
        "candidates": [
            _candidate("A", .1, .7, .3, .9),
            _candidate("B", .9, .6, .9, .02),
            _candidate("C", .2, .4, .2, .9),
        ],
    }
    report = evaluate_candidate_claim_program(
        sample_id="x", gold_answer="B", baseline_answer="A", fork=fork,
    )
    assert not report["conditions"]["target_unbound_claim_verification"]["correct"]
    assert report["conditions"]["authentic_bound_claim_program"]["correct"]
    assert not report["conditions"]["wrong_guard_bound_claim"]["correct"]
    assert report["authentic_action_contrast"]


def test_candidate_claim_program_preserves_clevrer_vector_contract():
    fork = {
        "complete": True,
        "answer_contract": "binary_vector",
        "candidates": [
            _candidate("0", .9, .4, .8, .3),
            _candidate("1", .9, .7, .2, .8),
            _candidate("2", .8, .2, .9, .2),
        ],
    }
    report = evaluate_candidate_claim_program(
        sample_id="v", gold_answer="101", baseline_answer="000", fork=fork,
    )
    assert report["conditions"]["authentic_bound_claim_program"]["committed_answer"] == "101"
    assert report["conditions"]["authentic_bound_claim_program"]["correct"]
