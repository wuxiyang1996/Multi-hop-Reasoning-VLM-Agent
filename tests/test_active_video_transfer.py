import json
from pathlib import Path

import numpy as np

from motif_transfer.active_video_transfer import (
    CandidateEffectRow,
    GroundedCandidateIntervention,
    CalibrationRow,
    GainRow,
    build_source_value_models,
    candidate_action_features,
    choose_candidate_action,
    choose_video_action,
    exact_binomial_two_sided,
    fit_calibration_head,
    fit_candidate_effect_grounder,
    fit_gain_grounder,
    normalized_probabilities,
    video_action_features,
)


REPO = Path(__file__).resolve().parents[1]


def _calibration_rows():
    rows = []
    for sample in range(8):
        answer = sample % 6
        for prefix in range(5):
            probabilities = np.full(6, 0.04)
            probabilities[answer] = 0.8
            probabilities[(answer + 1) % 6] += 0.0 if prefix else 0.1
            probabilities /= probabilities.sum()
            rows.append(CalibrationRow(
                sample_id=f"sample-{sample}",
                prefix_length=prefix,
                max_tests=4,
                mean_planner_score=prefix / 5,
                raw_probabilities=tuple(probabilities),
                answer_index=answer,
            ))
    return rows


def test_target_native_calibration_head_is_deterministic_and_normalized():
    rows = _calibration_rows()
    left = fit_calibration_head(rows, seed=7, epochs=300)
    right = fit_calibration_head(rows, seed=7, epochs=300)
    prediction = left.predict(rows[0].features())
    assert np.allclose(left.temperature_weights, right.temperature_weights)
    assert np.isclose(prediction.sum(), 1.0)
    assert int(np.argmax(prediction)) == rows[0].answer_index


def test_calibration_head_cannot_change_answer_slot_argmax():
    rows = _calibration_rows()
    model = fit_calibration_head(rows, seed=70, epochs=300)
    for row in rows:
        assert int(np.argmax(model.predict(row.features()))) == int(
            np.argmax(row.raw_probabilities)
        )


def test_gain_grounder_and_video_features_are_operational():
    gain_rows = [
        GainRow(
            sample_id=f"s{index}",
            current_belief=(0.4, 0.2, 0.1, 0.1, 0.1, 0.1),
            next_planner_score=index / 10,
            prefix_fraction=index / 10,
            information_gain=index / 30,
            confidence_gain=index / 40,
        )
        for index in range(10)
    ]
    model = fit_gain_grounder(gain_rows, seed=8, epochs=300)
    test, commits = video_action_features(
        gain_rows[0].current_belief,
        prefix_length=1,
        max_tests=4,
        next_planner_score=0.8,
        gain_grounder=model,
    )
    assert len(test) == 9
    assert len(commits) == 6
    assert test[0] == 1.0
    assert all(row[0] == 0.0 for row in commits)


def test_source_models_route_only_abstract_test_or_commit():
    config = json.loads((
        REPO / "configs/controlled_neural_symbolic_transfer_v3_formal.json"
    ).read_text())
    source_models = build_source_value_models(config, seed=9)
    gain = fit_gain_grounder([
        GainRow(
            sample_id=f"s{index}",
            current_belief=(1 / 6,) * 6,
            next_planner_score=index / 10,
            prefix_fraction=index / 10,
            information_gain=0.1,
            confidence_gain=0.05,
        )
        for index in range(10)
    ], seed=10, epochs=200)
    decision = choose_video_action(
        (1 / 6,) * 6,
        condition="authentic_source_plus_target",
        prefix_length=0,
        max_tests=4,
        next_planner_score=0.9,
        gain_grounder=gain,
        source_models=source_models,
        fallback_commit_threshold=0.72,
        uncertainty_scale=0.5,
        decision_margin=0.0025,
        information_gain_threshold=0.025,
    )
    assert decision.kind in {"TEST", "COMMIT"}
    if decision.kind == "COMMIT":
        assert 0 <= decision.answer_index < 6
    else:
        assert decision.answer_index is None


def test_candidate_grounder_ranks_matched_interventions_and_source_routes_ids():
    rows = []
    belief = (0.35, 0.2, 0.15, 0.1, 0.1, 0.1)
    for sample in range(12):
        for candidate in range(3):
            useful = candidate == sample % 3
            descriptor = [0.0] * 8
            descriptor[candidate] = 1.0
            rows.append(CandidateEffectRow(
                sample_id=f"s{sample}",
                candidate_id=f"c{candidate}",
                current_belief=belief,
                planner_score=0.9 if useful else 0.2,
                descriptor=tuple(descriptor),
                information_gain=0.2 if useful else 0.0,
                confidence_gain=0.15 if useful else 0.0,
                answer_quality_gain=0.4 if useful else -0.1,
            ))
    grounder = fit_candidate_effect_grounder(rows, seed=81, epochs=400)
    predictions = [grounder.predict(
        belief, planner_score=row.planner_score, descriptor=row.descriptor,
    ) for row in rows[:3]]
    assert all(len(row) == 3 for row in predictions)
    candidate_rows = [GroundedCandidateIntervention(
        candidate_id=f"c{index}", planner_score=0.8 - index * 0.1,
        predicted_information_gain=0.2 - index * 0.05,
        predicted_confidence_gain=0.1 - index * 0.02,
        predicted_answer_quality_gain=0.3 - index * 0.1,
        predicted_outcome_balance=0.8,
    ) for index in range(3)]
    tests, commits = candidate_action_features(
        belief, candidates=candidate_rows, remaining_test_fraction=1.0,
    )
    assert len(tests) == 3
    assert len(commits) == 6
    assert all(len(row) == 9 for row in tests + commits)
    source_config = json.loads((
        REPO / "configs/controlled_neural_symbolic_transfer_v3_formal.json"
    ).read_text())
    decision = choose_candidate_action(
        belief,
        condition="authentic_source_plus_target",
        candidates=candidate_rows,
        source_models=build_source_value_models(source_config, seed=82),
        uncertainty_scale=0.5,
        decision_margin=0.0025,
        fallback_commit_threshold=0.72,
        target_quality_threshold=0.0,
        information_gain_threshold=0.025,
    )
    assert decision.kind in {"TEST", "COMMIT"}
    if decision.kind == "TEST":
        assert decision.candidate_id in {"c0", "c1", "c2"}
        assert decision.answer_index is None
    else:
        assert decision.candidate_id is None
        assert 0 <= decision.answer_index < 6


def test_probability_and_paired_sign_contracts_fail_closed():
    assert np.allclose(
        normalized_probabilities({slot: 1 for slot in "ABCDEF"}),
        np.full(6, 1 / 6),
    )
    assert exact_binomial_two_sided(6, 0) == 0.03125
    assert exact_binomial_two_sided(0, 0) == 1.0


def test_binary_target_uses_native_answer_slots_without_padding():
    probabilities = normalized_probabilities(
        {"A": 3, "B": 1}, answer_slots=("A", "B"),
    )
    assert np.allclose(probabilities, (0.75, 0.25))
    rows = [
        CalibrationRow(
            sample_id=f"binary-{index}",
            prefix_length=0,
            max_tests=1,
            mean_planner_score=0.5,
            raw_probabilities=(0.8, 0.2) if index % 2 == 0 else (0.2, 0.8),
            answer_index=index % 2,
        )
        for index in range(8)
    ]
    head = fit_calibration_head(rows, seed=91, epochs=100)
    assert head.answer_slot_count == 2
    assert head.predict(rows[0].features()).shape == (2,)
