from __future__ import annotations

import numpy as np

from motif_transfer.hierarchical_skill_transfer import (
    collect_source_examples,
)
from motif_transfer.pairwise_option_advantage import (
    choose_option_against_fallback,
    conformal_error_quantile,
    deserialize_pairwise_ensemble,
    effect_blind_rows,
    fit_pairwise_ensemble,
    intervention_grounded_rows,
    pairwise_examples,
    pairwise_features,
    phase_blind_rows,
    serialize_pairwise_ensemble,
    within_state_effect_permutation,
)


def _rows():
    return collect_source_examples(
        surfaces=("game",),
        domains_per_surface=3,
        states_per_domain=8,
        seed=16,
        minimum_budget=7,
        maximum_budget=10,
        completion_probability_range=(0.5, 0.9),
        failure_cost_range=(0.01, 0.05),
    )


def test_pairwise_features_are_antisymmetric() -> None:
    rows = _rows()
    left, right = rows[:2]
    forward = np.asarray(pairwise_features(left.features, right.features))
    reverse = np.asarray(pairwise_features(right.features, left.features))
    assert np.allclose(forward, -reverse)


def test_pairwise_model_round_trip_and_conformal_choice() -> None:
    rows = _rows()
    pairs = pairwise_examples(rows)
    model = fit_pairwise_ensemble(pairs, seed=3, ensemble_size=3, alpha=0.1)
    restored = deserialize_pairwise_ensemble(serialize_pairwise_ensemble(model))
    predicted, deviation = restored.predict([row.features for row in pairs[:5]])
    assert predicted.shape == deviation.shape == (5,)
    quantile = conformal_error_quantile(restored, pairs, alpha=0.1)
    state = [row for row in rows if row.state_id == rows[0].state_id]
    features = {row.option: row.features for row in state}
    decision = choose_option_against_fallback(
        restored,
        features,
        fallback_option=state[0].option,
        conformal_error=quantile,
    )
    assert decision["option"] in features
    assert len(decision["all_comparisons"]) == len(features) - 1


def test_phase_blind_rows_remove_required_phase_only() -> None:
    rows = _rows()
    blind = phase_blind_rows(rows)
    assert [row.value for row in rows] == [row.value for row in blind]
    assert [row.option for row in rows] == [row.option for row in blind]
    assert all(sum(row.features[5:10]) == 0.0 for row in blind)
    assert all(row.features[10] == 0.0 for row in blind)


def test_intervention_grounded_rows_remove_oracle_symbols() -> None:
    rows = _rows()
    grounded = intervention_grounded_rows(
        rows, probe_trials=8, probe_seed=99
    )
    assert [row.value for row in rows] == [row.value for row in grounded]
    assert all(sum(row.features[5:10]) == 0.0 for row in grounded)
    assert all(row.features[10] == 0.0 for row in grounded)
    assert all(row.features[15] == 0.0 for row in grounded)
    assert all(row.features[18] == 0.0 for row in grounded)
    assert all(sum(row.features[19:22]) == 0.0 for row in grounded)
    assert all(0.0 < row.features[12] < 1.0 for row in grounded)
    assert all(np.isclose(row.features[12] + row.features[17], 1.0) for row in grounded)


def test_effect_controls_preserve_values_but_break_effect_binding() -> None:
    grounded = intervention_grounded_rows(
        _rows(), probe_trials=8, probe_seed=99
    )
    permuted = within_state_effect_permutation(grounded)
    blind = effect_blind_rows(grounded)
    assert {
        (row.state_id, row.option): row.value for row in grounded
    } == {
        (row.state_id, row.option): row.value for row in permuted
    }
    assert [row.value for row in grounded] == [row.value for row in blind]
    assert any(
        row.features[12] != changed.features[12]
        for row, changed in zip(grounded, permuted)
    )
    assert all(row.features[11] == row.features[12] == 0.0 for row in blind)
