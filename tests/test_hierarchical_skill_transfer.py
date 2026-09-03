import numpy as np

from motif_transfer.hierarchical_skill_transfer import (
    FEATURE_NAMES,
    OPTION_NAMES,
    collect_source_examples,
    fit_value_ensemble,
    option_features,
    phase_permuted_control,
)


def test_option_features_have_frozen_contract():
    features = option_features(
        option="ACQUIRE",
        required_option="ACQUIRE",
        precondition_satisfied=0.8,
        completion_probability=0.7,
        goal_binding_probability=0.9,
        remaining_budget_fraction=0.5,
        workflow_progress_fraction=0.25,
        action_repeat_fraction=0.0,
        noop_probability=0.3,
        stage_urgency=0.5,
        failure_cost=0.1,
    )
    assert len(features) == len(FEATURE_NAMES)
    assert features[OPTION_NAMES.index("ACQUIRE")] == 1.0
    assert features[2 * len(OPTION_NAMES)] == 1.0


def test_source_values_prefer_matching_high_completion_option():
    rows = collect_source_examples(
        surfaces=("game",),
        domains_per_surface=3,
        states_per_domain=8,
        seed=4,
        minimum_budget=7,
        maximum_budget=10,
        completion_probability_range=(0.5, 0.9),
        failure_cost_range=(0.01, 0.05),
    )
    groups = {}
    for row in rows:
        groups.setdefault(row.state_id, []).append(row)
    assert np.mean([
        max(group, key=lambda row: row.value).features[2 * len(OPTION_NAMES)]
        for group in groups.values()
    ]) > 0.8
    model = fit_value_ensemble(rows, seed=5, ensemble_size=3, alpha=0.1)
    predictions, deviations = model.predict([row.features for row in rows[:10]])
    assert predictions.shape == deviations.shape == (10,)


def test_phase_control_changes_required_option_without_changing_labels():
    rows = collect_source_examples(
        surfaces=("game",),
        domains_per_surface=1,
        states_per_domain=1,
        seed=7,
        minimum_budget=7,
        maximum_budget=8,
        completion_probability_range=(0.5, 0.9),
        failure_cost_range=(0.01, 0.05),
    )
    changed = phase_permuted_control(rows)
    assert [row.value for row in rows] == [row.value for row in changed]
    assert any(before.features != after.features for before, after in zip(rows, changed))
