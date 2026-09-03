import numpy as np

from motif_transfer.controlled_exploration_transfer import (
    AbstractAction,
    ActiveIdentificationDomain,
    FEATURE_NAMES,
    GroundedState,
    MatchedValueExample,
    action_features,
    calibrate_target_grounder,
    calibrate_target_neural_grounder,
    collect_matched_examples,
    fit_source_prior_residual_ensemble,
    fit_value_ensemble,
    make_domain,
    marginal_value_control,
    paired_bootstrap_delta,
    run_episode,
    shuffled_value_control,
)


def test_source_and_target_surfaces_share_no_tokens():
    source = make_domain(seed=11, surface="game")
    target = make_domain(seed=12, surface="diagnosis")
    source_tokens = set(source.hypothesis_tokens + source.test_tokens)
    target_tokens = set(target.hypothesis_tokens + target.test_tokens)
    assert not source_tokens & target_tokens


def test_target_native_grounding_produces_operational_features():
    domain = make_domain(seed=13, surface="diagnosis")
    grounded = calibrate_target_grounder(domain, samples_per_cell=8, seed=14)
    belief = tuple([0.25] * 4)
    state = GroundedState(belief, belief, domain.max_tests, tuple([0] * 5))
    test_action = next(action for action in domain.actions if action.kind == "TEST")
    commit_action = next(action for action in domain.actions if action.kind == "COMMIT")
    test_features = action_features(domain, state, grounded, test_action)
    commit_features = action_features(domain, state, grounded, commit_action)
    assert len(test_features) == len(FEATURE_NAMES)
    assert test_features[0] == 1.0
    assert commit_features[0] == 0.0
    assert test_features[1] > 0
    assert commit_features[7] == 0.25


def test_target_native_neural_grounder_is_deterministic_and_operational():
    domain = make_domain(seed=130, surface="diagnosis")
    kwargs = {
        "samples_per_cell": 24,
        "seed": 131,
        "hidden_units": 16,
        "epochs": 800,
        "learning_rate": 0.03,
        "l2": 1e-4,
    }
    left = calibrate_target_neural_grounder(domain, **kwargs)
    right = calibrate_target_neural_grounder(domain, **kwargs)
    assert left.shape == domain.likelihood_array.shape
    assert np.allclose(left, right)
    assert np.all((left > 0.0) & (left < 1.0))
    assert np.mean((left - domain.likelihood_array) ** 2) < 0.05


def test_neural_grounder_has_no_hidden_likelihood_shortcut():
    domain = make_domain(seed=140, surface="diagnosis")
    changed = ActiveIdentificationDomain(
        domain_id=domain.domain_id,
        surface=domain.surface,
        hypothesis_tokens=domain.hypothesis_tokens,
        test_tokens=domain.test_tokens,
        likelihood=tuple(
            tuple(1.0 - value for value in row) for row in domain.likelihood
        ),
        max_tests=domain.max_tests,
        test_cost=domain.test_cost,
    )
    # With zero epochs forbidden, the only environment access is the sampled
    # calibration receipt generation; identical seed remains deterministic.
    original = calibrate_target_neural_grounder(
        domain, samples_per_cell=16, seed=141, epochs=300
    )
    inverted = calibrate_target_neural_grounder(
        changed, samples_per_cell=16, seed=141, epochs=300
    )
    assert not np.allclose(original, inverted)


def test_controls_preserve_receipts_but_remove_action_value_relation():
    domain = make_domain(seed=15, surface="game")
    grounded = calibrate_target_grounder(domain, samples_per_cell=8, seed=16)
    examples = collect_matched_examples(domain, grounded, state_count=4, seed=17)
    shuffled = shuffled_value_control(examples, seed=18)
    marginal = marginal_value_control(examples)
    assert [row.state_id for row in shuffled] == [row.state_id for row in examples]
    for state_id in sorted({row.state_id for row in examples}):
        original_values = sorted(row.value for row in examples if row.state_id == state_id)
        shuffled_values = sorted(row.value for row in shuffled if row.state_id == state_id)
        assert np.allclose(original_values, shuffled_values)
    for kind in ("TEST", "COMMIT"):
        values = {round(row.value, 12) for row in marginal if row.action.kind == kind}
        assert len(values) == 1


def test_episode_and_paired_bootstrap_are_deterministic():
    domain = make_domain(seed=19, surface="diagnosis")
    grounded = calibrate_target_grounder(domain, samples_per_cell=8, seed=20)
    left = run_episode(
        domain, grounded, None, condition="left", episode_seed=21,
        uncertainty_scale=0.5, decision_margin=0.0, fallback_commit_threshold=0.72,
    )
    right = run_episode(
        domain, grounded, None, condition="right", episode_seed=21,
        uncertainty_scale=0.5, decision_margin=0.0, fallback_commit_threshold=0.72,
    )
    assert left.success == right.success
    assert left.net_return == right.net_return
    report = paired_bootstrap_delta(
        [left], [right], metric="net_return", seed=22, samples=20,
    )
    assert report == {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}


def test_target_residual_corrects_source_prior_without_replacing_it():
    action = AbstractAction("TEST", 0, "abstract-test")

    def rows(prefix, offset):
        output = []
        for index in range(8):
            x = index / 7
            features = (1.0, x, 0.2 * x, 0.5, 0.25, 1.0, 1.0, 0.0, 0.0)
            output.append(MatchedValueExample(
                f"{prefix}-{index}", action, features, 0.3 + 0.2 * x + offset,
            ))
        return tuple(output)

    source = rows("source", 0.0)
    target = rows("target", 0.25)
    source_model = fit_value_ensemble(
        source, (), seed=30, ensemble_size=5, alpha=0.1, target_mass=1.0,
    )
    adapted_model = fit_source_prior_residual_ensemble(
        source, target, seed=30, ensemble_size=5,
        source_alpha=0.1, residual_alpha=0.1, residual_scale=1.0,
    )
    assert source_model is not None
    assert adapted_model is not None
    features = [row.features for row in target]
    labels = np.asarray([row.value for row in target])
    source_predictions, _ = source_model.predict(features)
    adapted_predictions, _ = adapted_model.predict(features)
    assert np.mean((adapted_predictions - labels) ** 2) < np.mean(
        (source_predictions - labels) ** 2
    )
