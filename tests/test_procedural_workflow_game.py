from motif_transfer.hierarchical_skill_transfer import OPTION_NAMES
from motif_transfer.procedural_workflow_game import (
    collect_intervention_examples,
    make_domain,
)


def _collection(seed: int = 17):
    return collect_intervention_examples(
        surfaces=("grid_quest", "factory_quest"),
        domains_per_surface=2,
        states_per_domain=3,
        replicates_per_action=4,
        seed=seed,
        minimum_budget=3,
        maximum_budget=7,
        completion_probability_range=(0.45, 0.9),
        failure_cost_range=(0.01, 0.12),
        progress_reward=0.15,
        invalid_option_cost=0.18,
    )


def test_collection_is_matched_deterministic_and_action_complete() -> None:
    first = _collection()
    second = _collection()
    assert first == second
    assert first.domains == 4
    assert first.states == 12
    assert len(first.examples) == first.states * len(OPTION_NAMES)
    assert len(first.receipts) == len(first.examples) * 4
    by_state_replicate = {}
    for row in first.receipts:
        by_state_replicate.setdefault((row.state_id, row.replicate), set()).add(
            row.canonical_option
        )
    assert all(values == set(OPTION_NAMES) for values in by_state_replicate.values())


def test_native_tokens_are_domain_specific_and_not_exported_as_features() -> None:
    first = make_domain(
        surface="grid_quest", domain_index=0, seed=11, maximum_budget=8,
        completion_probability_range=(0.4, 0.9), failure_cost_range=(0.01, 0.1),
        progress_reward=0.15, invalid_option_cost=0.18,
    )
    second = make_domain(
        surface="grid_quest", domain_index=1, seed=11, maximum_budget=8,
        completion_probability_range=(0.4, 0.9), failure_cost_range=(0.01, 0.1),
        progress_reward=0.15, invalid_option_cost=0.18,
    )
    assert set(first.native_action_by_option.values()).isdisjoint(
        second.native_action_by_option.values()
    )
    collection = _collection()
    assert collection.alpha_renamed_native_actions
    assert not collection.raw_action_tokens_exported
    assert all(len(row.features) == 23 for row in collection.examples)


def test_matched_forks_preserve_draw_and_wrong_effect_is_noop() -> None:
    collection = _collection()
    grouped = {}
    for row in collection.receipts:
        grouped.setdefault((row.state_id, row.replicate), []).append(row)
    for rows in grouped.values():
        required = rows[0].required_option
        for row in rows:
            if row.canonical_option != required:
                assert row.first_transition_noop
                assert not row.first_transition_advanced
