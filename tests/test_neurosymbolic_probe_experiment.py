import pytest

from motif_transfer.neurosymbolic_probe_experiment import (
    FEATURE_NAMES,
    OperationalTransition,
    build_operational_probe_examples,
    split_source_examples,
)


def _transition(episode, step, before, after, reward=0.0, terminal=False, action="a"):
    return OperationalTransition(
        episode, step, tuple(before), action, tuple(after), reward, terminal,
    )


def test_operational_probe_uses_set_change_and_only_past_features():
    rows = build_operational_probe_examples((
        _transition("e", 0, ("a", "b"), ("b", "a")),
        _transition("e", 1, ("a", "b"), ("a",), reward=1.0, action="a"),
    ))
    assert len(rows[0].features) == len(FEATURE_NAMES)
    assert rows[0].labels == (0, 0, 0)
    assert rows[1].labels == (1, 1, 0)
    # The second row can see only the first row's effect, not its own label.
    assert rows[1].features[2:5] == (0.0, 0.0, 0.0)


def test_operational_probe_rejects_non_native_action():
    with pytest.raises(ValueError, match="not native-admissible"):
        build_operational_probe_examples((
            _transition("e", 0, ("a",), ("a",), action="invented"),
        ))


def test_source_split_is_episode_disjoint_and_deterministic():
    transitions = tuple(
        _transition(f"e{episode}", 0, ("a",), ("a",))
        for episode in range(6)
    )
    examples = build_operational_probe_examples(transitions)
    splits = split_source_examples(examples)
    episode_sets = [
        {row.episode_id for row in splits[name]}
        for name in ("train", "validation", "source_held_out")
    ]
    assert [len(rows) for rows in episode_sets] == [2, 2, 2]
    assert not (episode_sets[0] & episode_sets[1])
    assert not (episode_sets[0] & episode_sets[2])
    assert not (episode_sets[1] & episode_sets[2])
