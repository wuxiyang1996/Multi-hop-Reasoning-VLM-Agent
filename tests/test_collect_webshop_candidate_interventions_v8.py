from scripts.collect_webshop_candidate_interventions_v8 import (
    discover_exact_stalls,
    discover_no_progress_states,
)


def test_discover_exact_stalls_only_returns_explicit_observed_stalls() -> None:
    receipt = {
        "steps": [
            {"step": 0, "observed_exact_stall": False},
            {"step": 1, "observed_exact_stall": True},
            {"step": 2},
        ]
    }
    stalls = discover_exact_stalls([receipt])
    assert [step["step"] for _, step in stalls] == [1]


def test_discover_no_progress_states_uses_observed_before_after_hashes() -> None:
    receipt = {
        "steps": [
            {"step": 0, "before_hash": "a", "after_hash": "b"},
            {"step": 1, "before_hash": "b", "after_hash": "b"},
            {"step": 2, "before_hash": "b", "after_hash": "b"},
            {"step": 3, "before_hash": "b", "after_hash": "c"},
        ]
    }
    rows = discover_no_progress_states([receipt], minimum_no_effect_steps=2)
    assert [step["step"] for _, step in rows] == [3]


def test_discover_no_progress_states_rejects_nonpositive_window() -> None:
    try:
        discover_no_progress_states([], minimum_no_effect_steps=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
