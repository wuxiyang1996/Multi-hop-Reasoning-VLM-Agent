from __future__ import annotations

from scripts.run_intervention_grounded_outcome_forks_v19 import (
    _exact_one_sided_sign_p,
    _paired_comparison,
)


def test_exact_sign_test_requires_five_unopposed_wins() -> None:
    assert _exact_one_sided_sign_p(4, 0) == 0.0625
    assert _exact_one_sided_sign_p(5, 0) == 0.03125


def test_paired_bootstrap_uses_task_level_success_differences() -> None:
    authentic = [
        {"task_id": f"task-{index}", "official_success": True}
        for index in range(8)
    ]
    control = [
        {"task_id": f"task-{index}", "official_success": False}
        for index in range(8)
    ]
    result = _paired_comparison(
        authentic, control, bootstrap_seed=13, bootstrap_samples=100,
        alpha=0.05,
    )
    assert result["authentic_only_successes"] == 8
    assert result["comparator_only_successes"] == 0
    assert result["success_rate_difference"] == 1.0
    assert result["paired_task_bootstrap_lower_bound"] == 1.0
