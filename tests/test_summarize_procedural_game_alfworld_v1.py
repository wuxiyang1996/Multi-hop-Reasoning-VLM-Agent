from scripts.summarize_procedural_game_alfworld_v1 import (
    exact_sign_two_sided,
    paired_counts,
)


def test_paired_counts_use_task_identity_not_order() -> None:
    authentic = [
        {"task_id": "a", "official_success": True},
        {"task_id": "b", "official_success": False},
        {"task_id": "c", "official_success": True},
    ]
    comparator = [
        {"task_id": "c", "official_success": False},
        {"task_id": "a", "official_success": True},
        {"task_id": "b", "official_success": True},
    ]
    assert paired_counts(authentic, comparator) == {
        "wins": 1, "losses": 1, "ties": 1,
    }


def test_exact_sign_probability() -> None:
    assert exact_sign_two_sided(8, 0) == 0.0078125
    assert exact_sign_two_sided(0, 0) == 1.0
