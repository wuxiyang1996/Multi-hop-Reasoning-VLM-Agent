from __future__ import annotations

from scripts.summarize_four_domain_replication_v1 import exact_sign_p, paired_counts


def test_paired_counts_preserve_wins_losses_and_ties() -> None:
    authentic = {"a": True, "b": False, "c": True, "d": False}
    target = {"a": False, "b": True, "c": True, "d": False}
    assert paired_counts(authentic, target) == {"wins": 1, "losses": 1, "ties": 2}


def test_exact_sign_p_matches_small_known_cases() -> None:
    assert exact_sign_p(0, 0) == 1.0
    assert exact_sign_p(7, 0) == 0.015625
    assert exact_sign_p(3, 0) == 0.25


def test_paired_counts_reject_mismatched_coverage() -> None:
    try:
        paired_counts({"a": True}, {"b": False})
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("mismatched paired coverage must fail closed")
