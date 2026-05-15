"""Tests for skill_selection parse-path telemetry.

Locks in the May-2026 fix that surfaces silent fallbacks in
``parse_skill_selection``.  The legacy parser had 4 progressively
less-trustworthy recovery strategies — clean ``SKILL: N`` match,
trailing-number heuristic, candidate-name substring, default to
candidate 0 — but only the first was the "intended" path.  The
remaining 3 silently masked broken LoRA output (the reward log
recorded ``chosen_skill_id=candidates[0].skill_id`` for both
intelligent selections and total LoRA failure).

The fix adds:
  * a parse-path label on every parse result
  * module-level counters reset at each co-evolution step
  * exponential-checkpoint warnings (1, 10, 100, 1000 hits) for the
    heuristic + fallback paths
  * an opt-in 4-element return ``(idx, effects, decision, parse_path)``
    via ``return_parse_path=True`` (default keeps 3-tuple to preserve
    backwards compat for the labeling / unified-runner / scripts
    callers)
"""

from __future__ import annotations

import pytest

from decision_agents.skill_decision_core import (
    PARSE_PATH_EMPTY_REPLY,
    PARSE_PATH_FALLBACK_ZERO,
    PARSE_PATH_NAME_SUBSTRING,
    PARSE_PATH_SKILL_TAG,
    PARSE_PATH_TAIL_NUMBER,
    get_parse_stats,
    parse_skill_selection,
    reset_parse_stats,
)


@pytest.fixture(autouse=True)
def _reset_stats():
    reset_parse_stats()
    yield
    reset_parse_stats()


def _candidates(n: int = 3):
    return [
        {"skill_id": f"SKILL_{i}", "skill_name": f"skill_alpha_{i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Path 1: clean SKILL: N (the SFT-canonical case)
# ---------------------------------------------------------------------------


def test_parse_clean_skill_tag_counts_skill_tag_path():
    reply = "EFFECTS: enemy_hit, damage_taken\nDECISION: SWITCH\nSKILL: 2"
    idx, effects, decision = parse_skill_selection(reply, 3, _candidates(3))
    assert idx == 1   # 1-indexed → 0-indexed
    assert decision == "SWITCH"
    assert "enemy_hit" in effects
    stats = get_parse_stats()
    assert stats[PARSE_PATH_SKILL_TAG] == 1
    assert stats[PARSE_PATH_FALLBACK_ZERO] == 0


def test_parse_clean_skill_tag_returns_path_when_requested():
    reply = "DECISION: CONTINUE\nSKILL: 1"
    idx, effects, decision, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 0
    assert decision == "CONTINUE"
    assert parse_path == PARSE_PATH_SKILL_TAG


# ---------------------------------------------------------------------------
# Path 2: trailing-number heuristic (LoRA forgot the SKILL: tag)
# ---------------------------------------------------------------------------


def test_parse_trailing_number_heuristic_counts_tail_number_path():
    reply = "I think the best choice here is candidate number 3."
    idx, effects, decision, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 2   # "3" → 1-indexed → idx 2
    assert parse_path == PARSE_PATH_TAIL_NUMBER
    assert get_parse_stats()[PARSE_PATH_TAIL_NUMBER] == 1


def test_parse_trailing_number_picks_last_valid_number():
    """When multiple numbers appear, the LAST one in the tail wins
    (LLMs tend to say "from candidate 5 and 7, I pick 7")."""
    reply = "Between candidate 1 and option 3, my choice is 2"
    idx, _, _, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 1   # "2" → idx 1
    assert parse_path == PARSE_PATH_TAIL_NUMBER


def test_parse_trailing_number_rejects_out_of_range():
    """If the trailing number is out of range, falls through to
    further heuristics — must not silently clamp."""
    reply = "I want candidate number 99 because reasons"
    idx, _, _, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    # Falls through; no name substring match either
    assert parse_path == PARSE_PATH_FALLBACK_ZERO
    assert idx == 0


# ---------------------------------------------------------------------------
# Path 3: candidate-name substring
# ---------------------------------------------------------------------------


def test_parse_name_substring_match_counts_name_substring_path():
    reply = "Definitely going with skill_alpha_1 here"
    idx, _, _, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 1
    assert parse_path == PARSE_PATH_NAME_SUBSTRING
    assert get_parse_stats()[PARSE_PATH_NAME_SUBSTRING] == 1


def test_parse_name_substring_only_kicks_in_after_no_number():
    """If LoRA emits both a number AND a name, the number heuristic
    wins (it's higher in the precedence stack)."""
    reply = "Choosing skill_alpha_0 — final answer 2"
    idx, _, _, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 1   # tail number "2" → idx 1
    assert parse_path == PARSE_PATH_TAIL_NUMBER


# ---------------------------------------------------------------------------
# Path 4 + Empty-reply: the silent fallback paths (the bug-of-record)
# ---------------------------------------------------------------------------


def test_parse_empty_reply_is_separately_tracked():
    """``empty_reply`` is distinct from ``fallback_zero`` so we can
    distinguish "LoRA returned nothing" from "LoRA returned garbage"."""
    idx, effects, decision, parse_path = parse_skill_selection(
        "", 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 0
    assert effects == []
    assert decision == "SWITCH"
    assert parse_path == PARSE_PATH_EMPTY_REPLY
    stats = get_parse_stats()
    assert stats[PARSE_PATH_EMPTY_REPLY] == 1
    assert stats[PARSE_PATH_FALLBACK_ZERO] == 0


def test_parse_unparseable_garbage_falls_back_to_zero_loudly():
    """The bug-of-record: pre-fix this returned (0, [], "SWITCH")
    silently; lock in that we now (a) still return 0 (for backwards
    compat — episode_runner expects a valid idx) AND (b) count it +
    label it so monitoring / reward shaping can react."""
    reply = "Hmm, I'm not sure. Let me think about this more carefully..."
    idx, _, _, parse_path = parse_skill_selection(
        reply, 3, _candidates(3), return_parse_path=True,
    )
    assert idx == 0
    assert parse_path == PARSE_PATH_FALLBACK_ZERO
    assert get_parse_stats()[PARSE_PATH_FALLBACK_ZERO] == 1


def test_parse_counters_accumulate_across_calls():
    parse_skill_selection("SKILL: 1", 3, _candidates(3))
    parse_skill_selection("SKILL: 2", 3, _candidates(3))
    parse_skill_selection("garbage", 3, _candidates(3))
    parse_skill_selection("", 3, _candidates(3))
    parse_skill_selection("choose 3", 3, _candidates(3))

    stats = get_parse_stats()
    assert stats[PARSE_PATH_SKILL_TAG] == 2
    assert stats[PARSE_PATH_FALLBACK_ZERO] == 1
    assert stats[PARSE_PATH_EMPTY_REPLY] == 1
    assert stats[PARSE_PATH_TAIL_NUMBER] == 1


def test_reset_parse_stats_clears_all_counters():
    parse_skill_selection("SKILL: 1", 3, _candidates(3))
    parse_skill_selection("garbage", 3, _candidates(3))
    assert sum(get_parse_stats().values()) == 2

    reset_parse_stats()
    assert sum(get_parse_stats().values()) == 0


# ---------------------------------------------------------------------------
# Backwards compat: existing callers must still get a 3-tuple
# ---------------------------------------------------------------------------


def test_default_return_remains_3_tuple_for_back_compat():
    """``return_parse_path`` defaults to False so the labeling /
    unified-runner / scripts callers still unpack 3 values without
    modification."""
    result = parse_skill_selection("SKILL: 1", 3, _candidates(3))
    assert isinstance(result, tuple)
    assert len(result) == 3
    idx, effects, decision = result
    assert idx == 0
    assert isinstance(effects, list)
    assert decision in ("CONTINUE", "SWITCH", "SKIP")


def test_opt_in_4_tuple_when_return_parse_path_true():
    result = parse_skill_selection(
        "SKILL: 1", 3, _candidates(3), return_parse_path=True,
    )
    assert isinstance(result, tuple)
    assert len(result) == 4
    idx, effects, decision, parse_path = result
    assert parse_path in (
        PARSE_PATH_SKILL_TAG, PARSE_PATH_TAIL_NUMBER,
        PARSE_PATH_NAME_SUBSTRING, PARSE_PATH_FALLBACK_ZERO,
        PARSE_PATH_EMPTY_REPLY,
    )
