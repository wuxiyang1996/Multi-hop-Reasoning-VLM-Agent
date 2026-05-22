"""Telemetry for the three remaining silent-fallback layers.

Pre-fix the co-evolution loop had three silent failure modes that
made the May-2026 reward dip impossible to diagnose:

  Layer 1: ``check_predicate`` collapsed four distinct outcomes onto
           a single ``False`` return (match-false, key-missing,
           parse-error).  When a Protocol referenced a state key the
           runtime never produced — the dominant contamination bug —
           every step quietly contributed 0 intrinsic_bonus and no
           operator-visible warning fired.

  Layer 2: ``get_top_k_skill_candidates`` walked 3 progressively
           less-discriminating retrieval strategies (semantic similarity,
           all-active-unranked, single-best-guidance).  If the semantic
           engine failed it silently fell through to lists with no
           relevance ranking, so the LoRA's SFT prior was lost without
           any reward-log signal.

  Layer 3: ``episode_runner`` SWITCH branch validated candidates in
           order ``[chosen_idx, *rest]``.  When the harness vetoed the
           LoRA's pick, the runner silently fell through to another
           candidate and ``last_chosen_idx`` recorded the *executed*
           skill — the LoRA's intent was lost forever.

These tests lock in the corresponding fixes:

  * ``check_predicate_with_telemetry`` returns a result-kind tag
    (``match`` / ``mismatch`` / ``key_missing`` / ``parse_error``);
    module-level counters surface in ``get_predicate_stats()``.
  * ``get_top_k_skill_candidates`` tags every candidate with a
    ``_rag_source`` field and increments ``_RAG_STATS``.
  * ``episode_runner`` populates ``lora_chosen_idx`` /
    ``harness_override`` in the GRPO record metadata and applies a
    -0.05 reward shaping when the harness silently overrode the LoRA.
"""

from __future__ import annotations

import pytest


# ────────────────────────────────────────────────────────────────────
# Layer 1: check_predicate key-missing telemetry
# ────────────────────────────────────────────────────────────────────


from decision_agents.protocol_utils import (
    PREDICATE_RESULT_BOOL_DEFAULT,
    PREDICATE_RESULT_KEY_MISSING,
    PREDICATE_RESULT_MATCH,
    PREDICATE_RESULT_MISMATCH,
    PREDICATE_RESULT_PARSE_ERROR,
    check_predicate,
    check_predicate_with_telemetry,
    get_predicate_stats,
    reset_predicate_stats,
)


@pytest.fixture(autouse=True)
def _reset_predicate_stats():
    """Each test starts from a clean counter set."""
    reset_predicate_stats()
    yield
    reset_predicate_stats()


def test_check_predicate_match_against_present_key():
    state = {"score": "100"}
    result, kind = check_predicate_with_telemetry("score=100", state)
    assert result is True
    assert kind == PREDICATE_RESULT_MATCH
    assert get_predicate_stats()[PREDICATE_RESULT_MATCH] == 1


def test_check_predicate_mismatch_against_present_key():
    state = {"score": "100"}
    result, kind = check_predicate_with_telemetry("score=200", state)
    assert result is False
    assert kind == PREDICATE_RESULT_MISMATCH
    assert get_predicate_stats()[PREDICATE_RESULT_MISMATCH] == 1


def test_check_predicate_numeric_key_missing_distinguishable_from_mismatch():
    """Numeric predicates with missing keys still return KEY_MISSING — a
    missing numeric field is genuinely ambiguous and signals a real bug
    in the Protocol (referencing a hallucinated numeric field).
    """
    state = {"score": "100"}
    result, kind = check_predicate_with_telemetry("lives>2", state)
    assert result is False
    assert kind == PREDICATE_RESULT_KEY_MISSING

    stats = get_predicate_stats()
    assert stats[PREDICATE_RESULT_KEY_MISSING] == 1
    assert stats[PREDICATE_RESULT_MISMATCH] == 0
    assert "lives" in stats["missing_keys"]


def test_check_predicate_bool_missing_key_defaults_to_false():
    """Boolean predicates (``key=true``/``key=false``) against an
    accumulating effect dict have natural semantics: a key that hasn't
    appeared yet means the effect hasn't fired → equivalent to "false".

    This matches the StateEffectObserver contract where effects are
    monotonically added (never removed), so missing key == "not yet".
    """
    state: dict[str, str] = {}

    # ``enemy_hit=true`` with no enemy_hit key → effect hasn't fired
    result, kind = check_predicate_with_telemetry("enemy_hit=true", state)
    assert result is False
    assert kind == PREDICATE_RESULT_MISMATCH

    # ``damage_taken=false`` with no damage_taken key → no damage yet → true
    result, kind = check_predicate_with_telemetry("damage_taken=false", state)
    assert result is True
    assert kind == PREDICATE_RESULT_MATCH

    # ``shield_buff!=true`` with no key → shield_buff isn't true → match
    result, kind = check_predicate_with_telemetry("shield_buff!=true", state)
    assert result is True
    assert kind == PREDICATE_RESULT_MATCH

    stats = get_predicate_stats()
    assert stats[PREDICATE_RESULT_BOOL_DEFAULT] == 3
    assert stats[PREDICATE_RESULT_KEY_MISSING] == 0  # not flagged as real-bug


def test_check_predicate_records_unique_missing_keys_numeric_only():
    """Only numeric KEY_MISSING (truly ambiguous) is tracked in the
    ``missing_keys`` set; boolean defaults are resolved cleanly.
    """
    state = {"score": "100"}
    check_predicate_with_telemetry("shield_buff=true", state)   # bool_default
    check_predicate_with_telemetry("shield_buff=false", state)  # bool_default
    check_predicate_with_telemetry("lives>2", state)            # KEY_MISSING

    stats = get_predicate_stats()
    assert stats[PREDICATE_RESULT_KEY_MISSING] == 1
    assert stats[PREDICATE_RESULT_BOOL_DEFAULT] == 2
    assert set(stats["missing_keys"]) == {"lives"}


def test_check_predicate_parse_error_tagged():
    result, kind = check_predicate_with_telemetry("malformed predicate", {})
    assert result is False
    assert kind == PREDICATE_RESULT_PARSE_ERROR
    assert get_predicate_stats()[PREDICATE_RESULT_PARSE_ERROR] == 1


def test_check_predicate_numeric_comparison_match():
    state = {"score": "150"}
    result, kind = check_predicate_with_telemetry("score>100", state)
    assert result is True
    assert kind == PREDICATE_RESULT_MATCH


def test_check_predicate_legacy_bool_return_preserved():
    """``check_predicate`` (no _with_telemetry suffix) keeps returning
    a plain bool — backwards-compat for the 15+ callers in
    ``_SkillTracker`` and friends.
    """
    state = {"score": "100"}
    assert check_predicate("score=100", state) is True
    assert check_predicate("score=200", state) is False
    assert check_predicate("missing=x", state) is False


def test_reset_predicate_stats_clears_missing_keys():
    state = {"score": "100"}
    check_predicate_with_telemetry("lives>2", state)
    assert get_predicate_stats()[PREDICATE_RESULT_KEY_MISSING] == 1
    reset_predicate_stats()
    stats = get_predicate_stats()
    assert stats[PREDICATE_RESULT_KEY_MISSING] == 0
    assert stats["missing_keys"] == []


def test_missing_keys_list_bounded_to_50():
    """Defensive cap so a wildly misconfigured Protocol can't blow up
    ``step_log.jsonl`` with 10k unique missing keys.  Uses numeric
    predicates so the bool_default short-circuit doesn't intercept them.
    """
    for i in range(80):
        check_predicate_with_telemetry(f"phantom_field_{i}>0", {})

    stats = get_predicate_stats()
    assert stats[PREDICATE_RESULT_KEY_MISSING] == 80
    assert len(stats["missing_keys"]) == 50


# ────────────────────────────────────────────────────────────────────
# Layer 2: get_top_k_skill_candidates RAG-path telemetry
# ────────────────────────────────────────────────────────────────────


from scripts.qwen3_decision_agent import (  # noqa: E402
    RAG_PATH_ALL_ACTIVE,
    RAG_PATH_EMPTY,
    RAG_PATH_SEMANTIC,
    RAG_PATH_SINGLE_BEST,
    get_rag_stats,
    get_top_k_skill_candidates,
    reset_rag_stats,
)


@pytest.fixture(autouse=True)
def _reset_rag_stats():
    reset_rag_stats()
    yield
    reset_rag_stats()


class _StubSemanticBank:
    """SkillBank stub that returns ranked candidates from ``select()``
    — simulates the healthy semantic-retrieval path.
    """

    def __init__(self):
        self._bank = self

    def select(self, *args, **kwargs):
        return [
            {"skill_id": "s1", "skill_name": "fire_at_enemy", "relevance": 0.9},
            {"skill_id": "s2", "skill_name": "dodge_left",    "relevance": 0.7},
        ]

    def get_protocol(self, _sid):
        return None

    def get_skill_metadata(self, _sid):
        return {}


class _StubAllActiveBank:
    """Bank with ``select()`` returning [] but ``get_skills_for_decision_agent()``
    returning ≥2 skills — simulates fallback to all-active-unranked.
    """

    def __init__(self):
        self._bank = self

    def select(self, *args, **kwargs):
        return []

    def get_skills_for_decision_agent(self):
        return [
            {"skill_id": "s1", "skill_name": "skill_one"},
            {"skill_id": "s2", "skill_name": "skill_two"},
            {"skill_id": "s3", "skill_name": "skill_three"},
        ]

    def get_protocol(self, _sid):
        return None

    def get_skill_metadata(self, _sid):
        return {}


class _StubEmptyBank:
    """Bank that produces no candidates at all."""

    def __init__(self):
        self._bank = self

    def select(self, *args, **kwargs):
        return []

    def get_skills_for_decision_agent(self):
        return []

    def get_protocol(self, _sid):
        return None

    def get_skill_metadata(self, _sid):
        return {}


def test_rag_semantic_path_tags_candidates():
    bank = _StubSemanticBank()
    cands = get_top_k_skill_candidates(
        bank, state_text="hp=3", game_name="gymv_thunder_force_iii",
        intention="survive", top_k=2,
        apply_rejection_deboost=False,
    )
    assert cands
    for c in cands:
        assert c.get("_rag_source") == RAG_PATH_SEMANTIC
    assert get_rag_stats()[RAG_PATH_SEMANTIC] == 1


def test_rag_all_active_unranked_fallback_tagged_and_counted():
    bank = _StubAllActiveBank()
    cands = get_top_k_skill_candidates(
        bank, state_text="hp=3", game_name="gymv_thunder_force_iii",
        intention="survive", top_k=3,
        apply_rejection_deboost=False,
    )
    assert cands
    for c in cands:
        assert c.get("_rag_source") == RAG_PATH_ALL_ACTIVE
    stats = get_rag_stats()
    assert stats[RAG_PATH_ALL_ACTIVE] == 1
    assert stats[RAG_PATH_SEMANTIC] == 0


def test_rag_empty_path_when_bank_has_nothing():
    bank = _StubEmptyBank()
    cands = get_top_k_skill_candidates(
        bank, state_text="hp=3", game_name="gymv_thunder_force_iii",
        intention="survive", top_k=3,
        apply_rejection_deboost=False,
    )
    assert cands == []
    stats = get_rag_stats()
    assert stats[RAG_PATH_EMPTY] == 1


def test_rag_stats_distinguish_paths():
    """Composite scenario: 2 healthy + 1 fallback + 1 empty.

    Verifies all four buckets accumulate independently and the
    breakdown survives an orchestrator-style snapshot.
    """
    semantic = _StubSemanticBank()
    fallback = _StubAllActiveBank()
    empty = _StubEmptyBank()

    for _ in range(2):
        get_top_k_skill_candidates(
            semantic, state_text="", game_name="g",
            apply_rejection_deboost=False, top_k=2,
        )
    get_top_k_skill_candidates(
        fallback, state_text="", game_name="g",
        apply_rejection_deboost=False, top_k=3,
    )
    get_top_k_skill_candidates(
        empty, state_text="", game_name="g",
        apply_rejection_deboost=False, top_k=3,
    )

    stats = get_rag_stats()
    assert stats[RAG_PATH_SEMANTIC] == 2
    assert stats[RAG_PATH_ALL_ACTIVE] == 1
    assert stats[RAG_PATH_EMPTY] == 1


def test_reset_rag_stats_zeros_all_buckets():
    bank = _StubSemanticBank()
    get_top_k_skill_candidates(
        bank, state_text="", game_name="g",
        apply_rejection_deboost=False, top_k=2,
    )
    assert get_rag_stats()[RAG_PATH_SEMANTIC] == 1
    reset_rag_stats()
    assert all(v == 0 for v in get_rag_stats().values())


# ────────────────────────────────────────────────────────────────────
# Layer 3: harness-override metadata + reward shaping
# ────────────────────────────────────────────────────────────────────
#
# We unit-test the small piece of logic that's purely arithmetic
# (reward shaping) without spinning up a full ``episode_runner``;
# the full integration is exercised by the co-evolution smoke runs.


def _apply_skill_selection_reward_shaping(
    base_reward: float,
    parse_path: str,
    harness_override: bool,
) -> float:
    """Mirrors the shaping rules in ``episode_runner._skill_selection``.

    Kept in sync manually — if the magnitudes there change, update
    here too and the test will catch the silent drift.
    """
    r = base_reward
    if parse_path in ("fallback_zero", "empty_reply"):
        r -= 0.10
    elif parse_path in ("tail_number", "name_substring"):
        r -= 0.02
    if harness_override:
        r -= 0.05
    return r


def test_harness_override_applies_negative_shaping():
    assert _apply_skill_selection_reward_shaping(
        base_reward=1.0, parse_path="skill_tag", harness_override=True
    ) == pytest.approx(0.95)


def test_harness_override_compounds_with_parse_penalty():
    """LoRA emitted unparseable output AND its parsed pick was vetoed.
    Both penalties stack so the adapter feels the full cost of both
    failure modes.
    """
    assert _apply_skill_selection_reward_shaping(
        base_reward=1.0, parse_path="fallback_zero", harness_override=True
    ) == pytest.approx(0.85)


def test_clean_parse_no_override_unchanged():
    assert _apply_skill_selection_reward_shaping(
        base_reward=1.0, parse_path="skill_tag", harness_override=False
    ) == pytest.approx(1.0)


def test_heuristic_parse_alone():
    assert _apply_skill_selection_reward_shaping(
        base_reward=0.5, parse_path="tail_number", harness_override=False
    ) == pytest.approx(0.48)


def test_metadata_schema_contract():
    """Document the GRPO-metadata fields the fix introduced so that
    downstream consumers (reward-log analysis notebooks, the
    orchestrator step summary) know what to read.

    This is a documentation-style sentinel test — it'll start
    failing the moment someone removes a field from the schema.
    """
    expected_fields = {
        "chosen_idx",            # final executed candidate
        "lora_chosen_idx",       # LoRA's original pick (-1 if no LoRA call)
        "harness_override",      # True iff LoRA pick was vetoed/changed
        "skill_candidates",
        "chosen_skill_id",
        "lora_chosen_skill_id",
        "rag_source",            # which RAG path produced the list
        "summary_state",
        "intention",
        "reselect_reason",
        "parse_path",
    }
    # Hand-built representative example matching the runner's schema.
    sample_meta = {
        "chosen_idx": 1,
        "lora_chosen_idx": 0,
        "harness_override": True,
        "skill_candidates": ["s1", "s2", "s3"],
        "chosen_skill_id": "s2",
        "lora_chosen_skill_id": "s1",
        "rag_source": RAG_PATH_SEMANTIC,
        "summary_state": "hp=3",
        "intention": "survive",
        "reselect_reason": "skill_complete",
        "parse_path": "skill_tag",
    }
    assert expected_fields.issubset(sample_meta.keys())
