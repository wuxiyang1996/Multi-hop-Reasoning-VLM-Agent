"""Tests for trainer.coevolution.skill_reward_shaping."""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict

from trainer.coevolution.skill_reward_shaping import (
    SkillChainTracker,
    PositionCollapseTracker,
    exploration_bonus,
    premature_switch_penalty,
    reset_shaping_stats,
    get_shaping_stats,
    CHAIN_REWARD_HORIZON,
)


@dataclass
class _FakeRecord:
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── exploration_bonus ──────────────────────────────────────────────────

class TestExplorationBonus:
    def setup_method(self):
        reset_shaping_stats()

    def test_non_default_positive_reward_gives_bonus(self):
        b = exploration_bonus(chosen_idx=2, env_reward=10.0, n_candidates=5)
        assert b > 0

    def test_default_zero_reward_gives_penalty(self):
        b = exploration_bonus(chosen_idx=0, env_reward=0.0, n_candidates=3)
        assert b < 0

    def test_default_positive_reward_is_neutral(self):
        b = exploration_bonus(chosen_idx=0, env_reward=5.0, n_candidates=3)
        assert b == 0.0

    def test_non_default_zero_reward_is_neutral(self):
        b = exploration_bonus(chosen_idx=1, env_reward=0.0, n_candidates=3)
        assert b == 0.0

    def test_single_candidate_always_zero(self):
        b = exploration_bonus(chosen_idx=0, env_reward=10.0, n_candidates=1)
        assert b == 0.0

    def test_telemetry_counts(self):
        exploration_bonus(chosen_idx=1, env_reward=5.0, n_candidates=3)
        exploration_bonus(chosen_idx=0, env_reward=0.0, n_candidates=3)
        stats = get_shaping_stats()
        assert stats["exploration_bonus"] == 1
        assert stats["default_penalty"] == 1


# ── PositionCollapseTracker ────────────────────────────────────────────

class TestPositionCollapseTracker:
    def setup_method(self):
        reset_shaping_stats()

    def test_no_penalty_for_diverse_positions(self):
        t = PositionCollapseTracker(threshold=3)
        for pos in [0, 1, 2, 0, 1]:
            t.record(pos)
        assert t.penalty() == 0.0

    def test_penalty_after_consecutive_same(self):
        t = PositionCollapseTracker(threshold=3)
        for _ in range(4):
            t.record(0)
        pen = t.penalty()
        assert pen < 0

    def test_penalty_increases_with_repetition(self):
        t = PositionCollapseTracker(threshold=3)
        for _ in range(3):
            t.record(0)
        pen3 = t.penalty()

        t2 = PositionCollapseTracker(threshold=3)
        for _ in range(6):
            t2.record(0)
        pen6 = t2.penalty()
        assert pen6 < pen3  # more negative

    def test_no_penalty_below_threshold(self):
        t = PositionCollapseTracker(threshold=3)
        t.record(0)
        t.record(0)
        assert t.penalty() == 0.0


# ── SkillChainTracker ──────────────────────────────────────────────────

class TestSkillChainTracker:
    def test_positive_chain_reward(self):
        tracker = SkillChainTracker(horizon=3, gamma=0.9)
        records = [_FakeRecord(reward=0.5)]

        tracker.register(grpo_idx=0, step=0, current_score=0.0)
        for score in [10.0, 25.0, 50.0]:
            tracker.observe_step(score)

        n = tracker.finalize(records, current_score=50.0, weight=0.3)
        assert n == 1
        assert records[0].reward > 0.5
        assert "chain_reward" in records[0].metadata

    def test_negative_chain_reward(self):
        tracker = SkillChainTracker(horizon=3, gamma=0.9)
        records = [_FakeRecord(reward=0.5)]

        tracker.register(grpo_idx=0, step=0, current_score=100.0)
        for score in [90.0, 80.0, 70.0]:
            tracker.observe_step(score)

        tracker.finalize(records, current_score=70.0, weight=0.3)
        assert records[0].reward < 0.5
        assert records[0].metadata["chain_score_delta"] < 0

    def test_no_effect_without_register(self):
        tracker = SkillChainTracker()
        records = [_FakeRecord(reward=0.5)]
        for _ in range(5):
            tracker.observe_step(100.0)
        n = tracker.finalize(records, current_score=100.0)
        assert n == 0
        assert records[0].reward == 0.5

    def test_multiple_registrations(self):
        tracker = SkillChainTracker(horizon=5, gamma=0.9)
        records = [_FakeRecord(reward=0.3), _FakeRecord(reward=0.3)]

        tracker.register(grpo_idx=0, step=0, current_score=0.0)
        tracker.observe_step(10.0)
        tracker.observe_step(20.0)
        tracker.register(grpo_idx=1, step=2, current_score=20.0)
        tracker.observe_step(50.0)
        tracker.observe_step(80.0)

        n = tracker.finalize(records, current_score=80.0, weight=0.2)
        assert n == 2
        assert records[0].metadata.get("chain_reward") is not None
        assert records[1].metadata.get("chain_reward") is not None


# ── premature_switch_penalty ───────────────────────────────────────────

class TestPrematureSwitchPenalty:
    def test_penalty_for_early_stall(self):
        pen = premature_switch_penalty(
            protocol_completion_ratio=0.2,
            reselect_reason="zero_reward_stall",
        )
        assert pen < 0

    def test_no_penalty_for_completed_protocol(self):
        pen = premature_switch_penalty(
            protocol_completion_ratio=0.8,
            reselect_reason="zero_reward_stall",
        )
        assert pen == 0.0

    def test_no_penalty_for_success(self):
        pen = premature_switch_penalty(
            protocol_completion_ratio=0.1,
            reselect_reason="success:predicate",
        )
        assert pen == 0.0

    def test_no_penalty_for_abort(self):
        pen = premature_switch_penalty(
            protocol_completion_ratio=0.1,
            reselect_reason="abort:predicate",
        )
        assert pen == 0.0

    def test_no_penalty_for_duration_exceeded(self):
        pen = premature_switch_penalty(
            protocol_completion_ratio=0.3,
            reselect_reason="duration_exceeded",
        )
        assert pen == 0.0
