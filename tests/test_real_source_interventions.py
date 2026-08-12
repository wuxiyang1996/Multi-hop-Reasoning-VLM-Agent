from __future__ import annotations

from motif_transfer.real_source_interventions import (
    build_live_frozen_plan,
    split_seeds,
    summarize_source_gate,
    validate_plan,
)


class _DeterministicAdapter:
    def __init__(self, game: str, max_steps: int) -> None:
        self.game = game
        self.max_steps = max_steps
        self.seed = 0
        self.step_index = 0
        self.last_terminated = False
        self.last_truncated = False

    def reset(self, *, seed: int) -> str:
        self.seed = seed
        self.step_index = 0
        return self.state_receipt()

    def state_receipt(self) -> str:
        return f"{self.game}:{self.seed}:{self.step_index}"

    def admissible_actions(self) -> tuple[str, ...]:
        return ("left", "right", "test")

    def step(self, action: str) -> str:
        assert action in self.admissible_actions()
        self.step_index += 1
        return self.state_receipt()

    def close(self) -> None:
        pass


def test_split_seeds_is_deterministic_and_balanced() -> None:
    first = split_seeds(range(6), namespace="frozen")
    second = split_seeds(reversed(range(6)), namespace="frozen")
    assert first == second
    assert sorted(first.values()).count("development") == 2
    assert sorted(first.values()).count("qualification") == 2
    assert sorted(first.values()).count("heldout") == 2


def test_source_gate_requires_action_variation_in_every_split() -> None:
    rows = []
    for split in ("development", "qualification", "heldout"):
        for state in range(2):
            for reward in (0.0, 1.0):
                rows.append(
                    {
                        "split": split,
                        "snapshot_id": f"{split}-{state}",
                        "status": "VALID",
                        "immediate_reward": reward,
                    }
                )
    assert summarize_source_gate(rows)["status"] == "SOURCE_GATE_PASSED"
    for row in rows:
        if row["split"] == "heldout":
            row["immediate_reward"] = 0.0
    assert summarize_source_gate(rows)["status"] == "SOURCE_GATE_FAILED"


def test_source_gate_fails_on_non_reproducible_forks() -> None:
    rows = []
    for split in ("development", "qualification", "heldout"):
        for state in range(2):
            for action in range(20):
                rows.append(
                    {
                        "split": split,
                        "snapshot_id": f"{split}-{state}",
                        "status": "VALID" if action < 18 else "FORK_STATE_MISMATCH",
                        "immediate_reward": float(action % 2),
                    }
                )
    assert summarize_source_gate(rows)["status"] == "SOURCE_GATE_FAILED"


def test_live_plan_is_outcome_blind_and_valid() -> None:
    plan = build_live_frozen_plan(
        _DeterministicAdapter,
        game="fixture",
        seeds=range(6),
        namespace="frozen-live",
        max_steps=10,
        rollout_steps=8,
        snapshots_per_episode=2,
        actions_per_snapshot=3,
    )
    validate_plan(plan)
    assert len(plan["snapshots"]) == 12
    assert plan["selection"]["reward_read_during_plan_collection"] is False
    assert all(len(row["selected_actions"]) == 3 for row in plan["snapshots"])
