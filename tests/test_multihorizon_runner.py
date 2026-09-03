from __future__ import annotations

from motif_transfer.multihorizon_runner import (
    ForkState,
    ForkStep,
    run_matched_multihorizon_snapshot,
)


class ToyEnvironment:
    def __init__(self):
        self.position = 0
        self.closed = False

    def _state(self):
        return ForkState({"position": self.position}, ("LEFT", "RIGHT"))

    def reset(self, *, seed: int):
        self.position = seed % 2
        return self._state()

    def step(self, action: str):
        self.position += 1 if action == "RIGHT" else -1
        return ForkStep(self._state(), float(self.position > 1))

    def close(self):
        self.closed = True


class ToyPolicy:
    def choose_action(self, state, *, treatment, decision_index, history):
        return "RIGHT" if treatment == "G_PLUS_S" else "LEFT"


def test_runner_uses_closed_loop_common_continuation_and_full_regime():
    reference = ToyEnvironment()
    state = reference.reset(seed=0)
    state = reference.step("RIGHT").state
    rows = run_matched_multihorizon_snapshot(
        ToyEnvironment,
        ToyPolicy(),
        episode_seed=0,
        episode_id="episode",
        fork_step=1,
        prefix_actions=("RIGHT",),
        expected_fork_state_hash=state.receipt_hash,
    )
    assert len(rows) == 8
    assert all(row["status"] == "INTERVENTION_OBSERVED" for row in rows)
    common = next(
        row for row in rows
        if row["mode"] == "COMMON_G_MINUS_S_CONTINUATION"
        and row["treatment"] == "G_PLUS_S"
    )
    full = next(
        row for row in rows
        if row["mode"] == "FULL_TREATMENT_REGIME"
        and row["treatment"] == "G_PLUS_S"
    )
    assert common["actions"] == ["RIGHT"] + ["LEFT"] * 7
    assert full["actions"] == ["RIGHT"] * 8
    assert full["cumulative_returns"]["h8"] > common["cumulative_returns"]["h8"]


def test_runner_fails_closed_on_snapshot_mismatch():
    rows = run_matched_multihorizon_snapshot(
        ToyEnvironment,
        ToyPolicy(),
        episode_seed=0,
        episode_id="episode",
        fork_step=0,
        prefix_actions=(),
        expected_fork_state_hash="wrong",
    )
    assert all(row["status"] == "REPLAY_MISMATCH" for row in rows)
