from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .multihorizon_replay import (
    HORIZONS,
    MODES,
    TREATMENTS,
    cumulative_returns,
    stable_hash,
)


@dataclass(frozen=True)
class ForkState:
    """Only the environment adapter may assign semantics to the native state."""

    state: Any
    admissible_actions: tuple[str, ...]
    terminal: bool = False

    @property
    def receipt_hash(self) -> str:
        return stable_hash({
            "state": self.state,
            "admissible_actions": self.admissible_actions,
            "terminal": self.terminal,
        })


@dataclass(frozen=True)
class ForkStep:
    state: ForkState
    reward: float
    official_success: bool = False


class ForkEnvironment(Protocol):
    def reset(self, *, seed: int) -> ForkState: ...
    def step(self, action: str) -> ForkStep: ...
    def close(self) -> None: ...


class TreatmentPolicy(Protocol):
    def choose_action(
        self,
        state: ForkState,
        *,
        treatment: str,
        decision_index: int,
    ) -> str: ...


def run_matched_multihorizon_snapshot(
    environment_factory,
    policy: TreatmentPolicy,
    *,
    episode_seed: int,
    episode_id: str,
    fork_step: int,
    prefix_actions: Sequence[str],
    expected_fork_state_hash: str,
    maximum_horizon: int = max(HORIZONS),
) -> tuple[dict[str, Any], ...]:
    """Run both causal estimands from the same replay-verified snapshot.

    COMMON_G_MINUS_S_CONTINUATION changes only the first action and then uses
    G_MINUS_S on every state reached by that intervention. FULL_TREATMENT_REGIME
    keeps the assigned treatment throughout. Recorded historical actions are
    never used as a fake continuation.
    """

    if maximum_horizon < max(HORIZONS):
        raise ValueError("maximum_horizon must cover the frozen h8 primary endpoint")
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for treatment in TREATMENTS:
            env: ForkEnvironment = environment_factory()
            try:
                state = env.reset(seed=episode_seed)
                prefix_receipts = [state.receipt_hash]
                replay_failed = False
                for action in prefix_actions:
                    if state.terminal or action not in state.admissible_actions:
                        replay_failed = True
                        break
                    result = env.step(action)
                    state = result.state
                    prefix_receipts.append(state.receipt_hash)
                base = {
                    "episode_seed": episode_seed,
                    "episode_id": episode_id,
                    "fork_step": fork_step,
                    "mode": mode,
                    "treatment": treatment,
                    "prefix_actions": list(prefix_actions),
                    "prefix_state_hashes": prefix_receipts,
                    "expected_fork_state_hash": expected_fork_state_hash,
                    "observed_fork_state_hash": state.receipt_hash,
                }
                if replay_failed or state.receipt_hash != expected_fork_state_hash:
                    rows.append(base | {
                        "status": "REPLAY_MISMATCH",
                        "actions": [],
                        "step_rewards": [],
                    })
                    continue
                actions: list[str] = []
                rewards: list[float] = []
                success = False
                for decision_index in range(maximum_horizon):
                    if state.terminal:
                        break
                    active_treatment = (
                        treatment
                        if decision_index == 0 or mode == "FULL_TREATMENT_REGIME"
                        else "G_MINUS_S"
                    )
                    action = policy.choose_action(
                        state,
                        treatment=active_treatment,
                        decision_index=decision_index,
                    )
                    if action not in state.admissible_actions:
                        rows.append(base | {
                            "status": "POLICY_ACTION_INADMISSIBLE",
                            "actions": actions + [action],
                            "step_rewards": rewards,
                            "failed_decision_index": decision_index,
                            "active_treatment": active_treatment,
                        })
                        break
                    result = env.step(action)
                    actions.append(action)
                    rewards.append(float(result.reward))
                    state = result.state
                    success |= bool(result.official_success)
                else:
                    result = None
                if rows and rows[-1].get("episode_id") == episode_id \
                        and rows[-1].get("fork_step") == fork_step \
                        and rows[-1].get("mode") == mode \
                        and rows[-1].get("treatment") == treatment \
                        and rows[-1]["status"] == "POLICY_ACTION_INADMISSIBLE":
                    continue
                rows.append(base | {
                    "status": "INTERVENTION_OBSERVED",
                    "actions": actions,
                    "step_rewards": rewards,
                    "cumulative_returns": cumulative_returns(rewards or [0.0]),
                    "final_state_hash": state.receipt_hash,
                    "official_success": success,
                    "observed_horizon": len(rewards),
                })
            finally:
                env.close()
    return tuple(rows)


__all__ = [
    "ForkState", "ForkStep", "ForkEnvironment", "TreatmentPolicy",
    "run_matched_multihorizon_snapshot",
]
