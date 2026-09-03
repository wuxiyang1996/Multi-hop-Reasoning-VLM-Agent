from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

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
    truncated: bool = False
    observable: Any | None = None
    official_value: float | None = None

    @property
    def receipt_hash(self) -> str:
        return stable_hash({
            "state": self.state,
            "admissible_actions": self.admissible_actions,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "official_value": self.official_value,
        })

    @property
    def observable_hash(self) -> str:
        return stable_hash(self.state if self.observable is None else self.observable)


@dataclass(frozen=True)
class ForkStep:
    state: ForkState
    reward: float
    official_success: bool | None = None
    official_value: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyHistoryStep:
    decision_index: int
    treatment: str
    action: str
    reward: float
    before_state_hash: str
    after_state_hash: str
    official_value: float | None = None
    official_success: bool | None = None


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    prompt_sha256: str | None = None
    raw_response_sha256: str | None = None
    requested_adapter: str | None = None
    used_adapter: str | None = None
    request_seed: int | None = None
    source: str = "LIVE_POLICY"
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
        history: Sequence[PolicyHistoryStep],
    ) -> str | PolicyDecision: ...


def run_matched_multihorizon_snapshot(
    environment_factory,
    policy: TreatmentPolicy,
    *,
    episode_seed: int,
    episode_id: str,
    fork_step: int,
    prefix_actions: Sequence[str],
    expected_fork_state_hash: str | None = None,
    expected_fork_observable_hash: str | None = None,
    split: str | None = None,
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
    if (expected_fork_state_hash is None) == (expected_fork_observable_hash is None):
        raise ValueError(
            "supply exactly one of expected_fork_state_hash or "
            "expected_fork_observable_hash"
        )
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
                    "expected_fork_observable_hash": expected_fork_observable_hash,
                    "observed_fork_state_hash": state.receipt_hash,
                    "observed_fork_observable_hash": state.observable_hash,
                    **({"split": split} if split is not None else {}),
                }
                state_matches = (
                    state.receipt_hash == expected_fork_state_hash
                    if expected_fork_state_hash is not None
                    else state.observable_hash == expected_fork_observable_hash
                )
                if replay_failed or not state_matches:
                    rows.append(base | {
                        "status": "REPLAY_MISMATCH",
                        "actions": [],
                        "step_rewards": [],
                    })
                    continue
                actions: list[str] = []
                rewards: list[float] = []
                history: list[PolicyHistoryStep] = []
                success: bool | None = None
                official_values = [state.official_value]
                step_receipts: list[dict[str, Any]] = []
                for decision_index in range(maximum_horizon):
                    if state.terminal or state.truncated:
                        break
                    active_treatment = (
                        treatment
                        if decision_index == 0 or mode == "FULL_TREATMENT_REGIME"
                        else "G_MINUS_S"
                    )
                    proposal = policy.choose_action(
                        state,
                        treatment=active_treatment,
                        decision_index=decision_index,
                        history=tuple(history),
                    )
                    decision = (
                        proposal if isinstance(proposal, PolicyDecision)
                        else PolicyDecision(action=str(proposal))
                    )
                    action = decision.action
                    if action not in state.admissible_actions:
                        rows.append(base | {
                            "status": "POLICY_ACTION_INADMISSIBLE",
                            "actions": actions + [action],
                            "step_rewards": rewards,
                            "failed_decision_index": decision_index,
                            "active_treatment": active_treatment,
                            "policy_decision": asdict(decision),
                        })
                        break
                    before_hash = state.receipt_hash
                    result = env.step(action)
                    actions.append(action)
                    rewards.append(float(result.reward))
                    state = result.state
                    official_value = (
                        result.official_value
                        if result.official_value is not None
                        else state.official_value
                    )
                    official_values.append(official_value)
                    if result.official_success is not None:
                        success = bool(success) or bool(result.official_success)
                    history_step = PolicyHistoryStep(
                        decision_index=decision_index,
                        treatment=active_treatment,
                        action=action,
                        reward=float(result.reward),
                        before_state_hash=before_hash,
                        after_state_hash=state.receipt_hash,
                        official_value=official_value,
                        official_success=result.official_success,
                    )
                    history.append(history_step)
                    step_receipts.append({
                        "decision": asdict(decision),
                        "transition": asdict(history_step),
                        "state_terminal": state.terminal,
                        "state_truncated": state.truncated,
                        "metadata": dict(result.metadata),
                        "receipt_sha256": stable_hash({
                            "decision": asdict(decision),
                            "transition": asdict(history_step),
                            "state_terminal": state.terminal,
                            "state_truncated": state.truncated,
                            "metadata": dict(result.metadata),
                        }),
                    })
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
                    "official_values": official_values,
                    "official_value_delta": (
                        official_values[-1] - official_values[0]
                        if official_values
                        and official_values[0] is not None
                        and official_values[-1] is not None
                        else None
                    ),
                    "observed_horizon": len(rewards),
                    "first_positive_reward_step": next(
                        (index + 1 for index, reward in enumerate(rewards) if reward > 0),
                        None,
                    ),
                    "step_receipts": step_receipts,
                })
            finally:
                env.close()
    return tuple(rows)


__all__ = [
    "ForkState", "ForkStep", "PolicyHistoryStep", "PolicyDecision",
    "ForkEnvironment", "TreatmentPolicy",
    "run_matched_multihorizon_snapshot",
]
