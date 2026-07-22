from __future__ import annotations

from typing import Callable, Protocol, Sequence

from .contracts import DecisionCycleRecord, Observation, ReplayForkReceipt, stable_hash


class ReplayableEnvironment(Protocol):
    def reset(self, *, seed: int) -> Observation: ...

    def step(self, action: str) -> tuple[Observation, float]: ...


class ReplayMismatch(RuntimeError):
    pass


def replay_all_observed_alternatives(
    environment_factory: Callable[[], ReplayableEnvironment],
    records: Sequence[DecisionCycleRecord],
    *,
    seed: int,
) -> tuple[ReplayForkReceipt, ...]:
    """Replay every native alternative recorded at every decision point.

    There is no semantic selection or top-k ranking. An experiment that cannot
    afford exhaustive forks must pre-register a sampling design outside this
    function and report that design as part of its budget identity.
    """

    receipts: list[ReplayForkReceipt] = []
    prefix: list[str] = []
    for record in records:
        chosen = record.proposal_set.selected.action
        alternatives = tuple(action for action in record.before.native_actions if action != chosen)
        for alternative in alternatives:
            environment = environment_factory()
            observation = environment.reset(seed=seed)
            for prefix_action in prefix:
                observation, _ = environment.step(prefix_action)
            if stable_hash(observation.state) != stable_hash(record.before.state):
                raise ReplayMismatch("replayed fork state does not match the recorded state")
            if stable_hash(observation.native_actions) != stable_hash(record.before.native_actions):
                raise ReplayMismatch("replayed native action set does not match the recorded action set")
            if alternative not in observation.native_actions:
                raise ReplayMismatch("recorded alternative is no longer native-admissible")
            after, _ = environment.step(alternative)
            receipts.append(
                ReplayForkReceipt.create(
                    source_transition_id=record.transition.receipt_id,
                    prefix_hash=stable_hash(prefix),
                    fork_state_hash=stable_hash(observation.state),
                    admissible_actions_hash=stable_hash(observation.native_actions),
                    alternative_action=alternative,
                    alternative_after_hash=stable_hash(after.state),
                )
            )
        prefix.append(chosen)
    return tuple(receipts)
