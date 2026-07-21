"""Deterministic replay-to-fork verifier with explicit unsupported gaps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ReplayForkAdapter(Protocol):
    """Minimal domain-neutral adapter; semantic predicates are absent."""

    def reset(self, *, seed: int) -> Any: ...
    def state_receipt(self) -> Any: ...
    def admissible_actions(self) -> Sequence[str]: ...
    def step(self, action: str) -> Any: ...


@dataclass(frozen=True)
class ForkInterventionReceipt:
    intervention_id: str
    seed: int
    prefix_actions: Sequence[str]
    expected_fork_state_sha256: str
    replayed_fork_state_sha256: str | None
    alternative_action: str
    admissible_actions_sha256: str | None
    alternative_next_state_sha256: str | None
    status: str
    failure_codes: Sequence[str]

    def content_hash(self) -> str:
        return _hash(asdict(self))


class ReplayForkVerifier:
    def run(
        self,
        adapter: ReplayForkAdapter,
        *,
        intervention_id: str,
        seed: int,
        prefix_actions: Sequence[str],
        expected_fork_state_sha256: str,
        alternative_action: str,
    ) -> ForkInterventionReceipt:
        try:
            adapter.reset(seed=seed)
        except (NotImplementedError, TypeError) as exc:
            return self._gap(
                intervention_id, seed, prefix_actions, expected_fork_state_sha256,
                alternative_action, f"RESET_WITH_SEED_UNSUPPORTED:{type(exc).__name__}",
            )
        try:
            for action in prefix_actions:
                admissible = list(adapter.admissible_actions())
                if action not in admissible:
                    return self._gap(
                        intervention_id, seed, prefix_actions, expected_fork_state_sha256,
                        alternative_action, "PREFIX_ACTION_NOT_ADMISSIBLE_ON_REPLAY",
                        status="REPLAY_MISMATCH",
                    )
                adapter.step(action)
            replayed = _hash(adapter.state_receipt())
            if replayed != expected_fork_state_sha256:
                return ForkInterventionReceipt(
                    intervention_id, seed, tuple(prefix_actions),
                    expected_fork_state_sha256, replayed, alternative_action, None, None,
                    "REPLAY_MISMATCH", ("FORK_STATE_HASH_MISMATCH",),
                )
            admissible = list(adapter.admissible_actions())
            admissible_hash = _hash(admissible)
            if alternative_action not in admissible:
                return ForkInterventionReceipt(
                    intervention_id, seed, tuple(prefix_actions), expected_fork_state_sha256,
                    replayed, alternative_action, admissible_hash, None,
                    "INCONCLUSIVE", ("ALTERNATIVE_NOT_NATIVE_ADMISSIBLE",),
                )
            adapter.step(alternative_action)
            return ForkInterventionReceipt(
                intervention_id, seed, tuple(prefix_actions), expected_fork_state_sha256,
                replayed, alternative_action, admissible_hash,
                _hash(adapter.state_receipt()), "INTERVENTION_OBSERVED", (),
            )
        except Exception as exc:  # adapter boundary: preserve a typed gap
            return self._gap(
                intervention_id, seed, prefix_actions, expected_fork_state_sha256,
                alternative_action, f"ADAPTER_ERROR:{type(exc).__name__}",
            )

    @staticmethod
    def _gap(
        intervention_id: str,
        seed: int,
        prefix_actions: Sequence[str],
        expected: str,
        alternative: str,
        code: str,
        *,
        status: str = "UNSUPPORTED",
    ) -> ForkInterventionReceipt:
        return ForkInterventionReceipt(
            intervention_id, seed, tuple(prefix_actions), expected, None,
            alternative, None, None, status, (code,),
        )


__all__ = ["ForkInterventionReceipt", "ReplayForkAdapter", "ReplayForkVerifier"]
