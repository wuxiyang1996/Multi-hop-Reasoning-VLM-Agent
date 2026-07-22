from __future__ import annotations

from typing import Any

from .contracts import Observation


def _native_actions(info: dict[str, Any]) -> tuple[str, ...]:
    raw = info.get("action_names") or info.get("admissible_actions") or ()
    return tuple(str(value) for value in raw)


def _official_success(info: dict[str, Any]) -> bool:
    value = info.get("won")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    return bool(value) if isinstance(value, (bool, int, float)) else False


class GymLikeTextAdapter:
    """Narrow adapter for existing Gym-like game/ALFWorld wrappers."""

    def __init__(self, environment: Any, *, seed: int | None = None) -> None:
        self.environment = environment
        self.seed = seed

    def reset(self) -> Observation:
        observation, info = self.environment.reset(seed=self.seed)
        info = dict(info or {})
        return Observation(
            {"observation": str(observation), "structured_state": info.get("structured_state")},
            _native_actions(info),
            False,
            _official_success(info),
            0.0,
        )

    def step(self, action: str) -> tuple[Observation, float]:
        observation, reward, terminated, truncated, info = self.environment.step(action)
        info = dict(info or {})
        score = float(info.get("score", reward) or 0.0)
        return (
            Observation(
                {"observation": str(observation), "structured_state": info.get("structured_state")},
                _native_actions(info),
                bool(terminated or truncated),
                _official_success(info),
                score,
            ),
            float(reward),
        )

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()
