from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from .contracts import Observation


_TASK_GOAL = re.compile(r"(?:\A|\n)\s*Your task is to:\s*(?P<goal>[^\n]+)", re.IGNORECASE)


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _commands(info: Mapping[str, Any]) -> tuple[str, ...]:
    values = info.get("admissible_commands") or ()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return tuple(str(value) for value in values)


class ALFWorldTextEnvironment:
    """Batch-size-one ALFWorld adapter with official success and native actions."""

    def __init__(
        self,
        *,
        config_path: str,
        data_path: str,
        split: str = "eval_out_of_distribution",
        seed: int = 47,
        game_index: int = 0,
        max_steps: int = 30,
    ) -> None:
        import yaml
        from alfworld.agents.environment import get_environment

        os.environ["ALFWORLD_DATA"] = str(Path(data_path).resolve())
        with Path(config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        config.setdefault("env", {})["type"] = "AlfredTWEnv"
        config.setdefault("general", {})["random_seed"] = int(seed)
        factory = get_environment("AlfredTWEnv")(config, train_eval=split)
        if not getattr(factory, "game_files", None):
            raise RuntimeError(f"ALFWorld resolved zero games for split={split}")
        self.env = factory.init_env(batch_size=1)
        if not callable(getattr(self.env, "seed", None)):
            raise RuntimeError("ALFWorld environment does not expose deterministic seeding")
        self.env.seed(int(seed))
        if game_index:
            self.env.skip(int(game_index))
        self.max_steps = max_steps
        self.step_count = 0
        self.goal = ""

    def _observation(self, raw: Any, info: Mapping[str, Any], *, terminal: bool, score: float) -> Observation:
        text = str(_first(raw) or "").strip()
        match = _TASK_GOAL.search(text)
        if match is not None:
            self.goal = match.group("goal").strip()
        won = bool(_first(info.get("won")))
        return Observation(
            {
                "observation": text,
                "task_goal": self.goal,
                "step": self.step_count,
            },
            _commands(info),
            terminal,
            won,
            score,
        )

    def reset(self) -> Observation:
        self.step_count = 0
        raw, info = self.env.reset()
        return self._observation(raw, info or {}, terminal=False, score=0.0)

    def step(self, action: str) -> tuple[Observation, float]:
        self.step_count += 1
        raw, scores, dones, info = self.env.step([str(action)])
        reward = float(_first(scores) or 0.0)
        terminal = bool(_first(dones)) or self.step_count >= self.max_steps
        return self._observation(raw, info or {}, terminal=terminal, score=reward), reward

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
