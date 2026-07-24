from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from .contracts import Observation


_TASK_GOAL = re.compile(r"(?:\A|\n)\s*Your task is to:\s*(?P<goal>[^\n]+)", re.IGNORECASE)


def resolve_game_index(game_files: list[str] | tuple[str, ...], game_id: str) -> int:
    """Resolve a frozen split-relative game ID without semantic matching."""
    normalized = str(game_id).replace("\\", "/").lstrip("/")
    matches = [
        index for index, path in enumerate(game_files)
        if str(path).replace("\\", "/").endswith("/" + normalized)
        or str(path).replace("\\", "/") == normalized
    ]
    if len(matches) != 1:
        raise ValueError(f"game_id must resolve exactly once; id={game_id!r} matches={len(matches)}")
    return matches[0]


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
        game_id: str | None = None,
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
        if game_id is not None:
            if game_index:
                raise ValueError("pass game_index or game_id, not both")
            game_index = resolve_game_index(tuple(factory.game_files), game_id)
        self.resolved_game_index = int(game_index)
        self.resolved_game_file = str(factory.game_files[game_index])
        self.env = factory.init_env(batch_size=1)
        if not callable(getattr(self.env, "seed", None)):
            raise RuntimeError("ALFWorld environment does not expose deterministic seeding")
        self.env.seed(int(seed))
        if game_index:
            self.env.skip(int(game_index))
        self.max_steps = max_steps
        self.step_count = 0
        self.goal = ""
        self.last_info: dict[str, Any] = {}

    def _observation(self, raw: Any, info: Mapping[str, Any], *, terminal: bool, score: float) -> Observation:
        self.last_info = dict(info)
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

    def expert_action(self) -> str:
        """Return ALFWorld's official hand-coded expert action, if exposed."""
        plan: Any = self.last_info.get("extra.expert_plan")
        while isinstance(plan, (list, tuple)) and len(plan) == 1:
            plan = plan[0]
        if isinstance(plan, (list, tuple)) and plan:
            plan = plan[0]
        action = str(plan or "").strip()
        admissible = _commands(self.last_info)
        if not action:
            raise RuntimeError("ALFWorld did not expose an expert action")
        if action not in admissible:
            raise RuntimeError(f"expert action is not currently admissible: {action!r}")
        return action

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
