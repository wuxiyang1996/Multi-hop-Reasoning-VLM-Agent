"""Gym-like text wrapper for ALFWorld household tasks.

The wrapper exposes the same ``reset``/``step`` contract consumed by the
unified episode runner while keeping ALFWorld in its own conda environment.
Only text-mode ``AlfredTWEnv`` is required for the active transfer study.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


ALFWORLD_SPLITS = (
    "train",
    "eval_in_distribution",
    "eval_out_of_distribution",
)


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _commands_from_info(info: Dict[str, Any]) -> List[str]:
    raw_commands = info.get("admissible_commands")
    if not isinstance(raw_commands, (list, tuple)):
        return []
    commands = raw_commands
    if commands and isinstance(commands[0], (list, tuple)):
        commands = commands[0]
    return [str(command) for command in commands]


def alfworld_obs_to_natural_language(
    obs: Any,
    info: Optional[Dict[str, Any]] = None,
    *,
    include_admissible: bool = True,
    max_actions: int = 40,
) -> str:
    """Convert a batch-size-one ALFWorld observation to compact text."""
    parts = [str(_first(obs) or "").strip()]
    if include_admissible and info:
        commands = _commands_from_info(info)[:max_actions]
        if commands:
            parts.append("Admissible actions: " + "; ".join(commands))
    return "\n\n".join(part for part in parts if part)


@dataclass
class ALFWorldNLWrapper:
    """Wrap ALFWorld's batch API as a single-environment Gym-like API."""

    env: Any
    include_admissible: bool = True
    max_actions: int = 40
    max_steps: int = 50
    _step_count: int = 0
    _last_info: Dict[str, Any] = field(default_factory=dict)
    _last_observation: str = ""

    @property
    def action_names(self) -> List[str]:
        return _commands_from_info(self._last_info)

    @property
    def last_info(self) -> Dict[str, Any]:
        return self._build_info(self._last_info)

    @property
    def last_observation(self) -> str:
        return str(getattr(self, "_last_observation", ""))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        # ALFWorld's text environment does not expose Gymnasium seed/options
        # arguments.  Keep them in the wrapper contract for runner symmetry.
        del seed, options
        self._step_count = 0
        obs, info = self.env.reset()
        self._last_info = dict(info or {})
        text = alfworld_obs_to_natural_language(
            obs,
            self._last_info,
            include_admissible=self.include_admissible,
            max_actions=self.max_actions,
        )
        self._last_observation = text
        return text, self._build_info(self._last_info)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        command = str(action).strip()
        self._step_count += 1
        obs, scores, dones, info = self.env.step([command])
        self._last_info = dict(info or {})
        self._last_info["last_action"] = command
        reward = float(_first(scores) or 0.0)
        terminated = bool(_first(dones))
        truncated = self._step_count >= self.max_steps and not terminated
        text = alfworld_obs_to_natural_language(
            obs,
            self._last_info,
            include_admissible=self.include_admissible,
            max_actions=self.max_actions,
        )
        self._last_observation = text
        out_info = self._build_info(self._last_info)
        out_info["raw_env_reward"] = reward
        return text, reward, terminated, truncated, out_info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def skip_games(self, count: int) -> None:
        """Advance the deterministic gamefile iterator without opening games."""
        if count < 0:
            raise ValueError("count must be non-negative")
        skip = getattr(self.env, "skip", None)
        if not callable(skip):
            raise RuntimeError("underlying ALFWorld environment cannot skip games")
        if count:
            skip(int(count))

    def _build_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(info or {})
        commands = _commands_from_info(result)
        won = _first(result.get("won"))
        observation = self.last_observation.split(
            "\n\nAdmissible actions:", 1
        )[0]
        result.update(
            env="alfworld",
            env_name="alfworld",
            domain="alfworld",
            step=self._step_count,
            admissible_actions=commands,
            action_names=commands,
            structured_state={
                "observation": observation,
                "admissible_commands": commands,
                "won": bool(won),
                "last_action": result.get("last_action"),
            },
        )
        return result


def make_alfworld_env(
    *,
    split: str = "eval_out_of_distribution",
    env_type: str = "AlfredTWEnv",
    batch_size: int = 1,
    max_steps: int = 50,
    include_admissible: bool = True,
    config_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> ALFWorldNLWrapper:
    """Create a text-mode ALFWorld environment.

    ALFWorld is imported lazily so the main project environment does not need
    its TextWorld dependency stack.
    """
    if split not in ALFWORLD_SPLITS:
        raise ValueError(f"split must be one of {ALFWORLD_SPLITS}, got {split!r}")
    if batch_size != 1:
        raise ValueError("ALFWorldNLWrapper currently requires batch_size=1")

    # The repository installer uses ``.cache/alfworld_data``. Prefer that
    # populated location; the historical home-cache default is retained only
    # when no repository-local installation exists.
    repo_data = Path(__file__).resolve().parents[1] / ".cache" / "alfworld_data"
    default_data = repo_data if repo_data.is_dir() else Path.home() / ".cache" / "alfworld"
    os.environ.setdefault("ALFWORLD_DATA", str(default_data))

    try:
        from alfworld.agents.environment import get_environment  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise ImportError(
            "ALFWorld is not installed. Run install/install_alfworld.sh and "
            "activate the 'alfworld' conda environment."
        ) from exc

    default_config = Path(__file__).resolve().parents[1] / "configs" / "alfworld_base_config.yaml"
    resolved_config = Path(config_path).expanduser() if config_path else default_config
    if not resolved_config.is_file():
        raise FileNotFoundError(
            f"ALFWorld config not found: {resolved_config}. Pass config_path=... "
            "or restore configs/alfworld_base_config.yaml."
        )
    with resolved_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("env", {})["type"] = env_type
    if random_seed is not None:
        config.setdefault("general", {})["random_seed"] = int(random_seed)
    factory_env = get_environment(env_type)(config, train_eval=split)
    game_files = getattr(factory_env, "game_files", None)
    if game_files is not None and not game_files:
        raise RuntimeError(
            "ALFWorld resolved zero games for "
            f"split={split!r} with ALFWORLD_DATA={os.environ.get('ALFWORLD_DATA')!r}"
        )
    env = factory_env.init_env(batch_size=batch_size)
    # ALFWorld 0.4.2 does not forward ``general.random_seed`` into the
    # registered TextWorld Gym environment. Seed the actual gamefile iterator
    # explicitly; otherwise every run silently uses TextWorld's default 1234.
    if random_seed is not None:
        seed_fn = getattr(env, "seed", None)
        if not callable(seed_fn):
            raise RuntimeError("underlying ALFWorld environment cannot be seeded")
        seed_fn(int(random_seed))
    return ALFWorldNLWrapper(
        env=env,
        include_admissible=include_admissible,
        max_steps=max_steps,
    )


__all__ = [
    "ALFWORLD_SPLITS",
    "ALFWorldNLWrapper",
    "alfworld_obs_to_natural_language",
    "make_alfworld_env",
]
