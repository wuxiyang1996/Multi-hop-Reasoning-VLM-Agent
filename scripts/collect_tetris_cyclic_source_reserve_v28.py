#!/usr/bin/env python3
"""Collect fresh official-Tetris cyclic intervention forks without an LLM."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.cyclic_identity_induction import (  # noqa: E402
    DATASET_VERSION,
)


DEFAULT_CONFIG = REPO / "configs/tetris_cyclic_source_induction_v28.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


class _Observation:
    def __init__(
        self, img_path: str | None = None,
        textual_representation: str | None = None, **_: Any,
    ) -> None:
        self.img_path = img_path
        self.textual_representation = textual_representation


class _SourceOnlyAdapter:
    """Small adapter shim; official Tetris dynamics remain unchanged."""

    ACTIONS = {
        "no_op": 0,
        "noop": 0,
        "left": 1,
        "right": 2,
        "rotate_left": 3,
        "rotate_counterclockwise": 3,
        "rotate_right": 4,
        "rotate_clockwise": 4,
        "soft_drop": 5,
        "hard_drop": 6,
    }

    def __init__(
        self, game_name: str, observation_mode: str, agent_cache_dir: str,
        game_specific_config_path: str, max_steps_for_stuck: int | None,
    ) -> None:
        del game_name, agent_cache_dir, game_specific_config_path
        self.observation_mode = observation_mode
        self.current_episode_id = 0
        self.current_step_num = 0
        self.maximum_steps = int(max_steps_for_stuck or 1000)

    def reset_episode(self, episode_id: int) -> None:
        self.current_episode_id = int(episode_id)
        self.current_step_num = 0

    def calculate_perf_score(self, *_: Any) -> float:
        return 0.0

    def create_agent_observation(
        self, img_path: str | None, text: str | None, max_memory: int = 10,
    ) -> _Observation:
        del max_memory
        return _Observation(img_path, text)

    def map_agent_action_to_env_action(self, action: str) -> int | None:
        return self.ACTIONS.get(str(action))

    def increment_step(self) -> None:
        self.current_step_num += 1

    def verify_termination(
        self, observation: _Observation, terminated: bool, truncated: bool,
    ) -> tuple[bool, bool]:
        del observation
        return bool(terminated), bool(
            truncated or self.current_step_num >= self.maximum_steps
        )

    def log_step_data(self, **_: Any) -> None:
        return None

    def close_log_file(self) -> None:
        return None


def _load_official_tetris(path: Path):
    """Load the official environment while bypassing unrelated API clients."""

    for name in (
        "gamingagent", "gamingagent.envs", "gamingagent.modules",
        "gamingagent.envs.custom_04_tetris",
    ):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    core = types.ModuleType("gamingagent.modules.core_module")
    core.Observation = _Observation
    sys.modules[core.__name__] = core
    adapter = types.ModuleType("gamingagent.envs.gym_env_adapter")
    adapter.GymEnvAdapter = _SourceOnlyAdapter
    sys.modules[adapter.__name__] = adapter
    utilities = types.ModuleType("gamingagent.envs.env_utils")
    utilities.create_board_image_tetris = lambda **_: None
    sys.modules[utilities.__name__] = utilities
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")

    name = "gamingagent.envs.custom_04_tetris.tetrisEnv"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load official Tetris environment")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.TetrisEnv


def _matrix_key(matrix: np.ndarray) -> str:
    return stable_hash({
        "shape": list(matrix.shape),
        "values": matrix.astype(int).tolist(),
    })


def _new_env(environment_class: type[Any]):
    with redirect_stdout(StringIO()):
        return environment_class(
            render_mode=None,
            gravity=False,
            observation_mode_for_adapter="text",
            max_stuck_steps_for_adapter=100,
        )


def _close_env(env: Any) -> None:
    with redirect_stdout(StringIO()):
        env.close()


def _step(env: Any, action: str) -> None:
    _, _, terminated, truncated, _, _ = env.step(action)
    if terminated or truncated:
        raise RuntimeError("official Tetris fork terminated during rotation")


def _group_contract(environment_class: type[Any], seed: int) -> tuple[int, list[str]]:
    env = _new_env(environment_class)
    try:
        env.reset(seed=seed, episode_id=seed)
        keys = [_matrix_key(env.active_tetromino.matrix)]
        for _ in range(1, 9):
            _step(env, "rotate_right")
            key = _matrix_key(env.active_tetromino.matrix)
            if key == keys[0]:
                return len(keys), keys
            if key in keys:
                raise RuntimeError("official Tetris rotation did not form a cycle")
            keys.append(key)
    finally:
        _close_env(env)
    raise RuntimeError("official Tetris rotation order exceeded bound")


def _transition(
    *, phase: str, before: int, after: int, order: int,
) -> dict[str, Any]:
    body = {
        "state_element": int(before),
        "anonymous_intervention_phase": str(phase),
        "effect_element": int((after - before) % order),
        "next_state_element": int(after),
        "raw_action_exported": False,
    }
    return body | {"transition_sha256": stable_hash(body)}


def _execute_actions(
    env: Any, actions: Sequence[str], *, phase: str,
    key_to_element: Mapping[str, int], order: int,
) -> list[dict[str, Any]]:
    rows = []
    for action in actions:
        before_key = _matrix_key(env.active_tetromino.matrix)
        if before_key not in key_to_element:
            raise RuntimeError("source state left the induced cyclic orbit")
        before = int(key_to_element[before_key])
        _step(env, action)
        after_key = _matrix_key(env.active_tetromino.matrix)
        if after_key not in key_to_element:
            raise RuntimeError("source state left the induced cyclic orbit")
        rows.append(_transition(
            phase=phase, before=before,
            after=int(key_to_element[after_key]), order=order,
        ))
    return rows


def _candidate_actions(order: int, probe_steps: int) -> list[tuple[str, list[str]]]:
    candidates = [
        ("left_probe_count", ["rotate_left"] * probe_steps),
        ("right_probe_count", ["rotate_right"] * probe_steps),
        ("identity", ["no_op"]),
        ("right_half_order", ["rotate_right"] * (order // 2)),
        ("right_generator", ["rotate_right"]),
        ("left_generator", ["rotate_left"]),
    ]
    # The learner observes aggregate effects, not generator names.  Retain one
    # fork for every distinct recovery effect in the official environment.
    return candidates


def _collect_episode(
    environment_class: type[Any], *, seed: int, namespace: str,
) -> dict[str, Any] | None:
    order, orbit = _group_contract(environment_class, seed)
    if order != 4:
        return None
    key_to_element = {key: index for index, key in enumerate(orbit)}
    probe_steps = 1 if int(stable_hash({
        "namespace": namespace, "seed": seed,
    })[:8], 16) % 2 == 0 else 3
    observed: dict[int, dict[str, Any]] = {}
    for strategy, recovery_actions in _candidate_actions(order, probe_steps):
        env = _new_env(environment_class)
        try:
            env.reset(seed=seed, episode_id=seed)
            start_key = _matrix_key(env.active_tetromino.matrix)
            probe_rows = _execute_actions(
                env, ["rotate_right"] * probe_steps,
                phase="PROBE", key_to_element=key_to_element, order=order,
            )
            probe_effect = sum(
                int(row["effect_element"]) for row in probe_rows
            ) % order
            recovery_rows = _execute_actions(
                env, recovery_actions,
                phase="RECOVERY", key_to_element=key_to_element, order=order,
            )
            recovery_effect = sum(
                int(row["effect_element"]) for row in recovery_rows
            ) % order
            body = {
                "candidate_id": "CAND_" + stable_hash({
                    "namespace": namespace,
                    "recovery_effect_element": recovery_effect,
                })[:12],
                "probe_effect_element": probe_effect,
                "recovery_effect_element": recovery_effect,
                "primitive_transitions": [*probe_rows, *recovery_rows],
                "returned_to_identity": (
                    _matrix_key(env.active_tetromino.matrix) == start_key
                ),
                "raw_strategy_exported": False,
            }
            # De-duplicate extensionally identical native strategies.
            observed.setdefault(
                recovery_effect,
                body | {"candidate_sha256": stable_hash(body)},
            )
        finally:
            _close_env(env)
    if len(observed) != order:
        raise RuntimeError("fresh source forks did not cover the cyclic group")
    candidates = sorted(
        observed.values(), key=lambda row: str(row["candidate_id"]),
    )
    body = {
        "episode_id": "tetris-source-" + stable_hash({
            "namespace": namespace, "seed": seed,
        })[:16],
        "seed_commitment": stable_hash({"seed": seed, "namespace": namespace}),
        "group_order": order,
        "probe_primitive_steps": probe_steps,
        "candidates": candidates,
    }
    return body | {"episode_sha256": stable_hash(body)}


def collect(config: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_BEFORE_FRESH_SOURCE_COLLECTION":
        raise ValueError("Tetris source reserve is not frozen")
    if role not in config["source_splits"]:
        raise ValueError(f"unknown source role: {role}")
    environment_path = Path(str(config["official_tetris_environment"]))
    if _sha(environment_path) != config["official_tetris_environment_file_sha256"]:
        raise ValueError("official Tetris environment changed")
    environment_class = _load_official_tetris(environment_path)
    episodes = []
    for seed in map(int, config["source_splits"][role]):
        episode = _collect_episode(
            environment_class, seed=seed,
            namespace=str(config["source_namespace"]),
        )
        if episode is not None:
            episodes.append(episode)
    body = {
        "schema_version": DATASET_VERSION,
        "role": role,
        "config_sha256": str(config["config_sha256"]),
        "official_tetris_environment_file_sha256": str(
            config["official_tetris_environment_file_sha256"]
        ),
        "episodes": episodes,
        "attempted_seeds": len(config["source_splits"][role]),
        "retained_order_four_episodes": len(episodes),
        "raw_source_action_tokens_exported": False,
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--role", choices=("development", "qualification", "reserve"),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = _read(config_path)
    output = REPO / str(config["outputs"][args.role])
    if output.exists():
        raise SystemExit(f"refusing to overwrite source reserve: {output}")
    report = collect(config, role=args.role)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "role": args.role,
        "attempted_seeds": report["attempted_seeds"],
        "retained_order_four_episodes": report[
            "retained_order_four_episodes"
        ],
        "candidate_forks": sum(
            len(row["candidates"]) for row in report["episodes"]
        ),
        "primitive_transitions": sum(
            len(candidate["primitive_transitions"])
            for row in report["episodes"]
            for candidate in row["candidates"]
        ),
        "dataset_sha256": report["dataset_sha256"],
        "output": str(output),
    }, indent=2))
    return 0 if report["episodes"] else 2


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    raise SystemExit(main())
