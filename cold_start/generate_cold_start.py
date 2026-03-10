#!/usr/bin/env python
"""
Cold-start data generation for Game-AI-Agent.

Generates unlabeled trajectories using the prompt decision agent (VLMDecisionAgent)
and/or the dummy language agent, then labels them with GPT-5-mini to produce
initial seed data for the skill database.

Usage (from Game-AI-Agent root):

    export OPENAI_API_KEY="sk-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # Generate cold-start data for 2048
    python cold_start/generate_cold_start.py \
        --game twenty_forty_eight \
        --episodes 3 --max_steps 50 --model gpt-5-mini

    # Generate for sokoban with VLM decision agent
    python cold_start/generate_cold_start.py \
        --game sokoban \
        --agent_type vlm --episodes 2 --max_steps 100

    # Generate for all supported games
    python cold_start/generate_cold_start.py --all_games --episodes 2 --max_steps 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports from Game-AI-Agent
# ---------------------------------------------------------------------------
from API_func import ask_model
from data_structure.experience import Experience, Episode, Episode_Buffer
from decision_agents.agent import (
    VLMDecisionAgent,
    run_episode_vlm_agent,
    run_tool,
    TOOL_TAKE_ACTION,
    TOOL_REWARD,
)
from decision_agents.dummy_agent import (
    language_agent_action,
    detect_game,
    _default_action,
    GAME_GAMINGAGENT,
)
from decision_agents.reward_func import RewardConfig, RewardResult

# ---------------------------------------------------------------------------
# GamingAgent environment imports
# ---------------------------------------------------------------------------
try:
    from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv
except ImportError:
    TwentyFortyEightEnv = None

try:
    from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv
except ImportError:
    SokobanEnv = None

try:
    from gamingagent.envs.custom_03_candy_crush.candyCrushEnv import CandyCrushEnv
except ImportError:
    CandyCrushEnv = None

try:
    from gamingagent.envs.custom_04_tetris.tetrisEnv import TetrisEnv
except ImportError:
    TetrisEnv = None

try:
    from gamingagent.envs.custom_05_doom.doomEnv import DoomEnvWrapper as DoomEnv
except ImportError:
    DoomEnv = None

try:
    from gamingagent.envs.custom_06_pokemon_red.pokemonRedEnv import PokemonRedEnv
except ImportError:
    PokemonRedEnv = None

try:
    from gamingagent.envs.retro_01_super_mario_bros.superMarioBrosEnv import SuperMarioBrosEnv
except ImportError:
    SuperMarioBrosEnv = None

try:
    from gamingagent.envs.retro_02_ace_attorney.aceAttorneyEnv import AceAttorneyEnv
except ImportError:
    AceAttorneyEnv = None

try:
    from gamingagent.envs.retro_03_1942.NineteenFortyTwo_env import NineteenFortyTwoEnv
except ImportError:
    NineteenFortyTwoEnv = None

try:
    from gamingagent.envs.zoo_01_tictactoe.TicTacToeEnv import SingleTicTacToeEnv
except ImportError:
    SingleTicTacToeEnv = None

try:
    from gamingagent.envs.zoo_02_texasholdem.TexasHoldemEnv import SingleTexasHoldemEnv
except ImportError:
    SingleTexasHoldemEnv = None


# ---------------------------------------------------------------------------
# Game registry: name -> (env_class, config_path, action_names)
# ---------------------------------------------------------------------------

def _config_path(game_dir: str) -> str:
    return str(GAMINGAGENT_ROOT / "gamingagent" / "envs" / game_dir / "game_env_config.json")


def _adapter_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """Standard GymEnvAdapter kwargs (2048, sokoban, tetris)."""
    return dict(
        render_mode=None,
        observation_mode_for_adapter="text",
        agent_cache_dir_for_adapter=cache_dir,
        game_specific_config_path_for_adapter=config_path,
    )


def _candy_crush_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """CandyCrushEnv — adapter pattern but no render_mode parameter."""
    return dict(
        observation_mode_for_adapter="text",
        agent_cache_dir_for_adapter=cache_dir,
        game_specific_config_path_for_adapter=config_path,
    )


def _retro_kwargs(game_name: str):
    """Factory for retro envs (SuperMarioBros, NineteenFortyTwo) that take
    (game_name, config_dir_path, observation_mode, base_log_dir)."""
    def _build(cache_dir: str, config_path: str) -> Dict[str, Any]:
        return dict(
            game_name=game_name,
            config_dir_path=str(Path(config_path).parent),
            observation_mode="text",
            base_log_dir=cache_dir,
        )
    return _build


def _doom_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """DoomEnvWrapper — retro-style args plus render_mode."""
    return dict(
        game_name="doom",
        config_dir_path=str(Path(config_path).parent),
        observation_mode="text",
        base_log_dir=cache_dir,
        render_mode=None,
        headless=True,
    )


def _pokemon_red_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """PokemonRedEnv — adapter pattern but also needs rom_path from config."""
    import json as _json
    rom_path = None
    try:
        with open(config_path) as _f:
            _cfg = _json.load(_f)
            rom_path = _cfg.get("rom_path")
    except Exception:
        pass
    kwargs: Dict[str, Any] = dict(
        render_mode=None,
        observation_mode_for_adapter="text",
        agent_cache_dir_for_adapter=cache_dir,
        game_specific_config_path_for_adapter=config_path,
    )
    if rom_path:
        kwargs["rom_path"] = rom_path
    return kwargs


def _ace_attorney_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """AceAttorneyEnv — uses adapter_ prefixed args."""
    return dict(
        adapter_observation_mode="text",
        adapter_agent_cache_dir=cache_dir,
        adapter_config_path=config_path,
    )


GAME_REGISTRY: Dict[str, Dict[str, Any]] = {
    "twenty_forty_eight": {
        "env_class": TwentyFortyEightEnv,
        "config_path": _config_path("custom_01_2048"),
        "action_names": ["up", "down", "left", "right"],
        "task": "Achieve the highest possible tile in 2048 by merging tiles strategically.",
        "init_kwargs": _adapter_kwargs,
    },
    "sokoban": {
        "env_class": SokobanEnv,
        "config_path": _config_path("custom_02_sokoban"),
        "action_names": ["up", "down", "left", "right", "push up", "push down", "push left", "push right", "no_op"],
        "task": "Push all boxes onto goal positions in the Sokoban puzzle.",
        "init_kwargs": _adapter_kwargs,
    },
    "candy_crush": {
        "env_class": CandyCrushEnv,
        "config_path": _config_path("custom_03_candy_crush"),
        "action_names": [],  # dynamic: swap(row1,col1,row2,col2) on 8x8 board
        "task": "Match three or more candies in a row/column to clear the board and maximize score.",
        "init_kwargs": _candy_crush_kwargs,
    },
    "tetris": {
        "env_class": TetrisEnv,
        "config_path": _config_path("custom_04_tetris"),
        "action_names": ["no_op", "left", "right", "rotate_left", "rotate_right", "soft_drop", "hard_drop"],
        "task": "Clear as many lines as possible in Tetris by placing tetrominoes strategically.",
        "init_kwargs": _adapter_kwargs,
    },
    "doom": {
        "env_class": DoomEnv,
        "config_path": _config_path("custom_05_doom"),
        "action_names": ["move_left", "move_right", "attack"],
        "task": "Survive and defeat enemies in Doom by shooting and dodging.",
        "init_kwargs": _doom_kwargs,
    },
    "pokemon_red": {
        "env_class": PokemonRedEnv,
        "config_path": _config_path("custom_06_pokemon_red"),
        "action_names": ["a", "b", "start", "select", "up", "down", "left", "right"],
        "task": "Progress through Pokemon Red by exploring, battling, and catching Pokemon.",
        "init_kwargs": _pokemon_red_kwargs,
    },
    "super_mario_bros": {
        "env_class": SuperMarioBrosEnv,
        "config_path": _config_path("retro_01_super_mario_bros"),
        "action_names": ["noop", "right", "right_a", "right_b", "right_a_b", "a", "b", "left", "left_a", "left_b", "left_a_b", "down", "up"],
        "task": "Complete levels in Super Mario Bros by running, jumping, and avoiding enemies.",
        "init_kwargs": _retro_kwargs("super_mario_bros"),
    },
    "ace_attorney": {
        "env_class": AceAttorneyEnv,
        "config_path": _config_path("retro_02_ace_attorney"),
        "action_names": ["a", "b", "l", "r", "up", "down", "left", "right", "no_op"],
        "task": "Solve cases in Ace Attorney by investigating evidence and cross-examining witnesses.",
        "init_kwargs": _ace_attorney_kwargs,
    },
    "nineteen_forty_two": {
        "env_class": NineteenFortyTwoEnv,
        "config_path": _config_path("retro_03_1942"),
        "action_names": ["noop", "right", "right_b", "a", "b", "left", "left_b", "down", "up"],
        "task": "Survive waves of enemy aircraft in 1942 by dodging and shooting.",
        "init_kwargs": _retro_kwargs("nineteen_forty_two"),
    },
    "tic_tac_toe": {
        "env_class": SingleTicTacToeEnv,
        "config_path": _config_path("zoo_01_tictactoe"),
        "action_names": ["place 0", "place 1", "place 2", "place 3", "place 4", "place 5", "place 6", "place 7", "place 8"],
        "task": "Win at Tic-Tac-Toe by placing marks to get three in a row.",
        "init_kwargs": _adapter_kwargs,
    },
    "texas_holdem": {
        "env_class": SingleTexasHoldemEnv,
        "config_path": _config_path("zoo_02_texasholdem"),
        "action_names": ["call", "raise", "fold", "check"],
        "task": "Win at Texas Hold'em poker by making optimal betting decisions.",
        "init_kwargs": _adapter_kwargs,
    },
}


# ---------------------------------------------------------------------------
# Lightweight NL wrapper for GamingAgent envs
# ---------------------------------------------------------------------------

class ColdStartEnvWrapper:
    """
    Thin wrapper that adapts GamingAgent envs (which return Observation objects)
    into the standard (obs_str, info) interface used by decision agents.
    """

    def __init__(self, game_name: str, max_steps: int = 100):
        reg = GAME_REGISTRY.get(game_name)
        if reg is None or reg["env_class"] is None:
            raise ValueError(f"Game '{game_name}' not available. Install GamingAgent and check imports.")

        cache_dir = str(SCRIPT_DIR / "cache" / game_name)
        os.makedirs(cache_dir, exist_ok=True)

        kwargs_fn = reg.get("init_kwargs", _adapter_kwargs)
        env_kwargs = kwargs_fn(cache_dir, reg["config_path"])
        self._env = reg["env_class"](**env_kwargs)
        self._action_names = reg["action_names"]
        self._game_name = game_name
        self._max_steps = max_steps
        self._step_count = 0

    def reset(self, seed=None, options=None) -> Tuple[str, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        self._step_count = 0
        text = self._obs_to_text(obs)
        info["action_names"] = self._action_names
        info["game"] = GAME_GAMINGAGENT
        info["env_name"] = "gamingagent"
        info["game_name"] = self._game_name
        return text, info

    def step(self, action) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        action_str = str(action) if action is not None else _default_action(GAME_GAMINGAGENT)
        result = self._env.step(agent_action_str=action_str)
        # GamingAgent envs return 6-tuple: (obs, reward, term, trunc, info, perf_score)
        obs, reward, terminated, truncated, info = result[0], result[1], result[2], result[3], result[4]
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        text = self._obs_to_text(obs)
        info["action_names"] = self._action_names
        info["game"] = GAME_GAMINGAGENT
        info["env_name"] = "gamingagent"
        info["game_name"] = self._game_name
        info["perf_score"] = result[5] if len(result) > 5 else 0.0
        return text, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def _obs_to_text(self, obs) -> str:
        if isinstance(obs, str):
            return obs
        if hasattr(obs, "textual_representation") and obs.textual_representation:
            return str(obs.textual_representation)
        if isinstance(obs, dict):
            return str(obs.get("text", obs.get("textual_representation", str(obs))))
        return str(obs)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_dummy_agent_episode(
    env: ColdStartEnvWrapper,
    game_name: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with the dummy language_agent_action (GPT function calling)."""
    task = GAME_REGISTRY[game_name]["task"]
    action_names = GAME_REGISTRY[game_name]["action_names"]

    obs, info = env.reset()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0

    while step_count < max_steps:
        action = language_agent_action(
            state_nl=obs + f"\n\nValid actions: {', '.join(action_names)}. Choose one.",
            game=GAME_GAMINGAGENT,
            model=model,
            use_function_call=True,
            temperature=0.3,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            tasks=task,
        )
        exp.idx = step_count
        experiences.append(exp)

        if verbose:
            print(f"  step {step_count}: action={action}, reward={reward:.2f}, cum={total_reward:.2f}")

        obs = next_obs
        info = next_info
        step_count += 1

        if done:
            break

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=game_name,
    )
    episode.set_outcome()
    stats = {
        "game": game_name,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "dummy",
    }
    return episode, stats


def run_vlm_agent_episode(
    env: ColdStartEnvWrapper,
    game_name: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with VLMDecisionAgent (prompt decision agent).

    Delegates to run_episode_vlm_agent which returns a fully-populated
    Episode (Experience objects with summary_state, intentions, sub_tasks,
    reward_details) ready for direct ingestion by the skill agents pipeline.
    """
    task = GAME_REGISTRY[game_name]["task"]

    agent = VLMDecisionAgent(model=model, game=GAME_GAMINGAGENT)

    episode = run_episode_vlm_agent(
        env,
        agent=agent,
        task=task,
        max_steps=max_steps,
        verbose=verbose,
    )

    meta = episode.metadata or {}
    stats = {
        "game": game_name,
        "steps": meta.get("steps", len(episode.experiences)),
        "total_reward": episode.get_reward(),
        "total_shaped_reward": episode.get_total_reward(),
        "terminated": meta.get("done", False),
        "truncated": False,
        "model": model,
        "agent_type": "vlm",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Trajectory labeling with GPT-5-mini
# ---------------------------------------------------------------------------

def label_trajectory(episode: Episode, model: str) -> Episode:
    """
    Use GPT-5-mini to generate summaries and intention labels for each experience
    in the episode, and segment into sub-tasks for initial skill seeds.
    """
    print(f"  Labeling trajectory ({len(episode.experiences)} steps) with {model}...")

    for i, exp in enumerate(episode.experiences):
        history = episode.experiences[max(0, i - 3):i]
        try:
            exp.generate_summary()
        except Exception as e:
            exp.summary = f"Step {i}: action={exp.action}"
        try:
            exp.generate_intentions(history)
        except Exception as e:
            exp.intentions = "unknown"

    try:
        episode.generate_summary()
    except Exception:
        episode.summary = f"Episode with {len(episode.experiences)} steps, reward={episode.get_reward():.2f}"

    sub_episodes = episode.separate_into_sub_episodes(outcome_length=3)
    for sub_ep in sub_episodes:
        try:
            sub_ep.generate_summary()
        except Exception:
            pass
        try:
            sub_ep.sub_task_labeling()
        except Exception:
            pass

    return episode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate cold-start data for Game-AI-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", type=str, default="twenty_forty_eight",
                        choices=list(GAME_REGISTRY.keys()),
                        help="Game to generate data for")
    parser.add_argument("--all_games", action="store_true",
                        help="Generate data for all available games")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per game")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="LLM model to use for agent and labeling")
    parser.add_argument("--agent_type", type=str, default="dummy",
                        choices=["dummy", "vlm"],
                        help="Agent type: 'dummy' (language_agent_action) or 'vlm' (VLMDecisionAgent)")
    parser.add_argument("--label", action="store_true", default=True,
                        help="Label trajectories with LLM after generation")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/data/)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY not set. LLM calls will fail.")
        print("  Set it with: export OPENAI_API_KEY='sk-...'")

    games = list(GAME_REGISTRY.keys()) if args.all_games else [args.game]
    available_games = [g for g in games if GAME_REGISTRY[g]["env_class"] is not None]

    if not available_games:
        print("[ERROR] No games available. Ensure GamingAgent is installed.")
        sys.exit(1)

    print("=" * 78)
    print("  Cold-Start Data Generation")
    print("=" * 78)
    print(f"  Games:      {', '.join(available_games)}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Model:      {args.model}")
    print(f"  Agent:      {args.agent_type}")
    print(f"  Labeling:   {not args.no_label}")
    print(f"  Output:     {output_dir}")
    print("=" * 78)

    run_fn = run_vlm_agent_episode if args.agent_type == "vlm" else run_dummy_agent_episode
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for game_name in available_games:
        print(f"\n{'─' * 78}")
        print(f"  Game: {game_name}")
        print(f"{'─' * 78}")

        game_output_dir = output_dir / game_name
        game_output_dir.mkdir(parents=True, exist_ok=True)

        episode_buffer = Episode_Buffer(buffer_size=1000)

        for ep_idx in range(args.episodes):
            print(f"\n  Episode {ep_idx + 1}/{args.episodes}")

            try:
                env = ColdStartEnvWrapper(game_name, max_steps=args.max_steps)
                episode, stats = run_fn(
                    env=env,
                    game_name=game_name,
                    model=args.model,
                    max_steps=args.max_steps,
                    verbose=args.verbose,
                )
                env.close()

                print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

                if not args.no_label:
                    episode = label_trajectory(episode, args.model)

                episode_buffer.add_episode(episode)
                all_stats.append(stats)

                # Save individual episode
                ep_path = game_output_dir / f"episode_{ep_idx:03d}.json"
                ep_data = episode.to_dict()
                ep_data["metadata"] = stats
                with open(ep_path, "w", encoding="utf-8") as f:
                    json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            except Exception as e:
                print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save episode buffer for this game
        buffer_path = game_output_dir / "episode_buffer.json"
        episode_buffer.save_to_json(str(buffer_path))
        print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    elapsed = time.time() - t0

    # Save overall run summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": args.agent_type,
        "games": available_games,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "labeled": not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    summary_path = output_dir / "cold_start_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  COLD-START GENERATION COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Total episodes: {len(all_stats)}")
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
        print(f"  Mean reward:  {sum(rewards) / len(rewards):.2f}")
        print(f"  Mean steps:   {sum(steps) / len(steps):.1f}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Output:       {output_dir}")
    print(f"  Summary:      {summary_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
