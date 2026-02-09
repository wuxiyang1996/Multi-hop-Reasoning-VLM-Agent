"""
Test Avalon env wrapper with dummy (LLM) agent controlling all players.

Uses env_wrappers.AvalonNLWrapper and agents.dummy_agent.language_agent_action
to run one full game where every player's action is chosen by the same LLM.

Run from codebase root with AgentEvolver on PYTHONPATH so that
  from games.games.avalon.engine import ...  works.

  cd /path/to/codebase
  set PYTHONPATH=%CD%;%CD%\AgentEvolver
  python evaluation_evolver/test_avalon_dummy.py

Visualizer (watch rollout in browser):
  First start the games web server (see AgentEvolver/games/README.md):
    python AgentEvolver/games/web/server.py
  Open http://localhost:8000/avalon/observe in your browser.
  Then run this script with --visualizer (optionally --visualizer_url http://localhost:8000).
"""

import sys
from pathlib import Path

# Ensure codebase root and AgentEvolver are on path for env_wrappers and games
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
_agentevolver = _root / "AgentEvolver"
if _agentevolver.exists() and str(_agentevolver) not in sys.path:
    sys.path.insert(0, str(_agentevolver))

from typing import Any, Dict, Optional

import urllib.request
import json

from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
from agents.dummy_agent import GAME_AVALON, language_agent_action


def _avalon_visualizer_state(env: AvalonNLWrapper, info: dict, status: str = "running") -> Dict[str, Any]:
    """Build game state dict for the web visualizer (POST /api/push-state)."""
    roles_data = [
        {"role_id": int(rid), "role_name": str(rname), "is_good": bool(ig)}
        for rid, rname, ig in env.roles
    ]
    state = {
        "game": "avalon",
        "status": status,
        "phase": info.get("phase", 0),
        "mission_id": info.get("turn", 0),
        "round_id": info.get("round", 0),
        "leader": info.get("leader", 0),
        "roles": roles_data,
        "logs": getattr(env, "_discussion_log", []) or [],
    }
    if env.done and hasattr(env, "env") and env.env is not None:
        state["good_wins"] = getattr(env.env, "good_victory", None)
    return state


def _push_visualizer_state(visualizer_url: str, state: Dict[str, Any]) -> None:
    """POST state to the games web server push-state API."""
    url = f"{visualizer_url.rstrip('/')}/api/push-state"
    data = json.dumps(state).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"[visualizer] push-state returned {resp.status}")
    except Exception as e:
        print(f"[visualizer] failed to push state: {e}")


def run_avalon_with_dummy_agent(
    num_players: int = 5,
    max_phases: int = 50,
    model: str = "gpt-4o-mini",
    seed: int = 42,
    verbose: bool = True,
    visualizer_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one full Avalon game; all agents are controlled by the dummy language agent.

    If visualizer_url is set (e.g. http://localhost:8000), state is pushed after each step
    so you can watch the rollout in the browser (open /avalon/observe first).

    Returns:
        dict with "obs", "rewards", "info" from the final step, and "history" (list of obs/rewards per step).
    """
    env = AvalonNLWrapper(num_players=num_players, seed=seed)
    obs, info = env.reset()
    history = []
    rewards = {}
    step_count = 0

    if visualizer_url:
        _push_visualizer_state(visualizer_url, _avalon_visualizer_state(env, info, status="running"))
        if verbose:
            print(f"[visualizer] Pushed initial state to {visualizer_url}")

    while not env.done and step_count < max_phases:
        active = info.get("active_players", [])
        expected = info.get("expected_action", "")

        # Dummy agent: for each active player, call LLM with that player's observation
        actions: Dict[int, Any] = {}
        for pid in active:
            state_nl = obs.get(pid, "")
            if not state_nl:
                continue
            action = language_agent_action(
                state_nl,
                game=GAME_AVALON,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            actions[pid] = action
            if verbose:
                print(f"  Player {pid} action: {action!r}")

        obs, rewards, terminated, truncated, info = env.step(actions)
        history.append({"obs": obs, "rewards": rewards, "info": dict(info)})
        step_count += 1

        if visualizer_url:
            status = "finished" if env.done else "running"
            _push_visualizer_state(visualizer_url, _avalon_visualizer_state(env, info, status=status))

        if verbose:
            phase_name = info.get("phase_name", info.get("phase", ""))
            print(f"Step {step_count} | Phase: {phase_name} | Done: {terminated}")

    if visualizer_url and env.done:
        _push_visualizer_state(visualizer_url, _avalon_visualizer_state(env, info, status="finished"))

    if verbose and env.done:
        good_victory = info.get("good_victory")
        print(f"Game over. Good wins: {good_victory}. Rewards: {rewards}")

    return {
        "obs": obs,
        "rewards": rewards if env.done else {},
        "info": info,
        "history": history,
        "steps": step_count,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Avalon with dummy LLM agent for all players")
    parser.add_argument("--num_players", type=int, default=5)
    parser.add_argument("--max_phases", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--visualizer", action="store_true", help="Push state to games web server so you can watch rollout in browser")
    parser.add_argument("--visualizer_url", default="http://localhost:8000", help="Base URL of games web server (default: http://localhost:8000)")
    args = parser.parse_args()

    result = run_avalon_with_dummy_agent(
        num_players=args.num_players,
        max_phases=args.max_phases,
        model=args.model,
        seed=args.seed,
        verbose=not args.quiet,
        visualizer_url=args.visualizer_url if args.visualizer else None,
    )
    print(f"Finished in {result['steps']} steps. Good victory: {result['info'].get('good_victory')}")


if __name__ == "__main__":
    main()
