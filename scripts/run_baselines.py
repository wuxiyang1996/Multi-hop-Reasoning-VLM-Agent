#!/usr/bin/env python3
"""Run Tetris baselines using the co-evolution pipeline with external LLMs.

Uses the same env_wrapper (macro actions), XML state markup, and prompt
format as the co-evolution training — but replaces the local vLLM
action-taking call with GPT / Claude / Gemini / Qwen3 API calls.

Usage:
  python scripts/run_baselines.py \
      --model gpt-4o \
      --episodes 16 --max-steps 200 --seed 42
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_wrappers.game_configs import GAME_CONFIGS
from env_wrappers.gym_like import make_gaming_env
from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
from trainer.coevolution._game_schema import state_to_markup
from trainer.coevolution.episode_runner import (
    _parse_action_response,
    _format_numbered_actions,
    _generate_summary_state,
    SYSTEM_PROMPT,
)
from decision_agents.agent_helper import compact_text_observation

# ── API helpers ──────────────────────────────────────────────

def _call_openai(prompt: str, model: str, max_tokens: int = 512,
                 temperature: float = 0.3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt: str, model: str, max_tokens: int = 512,
                    temperature: float = 0.3) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _call_openrouter(prompt: str, model: str, max_tokens: int = 512,
                     temperature: float = 0.3) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_model(prompt: str, model: str, **kw) -> str:
    """Route to the right provider based on model name."""
    ml = model.lower()
    if "gpt" in ml or ml.startswith("o"):
        return _call_openai(prompt, model, **kw)
    elif "claude" in ml:
        return _call_anthropic(prompt, model, **kw)
    else:
        return _call_openrouter(prompt, model, **kw)


# ── Episode runner ───────────────────────────────────────────

def run_episode(
    model: str,
    episode_id: int,
    max_steps: int = 200,
    seed: int = 42,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Run a single Tetris episode with the given model."""
    game = "tetris"

    base_env = make_gaming_env(game=game, max_steps=max_steps)
    env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])

    try:
        info["state_markup"] = state_to_markup(
            obs_nl=obs_nl, info=info, game=game, step=0,
        )
    except Exception:
        info["state_markup"] = ""

    total_reward = 0.0
    step_count = 0
    recent_actions: List[str] = []
    recent_rewards: List[float] = []
    terminated = False
    truncated = False

    while step_count < max_steps and not terminated and not truncated:
        step_actions = action_names if action_names else ["stay"]

        summary_state = _generate_summary_state(
            obs_nl, game_name=game,
            step_idx=step_count, total_steps=max_steps,
            reward=total_reward,
        )

        _rich_markup = info.get("state_markup", "")
        _is_macro = getattr(env, "_is_macro_action", False)

        if _rich_markup and "<state>" in _rich_markup:
            summary_for_action = _rich_markup
        else:
            compact = compact_text_observation(obs_nl, max_chars=200)
            summary_for_action = compact if compact else obs_nl[:4000]

        if _is_macro and "<actions>" in summary_for_action:
            summary_for_action = re.sub(
                r"\n?<actions>\n.*?(?=\n<|\Z)", "",
                summary_for_action, flags=re.DOTALL,
            )

        _quality_sort_hint = ""
        if _is_macro:
            _quality_sort_hint = (
                "Actions are sorted best-first (fewest holes, most line clears). "
                "Prefer ACTION 1 unless you have a strong reason to pick another.\n"
            )

        recent_ctx = ""
        if recent_actions:
            last_n = list(zip(recent_actions[-5:], recent_rewards[-5:]))
            lines = [f"  {a} → reward {r:.1f}" for a, r in last_n]
            recent_ctx = "Recent actions:\n" + "\n".join(lines) + "\n"

        action_user = (
            f"Game state:\n\n{summary_for_action}\n\n"
            f"Subgoal: Play Tetris optimally\n"
            f"{recent_ctx}"
            f"{_quality_sort_hint}"
            f"Available actions (pick ONE by number):\n"
            f"{_format_numbered_actions(step_actions)}\n\n"
            f"Choose the best action. Output REASONING then ACTION number."
        )
        action_prompt = SYSTEM_PROMPT + "\n" + action_user

        t0 = time.time()
        try:
            reply = call_model(
                action_prompt, model,
                max_tokens=256, temperature=temperature,
            )
        except Exception as exc:
            print(f"  [step {step_count}] API error: {exc}")
            reply = ""
        api_ms = int((time.time() - t0) * 1000)

        action, reasoning, _ = _parse_action_response(reply, step_actions)
        action_str = str(action)

        obs_nl, reward, terminated, truncated, info = env.step(action_str)
        try:
            info["state_markup"] = state_to_markup(
                obs_nl=obs_nl, info=info, game=game, step=step_count + 1,
            )
        except Exception:
            info.setdefault("state_markup", "")

        total_reward += reward
        step_count += 1
        recent_actions.append(action_str)
        recent_rewards.append(float(reward))

        if step_count <= 5 or step_count % 20 == 0:
            print(
                f"  [ep {episode_id} step {step_count:>3d}] "
                f"action={action_str:<30s} rwd={reward:.1f} "
                f"cumul={total_reward:.1f}  ({api_ms}ms)"
            )

    env.close()
    print(
        f"  Episode {episode_id} done: {step_count} steps, "
        f"total_reward={total_reward:.1f}"
    )
    return {
        "episode_id": episode_id,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
    }


# ── Main ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    safe_name = args.model.replace("/", "_").replace(".", "_")
    out_dir = Path(f"runs/baselines/{safe_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Tetris Baseline: {args.model}")
    print(f"  episodes={args.episodes}, max_steps={args.max_steps}")
    print(f"  output → {out_dir}")
    print(f"{'='*60}")

    results = []
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep+1}/{args.episodes} ---")
        r = run_episode(
            model=args.model,
            episode_id=ep,
            max_steps=args.max_steps,
            seed=args.seed + ep,
            temperature=args.temperature,
        )
        results.append(r)

        with open(out_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")

    rewards = [r["total_reward"] for r in results]
    import numpy as np
    mean = np.mean(rewards)
    std = np.std(rewards, ddof=1) if len(rewards) > 1 else 0
    ci95 = 1.96 * std / np.sqrt(len(rewards)) if len(rewards) > 1 else 0

    summary = {
        "model": args.model,
        "episodes": len(results),
        "mean_reward": float(mean),
        "std_reward": float(std),
        "ci95": float(ci95),
        "min_reward": float(min(rewards)),
        "max_reward": float(max(rewards)),
        "all_rewards": rewards,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"  Mean reward: {mean:.1f} ± {ci95:.1f} (95% CI)")
    print(f"  Std:  {std:.1f}")
    print(f"  Range: [{min(rewards):.1f}, {max(rewards):.1f}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
