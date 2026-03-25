#!/usr/bin/env python3
"""Test: GPT-5.4 reads raw visual observations from real envs → structured schema.

Captures real screenshots from Gym-V games and BrowserGym web pages,
sends them to GPT-5.4, and prints the structured <state>…</state> schema.

Usage:
    # Run all tests (2 Gym-V games + 1 BrowserGym page)
    python test_schema_gen.py

    # Only Gym-V
    python test_schema_gen.py --gymv-only

    # Only BrowserGym
    python test_schema_gen.py --browser-only

    # Custom BrowserGym URL
    python test_schema_gen.py --browser-only --url https://en.wikipedia.org

    # Save captured screenshots to disk
    python test_schema_gen.py --save-images

Set OPENAI_API_KEY before running.
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_wrapper.schema import parse_schema_output, validate_schema
from vlm_wrapper.gymv_adapter import generate_label as gymv_generate_label
from vlm_wrapper.browser_adapter import generate_label as browser_generate_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL = "gpt-5.4"


# =========================================================================
# Gym-V: capture real game observations
# =========================================================================

def capture_gymv_obs(env_id: str, n_steps: int = 3, seed: int = 42) -> list[dict]:
    """Reset a Gym-V env, take n_steps, return list of observation dicts.

    Each dict has: image, text, description, task_id, step.
    """
    import gym_v

    log.info("Creating Gym-V env: %s", env_id)
    env = gym_v.make(env_id)
    obs_dict, info_dict = env.reset(seed=seed)
    obs = obs_dict["agent_0"]

    description = getattr(env, "description", "") or ""

    results = []
    results.append({
        "image": obs.image,
        "text": obs.text,
        "description": description,
        "task_id": env_id,
        "step": 0,
    })

    action_map = {
        "Games/Game2048-v0": ["[Up]", "[Left]", "[Down]", "[Right]", "[Up]"],
        "Games/Minesweeper-v0": ["reveal 0 0", "reveal 1 1", "reveal 2 2"],
        "Games/Sokoban-v0": ["[Up]", "[Right]", "[Down]", "[Left]", "[Up]"],
        "Games/Wordle-v0": ["CRANE", "SLOTH", "BUMPY"],
    }
    actions = action_map.get(env_id, ["[Up]"] * n_steps)

    for i, action in enumerate(actions[:n_steps]):
        try:
            obs_dict, rew, term, trunc, info = env.step({"agent_0": action})
            obs = obs_dict["agent_0"]
            results.append({
                "image": obs.image,
                "text": obs.text,
                "description": description,
                "task_id": env_id,
                "step": i + 1,
            })
            all_done = term.get("__all__", False) or trunc.get("__all__", False)
            if all_done:
                log.info("  %s terminated at step %d", env_id, i + 1)
                break
        except Exception as e:
            log.warning("  Step %d failed: %s", i + 1, e)
            break

    env.close()
    log.info("  Captured %d observations from %s", len(results), env_id)
    return results


# =========================================================================
# BrowserGym: capture real web page observations
# =========================================================================

def capture_browser_obs(
    url: str = "https://www.google.com",
    goal: str = "",
    n_steps: int = 1,
) -> list[dict]:
    """Reset a BrowserGym openended env, return observation dicts with real screenshots."""
    import gymnasium as gym
    import browsergym.core  # noqa: F401 — registers envs

    log.info("Creating BrowserGym env for: %s", url)
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": url},
        headless=True,
    )
    obs, info = env.reset()

    results = []

    screenshot = obs["screenshot"]
    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    else:
        image = screenshot

    results.append({
        "image": image,
        "goal": goal or f"Explore {url}",
        "url": obs.get("url", url),
        "task_id": "browsergym/openended",
        "step": 0,
        "last_action": "",
        "last_action_error": "",
    })

    for i in range(n_steps):
        obs, rew, term, trunc, info = env.step("noop()")
        screenshot = obs["screenshot"]
        image = Image.fromarray(screenshot) if isinstance(screenshot, np.ndarray) else screenshot
        results.append({
            "image": image,
            "goal": goal or f"Explore {url}",
            "url": obs.get("url", ""),
            "task_id": "browsergym/openended",
            "step": i + 1,
            "last_action": obs.get("last_action", ""),
            "last_action_error": obs.get("last_action_error", ""),
        })
        if term or trunc:
            break

    env.close()
    log.info("  Captured %d observations from BrowserGym", len(results))
    return results


# =========================================================================
# Send to GPT-5.4 and print
# =========================================================================

def run_gymv_test(obs_data: dict) -> dict:
    """Send a Gym-V observation to GPT-5.4."""
    goal_line = ""
    if obs_data["description"]:
        goal_line = obs_data["description"].strip().split("\n")[0]

    t0 = time.time()
    result = gymv_generate_label(
        obs_data["image"],
        goal=goal_line,
        task_id=obs_data["task_id"],
        step=obs_data["step"],
        game_rules=obs_data["description"],
        obs_text=obs_data["text"] or "",
        model=MODEL,
    )
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def run_browser_test(obs_data: dict) -> dict:
    """Send a BrowserGym observation to GPT-5.4."""
    t0 = time.time()
    result = browser_generate_label(
        obs_data["image"],
        goal=obs_data["goal"],
        task_id=obs_data["task_id"],
        step=obs_data["step"],
        url=obs_data["url"],
        last_action=obs_data.get("last_action", ""),
        last_action_error=obs_data.get("last_action_error", ""),
        model=MODEL,
    )
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def print_result(label: str, result: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"  model={result['model']}  elapsed={result.get('elapsed_s', '?')}s")
    print(sep)

    if result["schema"]:
        print(result["schema"])
    else:
        print("[NO SCHEMA PARSED]")
        print("Raw output:")
        print(textwrap.indent(result["raw"][:2000], "  "))

    if result["warnings"]:
        print(f"\nWarnings: {result['warnings']}")
    else:
        print("\nValidation: PASSED")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Test GPT-5.4 on real env observations")
    parser.add_argument("--gymv-only", action="store_true")
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--url", default="https://www.google.com",
                        help="BrowserGym URL to test")
    parser.add_argument("--goal", default="", help="BrowserGym task goal")
    parser.add_argument("--save-images", action="store_true",
                        help="Save captured screenshots to disk")
    parser.add_argument("--gymv-steps", type=int, default=2,
                        help="Number of game steps to capture per env")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_gymv = not args.browser_only
    run_browser = not args.gymv_only
    out_dir = Path(__file__).parent

    # ── Gym-V tests ──────────────────────────────────────────────────
    if run_gymv:
        gymv_envs = ["Games/Game2048-v0", "Games/Minesweeper-v0"]
        for env_id in gymv_envs:
            try:
                observations = capture_gymv_obs(env_id, n_steps=args.gymv_steps, seed=args.seed)
            except Exception as e:
                log.error("Failed to capture %s: %s", env_id, e)
                continue

            for obs_data in observations:
                step = obs_data["step"]
                tag = env_id.replace("/", "_")

                if args.save_images and obs_data["image"]:
                    fname = out_dir / f"real_{tag}_step{step}.png"
                    obs_data["image"].save(str(fname))
                    log.info("  Saved %s", fname.name)

                log.info("Sending %s step=%d to GPT-5.4 ...", env_id, step)
                result = run_gymv_test(obs_data)
                print_result(f"Gym-V: {env_id} (step {step})", result)

    # ── BrowserGym tests ─────────────────────────────────────────────
    if run_browser:
        urls = [
            (args.url, args.goal or f"Explore {args.url}"),
        ]
        if not args.gymv_only and args.url == "https://www.google.com":
            urls.append(("https://en.wikipedia.org/wiki/Reinforcement_learning",
                         "Find the section about temporal difference learning"))

        for url, goal in urls:
            try:
                observations = capture_browser_obs(url=url, goal=goal, n_steps=0)
            except Exception as e:
                log.error("Failed to capture %s: %s", url, e)
                continue

            for obs_data in observations:
                step = obs_data["step"]
                domain = url.split("//")[1].split("/")[0].replace(".", "_")

                if args.save_images and obs_data["image"]:
                    fname = out_dir / f"real_browser_{domain}_step{step}.png"
                    obs_data["image"].save(str(fname))
                    log.info("  Saved %s", fname.name)

                log.info("Sending %s step=%d to GPT-5.4 ...", url, step)
                result = run_browser_test(obs_data)
                print_result(f"Browser: {url} (step {step})", result)

    print("\n" + "=" * 70)
    print("  Done. All observations came from real environments.")
    print("=" * 70)


if __name__ == "__main__":
    main()
