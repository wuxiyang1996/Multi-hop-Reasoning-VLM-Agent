#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash
from motif_transfer.frozen_motif_agent import OpenAICompatibleBackend


def _json_object(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        raise ValueError("model must return one JSON object containing string action")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed live BrowserGym MiniWoB/WebShop baseline.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--miniwob-html", type=Path)
    parser.add_argument("--webshop-wrapper", type=Path)
    parser.add_argument("--webshop-base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cell = "webshop" if args.task_id.startswith("webshop.") else "miniwob"
    if cell == "miniwob" and (args.miniwob_html is None or not (args.miniwob_html / "click-button.html").is_file()):
        raise SystemExit(f"invalid MiniWoB HTML root: {args.miniwob_html}")
    if cell == "webshop" and (args.webshop_wrapper is None or not (args.webshop_wrapper / "webshop_wrapper/task.py").is_file()):
        raise SystemExit(f"invalid WebShop wrapper root: {args.webshop_wrapper}")
    values = runpy.run_path(str(args.keys))
    key = values.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    os.environ["DIAG_OPENROUTER_API_KEY"] = str(key)
    if cell == "miniwob":
        os.environ["MINIWOB_URL"] = args.miniwob_html.resolve().as_uri() + "/"
    else:
        os.environ["WEBSHOP_BASE_URL"] = args.webshop_base_url
        os.environ["WEBSHOP_NUM_GOALS"] = "50"

    import gymnasium as gym
    from browsergym.utils.obs import flatten_axtree_to_str
    if cell == "miniwob":
        import browsergym.miniwob  # noqa: F401
    else:
        sys.path.insert(0, str(args.webshop_wrapper))
        from webshop_wrapper import register_webshop_tasks
        register_webshop_tasks(50)

    gym_id = args.task_id if args.task_id.startswith("browsergym/") else f"browsergym/{args.task_id}"
    backend = OpenAICompatibleBackend(
        args.base_url, {"decision": args.model}, api_key_env="DIAG_OPENROUTER_API_KEY",
        json_mode=True, temperature=0, timeout_seconds=180,
    )
    env = gym.make(gym_id, headless=True)
    steps = []
    try:
        obs, info = env.reset(seed=args.seed)
        initial_goal = str(obs.get("goal") or obs.get("goal_object") or "")
        initial_state_hash = stable_hash({
            "goal": initial_goal,
            "axtree": obs.get("axtree_object"),
            "url": obs.get("url"),
        })
        terminated = truncated = False
        total_reward = 0.0
        for step_index in range(args.max_steps):
            axtree = flatten_axtree_to_str(
                obs["axtree_object"], extra_properties=obs.get("extra_element_properties", {}),
            )
            payload = {
                "goal": initial_goal,
                "accessibility_tree": axtree[:16000],
                "last_action": obs.get("last_action"),
                "last_action_error": obs.get("last_action_error") or "",
                "history": [{"action": row["action"], "reward": row["reward"]} for row in steps],
            }
            system = (
                "You are a target-native BrowserGym Decision Agent. Return exactly one JSON object "
                "with key action. Select one executable BrowserGym high-level action grounded in a BID "
                "from the supplied accessibility tree. Common forms are click('bid'), fill('bid','text'), "
                "press('bid','KEY'), scroll(x,y), and noop(). Do not invent a BID. No markdown or explanation."
            )
            raw = backend.complete("decision", system, payload)
            parsed = _json_object(raw)
            action = parsed["action"].strip()
            if not action or "\n" in action:
                raise ValueError("model returned an invalid multi-line/empty action")
            before_hash = stable_hash({"axtree": obs.get("axtree_object"), "url": obs.get("url")})
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps.append({
                "step": step_index,
                "prompt_sha256": stable_hash(payload),
                "response_sha256": stable_hash(raw),
                "before_hash": before_hash,
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "last_action_error": str(obs.get("last_action_error") or ""),
                "after_hash": stable_hash({"axtree": obs.get("axtree_object"), "url": obs.get("url")}),
                "usage": dict(backend.last_usage),
            })
            if terminated or truncated:
                break
    finally:
        env.close()
    payload = {
        "schema_version": 1,
        "cell": cell,
        "condition": "target_only",
        "executor_kind": "live_browsergym",
        "official_evaluator": "environment_reward",
        "task_id": args.task_id,
        "seed": args.seed,
        "model": args.model,
        "initial_goal": initial_goal,
        "initial_state_hash": initial_state_hash,
        "steps": steps,
        "total_reward": total_reward,
        "success": bool(total_reward > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if cell == "miniwob":
        payload["miniwob_commit"] = "7fd85d71a4b60325c6585396ec4f48377d049838"
    else:
        payload["webshop_base_url"] = args.webshop_base_url
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "task_id", "initial_goal", "total_reward", "success", "terminated", "truncated"
    )} | {"step_count": len(steps), "actions": [row["action"] for row in steps]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
