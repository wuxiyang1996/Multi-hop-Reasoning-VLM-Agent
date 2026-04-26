#!/usr/bin/env python
"""
Gym-V Temporal visual → structured ``<state>`` schema rollouts (GPT-5.5 by default).

Similar spirit to ``cold_start/generate_cold_start_orak.py``: steps a Gym-V
stable-retro env, calls :func:`gymv_wrapper.adapter.generate_label` on each
frame (vision primary), and fuses in :func:`gymv_wrapper.build_temporal_visual_schema`
as deterministic grounding metadata.

**Requires:** ``gym_v`` + ``stable_retro`` with ROMs imported for the chosen
``Temporal/...`` envs. API key via ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY``
(unless ``--dry_run``).

Output (``visual_grounding_tests/output/gpt55_gymv/<env_id_sanitized>/<run_id>/``):

  - ``steps.jsonl``   — one JSON object per step (VLM + heuristic grounding)
  - ``run_summary.json`` — timing, model, success counts

Usage (from ``Multi-hop-Reasoning-VLM-Agent`` / repo root on PYTHONPATH)::

    # One env, few steps, default model gpt-5.5
    export OPENAI_API_KEY=...
    python visual_grounding_tests/generate_gymv_visual_schema.py \\
        --envs Temporal/Airstriker-v0 --episodes 1 --max_steps 3 -v

    # Dry run: no API; save heuristic grounding + frame size only
    python visual_grounding_tests/generate_gymv_visual_schema.py --dry_run \\
        --envs Temporal/Airstriker-v0 --episodes 1 --max_steps 2

    # Override model (must match your provider)
    python visual_grounding_tests/generate_gymv_visual_schema.py \\
        --model gpt-5.5 --base_url https://api.openai.com/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))

# ---------------------------------------------------------------------------
# OpenAI / OpenRouter — reuse the project-wide helpers from API_func.py
# ---------------------------------------------------------------------------
from API_func import effective_openai_model, make_openai_client  # noqa: E402

MODEL_GPT55 = "gpt-5.5"
_DEFAULT_OUTPUT_TAG = "gpt55_gymv"


# ---------------------------------------------------------------------------
# Optional gym_v / stable_retro
# ---------------------------------------------------------------------------

def _import_gymv_stack() -> Tuple[Any, Any, Any, Any]:
    import gym_v
    from gymv_wrapper.adapter import generate_label
    from gymv_wrapper.temporal_visual_grounding import (
        TEMPORAL_GAME_SPECS,
        build_temporal_visual_schema,
    )

    return gym_v, generate_label, TEMPORAL_GAME_SPECS, build_temporal_visual_schema


def _rom_resolves(retro_game: str) -> bool:
    import stable_retro  # type: ignore

    try:
        stable_retro.data.get_romfile_path(retro_game)
        return True
    except (FileNotFoundError, OSError):
        if retro_game.endswith("-v0"):
            return False
        try:
            stable_retro.data.get_romfile_path(f"{retro_game}-v0")
            return True
        except (FileNotFoundError, OSError):
            return False


def _sanitize_env_id(env_id: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", env_id)


def _obs_image_size(o: Any) -> Optional[Tuple[int, int]]:
    img = getattr(o, "image", None)
    if img is None:
        return None
    if isinstance(img, list) and img:
        img = img[-1]
    try:
        w, h = img.size
        return (int(w), int(h))
    except Exception:
        return None


def run_one_episode(
    env_id: str,
    *,
    max_steps: int,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    api_key: Optional[str],
    base_url: Optional[str],
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gym_v, generate_label, _SPECS, build_temporal_visual_schema = _import_gymv_stack()
    if env_id not in _SPECS:
        raise ValueError(f"Unknown Temporal env: {env_id}")
    if not _rom_resolves(_SPECS[env_id].retro_game):
        raise FileNotFoundError(
            f"ROM not found for '{_SPECS[env_id].retro_game}' — import it into stable-retro."
        )

    env = gym_v.make(env_id)
    if not dry_run and not (api_key or make_openai_client()):
        print(
            "[WARNING] No API key: set OPENAI_API_KEY / OPENROUTER_API_KEY, "
            "or pass --api_key, or use --dry_run",
        )

    if dry_run or api_key:
        client_model = model
    else:
        client_model = effective_openai_model(model)
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    total_reward = 0.0
    vlm_ok = 0
    vlm_err = 0

    odict, _ = env.reset()
    agent_id = next(iter(odict))
    obs = odict[agent_id]
    unwrapped = env.unwrapped
    description = ""
    if hasattr(unwrapped, "description"):
        description = getattr(unwrapped, "description", "") or ""
    goal_line = description.strip().split("\n")[0] if description else ""
    if not goal_line and env.spec is not None:
        goal_line = str(env.spec.id or env_id)

    for step in range(max_steps):
        vg = build_temporal_visual_schema(env_id, obs)
        vlm_result: Optional[Dict[str, Any]] = None
        if not dry_run:
            try:
                img = obs.image
                if isinstance(img, list) and img:
                    img = img[-1]
                if img is None:
                    vlm_err += 1
                else:
                    vlm_result = generate_label(
                        img,
                        goal=goal_line,
                        task_id=env_id,
                        step=step,
                        game_rules=description,
                        obs_text=obs.text or "",
                        valid_actions=obs.metadata.get("available_actions")
                        if obs.metadata
                        else None,
                        model=client_model,
                        api_key=api_key,
                        base_url=base_url,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if vlm_result and vlm_result.get("schema"):
                        vlm_ok += 1
                    else:
                        vlm_err += 1
            except Exception as exc:  # noqa: BLE001
                vlm_err += 1
                vlm_result = {
                    "schema": None,
                    "raw": f"Error: {exc!r}",
                    "warnings": [str(exc)],
                    "validation": None,
                    "model": model,
                }
                if verbose:
                    traceback.print_exc()
        else:
            vlm_result = None

        rec: Dict[str, Any] = {
            "step": step,
            "env_id": env_id,
            "display_name": _SPECS[env_id].display_name,
            "image_size": _obs_image_size(obs),
            "heuristic_grounding": vg,
            "vlm_label": vlm_result,
            "model_scheduled": client_model,
            "dry_run": dry_run,
            "obs_text": obs.text,
        }
        records.append(rec)
        if verbose and not dry_run and vlm_result and vlm_result.get("schema"):
            sl = (vlm_result["schema"] or "")[:200]
            print(f"    step {step}: schema[0:200]={sl!r}...")

        if step >= max_steps - 1:
            break
        odict, reward, terminated, truncated, _ = env.step({agent_id: "NOOP"})
        total_reward += float(reward)
        obs = odict[agent_id]
        if bool(terminated.get("__all__", False) if isinstance(terminated, dict) else False):
            break
        if bool(truncated.get("__all__", False) if isinstance(truncated, dict) else False):
            break

    try:
        env.close()
    except Exception:
        pass

    elapsed = time.time() - t0
    stats: Dict[str, Any] = {
        "env_id": env_id,
        "max_steps": max_steps,
        "steps_recorded": len(records),
        "total_reward": total_reward,
        "elapsed_seconds": round(elapsed, 3),
        "model": model,
        "model_effective": client_model,
        "dry_run": dry_run,
        "vlm_schema_parsed_ok": vlm_ok,
        "vlm_parse_or_call_fail": vlm_err,
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gym-V Temporal: vision frames → <state> schema (GPT-5.5 default)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["Temporal/Airstriker-v0"],
        help="Gym-V Temporal/* env ids (default: Airstriker)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per env")
    parser.add_argument("--max_steps", type=int, default=3, help="Max steps (NOOP) per episode")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VLM_GYMV_SCHEMA_MODEL", MODEL_GPT55),
        help=f"VLM model (default: {MODEL_GPT55} or $VLM_GYMV_SCHEMA_MODEL)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Override API key (else env OPENAI / OPENROUTER)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Override OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not call the vision API; only heuristic visual_grounding + env steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root output dir (default: visual_grounding_tests/output/gpt55_gymv)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "output" / _DEFAULT_OUTPUT_TAG
    )
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        _import_gymv_stack()
    except ImportError as e:
        print(f"[FATAL] gym_v or gymv_wrapper not importable: {e}")
        sys.exit(1)

    from gymv_wrapper.temporal_visual_grounding import TEMPORAL_GAME_SPECS

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "dry_run": args.dry_run,
        "episodes_per_env": args.episodes,
        "max_steps": args.max_steps,
        "envs": args.envs,
    }
    all_summaries: List[Dict[str, Any]] = []

    for env_id in args.envs:
        if env_id not in TEMPORAL_GAME_SPECS:
            print(f"[SKIP] Unknown env (not in TEMPORAL_GAME_SPECS): {env_id}")
            continue
        if not _rom_resolves(TEMPORAL_GAME_SPECS[env_id].retro_game):
            print(
                f"[SKIP] {env_id}: ROM missing for {TEMPORAL_GAME_SPECS[env_id].retro_game}",
            )
            continue

        safe = _sanitize_env_id(env_id)
        for ep in range(args.episodes):
            ep_dir = out_root / safe / f"{run_id}_ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  -> {env_id}  episode {ep + 1}/{args.episodes}  -> {ep_dir}")
            try:
                records, stats = run_one_episode(
                    env_id,
                    max_steps=args.max_steps,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    dry_run=args.dry_run,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    verbose=args.verbose,
                )
                stats["episode"] = ep
                jsonl_path = ep_dir / "steps.jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                summary_path = ep_dir / "run_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
                print(
                    f"     wrote {len(records)} steps, vlm_ok={stats.get('vlm_schema_parsed_ok', 0)}",
                )
                all_summaries.append({**stats, "path": str(ep_dir)})
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {env_id} ep{ep}: {exc}")
                traceback.print_exc()
                all_summaries.append(
                    {
                        "env_id": env_id,
                        "episode": ep,
                        "error": str(exc),
                    },
                )

    master["runs"] = all_summaries
    master_path = out_root / f"batch_{run_id}.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Batch summary: {master_path}")


if __name__ == "__main__":
    main()
