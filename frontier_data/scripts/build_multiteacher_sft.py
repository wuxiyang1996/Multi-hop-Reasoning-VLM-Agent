#!/usr/bin/env python
"""Build multi-teacher high-reward SFT data for gymv games.

Scans cold-start rollouts from 4 frontier teachers (GPT-5.4, Claude-4.6,
Gemini-3.1, Qwen3-VL-235B), keeps only episodes with positive total reward,
and emits action_taking.jsonl per game in the same schema that
trainer/SFT/data_loader.py already consumes.

Two source pools:
  1. skip8 runs (80 steps, frame_skip=8)  — 8 games, 4 teachers
  2. legacy runs (100 steps, frame_skip=1) — 13 games, GPT-5.4 only (fallback)

For each game the script ranks episodes by reward across all teachers,
then emits up to --max-episodes-per-game (default 40) rows, keeping the
best-scoring episodes first.  When multiple teachers tie, diversity is
preferred (round-robin across teachers).

Output: frontier_data/output/decision_sft_multiteacher/<game>/action_taking.jsonl
        frontier_data/output/decision_sft_multiteacher/_build_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DOWNLOAD_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/main_project"
)

SKIP8_SOURCES: Dict[str, Path] = {
    "gpt54": DOWNLOAD_ROOT / "Cold-start-out-gymv"
             / "gpt54_skip8_e16_s80_20260503_093654",
    "claude": DOWNLOAD_ROOT / "openrouter-baselines-out"
              / "openrouter_skip8_e16_s80_20260503_093707" / "claude" / "gymv",
    "gemini": DOWNLOAD_ROOT / "openrouter-baselines-out"
              / "openrouter_skip8_e16_s80_20260503_093707" / "gemini" / "gymv",
    "qwen":  DOWNLOAD_ROOT / "qwen-baselines-out"
             / "qwen_vllm_skip8_e16_s80_20260503_093716" / "9B" / "gymv",
}

LEGACY_SOURCE: Dict[str, Path] = {
    "gpt54_legacy": DOWNLOAD_ROOT / "Cold-start-out-gymv"
                    / "sft_gpt5p4_e20_s100_stream_20260429_080127",
}

SKIP8_GAMES = [
    "Temporal_Airstriker-v0", "Temporal_AlteredBeast-v0",
    "Temporal_Columns-v0", "Temporal_DynamiteHeaddy-v0",
    "Temporal_SpaceHarrierII-v0", "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0", "Temporal_ThunderForceIII-v0",
]

LEGACY_ONLY_GAMES = [
    "Temporal_CastleOfIllusion-v0", "Temporal_CastlevaniaBloodlines-v0",
    "Temporal_GoldenAxe-v0", "Temporal_KidChameleon-v0",
    "Temporal_MortalKombatII-v0",
]

MODEL_NAMES = {
    "gpt54": "gpt-5.4",
    "claude": "claude-4.6-sonnet",
    "gemini": "gemini-3.1-pro",
    "qwen": "qwen3-vl-235b",
    "gpt54_legacy": "gpt-5.4",
}

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n"
)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_episode(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def episode_reward(ep: Dict[str, Any]) -> float:
    meta = ep.get("metadata") or {}
    tr = meta.get("total_reward")
    if tr is not None:
        return float(tr)
    exps = ep.get("experiences") or ep.get("steps") or []
    return sum(float(s.get("reward", 0)) for s in exps)


def collect_episodes(
    game: str,
    sources: Dict[str, Path],
    min_reward: float = 0.0,
) -> List[Tuple[str, Path, float]]:
    """Return [(teacher, ep_path, total_reward), ...] with reward > min_reward."""
    results: List[Tuple[str, Path, float]] = []
    for teacher, base_dir in sources.items():
        gdir = base_dir / game
        if not gdir.is_dir():
            continue
        for ep_path in sorted(gdir.glob("episode_*.json")):
            ep = load_episode(ep_path)
            if ep is None:
                continue
            r = episode_reward(ep)
            if r > min_reward:
                results.append((teacher, ep_path, r))
    return results


def select_top_episodes(
    candidates: List[Tuple[str, Path, float]],
    max_eps: int,
) -> List[Tuple[str, Path, float]]:
    """Pick top-max_eps episodes, preferring diversity across teachers."""
    if len(candidates) <= max_eps:
        return sorted(candidates, key=lambda x: -x[2])

    by_teacher: Dict[str, List[Tuple[str, Path, float]]] = defaultdict(list)
    for c in candidates:
        by_teacher[c[0]].append(c)
    for t in by_teacher:
        by_teacher[t].sort(key=lambda x: -x[2])

    selected: List[Tuple[str, Path, float]] = []
    teachers = sorted(by_teacher.keys())
    idx = {t: 0 for t in teachers}

    while len(selected) < max_eps:
        added_any = False
        for t in teachers:
            if idx[t] < len(by_teacher[t]) and len(selected) < max_eps:
                selected.append(by_teacher[t][idx[t]])
                idx[t] += 1
                added_any = True
        if not added_any:
            break

    selected.sort(key=lambda x: -x[2])
    return selected


# ---------------------------------------------------------------------------
# SFT row builders
# ---------------------------------------------------------------------------

def _extract_schema(step: Dict[str, Any]) -> str:
    meta = step.get("metadata") or {}
    schema = meta.get("schema")
    if isinstance(schema, str) and schema.strip():
        return schema.strip()
    summary = step.get("summary_state")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    state = step.get("state") or step.get("raw_state") or ""
    return str(state).strip()


def _format_actions(actions: List[str]) -> str:
    return "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))


def _action_index(action: str, valid: List[str]) -> int:
    target = (action or "").strip().lower()
    for i, a in enumerate(valid):
        if a == action:
            return i + 1
        if (a or "").strip().lower() == target:
            return i + 1
    return 1


def build_action_row(
    step: Dict[str, Any],
    *,
    game: str,
    episode_id: str,
    step_idx: int,
    teacher: str,
    episode_reward: float,
) -> Optional[Dict[str, Any]]:
    schema_text = _extract_schema(step)
    valid_actions = step.get("available_actions") or (
        (step.get("metadata") or {}).get("valid_actions") or []
    )
    action = step.get("action")
    if not schema_text or not valid_actions or not action:
        return None

    user_text = (
        f"Game state:\n\n{schema_text}\n\n"
        f"Available actions (pick ONE by number):\n"
        f"{_format_actions(valid_actions)}\n\n"
        f"Choose the best action. Output REASONING then ACTION number."
    )
    prompt = SYSTEM_PROMPT + "\n" + user_text

    action_num = _action_index(action, valid_actions)
    summary = (step.get("summary") or "").strip() or "Expert play."
    completion = f"REASONING: {summary[:200]}\nACTION: {action_num}"

    intention_raw = (step.get("intentions") or "").strip()
    intention = intention_raw if intention_raw.startswith("[") else "[EXECUTE] act in the game"

    return {
        "prompt": prompt,
        "completion": completion,
        "intention": intention,
        "active_skill": "",
        "game": game,
        "corpus": "gym_v",
        "episode_id": episode_id,
        "step_idx": step_idx,
        "valid_actions": list(valid_actions),
        "reward": step.get("reward"),
        "episode_reward": episode_reward,
        "teacher_model": MODEL_NAMES.get(teacher, teacher),
        "teacher_tag": teacher,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_game(
    game: str,
    sources: Dict[str, Path],
    output_root: Path,
    max_episodes: int,
    min_reward: float,
) -> Dict[str, Any]:
    candidates = collect_episodes(game, sources, min_reward=min_reward)
    selected = select_top_episodes(candidates, max_episodes)

    out_dir = output_root / game
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "action_taking.jsonl"

    teacher_counts: Dict[str, int] = defaultdict(int)
    total_rows = 0
    total_steps = 0
    rewards_list = []

    with open(out_file, "w") as f:
        for teacher, ep_path, ep_rwd in selected:
            ep = load_episode(ep_path)
            if ep is None:
                continue
            episode_id = str(ep.get("episode_id", ep_path.stem))
            exps = ep.get("experiences") or ep.get("steps") or []
            total_steps += len(exps)
            rewards_list.append(ep_rwd)
            teacher_counts[teacher] += 1

            for i, step in enumerate(exps):
                row = build_action_row(
                    step,
                    game=game,
                    episode_id=episode_id,
                    step_idx=i,
                    teacher=teacher,
                    episode_reward=ep_rwd,
                )
                if row:
                    f.write(json.dumps(row) + "\n")
                    total_rows += 1

    stats = {
        "game": game,
        "candidates_total": len(candidates),
        "selected_episodes": len(selected),
        "action_taking_rows": total_rows,
        "total_steps": total_steps,
        "teacher_breakdown": dict(teacher_counts),
        "reward_min": min(rewards_list) if rewards_list else 0,
        "reward_max": max(rewards_list) if rewards_list else 0,
        "reward_mean": sum(rewards_list) / len(rewards_list) if rewards_list else 0,
    }
    print(
        f"  {game}: {len(selected)} eps ({dict(teacher_counts)}), "
        f"{total_rows} rows, reward=[{stats['reward_min']:.0f}..{stats['reward_max']:.0f}] "
        f"mean={stats['reward_mean']:.0f}"
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-episodes-per-game", type=int, default=40,
                    help="Max episodes to keep per game (default 40)")
    ap.add_argument("--min-reward", type=float, default=0.0,
                    help="Minimum total_reward to include (default >0)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--include-legacy", action="store_true", default=True,
                    help="Include legacy 100-step GPT-5.4 for extra 5 games")
    args = ap.parse_args()

    output_root = args.output_dir or (
        REPO_ROOT / "frontier_data" / "output" / "decision_sft_multiteacher"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[build_multiteacher_sft] output: {output_root}")
    print(f"[build_multiteacher_sft] max_eps={args.max_episodes_per_game}, "
          f"min_reward={args.min_reward}")

    all_stats: List[Dict[str, Any]] = []

    print("\n--- skip8 games (4 teachers) ---")
    for game in SKIP8_GAMES:
        stats = process_game(
            game, SKIP8_SOURCES, output_root,
            args.max_episodes_per_game, args.min_reward,
        )
        all_stats.append(stats)

    if args.include_legacy:
        print("\n--- legacy-only games (GPT-5.4 only) ---")
        for game in LEGACY_ONLY_GAMES:
            stats = process_game(
                game, LEGACY_SOURCE, output_root,
                args.max_episodes_per_game, args.min_reward,
            )
            all_stats.append(stats)

    total_rows = sum(s["action_taking_rows"] for s in all_stats)
    total_eps = sum(s["selected_episodes"] for s in all_stats)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_episodes_per_game": args.max_episodes_per_game,
            "min_reward": args.min_reward,
            "include_legacy": args.include_legacy,
        },
        "totals": {
            "games": len(all_stats),
            "episodes": total_eps,
            "action_taking_rows": total_rows,
        },
        "per_game": all_stats,
    }
    summary_path = output_root / "_build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] {len(all_stats)} games, {total_eps} episodes, "
          f"{total_rows} action_taking rows")
    print(f"       summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
