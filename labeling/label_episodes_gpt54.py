#!/usr/bin/env python
"""
Label cold-start episode trajectories using GPT-5.4.

Reads existing episode JSON files and generates concise per-step annotations:

  - summary_state : compact state summary optimised for RAG retrieval
  - summary       : identical to summary_state
  - intentions    : concise reasoning (1-2 sentences) for the action taken
  - skills        : renamed from sub_tasks (null — populated downstream)

The labeling pipeline reuses deterministic helpers from
``decision_agents.agent_helper`` (compact_text_observation, get_state_summary,
infer_intention) and refines them through GPT-5.4 for conciseness.

Output structure (labeling/output/gpt54/<game_name>/):
  - episode_NNN.json        Labeled episode (original + new fields)
  - labeling_summary.json   Run statistics

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # Label all episodes for all games in the default cold-start output
    python labeling/label_episodes_gpt54.py

    # Label specific game(s)
    python labeling/label_episodes_gpt54.py --games tetris candy_crush

    # Label a single file
    python labeling/label_episodes_gpt54.py --input_file cold_start/output/gpt54/tetris/episode_000.json

    # Dry run (preview first episode without saving)
    python labeling/label_episodes_gpt54.py --dry_run --games tetris --max_episodes 1

    # Process exactly one rollout per game (quick test across all games)
    python labeling/label_episodes_gpt54.py --one_per_game -v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
# Imports from the project
# ---------------------------------------------------------------------------
from decision_agents.agent_helper import (
    compact_text_observation,
    get_state_summary,
    infer_intention,
    strip_think_tags,
    build_rag_summary,
    extract_game_facts,
    HARD_SUMMARY_CHAR_LIMIT,
    SUBGOAL_TAGS,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

try:
    import openai
    from api_keys import openai_api_key, open_router_api_key
except (ImportError, AttributeError):
    openai = None
    openai_api_key = None
    open_router_api_key = None

try:
    from API_func import OPENROUTER_BASE
except ImportError:
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"

_OUTPUT_ROOT = CODEBASE_ROOT / "cold_start" / "output"
DEFAULT_INPUT_DIRS: List[Path] = [
    _OUTPUT_ROOT / "gpt54",
    _OUTPUT_ROOT / "gpt54_evolver",
    _OUTPUT_ROOT / "gpt54_orak",
    _OUTPUT_ROOT / "gpt54_sokoban",
]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "gpt54"

SUMMARY_CHAR_BUDGET = 200
SUMMARY_PROSE_WORD_BUDGET = 30
INTENTION_WORD_BUDGET = 15

# ---------------------------------------------------------------------------
# GPT-5.4 chat helper (function-calling style for structured output)
# ---------------------------------------------------------------------------

def _ask_gpt54(
    prompt: str,
    *,
    system: str = "",
    model: str = MODEL_GPT54,
    temperature: float = 0.2,
    max_tokens: int = 150,
) -> Optional[str]:
    """Send a prompt to GPT-5.4 via ask_model and return the cleaned reply."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    if ask_model is not None:
        result = ask_model(full_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        if result and not result.startswith("Error"):
            return strip_think_tags(result).strip()

    return None


# ---------------------------------------------------------------------------
# Labeling functions — wrappers around decision_agents helpers + GPT-5.4
# ---------------------------------------------------------------------------

def generate_summary_state(
    state: str,
    game_name: str = "",
    step_idx: int = -1,
    total_steps: int = -1,
    reward: float = 0.0,
) -> str:
    """Produce a compact ``key=value`` state summary for RAG retrieval.

    Fully deterministic (no LLM call).  Uses game-aware fact extraction
    so the output matches the online agent's structured format and embeds
    well for cosine-similarity retrieval.
    """
    return build_rag_summary(
        state,
        game_name,
        step_idx=step_idx,
        total_steps=total_steps,
        reward=reward,
    )


def generate_intention(
    state: str,
    action: str,
    original_reasoning: Optional[str] = None,
    game_name: str = "",
    model: str = MODEL_GPT54,
) -> str:
    """Produce a ``[TAG] subgoal phrase`` for skill segmentation and indexing.

    Format: ``[TAG] short phrase`` (max ~15 words).
    Tags: SETUP, CLEAR, MERGE, ATTACK, DEFEND, NAVIGATE, POSITION,
    COLLECT, BUILD, SURVIVE, OPTIMIZE, EXPLORE, EXECUTE.

    When the episode already contains verbose reasoning (from cold-start),
    we condense it; otherwise we infer from state + action.  Output budget
    is kept very tight (~30 tokens) for 14B-model compatibility.
    """
    tags_str = "|".join(SUBGOAL_TAGS)

    if original_reasoning and len(original_reasoning) > 20:
        prompt = (
            f"Condense to [TAG] subgoal phrase (max {INTENTION_WORD_BUDGET} words).\n"
            f"Tags: {tags_str}\n"
            f"Reasoning: {original_reasoning[:300]}\n"
            f"Reply ONLY: [TAG] phrase\n"
            f"Example: [SETUP] Build flat stack for line clears\n"
            f"Subgoal:"
        )
    else:
        game_label = game_name.replace("_", " ") if game_name else "game"
        compact = compact_text_observation(state, max_chars=200)
        state_text = compact if compact else state[:1000]
        prompt = (
            f"{game_label}. Action: {action}\n"
            f"State: {state_text}\n"
            f"What multi-step subgoal? Reply ONLY: [TAG] phrase "
            f"(max {INTENTION_WORD_BUDGET} words)\n"
            f"Tags: {tags_str}\n"
            f"Example: [MERGE] Combine 4-tiles toward corner anchor\n"
            f"Subgoal:"
        )

    result = _ask_gpt54(prompt, model=model, max_tokens=40)
    if result:
        cleaned = result.split("\n")[0].strip().strip('"').strip("'")
        if not cleaned.startswith("["):
            cleaned = "[EXECUTE] " + cleaned
        return cleaned[:150]

    fallback = infer_intention(state, game=game_name, model=model)
    if fallback:
        return f"[EXECUTE] {fallback}"
    return f"[EXECUTE] {action}"


def generate_summary_prose(
    state: str,
    game_name: str = "",
    model: str = MODEL_GPT54,
) -> str:
    """Produce a 1-sentence prose summary for the LLM teacher and human review.

    Budget: max 30 words.  This is the only LLM call for the summary path;
    ``summary_state`` is fully deterministic.
    """
    compact = compact_text_observation(state, max_chars=250)
    state_text = compact if compact else state[:1500]
    game_label = game_name.replace("_", " ") if game_name else "game"

    prompt = (
        f"{game_label} state: {state_text}\n"
        f"Summarize in 1 sentence (max {SUMMARY_PROSE_WORD_BUDGET} words). "
        f"Include: key position, score, main threat or opportunity."
    )

    result = _ask_gpt54(prompt, model=model, max_tokens=60)
    if result:
        return result[:HARD_SUMMARY_CHAR_LIMIT]
    return (compact or state[:SUMMARY_CHAR_BUDGET])[:HARD_SUMMARY_CHAR_LIMIT]


# ---------------------------------------------------------------------------
# Per-experience labeling
# ---------------------------------------------------------------------------

def label_experience(
    exp: Dict[str, Any],
    game_name: str = "",
    model: str = MODEL_GPT54,
    delay: float = 0.0,
    step_idx: int = -1,
    total_steps: int = -1,
) -> Dict[str, Any]:
    """Label a single experience dict in-place and return it.

    Produces three complementary fields:

    * ``summary_state`` — compact ``key=value`` string (deterministic, no LLM)
      optimised for RAG embedding retrieval.
    * ``summary`` — concise prose sentence via LLM, for the LLM teacher /
      human review.
    * ``intentions`` — ``[TAG] subgoal phrase`` via LLM, for skill
      segmentation predicates and skill-bank indexing.
    """
    state = exp.get("state", "")
    action = exp.get("action", "")
    reward = exp.get("reward", 0.0)
    original_reasoning = exp.get("intentions")
    idx = exp.get("idx", step_idx)

    # --- summary_state (structured key=value, deterministic, 0 LLM calls) ---
    summary_state = generate_summary_state(
        state,
        game_name=game_name,
        step_idx=idx if isinstance(idx, int) else step_idx,
        total_steps=total_steps,
        reward=reward if isinstance(reward, (int, float)) else 0.0,
    )
    exp["summary_state"] = summary_state

    # --- summary (concise prose, 1 LLM call) ---
    summary = generate_summary_prose(state, game_name=game_name, model=model)
    exp["summary"] = summary

    if delay > 0:
        time.sleep(delay)

    # --- intention ([TAG] subgoal phrase, 1 LLM call) ---
    intention = generate_intention(
        state, action,
        original_reasoning=original_reasoning,
        game_name=game_name,
        model=model,
    )
    exp["intentions"] = intention

    # --- skills (renamed from sub_tasks, null for now) ---
    exp.pop("sub_tasks", None)
    exp["skills"] = None

    if delay > 0:
        time.sleep(delay)

    return exp


# ---------------------------------------------------------------------------
# Per-episode labeling
# ---------------------------------------------------------------------------

def label_episode(
    episode_data: Dict[str, Any],
    model: str = MODEL_GPT54,
    delay: float = 0.1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Label all experiences in an episode dict in-place."""
    game_name = episode_data.get("game_name", "")
    experiences = episode_data.get("experiences", [])
    n = len(experiences)

    for i, exp in enumerate(experiences):
        try:
            label_experience(
                exp, game_name=game_name, model=model, delay=delay,
                step_idx=i, total_steps=n,
            )
            if verbose:
                ss = (exp["summary_state"][:60] + "...") if len(exp.get("summary_state", "")) > 60 else exp.get("summary_state", "")
                it = (exp["intentions"][:60] + "...") if len(exp.get("intentions", "")) > 60 else exp.get("intentions", "")
                print(f"    step {exp.get('idx', i):>3d}/{n}: summary_state={ss}")
                print(f"           intention={it}")
        except Exception as exc:
            print(f"    [WARN] step {exp.get('idx', i)}: labeling failed ({exc})")
            exp.setdefault("summary_state", None)
            exp.setdefault("summary", None)
            exp.setdefault("intentions", exp.get("intentions"))
            exp.pop("sub_tasks", None)
            exp.setdefault("skills", None)

    return episode_data


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_episode_files(
    input_dirs: List[Path],
    games: Optional[List[str]] = None,
) -> List[Path]:
    """Find all episode_*.json files under one or more *input_dirs*.

    Scans every sub-directory that looks like a game folder.  When *games* is
    given, only matching folder names are included.
    """
    files: List[Path] = []
    seen_games: set = set()

    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for gd in sorted(input_dir.iterdir()):
            if not gd.is_dir():
                continue
            game_name = gd.name
            if games is not None and game_name not in games:
                continue
            for fp in sorted(gd.glob("episode_*.json")):
                if fp.name == "episode_buffer.json":
                    continue
                files.append(fp)
            if any(gd.glob("episode_*.json")):
                seen_games.add(game_name)
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label cold-start episodes with GPT-5.4 (summary_state, intention, skills)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, nargs="*", default=None,
                        help="Input director(ies) with game sub-folders (default: all gpt54* output dirs)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Label a single episode JSON file instead of scanning a directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory for labeled episodes (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Only label these games (default: all found)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model for labeling (default: {MODEL_GPT54})")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to label per game (default: all)")
    parser.add_argument("--one_per_game", action="store_true",
                        help="Process only the first episode for each game (quick test run)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay in seconds between API calls (default: 0.1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite already-labeled episodes in output dir")
    parser.add_argument("--in_place", action="store_true",
                        help="Write labels back to the original input files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview labeling on first episode without saving")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-step labeling details")

    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dir] if args.input_dir else DEFAULT_INPUT_DIRS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    if args.one_per_game:
        args.max_episodes = 1

    # ---- Validate API key ----
    has_key = bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or (open_router_api_key and str(open_router_api_key).strip())
    )
    if not has_key:
        print("[WARNING] No API key detected. LLM calls will fail.")
        print("  Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

    # ---- Collect files ----
    if args.input_file:
        files = [Path(args.input_file)]
        if not files[0].exists():
            print(f"[ERROR] File not found: {args.input_file}")
            sys.exit(1)
    else:
        files = find_episode_files(input_dirs, games=args.games)

    if not files:
        dirs_str = ", ".join(str(d) for d in input_dirs)
        print(f"[ERROR] No episode files found under: {dirs_str}")
        sys.exit(1)

    # Group by game
    game_files: Dict[str, List[Path]] = {}
    for fp in files:
        game = fp.parent.name
        game_files.setdefault(game, []).append(fp)

    if not args.dry_run and not args.in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  GPT-5.4 Episode Labeling")
    print("=" * 78)
    print(f"  Model:       {args.model}")
    if args.input_file:
        print(f"  Input:       {args.input_file}")
    else:
        for d in input_dirs:
            print(f"  Input:       {d}")
    print(f"  Output:      {'in-place' if args.in_place else output_dir}")
    print(f"  Games:       {', '.join(sorted(game_files.keys()))}")
    print(f"  Episodes:    {sum(len(v) for v in game_files.values())} total")
    per_game = args.max_episodes if args.max_episodes else "all"
    print(f"  Per game:    {per_game} episode(s)")
    print(f"  Delay:       {args.delay}s between calls")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 78)

    overall_t0 = time.time()
    all_stats: List[Dict[str, Any]] = []

    for game, gfiles in sorted(game_files.items()):
        episode_files = gfiles[:args.max_episodes] if args.max_episodes else gfiles
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game} ({len(episode_files)} episodes)")
        print(f"{'━' * 78}")

        game_out_dir = output_dir / game
        if not args.dry_run and not args.in_place:
            game_out_dir.mkdir(parents=True, exist_ok=True)

        game_t0 = time.time()
        game_labeled = 0

        for fp in episode_files:
            out_path = game_out_dir / fp.name if not args.in_place else fp

            if not args.overwrite and not args.in_place and out_path.exists():
                print(f"  [SKIP] {fp.name} (already labeled)")
                continue

            print(f"\n  Labeling {fp.name} ...")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    episode_data = json.load(f)
            except Exception as exc:
                print(f"    [ERROR] Failed to load: {exc}")
                continue

            n_steps = len(episode_data.get("experiences", []))
            t0 = time.time()

            label_episode(
                episode_data,
                model=args.model,
                delay=args.delay,
                verbose=args.verbose,
            )

            elapsed = time.time() - t0
            print(f"    Labeled {n_steps} steps in {elapsed:.1f}s")

            if args.dry_run:
                exps = episode_data.get("experiences", [])
                preview = exps[0] if exps else {}
                print("\n    --- Preview (step 0) ---")
                print(f"    summary_state: {preview.get('summary_state')}")
                print(f"    summary:       {preview.get('summary')}")
                print(f"    intentions:    {preview.get('intentions')}")
                print(f"    skills:        {preview.get('skills')}")
                if len(exps) > 1:
                    print(f"\n    --- Preview (step 1) ---")
                    print(f"    summary_state: {exps[1].get('summary_state')}")
                    print(f"    intentions:    {exps[1].get('intentions')}")
                print("    --- end preview ---\n")
                break

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=2, ensure_ascii=False, default=str)
                print(f"    Saved → {out_path}")
                game_labeled += 1
            except Exception as exc:
                print(f"    [ERROR] Failed to save: {exc}")

        game_elapsed = time.time() - game_t0
        stat = {
            "game": game,
            "episodes_labeled": game_labeled,
            "elapsed_seconds": game_elapsed,
        }
        all_stats.append(stat)

        if not args.dry_run and not args.in_place and game_labeled > 0:
            summary_path = game_out_dir / "labeling_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "game": game,
                    "model": args.model,
                    "timestamp": datetime.now().isoformat(),
                    "episodes_labeled": game_labeled,
                    "elapsed_seconds": game_elapsed,
                }, f, indent=2, ensure_ascii=False)

    overall_elapsed = time.time() - overall_t0

    print(f"\n{'=' * 78}")
    print("  LABELING COMPLETE")
    print(f"{'=' * 78}")
    total_labeled = sum(s["episodes_labeled"] for s in all_stats)
    print(f"  Episodes labeled: {total_labeled}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    if not args.dry_run and not args.in_place:
        print(f"  Output:           {output_dir}")

        master = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total_elapsed_seconds": overall_elapsed,
            "per_game": all_stats,
        }
        master_path = output_dir / "labeling_batch_summary.json"
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        print(f"  Summary:          {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
