#!/usr/bin/env python3
"""Fix the candidate-ordering bias in skill_selection SFT data.

Problem: all gymv games (and some env_wrapper games) have the oracle skill
at candidates[0] 100% of the time, so completion is always ``SKILL: 1``.
The LoRA learns ``any state → SKILL: 1`` instead of state-conditioned selection.

Fix: for each record, randomly permute the candidates list and rewrite both
the prompt (Strategies block) and the completion (SKILL: N) to match.

Usage::

    python -m tools.reshuffle_sft_candidates \
        --src  ../SFT_Data/high_reward/decision_sft_v2 \
        --dst  ../SFT_Data/high_reward/decision_sft_v3 \
        --games Temporal_ThunderForceIII-v0 Temporal_Airstriker-v0 ...

    # Or fix ALL games (safe — already-diverse games stay diverse):
    python -m tools.reshuffle_sft_candidates \
        --src  ../SFT_Data/high_reward/decision_sft_v2 \
        --dst  ../SFT_Data/high_reward/decision_sft_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reshuffle_sft")

# Deterministic per-record RNG for reproducibility
import hashlib
import random


def _record_seed(record: Dict[str, Any]) -> int:
    """Deterministic seed from (episode_id, step_idx) so re-runs are identical."""
    key = f"{record.get('episode_id', '')}__{record.get('step_idx', 0)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _rebuild_strategies_block(candidates: List[str]) -> str:
    """Build the numbered Strategies block for the prompt."""
    lines = []
    for i, name in enumerate(candidates, 1):
        lines.append(f"  {i}. {name}")
    return "\n".join(lines)


_STRATEGIES_RE = re.compile(
    r"(Strategies:\n)((?:\s+\d+\.\s+.+\n?)+)",
    re.MULTILINE,
)

_SKILL_RE = re.compile(r"(SKILL:\s*)(\d+)")


def reshuffle_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Shuffle candidates, rewrite prompt + completion + metadata."""
    record = dict(record)  # shallow copy
    candidates = list(record.get("candidates", []))
    selected = record.get("selected_skill_id", "")

    if len(candidates) < 2:
        return record

    rng = random.Random(_record_seed(record))
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    if selected and selected in shuffled:
        new_pos = shuffled.index(selected) + 1  # 1-indexed
    else:
        new_pos = 1

    # --- rewrite prompt Strategies block ---
    prompt = record.get("prompt", "")
    new_block = _rebuild_strategies_block(shuffled)
    prompt_new = _STRATEGIES_RE.sub(
        lambda m: m.group(1) + new_block + "\n",
        prompt,
        count=1,
    )

    # --- rewrite completion SKILL: N ---
    completion = record.get("completion", "")
    completion_new = _SKILL_RE.sub(rf"\g<1>{new_pos}", completion, count=1)

    record["candidates"] = shuffled
    record["prompt"] = prompt_new
    record["completion"] = completion_new

    return record


def process_game(
    src_dir: Path, dst_dir: Path, game: str,
) -> Dict[str, Any]:
    """Reshuffle one game's skill_selection.jsonl. Returns stats."""
    src = src_dir / game / "skill_selection.jsonl"
    if not src.exists():
        logger.warning("No data for %s", game)
        return {"game": game, "n": 0, "skipped": True}

    dst_game = dst_dir / game
    dst_game.mkdir(parents=True, exist_ok=True)
    dst_path = dst_game / "skill_selection.jsonl"

    from collections import Counter
    pos_before = Counter()
    pos_after = Counter()
    n = 0

    with open(src) as fin, open(dst_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n += 1

            # Track before
            cands = record.get("candidates", [])
            sel = record.get("selected_skill_id", "")
            if sel and sel in cands:
                pos_before[cands.index(sel)] += 1

            # Shuffle
            new_record = reshuffle_record(record)

            # Track after
            new_cands = new_record.get("candidates", [])
            if sel and sel in new_cands:
                pos_after[new_cands.index(sel)] += 1

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    # Also copy action_taking.jsonl if it exists (unchanged)
    at_src = src_dir / game / "action_taking.jsonl"
    if at_src.exists():
        import shutil
        shutil.copy2(at_src, dst_game / "action_taking.jsonl")

    stats = {
        "game": game,
        "n": n,
        "pos_before": dict(sorted(pos_before.items())),
        "pos_after": dict(sorted(pos_after.items())),
    }
    logger.info(
        "[%s] %d records  before=%s  after=%s",
        game, n, dict(pos_before), dict(pos_after),
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Reshuffle SFT candidates")
    parser.add_argument("--src", type=str,
                        default="../SFT_Data/high_reward/decision_sft_v2")
    parser.add_argument("--dst", type=str,
                        default="../SFT_Data/high_reward/decision_sft_v3")
    parser.add_argument("--games", nargs="*", default=None,
                        help="Games to fix (default: all subdirs)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if args.games:
        games = args.games
    else:
        games = sorted(d.name for d in src_dir.iterdir() if d.is_dir())

    logger.info("Source: %s", src_dir)
    logger.info("Output: %s", dst_dir)
    logger.info("Games: %s", games)

    all_stats = []
    for game in games:
        stats = process_game(src_dir, dst_dir, game)
        all_stats.append(stats)

    # Summary
    total = sum(s["n"] for s in all_stats)
    logger.info("=== Done: %d records across %d games → %s ===", total, len(games), dst_dir)

    summary_path = dst_dir / "RESHUFFLE_MANIFEST.json"
    with open(summary_path, "w") as f:
        json.dump({"total_records": total, "games": all_stats}, f, indent=2)
    logger.info("Manifest: %s", summary_path)


if __name__ == "__main__":
    main()
