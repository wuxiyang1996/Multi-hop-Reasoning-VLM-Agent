#!/usr/bin/env python3
"""Mine and inject step_checks for game skills from trajectory data.

This script solves the problem that game skill step_checks are all
empty (unlike QA skills which have operator-based predicates).

It works in two modes:

1. **Keyword mode** (default, no trajectory data needed):
   Parses protocol step descriptions and maps keywords to known
   game effect predicates.  E.g. "Apply transformation" matches
   ``board_transformed=true`` for 2048.

2. **Trajectory mode** (with --trajectories):
   Reads collected episode data, runs the StateEffectObserver
   on each step, and finds which effects consistently appear
   at each protocol step boundary.  More accurate but requires
   pre-collected rollout data.

Usage::

    # Keyword mode (works immediately, no data needed)
    python frontier_data/scripts/mine_game_step_checks.py

    # With trajectory data
    python frontier_data/scripts/mine_game_step_checks.py \\
        --trajectories rollout_data/*.jsonl

    # Dry run (show what would change)
    python frontier_data/scripts/mine_game_step_checks.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from decision_agents.protocol_utils import (
    generate_step_checks_from_effects,
)

logger = logging.getLogger("mine_game_step_checks")

GAME_TASKS = frozenset({
    "tetris", "candy_crush", "super_mario", "twenty_forty_eight",
    "2048", "tictactoe", "texasholdem",
    "Temporal_Airstriker-v0", "Temporal_AlteredBeast-v0",
    "Temporal_Columns-v0", "Temporal_DynamiteHeaddy-v0",
    "Temporal_SpaceHarrierII-v0", "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0", "Temporal_ThunderForceIII-v0",
})

DEFAULT_BANK_DIRS = [
    ROOT / "SFT_Data" / "skill_banks",
    ROOT / "SFT_Data" / "high_reward" / "skill_banks",
    ROOT / "SFT_Data" / "game_sft_balanced" / "skill_banks",
]


def _all_empty(checks: List[str]) -> bool:
    return all(not c for c in checks)


def inject_keyword_checks(bank_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """Generate step_checks from protocol step descriptions using keyword matching."""
    stats = {"total": 0, "injected": 0, "skipped_nongame": 0,
             "skipped_has_checks": 0, "skipped_no_steps": 0}

    task_name = bank_path.parent.name
    if task_name not in GAME_TASKS:
        stats["skipped_nongame"] = 1
        return stats

    records: List[Dict[str, Any]] = []
    with open(bank_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    stats["total"] = len(records)
    modified = False

    for rec in records:
        sk = rec.get("skill", rec)
        proto = sk.get("protocol")
        if not proto or not isinstance(proto, dict):
            stats["skipped_no_steps"] += 1
            continue

        existing = proto.get("step_checks", [])
        if existing and not _all_empty(existing):
            stats["skipped_has_checks"] += 1
            continue

        steps = proto.get("steps", [])
        if not steps:
            stats["skipped_no_steps"] += 1
            continue

        new_checks = generate_step_checks_from_effects(steps, game_name=task_name)
        if _all_empty(new_checks):
            stats["skipped_no_steps"] += 1
            continue

        proto["step_checks"] = new_checks

        req_effects = set()
        for check in new_checks:
            if check:
                key = check.split("=")[0].strip()
                if key:
                    req_effects.add(key)
        if req_effects:
            proto["required_effects"] = sorted(req_effects)

        stats["injected"] += 1
        modified = True

        sid = sk.get("skill_id", "?")
        logger.debug("  %s: steps=%s → checks=%s → required_effects=%s",
                      sid, steps, new_checks, sorted(req_effects))

    if modified and not dry_run:
        with open(bank_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank-dir", type=Path, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    bank_dirs = args.bank_dir if args.bank_dir else DEFAULT_BANK_DIRS

    total_stats = {"total": 0, "injected": 0, "skipped_nongame": 0,
                   "skipped_has_checks": 0, "skipped_no_steps": 0}

    for bdir in bank_dirs:
        if not bdir.exists():
            logger.warning("Bank dir not found: %s", bdir)
            continue
        for bank_file in sorted(bdir.rglob("skill_bank.jsonl")):
            try:
                label = str(bank_file.relative_to(ROOT))
            except ValueError:
                label = str(bank_file)
            logger.info("Processing %s", label)
            stats = inject_keyword_checks(bank_file, dry_run=args.dry_run)
            for k in total_stats:
                total_stats[k] += stats[k]
            if stats["injected"] > 0:
                tag = "(dry-run)" if args.dry_run else "(written)"
                logger.info("  -> injected %d/%d skills %s",
                            stats["injected"], stats["total"], tag)

    logger.info(
        "Done. total=%d injected=%d skipped_nongame=%d "
        "already_has_checks=%d no_steps=%d",
        total_stats["total"], total_stats["injected"],
        total_stats["skipped_nongame"], total_stats["skipped_has_checks"],
        total_stats["skipped_no_steps"],
    )


if __name__ == "__main__":
    main()
