#!/usr/bin/env python3
"""Populate empty step_checks in non-game skill banks using OPERATOR_TO_EFFECT.

For each skill whose ``protocol.step_checks`` are all empty, reads the
``template_signature`` (e.g. ``PERCEIVE → COMPARE → FILTER → DECIDE → VERIFY``)
and generates deterministic check predicates so ``_SkillTracker`` can give
meaningful intrinsic rewards based on 9B reasoning output.

Usage::

    python frontier_data/scripts/inject_step_checks.py
    python frontier_data/scripts/inject_step_checks.py --dry-run
    python frontier_data/scripts/inject_step_checks.py --bank-dir /path/to/banks
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

from decision_agents.protocol_utils import build_step_checks_from_signature

logger = logging.getLogger("inject_step_checks")

GAME_TASKS = frozenset({
    "tetris", "candy_crush", "super_mario", "twenty_forty_eight",
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


def inject_bank(bank_path: Path, dry_run: bool = False) -> Dict[str, int]:
    stats = {"total": 0, "injected": 0, "skipped_game": 0,
             "skipped_has_checks": 0, "skipped_no_sig": 0}

    task_name = bank_path.parent.name
    if task_name in GAME_TASKS:
        logger.info("Skipping game task: %s", task_name)
        stats["skipped_game"] = 1
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
            stats["skipped_no_sig"] += 1
            continue

        existing_checks = proto.get("step_checks", [])
        if existing_checks and not _all_empty(existing_checks):
            stats["skipped_has_checks"] += 1
            continue

        sig = sk.get("template_signature", "")
        vocab = proto.get("action_vocab", [])
        n_steps = len(proto.get("steps", []))

        if not sig and not vocab:
            stats["skipped_no_sig"] += 1
            continue

        new_checks = build_step_checks_from_signature(sig, vocab, n_steps)
        if _all_empty(new_checks):
            stats["skipped_no_sig"] += 1
            continue

        proto["step_checks"] = new_checks
        stats["injected"] += 1
        modified = True

        sid = sk.get("skill_id", "?")
        logger.debug("  %s: %s → %s", sid, sig, new_checks)

    if modified and not dry_run:
        with open(bank_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    bank_dirs = args.bank_dir if args.bank_dir else DEFAULT_BANK_DIRS

    total_stats = {"total": 0, "injected": 0, "skipped_game": 0,
                   "skipped_has_checks": 0, "skipped_no_sig": 0}

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
            stats = inject_bank(bank_file, dry_run=args.dry_run)
            for k in total_stats:
                total_stats[k] += stats[k]
            if stats["injected"] > 0:
                tag = "(dry-run)" if args.dry_run else "(written)"
                logger.info("  → injected %d/%d skills %s",
                            stats["injected"], stats["total"], tag)

    logger.info(
        "Done. total=%d injected=%d skipped_game=%d "
        "already_has_checks=%d no_signature=%d",
        total_stats["total"], total_stats["injected"],
        total_stats["skipped_game"], total_stats["skipped_has_checks"],
        total_stats["skipped_no_sig"],
    )


if __name__ == "__main__":
    main()
