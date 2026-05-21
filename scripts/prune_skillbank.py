"""One-shot skillbank pruner — drops dead / cross-game-translated skills.

Use after observing a coevo run where the live skillbank has grown
stale: skills with ``n_instances == 0`` after several outer steps,
seed skills whose ``protocol_raw.source_task`` doesn't match the
current game (i.e. translated from another game and never validated),
and skills whose ``predicate_success`` is the tautology
``state_observed=true`` (which makes any chain-reward check trivially
pass and gives the curator no discriminative signal).

Run example
-----------
::

    python3 scripts/prune_skillbank.py \\
        --bank runs/gymv_altered_beast_stage2_*/skillbank/gymv_altered_beast/skill_bank.jsonl \\
        --min-instances 1 \\
        --drop-cross-game \\
        --drop-tautology-predicates \\
        --dry-run

The ``--dry-run`` flag prints what would be removed without
modifying the file.  Without it, the original file is moved to
``<path>.bak-<timestamp>`` and a pruned copy takes its place.

Heuristic summary
-----------------
A skill is *kept* iff ALL of:

* ``n_instances >= --min-instances`` (default 1; i.e. drop strictly
  unused skills only — set to 0 to keep everything by default).
* If ``--drop-cross-game`` is set, the skill's ``protocol_raw.source_task``
  (when present) names the same game as inferred from the bank path,
  OR the skill has been validated (``n_instances >= 3`` AND
  ``overall_pass_rate > 0`` — i.e. it actually fires on this game).
* If ``--drop-tautology-predicates`` is set, ``predicate_success``
  is not exactly ``["state_observed=true"]`` (or empty).  Skills with
  richer predicates that *also* include ``state_observed=true`` are
  kept; only the pure-tautology case is dropped.

The script never modifies adapter weights and never touches the
seed bank — it operates exclusively on the live ``skill_bank.jsonl``
inside ``runs/<run>/skillbank/<game>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _infer_game_from_path(path: str) -> Optional[str]:
    """Walk the parent dirs looking for a ``gymv_*`` or known game name.

    Bank files live at::

        runs/<run>/skillbank/<game>/skill_bank.jsonl

    so the parent directory of ``skill_bank.jsonl`` is the game name.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent and parent != "skillbank":
        return parent
    return None


def _is_cross_game(skill: Dict[str, Any], target_game: str) -> bool:
    """Return True iff ``skill`` was translated from a different game."""
    if not target_game:
        return False
    pr = skill.get("protocol_raw") or {}
    src = pr.get("source_task")
    if not src:
        return False
    # Treat any non-target source as cross-game.  Compare via lowercased
    # substring so spellings like ``gymv_streets_of_rage_2`` vs
    # ``streets_of_rage_2`` both resolve correctly.
    src_l = str(src).lower()
    tgt_l = target_game.lower()
    return src_l != tgt_l and tgt_l not in src_l and src_l not in tgt_l


def _is_tautology_predicate(skill: Dict[str, Any]) -> bool:
    """Predicates that are vacuously satisfied → no discriminative power."""
    proto = skill.get("protocol") or {}
    preds = list(proto.get("predicate_success") or [])
    if not preds:
        # Pure-empty predicates still don't help the curator, but they
        # also don't actively contaminate scoring — keep them.
        return False
    norm = {p.strip().lower() for p in preds}
    return norm == {"state_observed=true"}


def _decision(
    entry: Dict[str, Any], *,
    target_game: str,
    min_instances: int,
    drop_cross_game: bool,
    drop_tautology: bool,
) -> Tuple[bool, str]:
    """Return (keep, reason). reason is informational only."""
    skill = entry.get("skill", entry)
    report = entry.get("report") or {}
    n_inst = int((skill.get("contract") or {}).get("n_instances", 0))
    pass_rate = float(report.get("overall_pass_rate") or 0.0)

    if n_inst < min_instances:
        return (False, f"n_instances={n_inst} < {min_instances}")

    if drop_cross_game and _is_cross_game(skill, target_game):
        # Validated cross-game skill (used at least 3x with >0 pass rate)
        # gets a pass — we don't want to nuke a skill that *does* transfer.
        if n_inst >= 3 and pass_rate > 0.0:
            pass
        else:
            return (
                False,
                f"cross-game from "
                f"{((skill.get('protocol_raw') or {}).get('source_task')) or '?'}",
            )

    if drop_tautology and _is_tautology_predicate(skill):
        return (False, "tautology predicate_success=[state_observed=true]")

    return (True, "kept")


def prune_one(
    path: str, *,
    min_instances: int,
    drop_cross_game: bool,
    drop_tautology: bool,
    dry_run: bool,
    inferred_game: Optional[str] = None,
) -> Dict[str, Any]:
    target_game = inferred_game or _infer_game_from_path(path) or ""
    with open(path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    kept: List[Dict[str, Any]] = []
    dropped: List[Tuple[str, str]] = []
    for entry in entries:
        skill = entry.get("skill", entry)
        sid = skill.get("skill_id", "?")
        keep, reason = _decision(
            entry,
            target_game=target_game,
            min_instances=min_instances,
            drop_cross_game=drop_cross_game,
            drop_tautology=drop_tautology,
        )
        if keep:
            kept.append(entry)
        else:
            dropped.append((sid, reason))

    summary = {
        "path": path,
        "target_game": target_game,
        "n_before": len(entries),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped": dropped,
    }

    if not dry_run and dropped:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{path}.bak-{ts}"
        shutil.copy2(path, backup)
        summary["backup"] = backup
        with open(path, "w") as f:
            for entry in kept:
                f.write(json.dumps(entry, default=str) + "\n")

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank", nargs="+", required=True,
        help="Path(s) to skill_bank.jsonl. Globs are allowed by the shell.",
    )
    p.add_argument(
        "--min-instances", type=int, default=1,
        help="Drop skills with strictly fewer instances than this "
             "(default: 1 — drop strictly unused skills only).",
    )
    p.add_argument(
        "--drop-cross-game", action="store_true",
        help="Drop skills whose protocol_raw.source_task names a "
             "different game, unless they have been validated "
             "(n_inst≥3 & pass_rate>0) on the current game.",
    )
    p.add_argument(
        "--drop-tautology-predicates", action="store_true",
        help='Drop skills whose predicate_success is exactly '
             '["state_observed=true"] (no discriminative power).',
    )
    p.add_argument(
        "--game", default=None,
        help="Override the target game name (otherwise inferred from path).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Don't modify files; just print what would change.",
    )
    args = p.parse_args(argv)

    total_summary = {"banks": [], "n_before": 0, "n_kept": 0, "n_dropped": 0}
    for path in args.bank:
        if not os.path.isfile(path):
            print(f"[skip] not a file: {path}", file=sys.stderr)
            continue
        s = prune_one(
            path,
            min_instances=args.min_instances,
            drop_cross_game=args.drop_cross_game,
            drop_tautology=args.drop_tautology_predicates,
            dry_run=args.dry_run,
            inferred_game=args.game,
        )
        total_summary["banks"].append(s)
        total_summary["n_before"] += s["n_before"]
        total_summary["n_kept"] += s["n_kept"]
        total_summary["n_dropped"] += s["n_dropped"]

        print(
            f"\n=== {s['path']} ===\n"
            f"  target_game={s['target_game'] or '(unknown)'}\n"
            f"  before={s['n_before']}  kept={s['n_kept']}  "
            f"dropped={s['n_dropped']}"
            + (f"  (dry-run)" if args.dry_run else
               f"  backup={s.get('backup','<no changes>')}")
        )
        for sid, reason in s["dropped"]:
            print(f"    - drop {sid:35s}  reason={reason}")

    print(
        f"\nTotal across {len(total_summary['banks'])} bank(s): "
        f"before={total_summary['n_before']}  "
        f"kept={total_summary['n_kept']}  "
        f"dropped={total_summary['n_dropped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
