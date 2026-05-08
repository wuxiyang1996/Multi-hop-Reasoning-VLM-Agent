#!/usr/bin/env python3
"""Promote the best step's adapters + per-game skill bank into the
LIVE run directory at a phase boundary.

Why this exists
---------------
GRPO's async pipeline produces oscillating reward — the FINAL step of a
phase is often NOT the peak.  When the curriculum advances Phase N → N+1
the ``run_coevolution.py --resume`` reload picks up the LIVE adapter
files (which are whatever the last optimizer step wrote) plus the LIVE
per-game ``skill_bank.jsonl``.  That throws away earlier, better policy
states.

This script atomically swaps the LIVE state to the *best* checkpoint
within a completed phase, before ``phase1_finalize.py`` runs its
crafter-v2 + cross-game translation pipeline (so the translator works
from the bank that pairs with the best LoRA, not a stale "last step"
bank).

Selection metric (configurable)
-------------------------------
Default: 3-step centered rolling mean of ``reward_per_game[<game>].
mean_reward`` from ``step_log.jsonl``.  Rolling smoothing avoids
single-spike outliers (bf16 sampling variance can swing ±100 reward
between adjacent steps in async GRPO).  Ties broken by LATER step
(more training accumulated).

What gets copied
----------------
LIVE ← ``checkpoints/step_<best>/`` (atomically, with backup):

* ``lora_adapters/decision/{action_taking,skill_selection}``
* ``lora_adapters/skillbank/{segment,contract,curator}``
* ``skillbank/<source_game>/skill_bank.jsonl``

A backup of the current LIVE state goes to
``runs/<run>/.live_backup_pre_promote_<phase>_<ts>/`` so a botched
promotion can be rolled back.

A marker file ``runs/<run>/promotion_log.jsonl`` records every
promotion (idempotent re-runs append a fresh entry).

Usage::

    python scripts/promote_best_checkpoint.py \\
        --run-dir runs/Qwen3.5-9B_20260507_192810 \\
        --phase-num 1 --source-game gymv_thunder_force_iii \\
        --window 3 --metric mean_reward

Set ``--dry-run`` to print the decision without touching files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_step_log(run_dir: Path, game: str) -> List[Tuple[int, Dict[str, Any]]]:
    """Return ``[(step, reward_dict), ...]`` for the given game in step
    order.  Skips lines that aren't valid JSON or don't carry the
    target game's reward block."""
    f = run_dir / "step_log.jsonl"
    if not f.exists():
        return []
    out: List[Tuple[int, Dict[str, Any]]] = []
    with open(f) as fh:
        for L in fh:
            try:
                d = json.loads(L)
            except Exception:
                continue
            rg = (d.get("reward_per_game") or {}).get(game)
            if rg is None:
                continue
            out.append((int(d.get("step", -1)), rg))
    out.sort(key=lambda r: r[0])
    return out


def find_best_step(
    rows: List[Tuple[int, Dict[str, Any]]],
    *, window: int = 3, metric: str = "mean_reward",
) -> Optional[Dict[str, Any]]:
    """Pick the step that maximizes the centered rolling-``window`` mean
    of ``metric`` (mean_reward by default).  Falls back to raw metric
    if rows < window.  Returns ``None`` when no rows."""
    if not rows:
        return None
    vals = [(s, float(r.get(metric, 0.0) or 0.0), r) for s, r in rows]
    n = len(vals)
    if n < window:
        # Fallback to raw — not enough data for smoothing.
        best = max(vals, key=lambda x: (x[1], x[0]))
        return {"step": best[0], "metric_value": best[1], "smoothed": False, "raw": best[2]}
    half = window // 2
    smoothed: List[Tuple[int, float, Dict[str, Any]]] = []
    for i, (s, v, r) in enumerate(vals):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = [vals[j][1] for j in range(lo, hi)]
        smoothed.append((s, sum(win) / len(win), r))
    # Tie-break: later step wins (more training).
    best = max(smoothed, key=lambda x: (x[1], x[0]))
    return {
        "step": best[0],
        "metric_value": best[1],
        "raw_metric_value": float(vals[[v[0] for v in vals].index(best[0])][1]),
        "smoothed": True,
        "window": window,
    }


def copytree_overwrite(src: Path, dst: Path) -> None:
    """``shutil.copytree`` that overwrites existing dst (we use
    ``dirs_exist_ok=True`` because LIVE adapter dirs always exist)."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Walk and overwrite each file — preserves any extra files
        # the LIVE adapter dir might have (rare, but safe).
        for r, _, files in __import__("os").walk(src):
            rel = Path(r).relative_to(src)
            (dst / rel).mkdir(parents=True, exist_ok=True)
            for fn in files:
                shutil.copy2(Path(r) / fn, dst / rel / fn)
    else:
        shutil.copytree(src, dst)


def backup_live(run_dir: Path, phase_num: int, source_game: str) -> Path:
    """Snapshot LIVE adapters + per-game bank to a timestamped
    backup dir.  Used by ``promote_step``."""
    ts = int(time.time())
    bk = run_dir / f".live_backup_pre_promote_p{phase_num:02d}_{ts}"
    bk.mkdir(parents=True, exist_ok=False)
    live_lora = run_dir / "lora_adapters"
    live_bank = run_dir / "skillbank" / source_game
    if live_lora.exists():
        shutil.copytree(live_lora, bk / "lora_adapters")
    if live_bank.exists():
        shutil.copytree(live_bank, bk / "skillbank" / source_game)
    return bk


def promote_step(
    run_dir: Path, step: int, source_game: str, *,
    promote_bank: bool, phase_num: int,
) -> Dict[str, Any]:
    """Copy ``checkpoints/step_<step>/`` adapters + bank → LIVE.
    Returns a metadata dict suitable for promotion_log.jsonl."""
    ckpt = run_dir / "checkpoints" / f"step_{step:04d}"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    backup_dir = backup_live(run_dir, phase_num, source_game)

    src_decision = ckpt / "adapters" / "decision"
    src_skillbank_lora = ckpt / "adapters" / "skillbank"
    dst_decision = run_dir / "lora_adapters" / "decision"
    dst_skillbank_lora = run_dir / "lora_adapters" / "skillbank"

    promoted: Dict[str, Any] = {
        "phase_num": phase_num,
        "source_game": source_game,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": step,
        "ckpt_dir": str(ckpt),
        "backup_dir": str(backup_dir),
        "copied": [],
        "promoted_bank": False,
    }

    if src_decision.is_dir():
        copytree_overwrite(src_decision, dst_decision)
        promoted["copied"].append(str(dst_decision.relative_to(run_dir)))

    if src_skillbank_lora.is_dir():
        copytree_overwrite(src_skillbank_lora, dst_skillbank_lora)
        promoted["copied"].append(str(dst_skillbank_lora.relative_to(run_dir)))

    if promote_bank:
        src_bank = ckpt / "banks" / source_game / "skill_bank.jsonl"
        dst_bank = run_dir / "skillbank" / source_game / "skill_bank.jsonl"
        if src_bank.is_file():
            dst_bank.parent.mkdir(parents=True, exist_ok=True)
            # Atomic via rename through a tmp path.
            tmp = dst_bank.with_suffix(dst_bank.suffix + ".tmp")
            shutil.copy2(src_bank, tmp)
            tmp.replace(dst_bank)
            promoted["promoted_bank"] = True
            promoted["copied"].append(str(dst_bank.relative_to(run_dir)))
            promoted["bank_n_skills"] = sum(1 for _ in open(dst_bank))
        else:
            promoted["bank_warning"] = f"missing source bank in checkpoint: {src_bank}"
    return promoted


def append_promotion_log(run_dir: Path, entry: Dict[str, Any]) -> Path:
    log = run_dir / "promotion_log.jsonl"
    with open(log, "a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase-num", type=int, required=True)
    ap.add_argument("--source-game", required=True)
    ap.add_argument("--window", type=int, default=3,
                    help="Centered rolling mean window for smoothing (default 3).")
    ap.add_argument("--metric", default="mean_reward",
                    help="Field on reward_per_game[<game>] to maximize.")
    ap.add_argument("--explicit-step", type=int, default=-1,
                    help="If set, promote this step; ignore selection metric.")
    ap.add_argument("--no-promote-bank", action="store_true",
                    help="Promote adapters only (LIVE bank stays — useful "
                         "when later-step skills must be preserved).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run-dir not found: {run_dir}")
        return 2

    rows = load_step_log(run_dir, args.source_game)
    if not rows:
        print(f"ERROR: no step_log rows for game={args.source_game}")
        return 3

    if args.explicit_step >= 0:
        chosen = {"step": args.explicit_step, "metric_value": None,
                  "smoothed": False, "explicit": True}
        print(f"explicit step requested: {args.explicit_step}")
    else:
        chosen = find_best_step(rows, window=args.window, metric=args.metric)
        if not chosen:
            print(f"ERROR: cannot select best step")
            return 4
        print(f"selected best step: {chosen['step']} "
              f"(smoothed-mean={chosen['metric_value']:.0f}, "
              f"raw={chosen.get('raw_metric_value', '?')})")

    last_step = rows[-1][0]
    chosen["last_step"] = last_step
    chosen["delta_steps_to_last"] = last_step - chosen["step"]
    print(f"   last step in phase = {last_step}; promoting back by "
          f"{chosen['delta_steps_to_last']} step(s)")

    if args.dry_run:
        print("\n--dry-run set; not touching files")
        print(json.dumps(chosen, indent=2, default=str))
        return 0

    # No-op shortcut: if best == last, the LIVE state already matches.
    if chosen["delta_steps_to_last"] == 0:
        print("   best == last; LIVE state already correct.  Recording no-op promotion.")
        entry = {**chosen, "phase_num": args.phase_num,
                 "source_game": args.source_game, "noop": True,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        append_promotion_log(run_dir, entry)
        return 0

    promoted = promote_step(
        run_dir, chosen["step"], args.source_game,
        promote_bank=not args.no_promote_bank,
        phase_num=args.phase_num,
    )
    promoted["selection"] = chosen
    print(f"   PROMOTED step={chosen['step']}  bank={promoted['promoted_bank']}")
    print(f"   backup at: {promoted['backup_dir']}")
    print(f"   copied: {promoted['copied']}")

    log = append_promotion_log(run_dir, promoted)
    print(f"   appended {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
