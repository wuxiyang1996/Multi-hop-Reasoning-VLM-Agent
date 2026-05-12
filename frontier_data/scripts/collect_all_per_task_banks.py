#!/usr/bin/env python
"""Collect per-task skill banks from ALL sources into a unified layout.

Reads from:
  1. labeling/skill_bank_out/<run>/env_wrappers/<game>/skill_bank.jsonl  (4 games)
  2. labeling/skill_bank_out/<run>/gym_v/<game>/skill_bank.jsonl         (13 games)
  3. skill_transfer_test/skill_bank_local/full_v5/<corpus>/archetype/skill_bank.jsonl
     or per_episode/skill_bank.jsonl                                     (6 non-game)

Writes a unified per_task_banks/ directory:
  frontier_data/output/per_task_banks/
    <task_name>/skill_bank.jsonl    (one per task)
    MANIFEST.json                   (task → count summary)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOWNLOAD_ROOT = Path(os.environ.get(
    "DOWNLOAD_ROOT",
    "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/main_project",
))

GYMV_GAMES = [
    "Temporal_Airstriker-v0", "Temporal_AlteredBeast-v0",
    "Temporal_CastleOfIllusion-v0", "Temporal_CastlevaniaBloodlines-v0",
    "Temporal_Columns-v0", "Temporal_DynamiteHeaddy-v0",
    "Temporal_GoldenAxe-v0", "Temporal_KidChameleon-v0",
    "Temporal_MortalKombatII-v0", "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0", "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
]
ENVW_GAMES = ["tetris", "super_mario", "candy_crush", "twenty_forty_eight"]
NONGAME_CORPORA = ["browsergym", "osworld", "siv_bench", "tir_bench",
                    "video_holmes", "visual_toolbench"]


def find_latest_run(root: Path) -> Path | None:
    """Return the newest run_* directory under *root*."""
    runs = sorted(root.glob("run_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def copy_bank(src: Path, dst_dir: Path, task: str) -> int:
    """Copy a skill_bank.jsonl into the unified layout, return line count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "skill_bank.jsonl"
    shutil.copy2(src, dst)
    with open(dst) as f:
        return sum(1 for _ in f)


def main() -> int:
    out_root = REPO_ROOT / "frontier_data" / "output" / "per_task_banks"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: OrderedDict[str, dict] = OrderedDict()
    total = 0

    # ── Source 1+2: labeling/skill_bank_out ─────────────────────────
    sbo_root = DOWNLOAD_ROOT / "labeling" / "skill_bank_out"
    sbo_run = find_latest_run(sbo_root)
    if sbo_run is None:
        print(f"[WARN] No run_* under {sbo_root}", file=sys.stderr)
    else:
        print(f"[INFO] Using labeling skill_bank_out: {sbo_run}")
        for corpus, games in [("env_wrappers", ENVW_GAMES), ("gym_v", GYMV_GAMES)]:
            for game in games:
                src = sbo_run / corpus / game / "skill_bank.jsonl"
                if not src.exists():
                    src2 = sbo_run / corpus / game / "reports" / "skill_bank.jsonl"
                    if src2.exists():
                        src = src2
                if src.exists():
                    n = copy_bank(src, out_root / game, game)
                    manifest[game] = {
                        "corpus": corpus, "source": str(src), "n_skills": n,
                    }
                    total += n
                    print(f"  {corpus}/{game}: {n} skills")
                else:
                    print(f"  [MISS] {corpus}/{game}: no skill_bank.jsonl at {src}")

    # ── Source 3: skill_transfer_test/skill_bank_local/full_v5 ──────
    stl_root = DOWNLOAD_ROOT / "skill_transfer_test" / "skill_bank_local" / "full_v5"
    if stl_root.is_dir():
        print(f"[INFO] Using skill_transfer_test full_v5: {stl_root}")
        for corpus in NONGAME_CORPORA:
            cdir = stl_root / corpus
            if not cdir.is_dir():
                print(f"  [MISS] {corpus}: not found")
                continue
            src = cdir / "archetype" / "skill_bank.jsonl"
            if not src.exists():
                src = cdir / "per_episode" / "skill_bank.jsonl"
            if not src.exists():
                src = cdir / "skill_bank.jsonl"
            if src.exists():
                n = copy_bank(src, out_root / corpus, corpus)
                manifest[corpus] = {
                    "corpus": "cross_domain", "source": str(src), "n_skills": n,
                }
                total += n
                print(f"  cross_domain/{corpus}: {n} skills")
            else:
                print(f"  [MISS] cross_domain/{corpus}: no skill_bank.jsonl")
    else:
        print(f"[WARN] No full_v5 at {stl_root}", file=sys.stderr)

    # ── Write manifest ──────────────────────────────────────────────
    manifest_out = out_root / "MANIFEST.json"
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_skills": total,
        "n_tasks": len(manifest),
        "tasks": manifest,
    }
    with open(manifest_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {total} skills across {len(manifest)} tasks → {out_root}")
    print(f"       Manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
