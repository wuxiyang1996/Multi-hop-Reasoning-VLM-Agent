#!/usr/bin/env python
"""Collect decision SFT data from all sources, note gaps.

Collects action_taking.jsonl and skill_selection.jsonl for all 18 tasks.
Missing tasks are logged in a GAP_REPORT.json for later filling.

Output:
  frontier_data/output/decision_sft/
    <task>/action_taking.jsonl
    <task>/skill_selection.jsonl
    MANIFEST.json
    GAP_REPORT.json
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

ALL_TASKS = [
    "Temporal_Airstriker-v0", "Temporal_AlteredBeast-v0",
    "Temporal_CastleOfIllusion-v0", "Temporal_CastlevaniaBloodlines-v0",
    "Temporal_Columns-v0", "Temporal_DynamiteHeaddy-v0",
    "Temporal_GoldenAxe-v0", "Temporal_KidChameleon-v0",
    "Temporal_MortalKombatII-v0", "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0", "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
    "tetris", "super_mario", "candy_crush", "twenty_forty_eight",
    "browsergym", "osworld", "siv_bench", "tir_bench",
    "video_holmes", "visual_toolbench", "miniwob", "webshop",
]

SEARCH_DIRS = [
    DOWNLOAD_ROOT / "labeling" / "decision_sft_jsonl",
]


def find_latest_run(root: Path) -> Path | None:
    runs = sorted(root.glob("run_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def find_sft_files(task: str) -> dict[str, Path]:
    """Search all candidate locations for action_taking + skill_selection."""
    found: dict[str, Path] = {}
    for sdir in SEARCH_DIRS:
        run = find_latest_run(sdir)
        if run is None:
            continue
        tdir = run / task
        for fname in ("action_taking.jsonl", "skill_selection.jsonl"):
            f = tdir / fname
            if f.exists() and fname not in found:
                found[fname] = f
    return found


def main() -> int:
    out_root = REPO_ROOT / "frontier_data" / "output" / "decision_sft"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: OrderedDict[str, dict] = OrderedDict()
    gaps: list[dict] = []
    total_at = 0
    total_ss = 0

    for task in ALL_TASKS:
        files = find_sft_files(task)
        task_dir = out_root / task

        if files:
            task_dir.mkdir(parents=True, exist_ok=True)
            entry: dict = {"task": task}
            for fname, src in files.items():
                dst = task_dir / fname
                shutil.copy2(src, dst)
                with open(dst) as f:
                    n = sum(1 for _ in f)
                entry[fname.replace(".jsonl", "_rows")] = n
                if "action_taking" in fname:
                    total_at += n
                else:
                    total_ss += n
            manifest[task] = entry
            print(f"  [OK]   {task}: {entry}")
        else:
            gaps.append({
                "task": task,
                "missing": ["action_taking.jsonl", "skill_selection.jsonl"],
                "reason": "no decision SFT found in any source",
            })
            print(f"  [GAP]  {task}: no SFT data found")

    # Write outputs
    with open(out_root / "MANIFEST.json", "w") as f:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "total_action_taking_rows": total_at,
            "total_skill_selection_rows": total_ss,
            "n_tasks_with_data": len(manifest),
            "tasks": manifest,
        }, f, indent=2)

    with open(out_root / "GAP_REPORT.json", "w") as f:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_gaps": len(gaps),
            "gaps": gaps,
            "remediation": (
                "Run: python labeling/build_decision_sft_jsonl.py "
                "to fill game gaps.  For non-game tasks, run: "
                "python scripts/build_multimodal_decision_sft.py "
                "or the QA labeling pipeline first."
            ),
        }, f, indent=2)

    print(f"\n[DONE] {len(manifest)}/{len(ALL_TASKS)} tasks have SFT data "
          f"(AT={total_at}, SS={total_ss})")
    print(f"       {len(gaps)} tasks are missing data — see GAP_REPORT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
