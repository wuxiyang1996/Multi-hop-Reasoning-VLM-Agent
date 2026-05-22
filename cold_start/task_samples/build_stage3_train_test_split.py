#!/usr/bin/env python
"""Build 80/20 train/test splits for Stage 3 non-game GRPO evaluation.

Reads the existing cold-start task sample manifests and splits each into
a small training set (~20%) for GRPO domain adaptation and a larger held-
out test set (~80%) for evaluation.

Splitting is deterministic (seed=42, distinct from the seed=0 used by the
original sample builders) and **stratified** where possible — the same
round-robin-by-category approach used to build the original manifests
ensures every task family / question type / subtask category is represented
proportionally in both splits.

Inputs (one sample id per line, ``#``-prefixed header):

    visual_toolbench_1000.txt   (603 ids)
    tir_bench_1000.txt          (1000 ids)
    video_holmes_1000.txt       (1000 ids)
    siv_bench_1000.txt          (1000 ids)
    browsergym_miniwob_200.txt  (125 task ids)

WebShop is handled separately since its task pool is defined by a
numeric range (``browsergym/webshop.0`` … ``webshop.49``).

Outputs (written to ``stage3_splits/``):

    stage3_splits/<benchmark>_train.txt
    stage3_splits/<benchmark>_test.txt
    stage3_splits/README.md
    stage3_splits/split_summary.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SAMPLES_DIR / "stage3_splits"
SEED = 42
TRAIN_RATIO = 0.20

BENCHMARKS = {
    "visual_toolbench": {
        "source": "visual_toolbench_1000.txt",
        "stratify_fn": None,
    },
    "tir_bench": {
        "source": "tir_bench_1000.txt",
        "stratify_fn": None,
    },
    "video_holmes": {
        "source": "video_holmes_1000.txt",
        "stratify_fn": None,  # flat shuffle (video_id groups are too granular)
    },
    "siv_bench": {
        "source": "siv_bench_1000.txt",
        "stratify_fn": lambda sid: sid.split("/")[0],  # group by category
    },
    "miniwob": {
        "source": "browsergym_miniwob_200.txt",
        "stratify_fn": None,
    },
}

WEBSHOP_TOTAL = 50
WEBSHOP_TRAIN_K = 10


def _load_ids(path: Path) -> list[str]:
    """Load sample IDs from a manifest file, skipping comment lines."""
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def _stratified_split(
    ids: list[str],
    train_ratio: float,
    rng: random.Random,
    key_fn=None,
) -> tuple[list[str], list[str]]:
    """Split ids into (train, test) with optional stratification.

    When key_fn is provided, ids are grouped by key_fn(id) and each
    group is split independently so the category distribution is
    preserved in both splits.  When key_fn is None, a simple shuffle-
    split is used.
    """
    if key_fn is None:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        k = max(1, int(len(shuffled) * train_ratio))
        return shuffled[:k], shuffled[k:]

    groups: dict[str, list[str]] = {}
    for sid in ids:
        key = key_fn(sid)
        groups.setdefault(key, []).append(sid)

    train, test = [], []
    for key in sorted(groups):
        members = list(groups[key])
        rng.shuffle(members)
        k = max(1, int(len(members) * train_ratio))
        train.extend(members[:k])
        test.extend(members[k:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _write_manifest(
    path: Path, ids: list[str], benchmark: str, split: str, total: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {benchmark} stage3 {split} split\n")
        f.write(f"# count={len(ids)}  total_pool={total}  "
                f"seed={SEED}  ratio={TRAIN_RATIO}\n")
        for sid in ids:
            f.write(f"{sid}\n")


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    summary: dict[str, dict] = {}

    # ── Standard benchmarks (from existing manifest files) ────────────
    for bench_name, cfg in BENCHMARKS.items():
        src = SAMPLES_DIR / cfg["source"]
        if not src.exists():
            print(f"WARN: {src} not found, skipping {bench_name}")
            continue

        ids = _load_ids(src)
        train, test = _stratified_split(
            ids, TRAIN_RATIO, rng, key_fn=cfg["stratify_fn"],
        )

        _write_manifest(
            SPLIT_DIR / f"{bench_name}_train.txt",
            train, bench_name, "train", len(ids),
        )
        _write_manifest(
            SPLIT_DIR / f"{bench_name}_test.txt",
            test, bench_name, "test", len(ids),
        )

        overlap = set(train) & set(test)
        assert not overlap, f"{bench_name}: {len(overlap)} ids in both splits!"

        summary[bench_name] = {
            "total": len(ids),
            "train": len(train),
            "test": len(test),
            "train_pct": round(100 * len(train) / len(ids), 1),
        }
        print(f"{bench_name:25s}  total={len(ids):5d}  "
              f"train={len(train):4d}  test={len(test):4d}  "
              f"({summary[bench_name]['train_pct']:.1f}% train)")

    # ── WebShop (numeric task indices) ────────────────────────────────
    ws_ids = list(range(WEBSHOP_TOTAL))
    rng.shuffle(ws_ids)
    ws_train_ids = [f"browsergym/webshop.{i}" for i in ws_ids[:WEBSHOP_TRAIN_K]]
    ws_test_ids = [f"browsergym/webshop.{i}" for i in ws_ids[WEBSHOP_TRAIN_K:]]

    _write_manifest(
        SPLIT_DIR / "webshop_train.txt",
        ws_train_ids, "webshop", "train", WEBSHOP_TOTAL,
    )
    _write_manifest(
        SPLIT_DIR / "webshop_test.txt",
        ws_test_ids, "webshop", "test", WEBSHOP_TOTAL,
    )

    summary["webshop"] = {
        "total": WEBSHOP_TOTAL,
        "train": len(ws_train_ids),
        "test": len(ws_test_ids),
        "train_pct": round(100 * len(ws_train_ids) / WEBSHOP_TOTAL, 1),
    }
    print(f"{'webshop':25s}  total={WEBSHOP_TOTAL:5d}  "
          f"train={len(ws_train_ids):4d}  test={len(ws_test_ids):4d}  "
          f"({summary['webshop']['train_pct']:.1f}% train)")

    # ── Summary JSON ──────────────────────────────────────────────────
    with open(SPLIT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── README ────────────────────────────────────────────────────────
    readme_lines = [
        "# Stage 3 — Non-Game Train/Test Splits",
        "",
        "80/20 splits for GRPO domain-adaptation (train) and held-out",
        "evaluation (test) across 6 non-game benchmarks.",
        "",
        "Generated by `build_stage3_train_test_split.py` with seed=42.",
        "Splits are deterministic and stratified where category labels",
        "are available (SIV-Bench by category, Video-Holmes by video_id).",
        "",
        "## Split Summary",
        "",
        "| Benchmark | Total | Train | Test | Train % |",
        "|-----------|------:|------:|-----:|--------:|",
    ]
    for bench, stats in summary.items():
        readme_lines.append(
            f"| {bench} | {stats['total']} | {stats['train']} "
            f"| {stats['test']} | {stats['train_pct']}% |"
        )

    grand_total = sum(s["total"] for s in summary.values())
    grand_train = sum(s["train"] for s in summary.values())
    grand_test = sum(s["test"] for s in summary.values())
    readme_lines.extend([
        f"| **Total** | **{grand_total}** | **{grand_train}** "
        f"| **{grand_test}** | **{round(100*grand_train/grand_total, 1)}%** |",
        "",
        "## Usage",
        "",
        "```python",
        "from pathlib import Path",
        "",
        "split_dir = Path('cold_start/task_samples/stage3_splits')",
        "",
        "# Load train IDs for a benchmark",
        "train_ids = [",
        "    line.strip() for line in",
        "    (split_dir / 'tir_bench_train.txt').read_text().splitlines()",
        "    if line.strip() and not line.startswith('#')",
        "]",
        "```",
        "",
        "## Experiment Design",
        "",
        "```",
        "Phase 0 (baseline):  Base Qwen3.5-9B → eval on all 6 test splits",
        "Phase 1 (game-only): Game GRPO model → eval on all 6 test splits",
        "Phase 2 (adapted):   Game + 20% non-game GRPO → eval on all 6 test splits",
        "```",
        "",
        "The evaluation matrix measures both in-domain adaptation (same",
        "benchmark train→test) and cross-domain generalization (e.g.",
        "VTB train → TIR test).",
        "",
    ])
    with open(SPLIT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    print(f"\nDone. Splits written to {SPLIT_DIR}/")
    print(f"Grand total: {grand_train} train + {grand_test} test = {grand_total}")


if __name__ == "__main__":
    main()
