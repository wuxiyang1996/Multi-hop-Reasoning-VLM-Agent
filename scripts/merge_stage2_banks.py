"""Merge best Stage 2 skill-bank snapshots with Stage 1 per-task banks.

For each of the 4 Stage 2 games, this script:
1. Loads the best checkpoint skill bank (chosen by reward).
2. Loads the Stage 1 per-task bank for the same game.
3. Clusters near-duplicate skills (by normalised name similarity).
4. Keeps the Stage 2 version when duplicates exist (it was GRPO-refined).
5. Writes a merged bank to the output directory.

Usage:
    python scripts/merge_stage2_banks.py [--output-dir DIR] [--dry-run]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BEST_CHECKPOINTS = {
    "gymv_airstriker": {
        "run": "gymv_airstriker_stage2_v9_20260521_180740",
        "step": "step_0004",
        "reason": "closest to best-reward step 3 (77.5)",
    },
    "gymv_altered_beast": {
        "run": "gymv_altered_beast_stage2_v9_20260521_214938",
        "step": "step_0003",
        "reason": "best resumed step (375.0, 24 skills)",
    },
    "gymv_dynamite_headdy": {
        "run": "gymv_dynamite_headdy_stage2_20260520_094617",
        "step": "step_0004",
        "reason": "max skill count (23); reward flat at 100",
    },
    "gymv_space_harrier_ii": {
        "run": "gymv_space_harrier_ii_stage2_20260520_094617",
        "step": "step_0009",
        "reason": "closest to best-reward step 8 (89558)",
    },
}

STAGE1_BANKS_DIR = REPO / "frontier_data" / "output" / "per_task_banks"
DEFAULT_OUTPUT = REPO / "frontier_data" / "output" / "stage2_merged_banks"

DEDUP_THRESHOLD = 0.70


def _normalise(name: str) -> str:
    """Lower-case, strip prefixes like 'seed.', 'early:', 'late:', 'mid:',
    collapse punctuation."""
    name = name.lower().strip()
    name = re.sub(r"^(seed\.|bootstrap\.)", "", name)
    name = re.sub(r"^(early|mid|late):", "", name)
    name = re.sub(r"[_/]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def load_bank(path: Path) -> list[dict]:
    skills = []
    if not path.exists():
        return skills
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                skills.append(json.loads(line))
    return skills


def get_skill_id(entry: dict) -> str:
    sk = entry.get("skill", entry)
    if isinstance(sk, str):
        sk = json.loads(sk)
    return sk.get("skill_id", "")


def get_skill_name(entry: dict) -> str:
    sk = entry.get("skill", entry)
    if isinstance(sk, str):
        sk = json.loads(sk)
    return sk.get("name", sk.get("skill_id", ""))


def get_skill_obj(entry: dict) -> dict:
    sk = entry.get("skill", entry)
    if isinstance(sk, str):
        sk = json.loads(sk)
    return sk


def cluster_and_merge(stage2_bank: list[dict],
                      stage1_bank: list[dict],
                      threshold: float = DEDUP_THRESHOLD) -> list[dict]:
    """Merge two banks. Stage 2 wins on duplicates."""
    merged = list(stage2_bank)
    s2_ids = {get_skill_id(e) for e in merged}
    s2_names = [get_skill_name(e) for e in merged]

    added = 0
    skipped = 0
    for s1_entry in stage1_bank:
        s1_id = get_skill_id(s1_entry)
        s1_name = get_skill_name(s1_entry)

        if s1_id in s2_ids:
            skipped += 1
            continue

        is_dup = False
        for s2_name in s2_names:
            if _similarity(s1_name, s2_name) >= threshold:
                is_dup = True
                break

        if is_dup:
            skipped += 1
        else:
            sk = get_skill_obj(s1_entry)
            tags = sk.get("tags", [])
            if "stage1_origin" not in tags:
                tags.append("stage1_origin")
                sk["tags"] = tags
            merged.append(s1_entry)
            s2_names.append(s1_name)
            s2_ids.add(s1_id)
            added += 1

    return merged, added, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str,
                        default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files")
    parser.add_argument("--threshold", type=float, default=DEDUP_THRESHOLD,
                        help=f"Name similarity threshold for dedup (default {DEDUP_THRESHOLD})")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("Stage 2 Best-Checkpoint → Stage 1 Merge")
    print("=" * 60)

    summary = {}

    for game, cfg in BEST_CHECKPOINTS.items():
        print(f"\n{'─' * 60}")
        print(f"  {game}")
        print(f"  run:  {cfg['run']}")
        print(f"  step: {cfg['step']}  ({cfg['reason']})")
        print(f"{'─' * 60}")

        s2_path = (REPO / "runs" / cfg["run"] / "checkpoints" /
                   cfg["step"] / "banks" / game / "skill_bank.jsonl")
        s1_path = STAGE1_BANKS_DIR / game / "skill_bank.jsonl"

        s2_bank = load_bank(s2_path)
        s1_bank = load_bank(s1_path)
        print(f"  Stage 2 bank: {len(s2_bank)} skills  ({s2_path.name})")
        print(f"  Stage 1 bank: {len(s1_bank)} skills  ({s1_path})")

        if not s2_bank:
            print(f"  ⚠ No Stage 2 bank found at {s2_path}")
            continue

        merged, added, skipped = cluster_and_merge(
            s2_bank, s1_bank, threshold=args.threshold)

        print(f"\n  Merge result:")
        print(f"    Stage 2 kept:   {len(s2_bank)}")
        print(f"    Stage 1 added:  {added}")
        print(f"    Stage 1 dedup:  {skipped}")
        print(f"    Total merged:   {len(merged)}")

        print(f"\n  Skills in merged bank:")
        for i, entry in enumerate(merged):
            sid = get_skill_id(entry)
            sname = get_skill_name(entry)
            conf = get_skill_obj(entry).get("confidence_tag", "?")
            tags = get_skill_obj(entry).get("tags", [])
            origin = "s1" if "stage1_origin" in tags else "s2"
            print(f"    [{origin}] {sid:45s} ({conf})")

        summary[game] = {
            "stage2_skills": len(s2_bank),
            "stage1_skills": len(s1_bank),
            "stage1_added": added,
            "stage1_deduped": skipped,
            "merged_total": len(merged),
        }

        if not args.dry_run:
            game_out = out_dir / game
            game_out.mkdir(parents=True, exist_ok=True)
            bank_path = game_out / "skill_bank.jsonl"
            with open(bank_path, "w") as f:
                for entry in merged:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"\n  → Written to {bank_path}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for game, s in summary.items():
        print(f"  {game:30s}  s2={s['stage2_skills']:2d} + s1_new={s['stage1_added']:2d}"
              f" (dedup={s['stage1_deduped']:2d}) = {s['merged_total']:2d} total")

    if not args.dry_run:
        summary_path = out_dir / "MERGE_SUMMARY.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary → {summary_path}")
        print(f"  Output  → {out_dir}")
    else:
        print("\n  [DRY RUN — no files written]")


if __name__ == "__main__":
    main()
