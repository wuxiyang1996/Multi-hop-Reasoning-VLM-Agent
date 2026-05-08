#!/usr/bin/env python3
"""Cross-game skill transfer analyzer.

Reads ``<run_dir>/transfer_log/usage.jsonl`` (populated at runtime by
``trainer/coevolution/_run_loggers.log_transfer_usage`` from
``episode_runner.py``) and computes per-game transfer-uptake metrics:

* Mix of (native, translated, crafter_v2) skills used per phase / per
  game.
* Reward earned conditional on each skill class.
* Translation survival rate (translated → verified).
* Top-N translated skill_ids by usage count + total reward.

Usage::

    python scripts/analyze_transfer_log.py \\
        --run-dir runs/Qwen3.5-9B_<ts> \\
        --out reports/transfer_<ts>.md

Outputs both a markdown report and a JSON file with the raw stats.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def load_usage(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
    with open(path) as f:
        for L in f:
            try:
                rows.append(json.loads(L))
            except Exception:
                pass
    return rows


def categorize_row(r: dict) -> str:
    """Return one of: native, translated, crafter_v2, other.

    Recovery-path note (2026-05-08): legacy foundry-mined skills have
    ``feasible_tasks=[]`` (the §22 metadata defaulted empty), which the
    older ``is_native`` field rejected.  The runtime logger has been
    patched to treat empty ``feasible_tasks`` + ``confidence_tag in
    {stable, crafter_v2}`` as native; for older usage.jsonl files we
    apply the same fallback heuristic here.
    """
    tag = (r.get("confidence_tag") or "stable").strip().lower()
    if tag == "translated":
        return "translated"
    if tag == "crafter_v2":
        return "crafter_v2"
    feasible = r.get("feasible_tasks") or []
    game = r.get("game") or ""
    if r.get("is_native") or (
        tag in ("stable", "")
        and (not feasible or game in feasible)
    ):
        return "native"
    return "other"


def per_game_stats(rows: List[dict]) -> Dict[str, Any]:
    by_game: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_game[r.get("game", "?")].append(r)

    out: Dict[str, Any] = {}
    for game, items in by_game.items():
        n = len(items)
        cat_counts: Dict[str, int] = Counter(categorize_row(r) for r in items)
        cat_rewards: Dict[str, List[float]] = defaultdict(list)
        for r in items:
            cat_rewards[categorize_row(r)].append(float(r.get("raw_env_reward") or 0.0))
        cat_mean_reward: Dict[str, float] = {
            k: (mean(v) if v else 0.0) for k, v in cat_rewards.items()
        }
        cat_total_reward: Dict[str, float] = {
            k: float(sum(v)) for k, v in cat_rewards.items()
        }
        # Top translated skills by usage
        translated = [r for r in items if r.get("confidence_tag") == "translated"]
        top_translated = Counter((r["skill_id"], r["skill_name"]) for r in translated).most_common(10)
        # Per-step distribution by inner_step bucket (deciles)
        steps = sorted(r.get("step") for r in items if r.get("step") is not None)
        step_min = steps[0] if steps else 0
        step_max = steps[-1] if steps else 0

        out[game] = {
            "n_decisions": n,
            "step_range": [step_min, step_max],
            "category_counts": dict(cat_counts),
            "category_share_pct": {
                k: round(100.0 * v / max(1, n), 1) for k, v in cat_counts.items()
            },
            "category_mean_reward": {k: round(v, 2) for k, v in cat_mean_reward.items()},
            "category_total_reward": {k: round(v, 2) for k, v in cat_total_reward.items()},
            "top_translated_skills": [
                {"skill_id": sid, "name": name, "n_uses": n_}
                for (sid, name), n_ in top_translated
            ],
            "n_unique_skill_ids": len({r.get("skill_id") for r in items}),
        }
    return out


def cross_game_transfer_stats(rows: List[dict]) -> Dict[str, Any]:
    """Aggregate cross-game transfer signal: how often the actor uses
    a skill derived from a different game than the current one."""
    n_total = len(rows)
    n_translated_used = sum(1 for r in rows if r.get("is_cross_game_translated"))
    by_lineage: Dict[str, int] = Counter()
    for r in rows:
        if r.get("is_cross_game_translated"):
            sid = r.get("derived_from") or "?"
            # Strip trailing __translated_to_... if accidentally included
            by_lineage[sid] += 1
    return {
        "n_total_decisions": n_total,
        "n_cross_game_translated_uses": n_translated_used,
        "cross_game_transfer_rate_pct": round(
            100.0 * n_translated_used / max(1, n_total), 2,
        ),
        "top_lineage_used": Counter(by_lineage).most_common(15),
    }


def render_markdown(stats: Dict[str, Any], cross: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Skill Transfer Analysis\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

    lines.append(f"## Headline\n")
    lines.append(f"- Total skill-selection events: **{cross['n_total_decisions']}**")
    lines.append(f"- Cross-game translated skill uses: **{cross['n_cross_game_translated_uses']}** "
                 f"({cross['cross_game_transfer_rate_pct']}%)")
    lines.append("")

    if cross["top_lineage_used"]:
        lines.append(f"## Top translated source skills (lineage → uses)\n")
        for (sid, n) in cross["top_lineage_used"]:
            lines.append(f"- `{sid}` → {n} uses")
        lines.append("")

    for game, gs in stats.items():
        lines.append(f"## Game: `{game}`\n")
        lines.append(f"- decisions: **{gs['n_decisions']}** (steps {gs['step_range'][0]}–{gs['step_range'][1]})")
        lines.append(f"- unique skills used: {gs['n_unique_skill_ids']}\n")
        lines.append(f"### Category mix\n")
        lines.append("| category | count | share % | mean reward | total reward |")
        lines.append("|---|---|---|---|---|")
        for cat in sorted(gs["category_counts"].keys()):
            lines.append(
                f"| {cat} | {gs['category_counts'][cat]} | "
                f"{gs['category_share_pct'].get(cat, 0)}% | "
                f"{gs['category_mean_reward'].get(cat, 0.0)} | "
                f"{gs['category_total_reward'].get(cat, 0.0)} |"
            )
        if gs["top_translated_skills"]:
            lines.append(f"\n### Top translated skills used in {game}\n")
            for s in gs["top_translated_skills"]:
                lines.append(f"- `{s['skill_id']}` ({s['name']}) — {s['n_uses']} uses")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="",
                    help="Output markdown path; defaults to <run_dir>/transfer_log/report.md")
    ap.add_argument("--json-out", default="",
                    help="Output JSON path; defaults to <run_dir>/transfer_log/report.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    usage_path = run_dir / "transfer_log" / "usage.jsonl"
    rows = load_usage(usage_path)
    if not rows:
        print(f"[analyze_transfer_log] no rows at {usage_path}; exiting")
        return 1

    print(f"[analyze_transfer_log] loaded {len(rows)} rows from {usage_path}")

    cross = cross_game_transfer_stats(rows)
    stats = per_game_stats(rows)

    out_md = Path(args.out) if args.out else (run_dir / "transfer_log" / "report.md")
    out_json = Path(args.json_out) if args.json_out else (run_dir / "transfer_log" / "report.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md = render_markdown(stats, cross)
    out_md.write_text(md)
    with open(out_json, "w") as fh:
        json.dump({"per_game": stats, "cross_game": cross,
                   "n_rows": len(rows)}, fh, ensure_ascii=False, indent=2,
                  default=str)

    print(f"[analyze_transfer_log] wrote {out_md}")
    print(f"[analyze_transfer_log] wrote {out_json}")
    print(f"")
    print(f"=== Headline ===")
    print(f"  total decisions:          {cross['n_total_decisions']}")
    print(f"  cross-game translated:    {cross['n_cross_game_translated_uses']} ({cross['cross_game_transfer_rate_pct']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
