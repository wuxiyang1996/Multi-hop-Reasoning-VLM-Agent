#!/usr/bin/env python
"""Generate Phase-2 seed banks for held-out games from the
LLM-judge mega-skills.

For each Phase-2 holdout game, this script ranks the available
mega-skills (from ``frontier_data/output/megaskills_2stage/mega_skills.jsonl``)
by *genre affinity to the target*, picks the top-K, and writes a seed
bank entry per mega containing:

  - ``skill_id``               : a seed id ``seed.mega.<mega-slug>``
  - ``name``                   : the mega-skill name
  - ``template_signature``     : abstract reasoning plan (e.g. EVALUATE → ACT → PERCEIVE → ACT)
  - ``protocol.preconditions`` : copied from mega's representative
  - ``protocol.steps``         : copied from mega's template_steps (rep's actual NL reasoning)
  - ``protocol.step_checks``   : copied from mega's representative
  - ``contract.description``   : composed from mega's judge_rationale
  - ``exemplars[0]``           : 1-shot ICL trace pulled from mega.icl_exemplar
  - ``tags``                   : provenance (``mega_skill_id``, ``exemplar_from``, ``ranked_by``)

Output:

    frontier_data/output/stage2_seeds_v3/<target>/skill_bank.jsonl

Usage::

    python scripts/stage2_seeds_from_megaskills.py \\
        --megaskills frontier_data/output/megaskills_2stage/mega_skills.jsonl \\
        --top-k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("stage2_seeds")


# ── Phase-2 holdout targets ──────────────────────────────────────────

PHASE2_TARGETS = [
    "gymv_space_harrier_ii",
    "gymv_airstriker",
    "gymv_altered_beast",
    "gymv_dynamite_headdy",
    "twenty_forty_eight",
    "super_mario",
]


# ── Genre affinity ───────────────────────────────────────────────────
# Genres assigned to all 11 games (5 sources + 6 targets) so we can rank
# mega-skills by how genre-aligned their binding tasks are to a target.
#
# Coarse genres (4 buckets):
#   puzzle / shmup / beatemup / platform
#
# These are coarse on purpose — the goal is to rank mega-skills, not to
# claim deep genre theory.  All buckets share the broad "score-and-clear"
# game loop, so weak cross-genre transfer is also allowed via a baseline
# match score.

GENRE_OF: Dict[str, str] = {
    # SOURCES (5 best GRPO banks)
    "candy_crush":              "puzzle",
    "gymv_columns":             "puzzle",
    "gymv_strider":             "platform",
    "gymv_thunder_force_iii":   "shmup",
    "gymv_streets_of_rage_2":   "beatemup",
    # TARGETS (Phase 2 holdouts)
    "twenty_forty_eight":       "puzzle",
    "super_mario":              "platform",
    "gymv_space_harrier_ii":    "shmup",
    "gymv_airstriker":          "shmup",
    "gymv_altered_beast":       "beatemup",
    "gymv_dynamite_headdy":     "platform",
}

# Pairwise genre affinity (used as binding-task weight when ranking
# mega-skills against a target).  Same genre = 1.0; cross-genre = 0.3
# baseline so even unrelated genres contribute a little.
GENRE_AFFINITY: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.3)
for g in {"puzzle", "shmup", "beatemup", "platform"}:
    GENRE_AFFINITY[(g, g)] = 1.0
# Specific cross-genre bonuses (action games share more with each other
# than with puzzles)
for a, b in [
    ("shmup", "platform"), ("shmup", "beatemup"),
    ("beatemup", "platform"),
]:
    GENRE_AFFINITY[(a, b)] = 0.55
    GENRE_AFFINITY[(b, a)] = 0.55


def affinity(target: str, task: str) -> float:
    return GENRE_AFFINITY[(GENRE_OF.get(target, "?"), GENRE_OF.get(task, "?"))]


# ── Mega-skill loading and ranking ───────────────────────────────────

def load_megaskills(path: Path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def score_mega_for_target(mega: dict, target: str) -> float:
    """Genre-weighted affinity of a mega-skill to a target.

    Score = sum_over_member_tasks(affinity(target, task)) / n_members.
    A small cluster-size bonus rewards mega-skills with broader empirical
    backing (more bindings).
    """
    members = mega.get("members") or []
    if not members:
        return 0.0
    aff = sum(affinity(target, m["task"]) for m in members) / len(members)
    # Cluster-size bonus: prefer megas with more bindings (more
    # empirical evidence of the pattern), capped.
    size_bonus = min(len(members), 10) / 100.0
    return aff + size_bonus


# ── Seed entry construction ──────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").lower()).strip("_") or "mega"


def build_seed_entry(target: str, mega: dict) -> dict:
    """Construct one seed bank entry from a mega-skill bundle."""
    mega_id = mega["mega_skill_id"]
    sig = mega.get("template_signature", "")
    rep = mega.get("representative", {}) or {}
    seed_slug = re.sub(r"^mega\.\d+\.", "", mega_id)

    # Compose a short contract description from top judge rationales.
    rationales = [
        e.get("shared", "")
        for e in (mega.get("judge_evidence") or [])
        if e.get("shared")
    ][:2]
    contract_desc = (
        rep.get("description", "")
        or " | ".join(rationales)
        or f"Reasoning template: {sig}"
    )

    exemplar = mega.get("icl_exemplar")
    exemplars = []
    if exemplar:
        exemplars.append({
            "source_task": exemplar.get("source_task"),
            "source_skill_id": exemplar.get("source_skill_id"),
            "source_kind": exemplar.get("source_kind", "protocol_raw"),
            "reasoning_steps": exemplar.get("steps", []),
        })

    tags = [
        f"mega_skill_id:{mega_id}",
        f"template_signature:{sig}",
        f"target:{target}",
        f"target_genre:{GENRE_OF.get(target, '?')}",
        f"ranked_by:genre_affinity",
        f"n_members:{mega.get('n_members', 0)}",
        f"n_tasks:{mega.get('n_tasks', 0)}",
    ]
    if exemplar:
        tags.append(f"exemplar_from:{exemplar.get('source_task','?')}")

    return {
        "skill_id": f"seed.{seed_slug}",
        "name": rep.get("name", "") or seed_slug.replace("_", " ").title(),
        "version": 1,
        "strategic_description": (rep.get("description") or "")[:600],
        "tags": tags,
        "template_signature": sig,
        "protocol": {
            "preconditions": mega.get("preconditions", []),
            "steps": mega.get("template_steps", []),
            "step_checks": mega.get("step_checks", []),
        },
        "contract": {
            "description": contract_desc[:600],
            "eff_add": [],
            "eff_del": [],
            "eff_event": [],
        },
        "exemplars": exemplars,
        "n_instances": 0,
        "retired": False,
        "feasible_tasks": [target],
        "verified_tasks": [],
        "provenance": {
            "kind": "megaskill_seed",
            "source_mega_skill": mega_id,
            "source_representative": rep,
            "source_members": [
                {"task": m["task"], "skill_id": m["skill_id"], "name": m.get("name", "")}
                for m in (mega.get("members") or [])
            ],
            "ranking_score": score_mega_for_target(mega, target),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--megaskills",
                    default=str(REPO_ROOT / "frontier_data/output/megaskills_2stage/mega_skills.jsonl"))
    ap.add_argument("--out-root",
                    default=str(REPO_ROOT / "frontier_data/output/stage2_seeds_v3"))
    ap.add_argument("--top-k", type=int, default=10,
                    help="how many mega-skills to include per target (default 10)")
    ap.add_argument("--targets", nargs="+", default=PHASE2_TARGETS,
                    help="target task list")
    args = ap.parse_args()

    megaskills = load_megaskills(Path(args.megaskills))
    logger.info("loaded %d mega-skills from %s", len(megaskills), args.megaskills)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    for target in args.targets:
        scored = [
            (score_mega_for_target(m, target), m) for m in megaskills
        ]
        scored.sort(key=lambda x: -x[0])
        picks = scored[: args.top_k]

        seed_dir = out_root / target
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_path = seed_dir / "skill_bank.jsonl"

        with open(seed_path, "w") as f:
            for score, mega in picks:
                seed = build_seed_entry(target, mega)
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")

        with_icl = sum(1 for _, m in picks if m.get("icl_exemplar"))
        within_genre = sum(
            1 for _, m in picks
            if any(GENRE_OF.get(mb["task"]) == GENRE_OF.get(target) for mb in m.get("members") or [])
        )
        logger.info(
            "[%s] genre=%s wrote %d seeds (%d with ICL, %d within-genre)",
            target, GENRE_OF.get(target, "?"), len(picks), with_icl, within_genre,
        )
        summary_rows.append({
            "target": target,
            "genre": GENRE_OF.get(target, "?"),
            "n_seeds": len(picks),
            "n_with_icl": with_icl,
            "n_within_genre": within_genre,
            "top_3": [
                {
                    "mega_skill_id": m["mega_skill_id"],
                    "score": round(score, 3),
                    "template_signature": m.get("template_signature", ""),
                    "n_members": m.get("n_members", 0),
                } for score, m in picks[:3]
            ],
        })

    # Top-level summary
    summary_path = out_root / "SUMMARY.json"
    summary_path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "megaskills_source": args.megaskills,
        "n_megaskills_available": len(megaskills),
        "top_k": args.top_k,
        "targets": summary_rows,
    }, indent=2, ensure_ascii=False))
    logger.info("wrote %s", summary_path)

    # Markdown summary
    md_path = out_root / "SUMMARY.md"
    lines = [
        f"# Phase-2 seed banks (v3, from LLM-judge mega-skills)",
        "",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        f"- mega-skills source: `{args.megaskills}`",
        f"- mega-skills available: **{len(megaskills)}**",
        f"- targets: {len(args.targets)}",
        "",
        "| target | genre | seeds | w/ICL | within-genre | top-1 mega |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        top1 = row["top_3"][0] if row["top_3"] else None
        top1_str = (
            f"`{top1['mega_skill_id']}` (sig={top1['template_signature']})"
            if top1 else "—"
        )
        lines.append(
            f"| {row['target']} | {row['genre']} | "
            f"{row['n_seeds']} | {row['n_with_icl']} | "
            f"{row['n_within_genre']} | {top1_str} |"
        )
    md_path.write_text("\n".join(lines))
    logger.info("wrote %s", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
