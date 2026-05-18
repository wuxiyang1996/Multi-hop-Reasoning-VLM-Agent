#!/usr/bin/env python3
"""Mega-skill coverage audit (Phase-1 / Phase-2 / Phase-3 headline).

Joins ``frontier_data/output/mega_skill_clusters.json`` against the live
Phase-1 / Phase-2 game split in ``trainer/coevolution/config.py`` and the
OOD corpora registered in
``frontier_data/output/per_task_banks/MANIFEST.json``, then emits the
three headline numbers the co-evolution plan actually cares about:

  H1 — Phase-1 family coverage:
        out of the N canonical mega-skill families, how many have at
        least one skill mined from a Phase-1 source game.

  H2 — Phase-2 transfer potential:
        for each Phase-2 hold-out game, the count of (family, P1-source)
        cells the seeded bank fires on — i.e. the mega-skill "links"
        that the §1 exhaustive-search split was optimised for. Sums to
        the published headline (50 links across 9 source→target bridges
        in the current optimal split; see
        ``frontier_data/PLAN_GAME_SPLIT_AND_NO_SFT_GRPO.md``).

  H3 — Phase-3 cross-domain coverage:
        for each non-game corpus (VR image/video, WEB), the # of
        canonical families that have BOTH at least one Phase-1-game
        skill AND at least one skill mined from that corpus. This is
        the upper bound on §8 cross-domain transfer (skills that map
        across domains by mega-skill family identity, before predicate
        translation / harness execution narrows it further).

Also emits a per-family Phase-2 bridge table (mirrors the 9-bridge
table in §1 of PLAN_GAME_SPLIT_AND_NO_SFT_GRPO.md) so cuts in the live
data are visible vs the published number.

Output:
  * stdout summary with three headline numbers + bridge table.
  * (optional) ``--output <path.json>`` machine-readable dump:
        {
          "phase1_coverage": ...,
          "phase2_links": {<p2_game>: {<p1_source>: <count>}, ...},
          "phase2_link_total": <int>,
          "phase3_coverage": {<corpus>: {"families": [...], "n": ...}},
          "families": [ {name, n_way, is_cross_domain, p1_tasks, p2_tasks,
                         vr_tasks, web_tasks, ...}, ... ]
        }

Usage:
    python frontier_data/scripts/coverage_audit.py
    python frontier_data/scripts/coverage_audit.py \
        --clusters frontier_data/output/mega_skill_clusters.json \
        --output   frontier_data/output/coverage_audit.json

Exit code:
    0 if all three headlines are computable; 1 if the join produces a
    P1/P2 game not present in any family (would indicate a stale
    cluster file or a config drift bug — actionable signal).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

# gymv_<slug>  →  Temporal_<Slug>-v0  (the name used in cluster `tasks`)
GYMV_SLUG_TO_TASK = {
    "gymv_airstriker":          "Temporal_Airstriker-v0",
    "gymv_altered_beast":       "Temporal_AlteredBeast-v0",
    "gymv_columns":             "Temporal_Columns-v0",
    "gymv_dynamite_headdy":     "Temporal_DynamiteHeaddy-v0",
    "gymv_space_harrier_ii":    "Temporal_SpaceHarrierII-v0",
    "gymv_streets_of_rage_2":   "Temporal_StreetsOfRage2-v0",
    "gymv_strider":             "Temporal_Strider-v0",
    "gymv_thunder_force_iii":   "Temporal_ThunderForceIII-v0",
}

# Phase-3 OOD corpora, split by stage (matches frontier_data/README.md
# §8 + run_full_pipeline.sh task lists).
VR_IMAGE_TASKS = {"tir_bench", "visual_toolbench"}
VR_VIDEO_TASKS = {"video_holmes", "siv_bench"}
WEB_TASKS      = {"miniwob", "webshop"}
OOD_CORPORA = {
    "vr_image": VR_IMAGE_TASKS,
    "vr_video": VR_VIDEO_TASKS,
    "web":      WEB_TASKS,
}


def _slug_to_task(slug: str) -> str:
    """Map a config-side slug onto the task name used in the cluster file."""
    if slug.startswith("gymv_"):
        return GYMV_SLUG_TO_TASK.get(slug, slug)
    return slug


def _load_phase_split() -> Tuple[List[str], List[str]]:
    """Return (PHASE1_DEFAULT_GAMES, PHASE2_HOLDOUT_GAMES) as live in config.py.

    Falls back to a hardcoded copy of the 2026-05-12 mega-skill-optimal
    split if the trainer module is not importable (e.g. running from a
    minimal environment without the rest of the project on PYTHONPATH).
    Emits a warning in that case so the caller knows the split could
    drift silently.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from trainer.coevolution.config import (  # type: ignore[import-not-found]
            PHASE1_DEFAULT_GAMES,
            PHASE2_HOLDOUT_GAMES,
        )
        return list(PHASE1_DEFAULT_GAMES), list(PHASE2_HOLDOUT_GAMES)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[coverage_audit] WARN: cannot import trainer.coevolution.config "
            f"({exc!r}); falling back to hard-coded 2026-05-12 split",
            file=sys.stderr,
        )
        return (
            [
                "gymv_thunder_force_iii",
                "gymv_streets_of_rage_2",
                "gymv_strider",
                "gymv_columns",
                "tetris",
                "candy_crush",
            ],
            [
                "gymv_space_harrier_ii",
                "gymv_airstriker",
                "gymv_altered_beast",
                "gymv_dynamite_headdy",
                "twenty_forty_eight",
                "super_mario",
            ],
        )


def _index_family_skills(family: dict) -> Dict[str, List[dict]]:
    """Bucket a family's ``skills`` list by source task name."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for s in family.get("skills", []):
        task = s.get("task")
        if task:
            out[task].append(s)
    return dict(out)


def audit(clusters: dict, phase1: List[str], phase2: List[str]) -> dict:
    """Compute the three headline numbers + per-family detail.

    Returns a dict suitable for JSON serialisation and pretty printing.
    """
    p1_tasks = {_slug_to_task(s) for s in phase1}
    p2_tasks = {_slug_to_task(s) for s in phase2}

    families = clusters.get("families", {})
    n_families = len(families)

    # H1 — Phase-1 family coverage.
    families_with_p1 = []
    families_without_p1 = []
    for fname, fam in families.items():
        fam_tasks = set(fam.get("tasks", []))
        if fam_tasks & p1_tasks:
            families_with_p1.append(fname)
        else:
            families_without_p1.append(fname)

    # H2 — Phase-2 transfer potential.
    # For each Phase-2 game, count (family, P1-source) pairs where the
    # family contains both a P1-task skill and a P2-task skill. That's
    # the same "link" semantics as §1 PLAN_GAME_SPLIT_AND_NO_SFT_GRPO.md.
    p2_links: Dict[str, Dict[str, int]] = {p2: defaultdict(int) for p2 in phase2}
    bridge_families: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for fname, fam in families.items():
        idx = _index_family_skills(fam)
        # The skills mined from each P1 task within this family
        p1_present = {p1 for p1 in phase1 if _slug_to_task(p1) in idx}
        p2_present = {p2 for p2 in phase2 if _slug_to_task(p2) in idx}
        if not p1_present or not p2_present:
            continue
        for p2 in p2_present:
            for p1 in p1_present:
                p2_links[p2][p1] += 1
                bridge_families[(p1, p2)].add(fname)

    p2_links_plain = {p2: dict(d) for p2, d in p2_links.items()}
    total_p2_links = sum(sum(d.values()) for d in p2_links_plain.values())

    # H3 — Phase-3 cross-domain coverage.
    p3_coverage: Dict[str, Dict[str, object]] = {}
    for corpus_name, corpus_tasks in OOD_CORPORA.items():
        hit_families: List[str] = []
        for fname, fam in families.items():
            fam_tasks = set(fam.get("tasks", []))
            has_p1 = bool(fam_tasks & p1_tasks)
            has_corpus = bool(fam_tasks & corpus_tasks)
            if has_p1 and has_corpus:
                hit_families.append(fname)
        p3_coverage[corpus_name] = {
            "n_families": len(hit_families),
            "families":   hit_families,
        }

    # Family-by-family detail (sorted by descending skill count)
    family_detail = []
    for fname, fam in sorted(
        families.items(), key=lambda kv: -kv[1].get("count", 0)
    ):
        idx = _index_family_skills(fam)
        family_detail.append({
            "name":             fname,
            "count":            fam.get("count", 0),
            "n_way":            fam.get("n_way"),
            "is_cross_domain":  fam.get("is_cross_domain"),
            "domains":          fam.get("domains", []),
            "p1_tasks":         sorted(
                _slug_to_task(p1) for p1 in phase1 if _slug_to_task(p1) in idx
            ),
            "p2_tasks":         sorted(
                _slug_to_task(p2) for p2 in phase2 if _slug_to_task(p2) in idx
            ),
            "vr_image_tasks":   sorted(set(idx) & VR_IMAGE_TASKS),
            "vr_video_tasks":   sorted(set(idx) & VR_VIDEO_TASKS),
            "web_tasks":        sorted(set(idx) & WEB_TASKS),
        })

    return {
        "phase1_games": phase1,
        "phase2_games": phase2,
        "n_families": n_families,
        "phase1_coverage": {
            "n_with_p1":      len(families_with_p1),
            "pct_with_p1":    (len(families_with_p1) / n_families) if n_families else 0.0,
            "families_with_p1": families_with_p1,
            "families_without_p1": families_without_p1,
        },
        "phase2_links":      p2_links_plain,
        "phase2_link_total": total_p2_links,
        "phase2_bridges": {
            f"{p1} -> {p2}": sorted(families)
            for (p1, p2), families in sorted(bridge_families.items())
        },
        "phase3_coverage":   p3_coverage,
        "families":          family_detail,
    }


def _print_summary(result: dict) -> None:
    nf = result["n_families"]
    p1 = result["phase1_coverage"]
    print()
    print("=" * 70)
    print("  Mega-skill coverage audit")
    print("=" * 70)
    print(f"  Canonical families:        {nf}")
    print(f"  Phase-1 games (live split): {result['phase1_games']}")
    print(f"  Phase-2 games (live split): {result['phase2_games']}")
    print()
    print("─" * 70)
    print(f"  H1 — Phase-1 family coverage")
    print("─" * 70)
    print(
        f"  Families with ≥1 P1-game skill: "
        f"{p1['n_with_p1']}/{nf}  ({p1['pct_with_p1']:.0%})"
    )
    if p1["families_without_p1"]:
        print(f"  Families with NO P1 source (gap):")
        for fname in p1["families_without_p1"]:
            print(f"    - {fname}")
    print()
    print("─" * 70)
    print(f"  H2 — Phase-2 transfer potential (mega-skill links)")
    print("─" * 70)
    print(f"  Total (family × P1-source × P2-target) links: {result['phase2_link_total']}")
    print(f"  Per Phase-2 hold-out game (sources contributing skills):")
    for p2, src_counts in result["phase2_links"].items():
        if not src_counts:
            print(f"    {p2:30s} (no P1 source — ISOLATED)")
            continue
        srcs = sorted(src_counts.items(), key=lambda kv: -kv[1])
        bits = ", ".join(f"{p1}={n}" for p1, n in srcs)
        total = sum(src_counts.values())
        print(f"    {p2:30s} total={total:>3d}  ({bits})")
    print()
    print("─" * 70)
    print(f"  H3 — Phase-3 cross-domain coverage (P1 ∩ OOD per corpus)")
    print("─" * 70)
    for corpus, info in result["phase3_coverage"].items():
        print(
            f"  {corpus:8s} families with P1+corpus skills: "
            f"{info['n_families']}/{nf}"
        )
        for fname in info["families"]:
            print(f"    - {fname}")
    print()
    print("─" * 70)
    print(f"  9-bridge table (per the published §1 split)")
    print("─" * 70)
    sorted_bridges = sorted(
        result["phase2_bridges"].items(),
        key=lambda kv: -len(kv[1]),
    )
    for bridge, families in sorted_bridges:
        if not families:
            continue
        print(f"  {bridge:60s} {len(families)} families")
    print()
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--clusters", type=Path,
        default=REPO_ROOT / "frontier_data" / "output" / "mega_skill_clusters.json",
        help="Path to mega_skill_clusters.json (default: frontier_data/output/...)",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to dump the full audit as JSON.",
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress stdout summary; only write --output if set.",
    )
    args = ap.parse_args()

    if not args.clusters.is_file():
        print(
            f"[coverage_audit] ERROR: clusters file not found: {args.clusters}\n"
            "  Run frontier_data/scripts/cluster_mega_skills.py first.",
            file=sys.stderr,
        )
        return 2

    with args.clusters.open() as f:
        clusters = json.load(f)

    phase1, phase2 = _load_phase_split()
    result = audit(clusters, phase1, phase2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2)
        if not args.quiet:
            print(f"[coverage_audit] wrote audit → {args.output}", file=sys.stderr)

    if not args.quiet:
        _print_summary(result)

    # Drift check — any P1/P2 game with no mined skill in the cluster
    # file means the cluster is stale relative to config (e.g. a roster
    # change landed in config.py without re-running cluster_mega_skills.py).
    drifted = []
    family_task_union: Set[str] = set()
    for fam in clusters.get("families", {}).values():
        family_task_union.update(fam.get("tasks", []))
    for slug in phase1 + phase2:
        task = _slug_to_task(slug)
        if task not in family_task_union:
            drifted.append(slug)
    if drifted:
        print(
            f"[coverage_audit] WARN: roster drift — these games are in "
            f"config.py but NOT in any cluster family: {drifted}. "
            f"Re-run frontier_data/scripts/run_full_pipeline.sh stages 4-5.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
