#!/usr/bin/env python
"""Generate Stage 3 seed banks for non-game benchmarks from mega-skills.

Unlike the Phase-2 script (``stage2_seeds_from_megaskills.py``) which
ranks mega-skills by *genre affinity* (puzzle/shmup/beatemup/platform),
this script uses **reasoning-plan affinity**: mega-skills whose abstract
reasoning plan matches the target domain's dominant reasoning patterns
score highest.

Two mega-skill sources are merged:

  1. ``megaskills_all_stages/mega_skills.jsonl`` — 20 game-derived megas
     with full ICL exemplars, contracts, and protocol steps.
  2. ``reasoning_aligned_mega_skills.json`` — 59 cross-domain reasoning
     plans with domain membership (GAME/VR/WEB/OTHER) and per-domain
     representative members.

Ranking heuristic per (mega, target) pair:

  score = domain_membership_weight
        + reasoning_plan_affinity   (VR targets prefer VERIFY/CHECK plans)
        + cluster_size_bonus        (broader evidence = more reliable)

Output::

    frontier_data/output/stage3_seed_banks/<task>/skill_bank.jsonl

Usage::

    python scripts/stage3_seeds_from_megaskills.py --top-k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("stage3_seeds")

STAGE3_TARGETS = [
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
    "alfworld",
    "miniwob",
    "webshop",
]

# Maps each target to its primary domain tag (matches reasoning_aligned
# mega-skills' ``domains`` field).
TARGET_DOMAIN: Dict[str, str] = {
    "visual_toolbench": "VR",
    "tir_bench": "VR",
    "video_holmes": "VR",
    "siv_bench": "VR",
    "alfworld": "ALFWORLD",
    "miniwob": "WEB",
    "webshop": "WEB",
}

# Reasoning operators that each target domain relies on most heavily.
# Used to boost mega-skills whose compressed_plan contains these ops.
DOMAIN_PREFERRED_OPS: Dict[str, List[str]] = {
    "VR": ["VERIFY", "CHECK", "PERCEIVE", "DECIDE"],
    "WEB": ["ACT", "NAVIGATE", "DECIDE", "VERIFY"],
    "ALFWORLD": ["PERCEIVE", "NAVIGATE", "ACT", "VERIFY"],
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").lower()).strip("_") or "mega"


# ── Load mega-skills from both sources ──────────────────────────────

def load_judge_megaskills(path: Path) -> List[dict]:
    """Load game-derived mega-skills (JSONL, from cluster_all_into_megaskills)."""
    out = []
    if not path.exists():
        logger.warning("judge mega-skills not found: %s", path)
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_reasoning_megaskills(path: Path) -> List[dict]:
    """Load reasoning-aligned mega-skills (JSON, from build_reasoning_aligned_bank)."""
    if not path.exists():
        logger.warning("reasoning mega-skills not found: %s", path)
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("mega_skills", [])


# ── Scoring ──────────────────────────────────────────────────────────

def score_reasoning_mega(mega: dict, target: str) -> float:
    """Score a reasoning-aligned mega-skill for a target benchmark."""
    domain = TARGET_DOMAIN.get(target, "OTHER")

    # Domain membership: does this mega have members from the target's domain?
    mega_domains = set(mega.get("domains", []))
    if domain in mega_domains:
        domain_score = 1.0
    elif "OTHER" in mega_domains:
        domain_score = 0.5
    else:
        domain_score = 0.2

    # Reasoning plan affinity: do the ops match what the domain needs?
    plan_ops = set(mega.get("compressed_plan", []))
    preferred = set(DOMAIN_PREFERRED_OPS.get(domain, []))
    if plan_ops and preferred:
        overlap = len(plan_ops & preferred) / max(len(preferred), 1)
    else:
        overlap = 0.3
    plan_score = overlap

    size_bonus = min(mega.get("n_members", 0), 20) / 200.0
    task_bonus = min(mega.get("n_tasks", 0), 10) / 100.0

    return domain_score + plan_score + size_bonus + task_bonus


def score_judge_mega(mega: dict, target: str) -> float:
    """Score a judge-derived mega-skill for a target benchmark."""
    domain = TARGET_DOMAIN.get(target, "OTHER")
    tasks = set(mega.get("tasks", []))

    # Check if any member task is in the target's domain
    target_in_tasks = target in tasks
    domain_match = any(
        t.startswith("miniwob") or t.startswith("webshop")
        for t in tasks
    ) if domain == "WEB" else any(
        t in ("visual_toolbench", "tir_bench", "video_holmes", "siv_bench")
        for t in tasks
    ) if domain == "VR" else False

    base = 0.5
    if target_in_tasks:
        base = 1.5
    elif domain_match:
        base = 1.0

    sig = mega.get("template_signature", "")
    sig_ops = set(sig.replace("→", " ").replace("->", " ").split())
    preferred = set(DOMAIN_PREFERRED_OPS.get(domain, []))
    plan_score = len(sig_ops & preferred) / max(len(preferred), 1) if preferred else 0.3

    size_bonus = min(mega.get("n_members", 0), 20) / 200.0

    return base + plan_score + size_bonus


# ── Seed entry builders ──────────────────────────────────────────────

def build_seed_from_reasoning_mega(target: str, mega: dict) -> dict:
    """Build a seed bank entry from a reasoning-aligned mega-skill."""
    mega_id = mega["mega_skill_id"]
    plan = mega.get("reasoning_plan", "")
    reps = mega.get("representatives", {})

    # Pick the best representative: prefer target's domain, then OTHER
    domain = TARGET_DOMAIN.get(target, "OTHER")
    rep = reps.get(domain) or reps.get("OTHER") or reps.get("GAME") or {}
    if isinstance(rep, list):
        rep = rep[0] if rep else {}

    rep_name = rep.get("name", "") or mega_id.replace("plan.", "").replace("_", " ").title()
    rep_task = rep.get("task", "")

    return {
        "skill_id": f"seed.s3.{_slug(mega_id)}",
        "name": rep_name,
        "version": 1,
        "strategic_description": f"Reasoning pattern: {plan}. Applicable across {mega.get('n_domains', 0)} domains.",
        "tags": [
            f"mega_skill_id:{mega_id}",
            f"reasoning_plan:{plan}",
            f"target:{target}",
            f"source:reasoning_aligned",
            f"n_domains:{mega.get('n_domains', 0)}",
            f"n_tasks:{mega.get('n_tasks', 0)}",
        ],
        "template_signature": plan,
        "protocol": {
            "preconditions": [],
            "steps": [op.strip() for op in plan.split("→") if op.strip()],
            "step_checks": [],
        },
        "contract": {
            "description": f"Cross-domain reasoning template: {plan}",
            "eff_add": [],
            "eff_del": [],
            "eff_event": [],
        },
        "exemplars": [],
        "n_instances": 0,
        "retired": False,
        "feasible_tasks": [target],
        "verified_tasks": [],
        "provenance": {
            "kind": "megaskill_seed_stage3",
            "source_mega_skill": mega_id,
            "source_type": "reasoning_aligned",
            "representative_task": rep_task,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def build_seed_from_judge_mega(target: str, mega: dict) -> dict:
    """Build a seed bank entry from a judge-derived mega-skill."""
    mega_id = mega["mega_skill_id"]
    sig = mega.get("template_signature", "")
    rep = mega.get("representative", {}) or {}
    seed_slug = re.sub(r"^mega\.\d+\.", "", mega_id)

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

    return {
        "skill_id": f"seed.s3.{_slug(seed_slug)}",
        "name": rep.get("name", "") or seed_slug.replace("_", " ").title(),
        "version": 1,
        "strategic_description": (rep.get("description") or "")[:600],
        "tags": [
            f"mega_skill_id:{mega_id}",
            f"template_signature:{sig}",
            f"target:{target}",
            f"source:judge_megaskill",
            f"n_members:{mega.get('n_members', 0)}",
            f"n_tasks:{mega.get('n_tasks', 0)}",
        ],
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
            "kind": "megaskill_seed_stage3",
            "source_mega_skill": mega_id,
            "source_type": "judge_megaskill",
            "source_representative": rep,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ── Also include existing per-task bank skills (from cold-start) ─────

def load_per_task_bank(task: str) -> List[dict]:
    """Load pre-existing per-task bank skills from cold-start extraction."""
    bank_path = (
        REPO_ROOT / "frontier_data" / "output" / "per_task_banks"
        / task / "skill_bank.jsonl"
    )
    if not bank_path.exists():
        return []
    skills = []
    with open(bank_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                skills.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return skills


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--judge-megaskills",
        default=str(REPO_ROOT / "frontier_data/output/megaskills_all_stages/mega_skills.jsonl"),
    )
    ap.add_argument(
        "--reasoning-megaskills",
        default=str(REPO_ROOT / "frontier_data/output/reasoning_aligned_mega_skills.json"),
    )
    ap.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "frontier_data/output/stage3_seed_banks"),
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--include-per-task-bank", action="store_true", default=True,
                    help="also include existing per-task cold-start skills")
    ap.add_argument("--targets", nargs="+", default=STAGE3_TARGETS)
    args = ap.parse_args()

    judge_megas = load_judge_megaskills(Path(args.judge_megaskills))
    reasoning_megas = load_reasoning_megaskills(Path(args.reasoning_megaskills))
    logger.info(
        "loaded %d judge mega-skills + %d reasoning mega-skills",
        len(judge_megas), len(reasoning_megas),
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []

    for target in args.targets:
        # Score and rank both sources
        scored: List[tuple[float, dict, str]] = []

        for m in reasoning_megas:
            s = score_reasoning_mega(m, target)
            scored.append((s, m, "reasoning"))

        for m in judge_megas:
            s = score_judge_mega(m, target)
            scored.append((s, m, "judge"))

        scored.sort(key=lambda x: -x[0])

        # Deduplicate by seed slug
        seen_slugs: set = set()
        picks: List[tuple[float, dict, str]] = []
        for score, mega, source in scored:
            mid = mega.get("mega_skill_id", "")
            slug = _slug(mid)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            picks.append((score, mega, source))
            if len(picks) >= args.top_k:
                break

        # Build seed entries
        seeds: List[dict] = []
        for score, mega, source in picks:
            if source == "reasoning":
                seeds.append(build_seed_from_reasoning_mega(target, mega))
            else:
                seeds.append(build_seed_from_judge_mega(target, mega))

        # Optionally merge in per-task bank skills
        n_per_task = 0
        if args.include_per_task_bank:
            existing = load_per_task_bank(target)
            existing_ids = {s.get("skill_id") for s in seeds}
            for sk in existing:
                if sk.get("skill_id") not in existing_ids:
                    seeds.append(sk)
                    n_per_task += 1

        # Write seed bank
        seed_dir = out_root / target
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_path = seed_dir / "skill_bank.jsonl"
        with open(seed_path, "w") as f:
            for seed in seeds:
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")

        n_reasoning = sum(1 for _, _, s in picks if s == "reasoning")
        n_judge = sum(1 for _, _, s in picks if s == "judge")
        logger.info(
            "[%s] domain=%s wrote %d seeds (%d reasoning + %d judge + %d per-task)",
            target, TARGET_DOMAIN.get(target, "?"),
            len(seeds), n_reasoning, n_judge, n_per_task,
        )
        summary_rows.append({
            "target": target,
            "domain": TARGET_DOMAIN.get(target, "?"),
            "n_seeds_total": len(seeds),
            "n_reasoning": n_reasoning,
            "n_judge": n_judge,
            "n_per_task": n_per_task,
            "top_3": [
                {
                    "mega_skill_id": m.get("mega_skill_id", ""),
                    "score": round(sc, 3),
                    "source": src,
                }
                for sc, m, src in picks[:3]
            ],
        })

    # Summary
    summary_path = out_root / "SUMMARY.json"
    summary_path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "judge_megaskills_source": args.judge_megaskills,
        "reasoning_megaskills_source": args.reasoning_megaskills,
        "n_judge_available": len(judge_megas),
        "n_reasoning_available": len(reasoning_megas),
        "top_k": args.top_k,
        "include_per_task_bank": args.include_per_task_bank,
        "targets": summary_rows,
    }, indent=2, ensure_ascii=False))
    logger.info("summary → %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
