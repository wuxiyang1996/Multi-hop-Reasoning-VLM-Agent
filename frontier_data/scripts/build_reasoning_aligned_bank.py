#!/usr/bin/env python
"""Build a reasoning-plan-aligned shared skill bank.

Instead of clustering by skill_id_stem (name), this script:
1. Normalizes every protocol step to a domain-agnostic REASONING INTENT
2. Extracts a "reasoning plan signature" per skill
3. Clusters skills by plan signature across domains
4. Builds cross-domain mega-skills where game/web/VR skills share the
   SAME multi-step reasoning procedure

Reasoning intent vocabulary (7 canonical intents):
  PERCEIVE   - observe / scan / inspect the current state
  RECALL     - retrieve prior knowledge or memory
  EVALUATE   - compare, assess, or reason about options
  DECIDE     - select among alternatives
  NAVIGATE   - move to a target location / scroll / approach
  ACT        - execute a concrete action (click, attack, answer)
  VERIFY     - confirm outcome, check result, validate

A reasoning plan is the ORDERED SEQUENCE of these intents.
Two skills share a mega-skill IFF their reasoning plan matches.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("reasoning_aligned_bank")

REPO = Path(__file__).resolve().parents[2]

CANONICAL_INTENTS = [
    "PERCEIVE", "RECALL", "EVALUATE", "DECIDE",
    "NAVIGATE", "ACT", "VERIFY",
]

# ── Step-level intent classifier ──────────────────────────────────

# Op verb → intent mapping (covers game, VR, web vocabularies)
OP_TO_INTENT = {
    # Perception
    "INSPECT":   "PERCEIVE",
    "PERCEIVE":  "PERCEIVE",
    "OBSERVE":   "PERCEIVE",
    "SCAN":      "PERCEIVE",
    "LOOK":      "PERCEIVE",
    # Recall / memory
    "RECALL":    "RECALL",
    "REMEMBER":  "RECALL",
    # Evaluation / reasoning
    "EVALUATE":  "EVALUATE",
    "COMPARE":   "EVALUATE",
    "ASSESS":    "EVALUATE",
    "FILTER":    "EVALUATE",
    "DECIDE":    "DECIDE",
    "SELECT":    "DECIDE",
    "PREFER":    "DECIDE",
    "CHOOSE":    "DECIDE",
    # Navigation / movement
    "MOVE":      "NAVIGATE",
    "APPROACH":  "NAVIGATE",
    "NAVIGATE":  "NAVIGATE",
    "SLIDE":     "NAVIGATE",
    # Action / execution
    "EXEC":      "ACT",
    "EXECUTE":   "ACT",
    "COMMIT":    "ACT",
    "PLACE":     "ACT",
    "DROP":      "ACT",
    "ATTACK":    "ACT",
    # Verification
    "VERIFY":    "VERIFY",
    "CONFIRM":   "VERIFY",
    "CHECK":     "VERIFY",
    "KEEP":      "VERIFY",
    "TRACK":     "VERIFY",
    "CONTINUE":  "VERIFY",
}

# Evidence role → intent (fallback when op verb is ambiguous)
EVIDENCE_TO_INTENT = {
    "GATHER":    "PERCEIVE",
    "REASON":    "EVALUATE",
    "COMMIT":    "ACT",
    "VERIFY":    "VERIFY",
    "OBSERVATION": "PERCEIVE",
    "RECOVER":   "VERIFY",
}

# Keywords in notes for disambiguation
NOTES_PATTERNS = [
    (re.compile(r"inspect|scan|look|observe|examine|read", re.I), "PERCEIVE"),
    (re.compile(r"recall|remember|prior|memory|history", re.I), "RECALL"),
    (re.compile(r"compar|evaluat|assess|count|reason|analyz|measure", re.I), "EVALUATE"),
    (re.compile(r"select|choose|decide|pick|best", re.I), "DECIDE"),
    (re.compile(r"move|scroll|navigate|approach|drag|slide", re.I), "NAVIGATE"),
    (re.compile(r"click|press|fill|type|attack|submit|buy|search|fire|noop|execute|emit|answer", re.I), "ACT"),
    (re.compile(r"verify|confirm|check|wait|maintain|keep|hold|ensure|cross.?check", re.I), "VERIFY"),
]


def classify_step_intent(step: Dict[str, Any]) -> str:
    """Classify a single protocol step into a canonical reasoning intent."""
    op = str(step.get("op", step.get("action", ""))).upper().strip()
    evidence_role = str(step.get("evidence_role", "")).upper().strip()
    notes = str(step.get("notes", ""))

    # 1. Direct op mapping (highest confidence)
    if op in OP_TO_INTENT:
        return OP_TO_INTENT[op]

    # 2. Evidence role mapping
    if evidence_role in EVIDENCE_TO_INTENT:
        return EVIDENCE_TO_INTENT[evidence_role]

    # 3. Notes-based classification
    for pattern, intent in NOTES_PATTERNS:
        if pattern.search(notes):
            return intent

    return "ACT"


def extract_reasoning_plan(protocol: List[Dict]) -> List[str]:
    """Extract a reasoning plan = ordered sequence of intents."""
    intents = []
    for step in protocol:
        if isinstance(step, dict):
            intent = classify_step_intent(step)
            intents.append(intent)
    return intents


def compress_plan(intents: List[str]) -> List[str]:
    """Compress consecutive repeated intents for plan matching.

    E.g., [PERCEIVE, PERCEIVE, EVALUATE, ACT, ACT, VERIFY]
       → [PERCEIVE, EVALUATE, ACT, VERIFY]

    This captures the REASONING STRUCTURE (what phases the plan goes through)
    independent of how many steps each phase takes.
    """
    if not intents:
        return []
    compressed = [intents[0]]
    for i in intents[1:]:
        if i != compressed[-1]:
            compressed.append(i)
    return compressed


def plan_to_signature(plan: List[str]) -> str:
    """Convert a plan to a string signature for grouping."""
    return " → ".join(plan) if plan else "(empty)"


def _cohort_of(task: str) -> str:
    if task.startswith("Temporal_"):
        return "gymv_game"
    if task in ("candy_crush", "tetris", "super_mario", "twenty_forty_eight"):
        return "env_wr_game"
    if task in ("miniwob", "webshop"):
        return "web"
    if task in ("tir_bench", "visual_toolbench"):
        return "vr_image"
    if task in ("siv_bench", "video_holmes"):
        return "vr_video"
    return "other"


def _domain_group(cohort: str) -> str:
    if cohort in ("gymv_game", "env_wr_game"):
        return "GAME"
    if cohort == "web":
        return "WEB"
    if cohort in ("vr_image", "vr_video"):
        return "VR"
    return "OTHER"


def load_all_skills(bank_root: Path) -> List[Dict]:
    """Load all per-task skills with reasoning plan analysis."""
    all_skills = []
    for task_dir in sorted(os.listdir(bank_root)):
        sb = bank_root / task_dir / "skill_bank.jsonl"
        if not sb.is_file():
            continue
        with open(sb) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sk = r.get("skill", r)
                protocol = sk.get("protocol", [])
                if not isinstance(protocol, list):
                    protocol = []

                raw_plan = extract_reasoning_plan(protocol)
                compressed = compress_plan(raw_plan)
                sig = plan_to_signature(compressed)
                cohort = _cohort_of(task_dir)

                all_skills.append({
                    "task": task_dir,
                    "cohort": cohort,
                    "domain_group": _domain_group(cohort),
                    "skill_id": sk.get("skill_id", "?"),
                    "name": sk.get("name", "?"),
                    "raw_plan": raw_plan,
                    "compressed_plan": compressed,
                    "plan_signature": sig,
                    "n_steps": len(protocol),
                    "protocol": protocol,
                    "strategic_description": sk.get("strategic_description", ""),
                })
    return all_skills


def build_reasoning_mega_skills(
    all_skills: List[Dict],
    min_domains: int = 2,
) -> Tuple[List[Dict], Dict]:
    """Cluster skills by compressed reasoning plan signature.

    Only promote to mega-skill if the plan spans ≥ min_domains
    distinct domain groups (GAME, WEB, VR).
    """
    by_plan = defaultdict(list)
    for s in all_skills:
        if len(s["compressed_plan"]) >= 2:
            by_plan[s["plan_signature"]].append(s)

    mega_skills = []
    stats = {
        "total_plans": len(by_plan),
        "cross_domain_plans": 0,
        "total_skills_in_cross_plans": 0,
        "domain_coverage": defaultdict(int),
    }

    for sig, skills in sorted(by_plan.items(), key=lambda x: -len(x[1])):
        domains = set(s["domain_group"] for s in skills)
        tasks = set(s["task"] for s in skills)

        if len(domains) < min_domains:
            continue

        stats["cross_domain_plans"] += 1
        stats["total_skills_in_cross_plans"] += len(skills)

        # Pick representative skill per domain (highest n_steps for richest protocol)
        representatives = {}
        for s in skills:
            dg = s["domain_group"]
            if dg not in representatives or s["n_steps"] > representatives[dg]["n_steps"]:
                representatives[dg] = s

        mega_skill_id = f"plan.{'_'.join(sig.replace(' → ', '_').split())}"

        members_by_domain = defaultdict(list)
        for s in skills:
            members_by_domain[s["domain_group"]].append({
                "task": s["task"],
                "skill_id": s["skill_id"],
                "name": s["name"][:60],
                "n_steps": s["n_steps"],
                "raw_plan": s["raw_plan"],
            })

        mega = {
            "mega_skill_id": mega_skill_id,
            "reasoning_plan": sig,
            "compressed_plan": skills[0]["compressed_plan"],
            "n_domains": len(domains),
            "domains": sorted(domains),
            "n_tasks": len(tasks),
            "tasks": sorted(tasks),
            "n_members": len(skills),
            "members_by_domain": dict(members_by_domain),
            "representatives": {
                dg: {
                    "task": s["task"],
                    "skill_id": s["skill_id"],
                    "name": s["name"][:60],
                    "strategic_description": s["strategic_description"][:200],
                    "raw_plan": s["raw_plan"],
                    "n_steps": s["n_steps"],
                }
                for dg, s in representatives.items()
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        mega_skills.append(mega)

        for d in domains:
            stats["domain_coverage"][d] += 1

    return mega_skills, dict(stats)


def print_report(mega_skills: List[Dict], stats: Dict):
    print("\n" + "=" * 78)
    print("REASONING-PLAN-ALIGNED CROSS-DOMAIN MEGA-SKILLS")
    print("=" * 78)
    print(f"\nTotal unique plans (≥2 steps): {stats['total_plans']}")
    print(f"Cross-domain plans (≥2 domain groups): {stats['cross_domain_plans']}")
    print(f"Skills covered: {stats['total_skills_in_cross_plans']}")
    print(f"Domain coverage: {dict(stats.get('domain_coverage', {}))}")

    for i, mega in enumerate(mega_skills, 1):
        print(f"\n{'─' * 78}")
        print(f"Mega-skill #{i}: {mega['reasoning_plan']}")
        print(f"  Domains: {mega['domains']}  Tasks: {mega['n_tasks']}  Members: {mega['n_members']}")

        for dg in ["GAME", "WEB", "VR"]:
            members = mega["members_by_domain"].get(dg, [])
            rep = mega["representatives"].get(dg)
            if not members:
                continue
            print(f"\n  [{dg}] {len(members)} skills:")
            for m in members[:4]:
                label = " ★" if rep and m["skill_id"] == rep["skill_id"] else ""
                print(f"    {m['task']:<30} {m['skill_id'][:30]:<30}{label}")
                print(f"      raw_plan: {' → '.join(m['raw_plan'][:6])}")
            if len(members) > 4:
                print(f"    ... and {len(members)-4} more")

            if rep:
                print(f"  Representative ({dg}): {rep['name']}")
                print(f"    \"{rep['strategic_description'][:120]}\"")

    print(f"\n{'=' * 78}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank-root",
        default=str(REPO / "frontier_data" / "output" / "per_task_banks"),
    )
    p.add_argument("--min-domains", type=int, default=2,
                   help="Minimum domain groups for cross-domain mega-skill")
    p.add_argument("--output",
                   default=str(REPO / "frontier_data" / "output" / "reasoning_aligned_mega_skills.json"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    bank_root = Path(args.bank_root)
    all_skills = load_all_skills(bank_root)
    logger.info("Loaded %d skills from %s", len(all_skills), bank_root)

    # Show plan distribution per domain first
    plan_by_domain = defaultdict(lambda: defaultdict(int))
    for s in all_skills:
        if len(s["compressed_plan"]) >= 2:
            plan_by_domain[s["domain_group"]][s["plan_signature"]] += 1

    print("\n" + "=" * 78)
    print("TOP REASONING PLANS PER DOMAIN (compressed)")
    print("=" * 78)
    for dg in ["GAME", "WEB", "VR"]:
        plans = plan_by_domain.get(dg, {})
        print(f"\n{dg}:")
        for sig, cnt in sorted(plans.items(), key=lambda x: -x[1])[:8]:
            print(f"  {cnt:3d}× {sig}")

    mega_skills, stats = build_reasoning_mega_skills(all_skills, min_domains=args.min_domains)
    print_report(mega_skills, stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "meta": {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n_input_skills": len(all_skills),
                **stats,
            },
            "mega_skills": mega_skills,
        }, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d mega-skills to %s", len(mega_skills), out_path)


if __name__ == "__main__":
    main()
