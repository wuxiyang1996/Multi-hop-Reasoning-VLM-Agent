"""Stage 2: Mega-skill + 1-shot ICL transfer to new domain.

Aligned with PLAN_FEW_SHOT_SKILL_BANK.md and PIPELINE_GUIDE.md §4d.

Paradigm:
  - Mega-skill clustering (build_reasoning_aligned_bank.py) produces
    reasoning-plan signatures across 324 Stage-1 skills.
  - For each new target task T, we transfer the (mega-skill template +
    1-shot ICL exemplar) bundle, NOT concrete game skills.
  - Concrete game skills do NOT transfer to VR/Web (J_tok ≈ 0.02).
    What transfers is the abstract reasoning structure.

For each target T:
  1. Pick top-K mega-skill templates relevant to T's domain.
     - GAME target: same-genre Phase 1 game templates.
     - VR target: cross-domain mega-skills + within-cohort exemplars.
     - WEB target: cross-domain mega-skills (web exemplar TBD via teacher demos).
  2. For each mega-skill, build a seed skill containing:
     - template_signature (e.g. "PERCEIVE → COMPARE → DECIDE → ACT")
     - protocol.steps (the abstract reasoning plan, rendered to NL)
     - exemplar: 1-shot ICL from protocol_raw of a within-cohort member
     - step_checks: derived from template ops
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
BANKS_DIR = ROOT / "frontier_data" / "output" / "per_task_banks"
MEGA_FILE = ROOT / "frontier_data" / "output" / "reasoning_aligned_mega_skills.json"
OUT_DIR = ROOT / "frontier_data" / "output" / "stage2_seed_v2"

MAX_SEEDS = 12  # mega-skills per target


# ── Domain & genre ─────────────────────────────────────────────────

GAME_TARGETS = {
    "gymv_space_harrier_ii":  {"domain": "GAME", "genre": "shooter"},
    "gymv_airstriker":        {"domain": "GAME", "genre": "shooter"},
    "gymv_altered_beast":     {"domain": "GAME", "genre": "brawler"},
    "gymv_dynamite_headdy":   {"domain": "GAME", "genre": "platformer"},
    "twenty_forty_eight":     {"domain": "GAME", "genre": "puzzle"},
    "super_mario":            {"domain": "GAME", "genre": "platformer"},
}

WEB_TARGETS = {
    "webshop_new":    {"domain": "WEB", "genre": "web"},
    "miniwob_unseen": {"domain": "WEB", "genre": "web"},
}

VR_TARGETS = {
    "vr_new_bench": {"domain": "VR", "genre": "vr"},
}

GENRE_OF_SOURCE = {
    # Phase 1 trained games (use gymv_* canonical names)
    "gymv_thunder_force_iii": "shooter",
    "gymv_airstriker":        "shooter",
    "gymv_space_harrier_ii":  "shooter",
    "gymv_streets_of_rage_2": "brawler",
    "gymv_altered_beast":     "brawler",
    "gymv_strider":           "platformer",
    "gymv_dynamite_headdy":   "platformer",
    "gymv_columns":           "puzzle",
    "candy_crush":            "puzzle",
    "tetris":                 "puzzle",
    "twenty_forty_eight":     "puzzle",
    "super_mario":            "platformer",
    # Temporal_ duplicates
    "Temporal_ThunderForceIII-v0": "shooter",
    "Temporal_Airstriker-v0":      "shooter",
    "Temporal_SpaceHarrierII-v0":  "shooter",
    "Temporal_StreetsOfRage2-v0":  "brawler",
    "Temporal_AlteredBeast-v0":    "brawler",
    "Temporal_Strider-v0":         "platformer",
    "Temporal_DynamiteHeaddy-v0":  "platformer",
    "Temporal_Columns-v0":         "puzzle",
    # Non-game
    "miniwob":     "web",
    "webshop":     "web",
    "siv_bench":   "vr",
    "tir_bench":   "vr",
    "video_holmes":"vr",
    "visual_toolbench": "vr",
}

GENRE_AFFINITY = {
    ("shooter", "shooter"): 1.0, ("shooter", "platformer"): 0.4,
    ("brawler", "brawler"): 1.0, ("brawler", "platformer"): 0.6,
    ("platformer", "platformer"): 1.0, ("platformer", "shooter"): 0.4,
    ("puzzle", "puzzle"): 1.0,
    ("web", "web"): 1.0,
    ("vr", "vr"): 1.0,
}


def determine_domain(task: str) -> str:
    if task.startswith("gymv_") or task.startswith("Temporal_"):
        return "GAME"
    if task in ("candy_crush", "tetris", "twenty_forty_eight", "super_mario"):
        return "GAME"
    if task in ("miniwob", "webshop"):
        return "WEB"
    if task in ("siv_bench", "tir_bench", "video_holmes", "visual_toolbench"):
        return "VR"
    return "OTHER"


def genre_sim(g1: str, g2: str) -> float:
    if g1 == g2:
        return 1.0
    return GENRE_AFFINITY.get((g1, g2), GENRE_AFFINITY.get((g2, g1), 0.1))


# ── Skill loading ──────────────────────────────────────────────────

def load_skill(task: str, sid: str) -> Optional[Dict]:
    """Load a single skill record from per_task_banks/<task>/skill_bank.jsonl."""
    sb = BANKS_DIR / task / "skill_bank.jsonl"
    if not sb.exists():
        return None
    with open(sb) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sk = entry.get("skill", entry)
            if sk.get("skill_id") == sid:
                return sk
    return None


def has_usable_icl(skill: Dict) -> bool:
    """A skill has usable ICL exemplar if protocol_raw.steps contains
    non-empty natural-language reasoning."""
    raw = skill.get("protocol_raw", {})
    if not isinstance(raw, dict):
        return False
    steps = raw.get("steps", [])
    return any(isinstance(s, str) and s.strip() for s in steps)


def extract_icl_exemplar(skill: Dict, source_task: str) -> Optional[Dict]:
    """Extract a 1-shot ICL exemplar from a skill's protocol_raw."""
    raw = skill.get("protocol_raw", {})
    if not isinstance(raw, dict):
        return None
    steps = [s for s in raw.get("steps", []) if isinstance(s, str) and s.strip()]
    if not steps:
        return None

    name = skill.get("name", "")
    strategic = skill.get("strategic_description", "")[:300]

    return {
        "source_task":        source_task,
        "source_skill_id":    skill.get("skill_id", ""),
        "source_skill_name":  name[:100],
        "reasoning_steps":    steps[:6],  # cap to avoid token bloat
        "strategic_context":  strategic,
        "source_model":       "gpt-5.4",  # per provenance: per_task_banks were extracted by GPT-5.4
    }


# ── Mega-skill loading ─────────────────────────────────────────────

def load_mega_skills() -> List[Dict]:
    data = json.load(open(MEGA_FILE))
    return data["mega_skills"]


def get_mega_skill_members_with_icl(mega: Dict) -> Dict[str, List[Dict]]:
    """For each domain, return members whose source skill has usable ICL."""
    result = defaultdict(list)
    for dg, members in mega.get("members_by_domain", {}).items():
        if dg not in ("GAME", "WEB", "VR"):
            continue
        for m in members:
            sk = load_skill(m["task"], m["skill_id"])
            if sk and has_usable_icl(sk):
                result[dg].append({"member": m, "skill": sk})
    return result


# ── Step-checks from canonical intent ──────────────────────────────

INTENT_TO_CHECK = {
    "PERCEIVE": "entity_grounded=true",
    "RECALL":   "context_recalled=true",
    "EVALUATE": "options_compared=true",
    "DECIDE":   "answer_selected=true",
    "NAVIGATE": "target_reached=true",
    "ACT":      "action_executed=true",
    "VERIFY":   "outcome_confirmed=true",
}

INTENT_TO_PROMPT = {
    "PERCEIVE": "Observe the current state — identify the relevant entities, "
                "options, or visual evidence.",
    "RECALL":   "Recall any task context, prior observations, or constraints "
                "that apply.",
    "EVALUATE": "Compare or evaluate the candidates against the goal "
                "criteria; eliminate unsupported options.",
    "DECIDE":   "Choose the action or answer most strongly supported by "
                "the evidence.",
    "NAVIGATE": "Move toward the target location or scroll to bring the "
                "target into reach.",
    "ACT":      "Execute the chosen action (click, type, attack, submit, "
                "answer, …).",
    "VERIFY":   "Confirm the action produced the intended effect; "
                "check the outcome.",
}


def build_protocol_steps(compressed_plan: List[str]) -> List[str]:
    """Render compressed plan into NL protocol steps."""
    return [INTENT_TO_PROMPT.get(op, f"Execute {op.lower()} step.")
            for op in compressed_plan]


def build_step_checks(compressed_plan: List[str]) -> List[str]:
    return [INTENT_TO_CHECK.get(op, "") for op in compressed_plan]


# ── Source-side member selection (within-cohort exemplar preference) ──

def select_icl_member(
    mega: Dict,
    target_domain: str,
    target_genre: str,
    members_with_icl: Dict[str, List[Dict]],
) -> Optional[Dict]:
    """Pick the BEST member to serve as 1-shot exemplar for target.

    Priority (per PLAN_FEW_SHOT_SKILL_BANK.md §"What non-game tasks
    contribute to each other"): within-cohort > cross-domain.

      1. Same domain as target + best genre match
      2. Same domain (any genre)
      3. Cross-domain (last resort; rarely useful for VR)
    """
    # Priority 1: same domain
    same_domain = members_with_icl.get(target_domain, [])
    if same_domain:
        if target_domain == "GAME":
            # Rank by genre similarity
            def score(item):
                task = item["member"]["task"]
                src_genre = GENRE_OF_SOURCE.get(task, "unknown")
                return genre_sim(src_genre, target_genre)
            same_domain.sort(key=score, reverse=True)
        return same_domain[0]

    # Priority 2: VR can borrow from WEB (both single-shot reasoning),
    # WEB can borrow from VR — but GAME concrete skills don't help.
    if target_domain == "VR":
        return (members_with_icl.get("WEB", []) or [None])[0]
    if target_domain == "WEB":
        return (members_with_icl.get("VR", []) or [None])[0]

    # GAME target: prefer game exemplar (already handled above);
    # do NOT use VR/WEB exemplars for game targets — concrete steps incompatible.
    return None


# ── Mega-skill ranking per target ──────────────────────────────────

def rank_mega_skills_for_target(
    mega_skills: List[Dict],
    target_domain: str,
    target_genre: str,
) -> List[Tuple[Dict, float]]:
    """Score and sort mega-skills by relevance to target."""
    scored = []
    for mega in mega_skills:
        domains = set(mega["domains"])
        n_members = mega["n_members"]
        compressed = mega.get("compressed_plan", [])

        if len(compressed) < 2:
            continue

        # Must contain target's domain OR be a cross-domain bridge
        is_in_target_domain = target_domain in domains
        is_cross_bridge = len(domains) >= 2 and (
            ("GAME" in domains and ("WEB" in domains or "VR" in domains))
            or ("WEB" in domains and "VR" in domains)
        )

        score = 0.0
        if is_in_target_domain:
            score += 3.0
        if is_cross_bridge:
            score += 2.0
        if len(domains) >= 3:
            score += 1.5  # 3-way is gold
        # Reward size (well-attested patterns)
        score += min(n_members, 30) / 10.0

        # For GAME targets, additionally reward plans common in target's genre
        # (no direct measure; use plan length as proxy for richer reasoning)
        score += len(compressed) * 0.1

        if score > 0:
            scored.append((mega, score))

    scored.sort(key=lambda x: -x[1])
    return scored


# ── Build seed bank entries ────────────────────────────────────────

def build_seed_entry(
    mega: Dict,
    target_task: str,
    target_domain: str,
    icl_item: Optional[Dict],
    rank: int,
) -> Dict:
    """Build a single seed-bank entry combining mega-skill template + ICL."""
    compressed = mega.get("compressed_plan", [])
    plan_sig = mega["reasoning_plan"]
    domains = mega["domains"]

    protocol_steps = build_protocol_steps(compressed)
    step_checks = build_step_checks(compressed)

    # Construct seed skill ID
    plan_token = "_".join(compressed).lower()
    skill_id = f"seed.mega.{plan_token[:48]}.{rank:02d}"

    # Build exemplar block
    exemplar = None
    exemplar_tags = ["no_exemplar"]
    if icl_item:
        exemplar = extract_icl_exemplar(icl_item["skill"], icl_item["member"]["task"])
        if exemplar:
            exemplar_tags = [
                f"exemplar_from:{exemplar['source_task']}",
                f"exemplar_source_model:{exemplar['source_model']}",
            ]

    strategic_desc = (
        f"Mega-skill template '{plan_sig}' (cross-domain coverage: {'+'.join(sorted(domains))}). "
        f"Apply this {len(compressed)}-step reasoning structure to the current "
        f"{target_domain} task. See ICL exemplar for one concrete instantiation."
    )

    seed_skill = {
        "skill_id":              skill_id,
        "version":               1,
        "name":                  f"MegaSkill[{plan_sig}]",
        "strategic_description": strategic_desc,
        "tags": [
            "seed_stage2_mega",
            f"target:{target_task}",
            f"target_domain:{target_domain}",
            f"plan_signature:{plan_sig.replace(' ', '')}",
            f"mega_domains:{'+'.join(sorted(domains))}",
            f"mega_n_members:{mega['n_members']}",
            *exemplar_tags,
        ],
        "template_signature":    plan_sig,
        "protocol": {
            "preconditions":     [],
            "steps":             protocol_steps,
            "success_criteria":  [step_checks[-1]] if step_checks and step_checks[-1] else [],
            "abort_criteria":    ["No progress toward skill objective after several moves"],
            "expected_duration": max(len(protocol_steps) * 3, 6),
            "step_checks":       step_checks,
            "predicate_success": [c for c in step_checks if c],
            "predicate_abort":   [],
            "source":            "stage2_mega_skill",
        },
        "contract": {
            "eff_add":      [c for c in step_checks if c],
            "eff_del":      [],
            "eff_event":    [],
        },
        "exemplars":           [exemplar] if exemplar else [],
        "failure_exemplars":   [],
        "sub_episodes":        [],
        "expected_tag_pattern": compressed,
        "feasible_tasks":      [target_task],
        "verified_tasks":      [],
        "confidence_tag":      "candidate",
        "retired":             False,
        "n_instances":         0,
    }

    return {"skill": seed_skill, "report": {"skill_id": skill_id, "n_instances": 0}}


# ── Driver ─────────────────────────────────────────────────────────

def generate_for_target(
    target_task: str,
    target_domain: str,
    target_genre: str,
    mega_skills: List[Dict],
    max_seeds: int,
) -> Tuple[List[Dict], Dict]:
    """Generate seed bank entries for one target task."""
    ranked = rank_mega_skills_for_target(mega_skills, target_domain, target_genre)

    seeds = []
    coverage_stats = {
        "total_mega_skills_considered": len(ranked),
        "seeds_with_icl": 0,
        "seeds_without_icl": 0,
        "exemplar_sources": Counter(),
        "plan_signatures": [],
    }

    for rank, (mega, score) in enumerate(ranked[:max_seeds]):
        members_icl = get_mega_skill_members_with_icl(mega)
        icl_item = select_icl_member(mega, target_domain, target_genre, members_icl)

        entry = build_seed_entry(mega, target_task, target_domain, icl_item, rank)
        seeds.append(entry)

        coverage_stats["plan_signatures"].append(mega["reasoning_plan"])
        if icl_item:
            coverage_stats["seeds_with_icl"] += 1
            coverage_stats["exemplar_sources"][icl_item["member"]["task"]] += 1
        else:
            coverage_stats["seeds_without_icl"] += 1

    return seeds, coverage_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-targets", action="store_true", default=True)
    ap.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    mega_skills = load_mega_skills()
    print(f"Loaded {len(mega_skills)} mega-skill templates from {MEGA_FILE.name}")

    targets = []
    for t, info in GAME_TARGETS.items():
        targets.append((t, info["domain"], info["genre"]))
    for t, info in WEB_TARGETS.items():
        targets.append((t, info["domain"], info["genre"]))
    for t, info in VR_TARGETS.items():
        targets.append((t, info["domain"], info["genre"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting seed banks to {args.out_dir}\n")

    print(f"{'Target':<25} {'Dom':<5} {'Genre':<11} {'Seeds':>5} {'ICL':>4} {'Exemplar sources'}")
    print("-" * 100)

    summary = {}
    for target_task, target_domain, target_genre in targets:
        seeds, stats = generate_for_target(
            target_task, target_domain, target_genre,
            mega_skills, args.max_seeds,
        )

        out_path = args.out_dir / target_task / "skill_bank.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for entry in seeds:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Print summary line
        exemplars_str = ", ".join(
            f"{t}({c})" for t, c in stats["exemplar_sources"].most_common(3)
        ) or "(none)"
        print(f"  {target_task:<23} {target_domain:<5} {target_genre:<11} "
              f"{len(seeds):>5} {stats['seeds_with_icl']:>4} {exemplars_str}")

        summary[target_task] = {
            "domain":              target_domain,
            "genre":               target_genre,
            "n_seeds":             len(seeds),
            "n_seeds_with_icl":    stats["seeds_with_icl"],
            "n_seeds_no_icl":      stats["seeds_without_icl"],
            "exemplar_sources":    dict(stats["exemplar_sources"]),
            "plan_signatures":     stats["plan_signatures"],
            "output_path":         str(out_path),
        }

    # Write top-level summary
    summary_path = args.out_dir / "SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
