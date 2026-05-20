"""Stage 2 seed generator: use mega-skill clusters to transfer
best-run skills from Phase 1 games to Phase 2 holdout games.

Phase 1→2 transfer map is computed AUTOMATICALLY via LLM-based game
similarity scoring (--auto-match, default) or can be overridden with
a hardcoded map (--manual-match).
"""
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BANKS_DIR = REPO / "frontier_data" / "output" / "per_task_banks"
CLUSTERS_FILE = REPO / "frontier_data" / "output" / "mega_skill_clusters.json"
LABELS_FILE = REPO / "frontier_data" / "output" / "mega_skill_labels_final.json"
OUT_DIR = REPO / "frontier_data" / "output" / "stage2_seed_banks"
MATCH_CACHE = REPO / "frontier_data" / "output" / "auto_transfer_map.json"

PHASE1_GAMES = [
    "gymv_thunder_force_iii",
    "gymv_streets_of_rage_2",
    "gymv_strider",
    "gymv_columns",
    "tetris",
    "candy_crush",
]

PHASE2_GAMES = [
    "gymv_space_harrier_ii",
    "gymv_airstriker",
    "gymv_altered_beast",
    "gymv_dynamite_headdy",
    "twenty_forty_eight",
    "super_mario",
]

MANUAL_TRANSFER_MAP = {
    "gymv_space_harrier_ii":  ["gymv_thunder_force_iii"],
    "gymv_airstriker":        ["gymv_thunder_force_iii", "gymv_strider"],
    "gymv_altered_beast":     ["gymv_streets_of_rage_2", "gymv_strider"],
    "gymv_dynamite_headdy":   ["gymv_strider", "gymv_thunder_force_iii", "gymv_columns"],
    "twenty_forty_eight":     ["gymv_columns", "candy_crush"],
    "super_mario":            ["gymv_strider", "gymv_streets_of_rage_2"],
}

MAX_SEEDS_PER_SOURCE = 16


# --------------- Auto matching via LLM + mega-skill overlap ---------------

def _get_skill_families_per_game(skill_to_family):
    """Return {game: set_of_families} from the labels."""
    game_families = defaultdict(set)
    for (task, sid), family in skill_to_family.items():
        game_families[task].add(family)
    return dict(game_families)


def _mega_skill_overlap(game_families, src, tgt):
    """Jaccard overlap of mega-skill families between two games."""
    s = game_families.get(src, set())
    t = game_families.get(tgt, set())
    if not s and not t:
        return 0.0
    return len(s & t) / len(s | t) if (s | t) else 0.0


def _build_game_descriptions():
    """Build a short description for each game for LLM similarity scoring."""
    descriptions = {
        "gymv_thunder_force_iii": "ThunderForce III: side-scrolling space shooter, dodge bullets, shoot enemies, collect power-ups",
        "gymv_streets_of_rage_2": "Streets of Rage 2: side-scrolling beat-em-up brawler, melee combat, combos, health management",
        "gymv_strider": "Strider: side-scrolling action platformer, wall climbing, sword attacks, boss fights, traversal",
        "gymv_columns": "Columns: falling-block puzzle, match 3+ gems vertically/horizontally/diagonally, clear rows",
        "tetris": "Tetris: falling-block spatial puzzle, rotate and place tetrominoes, clear complete rows",
        "candy_crush": "Candy Crush: match-3 swap puzzle, create combos, clear objectives within move limits",
        "gymv_space_harrier_ii": "Space Harrier II: third-person rail shooter, dodge obstacles, shoot enemies, fast reflexes",
        "gymv_airstriker": "Airstriker: top-down vertical shooter, dodge bullets, shoot waves of enemies, collect upgrades",
        "gymv_altered_beast": "Altered Beast: side-scrolling brawler/platformer, punch/kick enemies, collect power orbs, transform",
        "gymv_dynamite_headdy": "Dynamite Headdy: side-scrolling action platformer, swap head powers, boss patterns, precise jumping",
        "twenty_forty_eight": "2048: number tile puzzle, slide tiles to merge matching numbers, strategic positioning",
        "super_mario": "Super Mario: side-scrolling platformer, jump on enemies, collect coins, navigate obstacles, reach flag",
    }
    return descriptions


def auto_match_via_llm(skill_to_family, top_k=3):
    """Use LLM to score game similarity and pick best sources per holdout."""
    if MATCH_CACHE.exists():
        cached = json.load(open(MATCH_CACHE))
        print(f"[auto-match] Loaded cached transfer map from {MATCH_CACHE}")
        return cached

    from openai import OpenAI
    sys.path.insert(0, "/workspace")
    from keys import openai_api_key
    client = OpenAI(api_key=openai_api_key)

    game_families = _get_skill_families_per_game(skill_to_family)
    descriptions = _build_game_descriptions()

    overlap_info = []
    for tgt in PHASE2_GAMES:
        for src in PHASE1_GAMES:
            jacc = _mega_skill_overlap(game_families, src, tgt)
            # Also check Temporal_ variants
            for alt_src in TEMPORAL_MAP.values():
                j2 = _mega_skill_overlap(game_families, alt_src, tgt)
                jacc = max(jacc, j2)
            for alt_tgt in TEMPORAL_MAP.values():
                j3 = _mega_skill_overlap(game_families, src, alt_tgt)
                jacc = max(jacc, j3)
            overlap_info.append((tgt, src, jacc))

    # Build prompt with both descriptions and skill overlap data
    lines = []
    lines.append("## Phase 1 Source Games (with descriptions)")
    for g in PHASE1_GAMES:
        desc = descriptions.get(g, g)
        fams = game_families.get(g, set())
        for alt in [g, TEMPORAL_MAP.get(g, "")]:
            fams |= game_families.get(alt, set())
        lines.append(f"- {g}: {desc} [{len(fams)} mega-skill families]")

    lines.append("\n## Phase 2 Target Games (need skill transfer)")
    for g in PHASE2_GAMES:
        desc = descriptions.get(g, g)
        lines.append(f"- {g}: {desc}")

    lines.append("\n## Mega-skill Family Overlap (Jaccard similarity)")
    for tgt in PHASE2_GAMES:
        relevant = [(s, j) for t, s, j in overlap_info if t == tgt]
        relevant.sort(key=lambda x: -x[1])
        scores = ", ".join(f"{s}={j:.2f}" for s, j in relevant)
        lines.append(f"- {tgt}: {scores}")

    prompt = "\n".join(lines)

    system = f"""\
You are a game AI researcher selecting the best source games for skill transfer.

For each Phase 2 target game, select the top {top_k} Phase 1 source games
that would provide the most useful transferable skills. Consider:

1. GAMEPLAY SIMILARITY is the PRIMARY signal. Same genre/mechanics transfer
   best: shooter→shooter, puzzle→puzzle, platformer→platformer, brawler→brawler.
   A puzzle game (candy_crush, tetris, gymv_columns) should ALWAYS be a top
   source for another puzzle game (twenty_forty_eight).
2. MEGA-SKILL OVERLAP: Use as a secondary signal — higher overlap confirms
   the gameplay match. But do NOT let a high-skill-count action game dominate
   over a lower-count same-genre game.
3. COMPLEMENTARY SKILLS: After picking the best same-genre source(s), add
   cross-genre sources for cognitive diversity.
4. IMPORTANT: Only select from Phase 1 source games. Do NOT select Phase 2
   holdout games as sources (they have no trained skill banks).
5. Every Phase 1 source game should appear at least once across all targets.
   candy_crush and tetris are valid puzzle sources — do not ignore them.

Output a JSON object mapping each Phase 2 game to a ranked list of 1-{top_k}
Phase 1 source games (best first). Include a brief reason for each pick.

Format:
{{
  "matches": {{
    "<target_game>": {{
      "sources": ["<src1>", "<src2>", ...],
      "reason": "<brief explanation>"
    }},
    ...
  }}
}}

Output ONLY the JSON."""

    print("[auto-match] Querying LLM for game similarity matching...")
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_completion_tokens=2000,
    )

    text = resp.choices[0].message.content.strip()
    match = re.search(r'\{[\s\S]+\}', text)
    if not match:
        print("[auto-match] WARNING: LLM returned no valid JSON, falling back to manual map")
        return MANUAL_TRANSFER_MAP

    result = json.loads(match.group())
    matches = result.get("matches", result)

    transfer_map = {}
    print("\n[auto-match] LLM Transfer Map:")
    for tgt, info in matches.items():
        if isinstance(info, dict):
            sources = info.get("sources", [])
            reason = info.get("reason", "")
        elif isinstance(info, list):
            sources = info
            reason = ""
        else:
            continue
        transfer_map[tgt] = sources
        print(f"  {tgt} ← {sources}")
        if reason:
            print(f"    reason: {reason}")

    # Cache the result
    MATCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MATCH_CACHE, "w") as f:
        json.dump({"transfer_map": transfer_map, "llm_response": result}, f, indent=2)
    print(f"\n[auto-match] Cached → {MATCH_CACHE}")

    return transfer_map

TEMPORAL_MAP = {
    "gymv_thunder_force_iii": "Temporal_ThunderForceIII-v0",
    "gymv_strider": "Temporal_Strider-v0",
    "gymv_columns": "Temporal_Columns-v0",
    "gymv_streets_of_rage_2": "Temporal_StreetsOfRage2-v0",
    "gymv_space_harrier_ii": "Temporal_SpaceHarrierII-v0",
    "gymv_airstriker": "Temporal_Airstriker-v0",
    "gymv_altered_beast": "Temporal_AlteredBeast-v0",
    "gymv_dynamite_headdy": "Temporal_DynamiteHeaddy-v0",
}


def _find_family(mapping, task, sid):
    family = mapping.get((task, sid))
    if family:
        return family
    alt = TEMPORAL_MAP.get(task, task)
    if alt != task:
        family = mapping.get((alt, sid))
    return family


def load_bank(task):
    path = BANKS_DIR / task / "skill_bank.jsonl"
    if not path.exists():
        return []
    skills = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                skills.append(json.loads(line))
    return skills


def build_skill_to_family():
    """Map (task, skill_id) → mega-skill family using labels file."""
    # Try final labels first, fall back to raw labels
    for path in [LABELS_FILE, REPO / "frontier_data" / "output" / "mega_skill_labels.json"]:
        if not path.exists():
            continue
        data = json.load(open(path))
        mapping = {}
        mega = data.get("mega_skills", {})
        for family_name, info in mega.items():
            for skill_entry in info.get("skills", []):
                task = skill_entry.get("task", "")
                sid = skill_entry.get("skill_id", "")
                if task and sid:
                    mapping[(task, sid)] = family_name
        if mapping:
            return mapping
    return {}


def adapt_skill_for_target(skill_entry, target_game, source_game, family):
    """Adapt a skill entry for a new target game."""
    entry = json.loads(json.dumps(skill_entry))
    sk = entry.get("skill", entry)
    
    tags = sk.get("tags", [])
    tags.extend([
        f"transferred_from:{source_game}",
        f"mega_family:{family}",
        "stage2_seed",
    ])
    sk["tags"] = list(set(tags))
    
    sk["derived_from"] = f"{source_game}::{sk.get('skill_id', '')}"
    sk["confidence_tag"] = "candidate"
    
    if "feasible_tasks" in sk:
        sk["feasible_tasks"] = [target_game]
    
    if "skill" in entry:
        entry["skill"] = sk
    return entry


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-match", action="store_true",
                        help="Use hardcoded TRANSFER_MAP instead of LLM auto-matching")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Max source games per holdout (default: 3)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Force re-compute auto transfer map")
    args = parser.parse_args()

    skill_to_family = build_skill_to_family()

    if args.manual_match:
        transfer_map = MANUAL_TRANSFER_MAP
        print("[mode] Using MANUAL hardcoded transfer map\n")
    else:
        if args.clear_cache and MATCH_CACHE.exists():
            MATCH_CACHE.unlink()
        transfer_map = auto_match_via_llm(skill_to_family, top_k=args.top_k)
        print(f"[mode] Using AUTO LLM-based transfer map\n")

    # Build family → skills index from source banks
    source_skills_by_family = defaultdict(list)
    for sources in transfer_map.values():
        for src in sources:
            bank = load_bank(src)
            for entry in bank:
                sk = entry.get("skill", entry)
                sid = sk.get("skill_id", "")
                # Try gymv_ key and Temporal_ key
                family = _find_family(skill_to_family, src, sid)
                if family:
                    source_skills_by_family[family].append((src, entry))
    
    print(f"Loaded {len(skill_to_family)} skill→family mappings")
    print(f"Found {len(source_skills_by_family)} families with source skills")
    print()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for target, sources in transfer_map.items():
        print(f"=== {target} ← {sources} ===")
        
        # Collect skills from source games, prioritize by family diversity
        seed_skills = []
        seen_families = set()
        
        for src in sources:
            bank = load_bank(src)
            src_with_family = []
            for entry in bank:
                sk = entry.get("skill", entry)
                sid = sk.get("skill_id", "")
                family = _find_family(skill_to_family, src, sid) or "unknown"
                src_with_family.append((entry, family))
            
            # First pass: one skill per family for diversity
            for entry, family in src_with_family:
                if family not in seen_families and len(seed_skills) < MAX_SEEDS_PER_SOURCE * len(sources):
                    adapted = adapt_skill_for_target(entry, target, src, family)
                    seed_skills.append(adapted)
                    seen_families.add(family)
            
            # Second pass: fill remaining up to cap
            for entry, family in src_with_family:
                if len(seed_skills) >= MAX_SEEDS_PER_SOURCE * len(sources):
                    break
                sk = entry.get("skill", entry)
                sid = sk.get("skill_id", "")
                if not any(s.get("skill", s).get("skill_id") == sid for s in seed_skills):
                    adapted = adapt_skill_for_target(entry, target, src, family)
                    seed_skills.append(adapted)
        
        # Write seed bank
        out_path = OUT_DIR / target / "skill_bank.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for entry in seed_skills:
                f.write(json.dumps(entry) + "\n")
        
        families_used = set()
        for entry in seed_skills:
            for t in entry.get("skill", entry).get("tags", []):
                if t.startswith("mega_family:"):
                    families_used.add(t.split(":", 1)[1])
        
        print(f"  Seeds: {len(seed_skills)} skills, {len(families_used)} families")
        print(f"  → {out_path}")
        print()


if __name__ == "__main__":
    main()
