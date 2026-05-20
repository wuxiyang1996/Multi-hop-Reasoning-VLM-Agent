"""Stage 2 seed generator: use mega-skill clusters to transfer
best-run skills from Phase 1 games to Phase 2 holdout games.

For each holdout game, finds skills from its genre-matched source games
that share a mega-skill family, then copies them with adapted metadata.
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BANKS_DIR = REPO / "frontier_data" / "output" / "per_task_banks"
CLUSTERS_FILE = REPO / "frontier_data" / "output" / "mega_skill_clusters.json"
LABELS_FILE = REPO / "frontier_data" / "output" / "mega_skill_labels_final.json"
OUT_DIR = REPO / "frontier_data" / "output" / "stage2_seed_banks"

# Phase 2 holdout → genre-matched Phase 1 sources
TRANSFER_MAP = {
    "gymv_space_harrier_ii":  ["gymv_thunder_force_iii"],
    "gymv_airstriker":        ["gymv_thunder_force_iii", "gymv_strider"],
    "gymv_altered_beast":     ["gymv_streets_of_rage_2", "gymv_strider"],
    "gymv_dynamite_headdy":   ["gymv_strider", "gymv_thunder_force_iii", "gymv_columns"],
    "twenty_forty_eight":     ["gymv_columns", "candy_crush"],
    "super_mario":            ["gymv_strider", "gymv_streets_of_rage_2"],
}

MAX_SEEDS_PER_SOURCE = 16

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
    skill_to_family = build_skill_to_family()
    
    # Build family → skills index from source banks
    source_skills_by_family = defaultdict(list)
    for sources in TRANSFER_MAP.values():
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
    
    for target, sources in TRANSFER_MAP.items():
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
