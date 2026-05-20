#!/usr/bin/env python3
"""Stage 2 seed generator using cognitive-signature-based clustering.

Selects seeds for Phase 2 holdout targets (games, web tasks, VR tasks)
by matching cognitive loop families + intent sub-families between source
and target domains.

Transfer is driven by cognitive signature affinity, NOT genre heuristics.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BANKS_DIR = ROOT / "frontier_data" / "output" / "per_task_banks"
CLUSTERS_FILE = ROOT / "frontier_data" / "output" / "mega_skill_clusters_v2.json"
REASONING_FILE = ROOT / "frontier_data" / "output" / "reasoning_aligned_mega_skills.json"
OUT_DIR = ROOT / "frontier_data" / "output" / "stage2_seed_banks"

# Source tasks with trained skill banks (Stage 1 best runs)
SOURCE_TASKS = [
    "gymv_thunder_force_iii",
    "gymv_streets_of_rage_2",
    "gymv_strider",
    "gymv_columns",
    "candy_crush",
]

# Also include Temporal_ variants as aliases
TEMPORAL_MAP = {
    "gymv_thunder_force_iii": "Temporal_ThunderForceIII-v0",
    "gymv_strider": "Temporal_Strider-v0",
    "gymv_columns": "Temporal_Columns-v0",
    "gymv_streets_of_rage_2": "Temporal_StreetsOfRage2-v0",
    "Temporal_ThunderForceIII-v0": "gymv_thunder_force_iii",
    "Temporal_Strider-v0": "gymv_strider",
    "Temporal_Columns-v0": "gymv_columns",
    "Temporal_StreetsOfRage2-v0": "gymv_streets_of_rage_2",
    "Temporal_Airstriker-v0": "gymv_airstriker",
    "Temporal_AlteredBeast-v0": "gymv_altered_beast",
    "Temporal_DynamiteHeaddy-v0": "gymv_dynamite_headdy",
    "Temporal_SpaceHarrierII-v0": "gymv_space_harrier_ii",
}

# All available source tasks (including Temporal variants and non-game banks)
ALL_SOURCE_BANKS = set()

# Target tasks with genre hints for intra-domain differentiation
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

# Genre-to-intent affinity: which intents are most useful for each genre
GENRE_INTENT_AFFINITY = {
    "shooter":    ["ATTACK", "EVADE", "NAVIGATE", "SURVIVE", "POSITION", "COLLECT"],
    "brawler":    ["ATTACK", "EVADE", "SURVIVE", "EXPLORE", "DEFEND", "POSITION"],
    "platformer": ["NAVIGATE", "EXPLORE", "ATTACK", "COLLECT", "EVADE", "SURVIVE"],
    "puzzle":     ["CLEAR", "SETUP", "POSITION", "EXECUTE", "OPTIMIZE", "MERGE"],
    "web":        [],  # no intent filtering for web
    "vr":         [],  # no intent filtering for vr
}

# Source genre classification
SOURCE_GENRES = {
    "gymv_thunder_force_iii": "shooter",
    "gymv_streets_of_rage_2": "brawler",
    "gymv_strider":           "platformer",
    "gymv_columns":           "puzzle",
    "candy_crush":            "puzzle",
    "tetris":                 "puzzle",
    "twenty_forty_eight":     "puzzle",
    "super_mario":            "platformer",
    "Temporal_ThunderForceIII-v0": "shooter",
    "Temporal_StreetsOfRage2-v0":  "brawler",
    "Temporal_Strider-v0":        "platformer",
    "Temporal_Columns-v0":        "puzzle",
    "Temporal_Airstriker-v0":     "shooter",
    "Temporal_AlteredBeast-v0":   "brawler",
    "Temporal_DynamiteHeaddy-v0": "platformer",
    "Temporal_SpaceHarrierII-v0": "shooter",
    "miniwob":                "web",
    "webshop":                "web",
    "siv_bench":              "vr",
    "tir_bench":              "vr",
    "video_holmes":           "vr",
    "visual_toolbench":       "vr",
}

# Genre similarity for source ranking
GENRE_SIMILARITY = {
    ("shooter", "shooter"): 1.0,
    ("shooter", "brawler"): 0.5,
    ("shooter", "platformer"): 0.4,
    ("shooter", "puzzle"): 0.1,
    ("brawler", "brawler"): 1.0,
    ("brawler", "platformer"): 0.6,
    ("brawler", "puzzle"): 0.1,
    ("platformer", "platformer"): 1.0,
    ("platformer", "puzzle"): 0.2,
    ("puzzle", "puzzle"): 1.0,
    ("web", "web"): 1.0,
    ("vr", "vr"): 1.0,
    ("web", "vr"): 0.5,
    ("vr", "web"): 0.5,
}

MAX_SEEDS = 50


def determine_domain(task):
    if task.startswith("gymv_") or task.startswith("Temporal_"):
        return "GAME"
    if task in ("candy_crush", "tetris", "twenty_forty_eight", "super_mario"):
        return "GAME"
    if task in ("miniwob", "webshop") or task.startswith("miniwob") or task.startswith("webshop"):
        return "WEB"
    return "VR"


def load_clusters():
    with open(CLUSTERS_FILE) as f:
        return json.load(f)


def load_reasoning_mega_skills():
    """Load cross-domain mega-skills from build_reasoning_aligned_bank.py."""
    if not REASONING_FILE.exists():
        return {}
    with open(REASONING_FILE) as f:
        data = json.load(f)
    # Build (task, skill_id) → reasoning_plan mapping
    mapping = {}
    for mega in data.get("mega_skills", []):
        plan = mega.get("reasoning_plan", "")
        domains = set(mega.get("domains", []))
        is_cross_domain = len(domains) >= 2 and (
            ("GAME" in domains and ("WEB" in domains or "VR" in domains))
            or ("WEB" in domains and "VR" in domains)
        )
        for domain_group, members in mega.get("members_by_domain", {}).items():
            for m in members:
                key = (m["task"], m["skill_id"])
                mapping[key] = {
                    "reasoning_plan": plan,
                    "is_cross_domain_bridge": is_cross_domain,
                    "mega_domains": sorted(domains),
                }
    return mapping


def build_skill_index(clusters):
    """Build (task, skill_id) → {family, intent, signature, domain} index.

    Enriches with reasoning-plan-aligned mega-skill data when available.
    """
    reasoning = load_reasoning_mega_skills()

    index = {}
    for family_name, family_info in clusters["families"].items():
        for sk in family_info["skills"]:
            key = (sk["task"], sk["skill_id"])
            entry = {
                "family": family_name,
                "intent": sk.get("intent", ""),
                "signature": sk.get("signature", ""),
                "domain": sk.get("domain", ""),
                "reasoning_plan": "",
                "is_cross_domain_bridge": False,
            }
            if key in reasoning:
                entry["reasoning_plan"] = reasoning[key]["reasoning_plan"]
                entry["is_cross_domain_bridge"] = reasoning[key]["is_cross_domain_bridge"]
            index[key] = entry
    return index


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


def compute_affinity(source_profile, target_domain):
    """Compute transfer affinity score for a source task to a target domain.

    Based on how many of the source's skills belong to cognitive families
    that are transferable to the target domain.
    """
    TRANSFER_TARGETS = {
        "reactive_execute": {"GAME"},
        "act_and_verify": {"GAME", "WEB", "VR"},
        "decide_act_verify": {"GAME", "WEB", "VR"},
        "observe_decide_verify": {"GAME", "WEB", "VR"},
        "perceive_decide_act": {"GAME", "WEB", "VR"},
        "evaluate_act_perceive": {"GAME", "WEB"},
        "deliberate_reason": {"GAME", "WEB", "VR"},
        "strategic_navigate": {"GAME"},
        "navigate_orient": {"GAME"},
    }

    total = 0
    transferable = 0
    for family, count in source_profile.items():
        total += count
        targets = TRANSFER_TARGETS.get(family, set())
        if target_domain in targets:
            transferable += count

    return transferable / max(total, 1)


def get_source_profile(task, skill_index):
    """Get cognitive family distribution for a source task."""
    profile = Counter()
    bank = load_bank(task)
    for entry in bank:
        sk = entry.get("skill", entry)
        sid = sk.get("skill_id", "")
        info = skill_index.get((task, sid))
        if not info:
            alt = TEMPORAL_MAP.get(task)
            if alt:
                info = skill_index.get((alt, sid))
        if info:
            profile[info["family"]] += 1
    return profile


def get_genre_sim(g1, g2):
    if g1 == g2:
        return 1.0
    return GENRE_SIMILARITY.get((g1, g2), GENRE_SIMILARITY.get((g2, g1), 0.2))


def select_seeds(source_tasks, target_domain, target_task, target_genre,
                 skill_index, max_seeds=MAX_SEEDS):
    """Select seed skills prioritizing cognitive transferability + genre affinity."""
    TRANSFER_TARGETS = {
        "reactive_execute": {"GAME"},
        "act_and_verify": {"GAME", "WEB", "VR"},
        "decide_act_verify": {"GAME", "WEB", "VR"},
        "observe_decide_verify": {"GAME", "WEB", "VR"},
        "perceive_decide_act": {"GAME", "WEB", "VR"},
        "evaluate_act_perceive": {"GAME", "WEB"},
        "deliberate_reason": {"GAME", "WEB", "VR"},
        "strategic_navigate": {"GAME"},
        "navigate_orient": {"GAME"},
    }

    preferred_intents = set(GENRE_INTENT_AFFINITY.get(target_genre, []))

    candidates = []
    for src in source_tasks:
        src_genre = SOURCE_GENRES.get(src, "unknown")
        genre_sim = get_genre_sim(src_genre, target_genre)

        bank = load_bank(src)
        for entry in bank:
            sk = entry.get("skill", entry)
            sid = sk.get("skill_id", "")
            info = skill_index.get((src, sid))
            if not info:
                alt = TEMPORAL_MAP.get(src)
                if alt:
                    info = skill_index.get((alt, sid))
            if not info:
                info = {"family": "unknown", "intent": "GENERIC", "signature": ""}

            family = info["family"]
            intent = info["intent"]
            targets = TRANSFER_TARGETS.get(family, set())
            is_transferable = target_domain in targets
            is_reasoning_bridge = info.get("is_cross_domain_bridge", False)

            src_domain = determine_domain(src)

            score = 0.0
            if is_transferable:
                score += 2.0
            if family in ("decide_act_verify", "observe_decide_verify",
                          "perceive_decide_act", "deliberate_reason"):
                score += 1.0
            if is_reasoning_bridge:
                score += 3.0
            if src_domain == target_domain:
                score += 4.0  # same-domain source gets highest priority
            score += genre_sim * 2.0
            if intent in preferred_intents:
                score += 1.5

            candidates.append({
                "entry": entry,
                "source": src,
                "family": family,
                "intent": intent,
                "signature": info["signature"],
                "score": score,
                "is_transferable": is_transferable,
                "genre_sim": genre_sim,
            })

    candidates.sort(key=lambda x: -x["score"])

    # Separate cross-domain bridge candidates
    cross_domain_bridges = [
        c for c in candidates
        if determine_domain(c["source"]) != target_domain
        and c["is_transferable"]
    ]
    cross_domain_bridges.sort(key=lambda x: -x["score"])

    # Reserve slots for cross-domain bridge skills (min 20% of seeds)
    min_cross = max(8, max_seeds // 5)

    selected = []
    seen_family_intent = set()
    seen_sids = set()

    # Pass 1: cross-domain bridges first (one per family for diversity)
    cross_families_seen = set()
    for c in cross_domain_bridges:
        if len(selected) >= min_cross:
            break
        sid = c["entry"].get("skill", c["entry"]).get("skill_id", "")
        family = c["family"]
        if family not in cross_families_seen and sid not in seen_sids:
            adapted = adapt_skill(c["entry"], target_task, c["source"], c["family"], c["intent"])
            selected.append(adapted)
            seen_family_intent.add((family, c["intent"]))
            seen_sids.add(sid)
            cross_families_seen.add(family)

    # Pass 1b: fill remaining cross-domain quota with best bridge skills
    for c in cross_domain_bridges:
        if len(selected) >= min_cross:
            break
        sid = c["entry"].get("skill", c["entry"]).get("skill_id", "")
        if sid not in seen_sids:
            adapted = adapt_skill(c["entry"], target_task, c["source"], c["family"], c["intent"])
            selected.append(adapted)
            seen_family_intent.add((c["family"], c["intent"]))
            seen_sids.add(sid)

    # Pass 2: one per (family, intent) for diversity from ALL candidates
    for c in candidates:
        if len(selected) >= max_seeds:
            break
        key = (c["family"], c["intent"])
        sid = c["entry"].get("skill", c["entry"]).get("skill_id", "")
        if key not in seen_family_intent and sid not in seen_sids:
            adapted = adapt_skill(c["entry"], target_task, c["source"], c["family"], c["intent"])
            selected.append(adapted)
            seen_family_intent.add(key)
            seen_sids.add(sid)

    # Pass 3: fill remaining slots with highest-scored skills
    for c in candidates:
        if len(selected) >= max_seeds:
            break
        sid = c["entry"].get("skill", c["entry"]).get("skill_id", "")
        if sid not in seen_sids:
            adapted = adapt_skill(c["entry"], target_task, c["source"], c["family"], c["intent"])
            selected.append(adapted)
            seen_sids.add(sid)

    return selected


def adapt_skill(entry, target_task, source_task, family, intent):
    """Tag a skill for transfer tracking."""
    entry = json.loads(json.dumps(entry))
    sk = entry.get("skill", entry)

    tags = sk.get("tags", [])
    tags.extend([
        f"transferred_from:{source_task}",
        f"cognitive_family:{family}",
        f"intent:{intent}",
        "stage2_seed",
    ])
    sk["tags"] = list(set(tags))
    sk["derived_from"] = f"{source_task}::{sk.get('skill_id', '')}"
    sk["confidence_tag"] = "candidate"

    if "feasible_tasks" in sk:
        sk["feasible_tasks"] = [target_task]

    if "skill" in entry:
        entry["skill"] = sk
    return entry


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-targets", action="store_true", default=True)
    parser.add_argument("--web-targets", action="store_true", default=False)
    parser.add_argument("--vr-targets", action="store_true", default=False)
    parser.add_argument("--all-targets", action="store_true", default=False,
                        help="Generate seeds for game + web + vr targets")
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    args = parser.parse_args()

    clusters = load_clusters()
    skill_index = build_skill_index(clusters)

    # Auto-discover ALL source banks
    # Tasks that are ONLY targets (no pre-trained bank of their own)
    pure_targets = {"gymv_airstriker", "gymv_altered_beast",
                    "gymv_dynamite_headdy", "gymv_space_harrier_ii",
                    "miniwob_unseen", "webshop_new", "vr_new_bench"}

    available_sources = []
    web_sources = []
    vr_sources = []
    for task_dir in sorted(BANKS_DIR.iterdir()):
        task = task_dir.name
        if not (task_dir / "skill_bank.jsonl").exists():
            continue
        if task in pure_targets:
            continue
        domain = determine_domain(task)
        if domain == "GAME":
            available_sources.append(task)
        elif domain == "WEB":
            web_sources.append(task)
        elif domain == "VR":
            vr_sources.append(task)

    print(f"Available GAME sources: {available_sources}")
    print(f"Available WEB sources: {web_sources}")
    print(f"Available VR sources: {vr_sources}")
    print()

    # Build source profiles
    print("=== Source Task Cognitive Profiles ===")
    for src in available_sources + web_sources + vr_sources:
        profile = get_source_profile(src, skill_index)
        domain = determine_domain(src)
        game_aff = compute_affinity(profile, "GAME")
        web_aff = compute_affinity(profile, "WEB")
        vr_aff = compute_affinity(profile, "VR")
        fam_str = ", ".join(f"{f}={c}" for f, c in profile.most_common(4))
        print(f"  {src} [{domain}]: GAME={game_aff:.0%} WEB={web_aff:.0%} VR={vr_aff:.0%}")
        print(f"    {fam_str}")
    print()

    # Determine targets
    all_sources = available_sources + web_sources + vr_sources

    targets = []
    if args.all_targets or args.game_targets:
        for t, info in GAME_TARGETS.items():
            targets.append((t, info["domain"], info["genre"], all_sources))
    if args.all_targets or args.web_targets:
        for t, info in WEB_TARGETS.items():
            targets.append((t, info["domain"], info["genre"], all_sources))
    if args.all_targets or args.vr_targets:
        for t, info in VR_TARGETS.items():
            targets.append((t, info["domain"], info["genre"], all_sources))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating Seed Banks ===")
    print()
    for target_task, target_domain, target_genre, sources in targets:
        seeds = select_seeds(sources, target_domain, target_task, target_genre,
                             skill_index, max_seeds=args.max_seeds)

        # Write
        out_path = OUT_DIR / target_task / "skill_bank.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for entry in seeds:
                f.write(json.dumps(entry) + "\n")

        # Stats
        family_dist = Counter()
        intent_dist = Counter()
        source_dist = Counter()
        bridge_count = 0
        for entry in seeds:
            sk = entry.get("skill", entry)
            for tag in sk.get("tags", []):
                if tag.startswith("cognitive_family:"):
                    fam = tag.split(":", 1)[1]
                    family_dist[fam] += 1
                    if fam in ("act_and_verify", "decide_act_verify",
                               "perceive_decide_act", "observe_decide_verify",
                               "deliberate_reason"):
                        bridge_count += 1
                elif tag.startswith("intent:"):
                    intent_dist[tag.split(":", 1)[1]] += 1
                elif tag.startswith("transferred_from:"):
                    source_dist[tag.split(":", 1)[1]] += 1

        fam_str = ", ".join(f"{f}({c})" for f, c in family_dist.most_common(5))
        src_str = ", ".join(f"{s}({c})" for s, c in source_dist.most_common())
        intent_str = ", ".join(f"{i}({c})" for i, c in intent_dist.most_common(6))

        # Cross-domain stats
        domain_dist = Counter()
        for s, c in source_dist.items():
            domain_dist[determine_domain(s)] += c
        same_d = domain_dist.get(target_domain, 0)
        cross_d = len(seeds) - same_d
        dom_str = ", ".join(f"{d}={c}" for d, c in domain_dist.most_common())

        print(f"  {target_task} [{target_domain}]:")
        print(f"    Seeds: {len(seeds)}, Bridge families: {bridge_count}/{len(seeds)}")
        print(f"    Domain mix: {dom_str} (cross-domain: {cross_d}/{len(seeds)})")
        print(f"    Families: {fam_str}")
        print(f"    Sources: {src_str}")
        print(f"    → {out_path}")
        print()


if __name__ == "__main__":
    main()
