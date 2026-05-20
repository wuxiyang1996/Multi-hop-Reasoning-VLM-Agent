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
}

MAX_SEEDS = 40


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


def build_skill_index(clusters):
    """Build (task, skill_id) → {family, intent, signature, domain} index."""
    index = {}
    for family_name, family_info in clusters["families"].items():
        for sk in family_info["skills"]:
            key = (sk["task"], sk["skill_id"])
            index[key] = {
                "family": family_name,
                "intent": sk.get("intent", ""),
                "signature": sk.get("signature", ""),
                "domain": sk.get("domain", ""),
            }
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
    # Family transferability from clusters_v2
    TRANSFER_TARGETS = {
        "reactive_execute": {"GAME"},
        "deliberate_select": {"GAME", "WEB"},
        "inferential_reason": {"VR", "WEB"},
        "retrieve_match_act": {"WEB", "VR"},
        "plan_transform": {"GAME", "WEB"},
        "explore_monitor": {"GAME"},
        "filter_aggregate": {"VR", "WEB"},
        "sequence_chain": {"GAME"},
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
        "deliberate_select": {"GAME", "WEB"},
        "inferential_reason": {"VR", "WEB"},
        "retrieve_match_act": {"WEB", "VR"},
        "plan_transform": {"GAME", "WEB"},
        "explore_monitor": {"GAME"},
        "filter_aggregate": {"VR", "WEB"},
        "sequence_chain": {"GAME"},
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

            # Score: combines cognitive transferability + genre match + intent match
            score = 0.0
            if is_transferable:
                score += 2.0
            if family in ("deliberate_select", "plan_transform",
                          "inferential_reason", "retrieve_match_act"):
                score += 1.0  # bridge family bonus
            score += genre_sim * 2.0  # genre similarity (0-2)
            if intent in preferred_intents:
                score += 1.5  # intent affinity bonus

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

    # Select with diversity constraints
    selected = []
    seen_family_intent = set()
    seen_sids = set()

    # Pass 1: one per (family, intent) for diversity, highest score first
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

    # Pass 2: fill remaining slots with highest-scored skills
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

    # Determine which source banks exist
    available_sources = []
    for src in SOURCE_TASKS:
        if (BANKS_DIR / src / "skill_bank.jsonl").exists():
            available_sources.append(src)
    # Also add non-game banks as sources for web/vr targets
    web_sources = []
    vr_sources = []
    for task_dir in BANKS_DIR.iterdir():
        task = task_dir.name
        if (task_dir / "skill_bank.jsonl").exists():
            domain = determine_domain(task)
            if domain == "WEB" and task not in ("miniwob_unseen", "webshop_new"):
                web_sources.append(task)
            elif domain == "VR" and task not in ("vr_new_bench",):
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
    targets = []
    if args.all_targets or args.game_targets:
        for t, info in GAME_TARGETS.items():
            targets.append((t, info["domain"], info["genre"], available_sources))
    if args.all_targets or args.web_targets:
        for t, info in WEB_TARGETS.items():
            targets.append((t, info["domain"], info["genre"],
                           available_sources + web_sources))
    if args.all_targets or args.vr_targets:
        for t, info in VR_TARGETS.items():
            targets.append((t, info["domain"], info["genre"],
                           available_sources + web_sources + vr_sources))

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
                    if fam in ("deliberate_select", "plan_transform",
                               "inferential_reason", "retrieve_match_act"):
                        bridge_count += 1
                elif tag.startswith("intent:"):
                    intent_dist[tag.split(":", 1)[1]] += 1
                elif tag.startswith("transferred_from:"):
                    source_dist[tag.split(":", 1)[1]] += 1

        fam_str = ", ".join(f"{f}({c})" for f, c in family_dist.most_common(5))
        src_str = ", ".join(f"{s}({c})" for s, c in source_dist.most_common())
        intent_str = ", ".join(f"{i}({c})" for i, c in intent_dist.most_common(6))

        print(f"  {target_task} [{target_domain}]:")
        print(f"    Seeds: {len(seeds)}, Bridge skills: {bridge_count}/{len(seeds)}")
        print(f"    Families: {fam_str}")
        print(f"    Sources: {src_str}")
        print(f"    Top intents: {intent_str}")
        print(f"    → {out_path}")
        print()


if __name__ == "__main__":
    main()
