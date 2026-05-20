#!/usr/bin/env python3
"""
Phase 2: Cluster raw mega-skill labels into canonical families.

Takes the free-form raw labels from extract_mega_skills.py and asks LLM
to merge semantically equivalent labels into a smaller canonical set.
Then re-assigns every skill to its canonical family and outputs a
codebook for the re-labeling pass (extract_mega_skills.py --relabel).

Usage:
    python frontier_data/scripts/cluster_mega_skills.py
"""

import json, sys, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

import importlib.util
_keys_path = ROOT.parent / "keys.py"
_spec = importlib.util.spec_from_file_location("keys", _keys_path)
_keys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_keys)

from openai import OpenAI

client = OpenAI(api_key=_keys.openai_api_key)

OUT_DIR = ROOT / "frontier_data" / "output"
RAW_FILE = OUT_DIR / "mega_skill_labels.json"
CLUSTERED_FILE = OUT_DIR / "mega_skill_clusters.json"
CODEBOOK_FILE = OUT_DIR / "mega_skill_codebook.json"

MERGE_SYSTEM = """\
You are a cognitive-science expert. Given a list of mega-skill labels \
with their procedures and skill counts, merge semantically equivalent \
labels into canonical families.

Rules:
1. Two labels are equivalent if they describe the SAME cognitive procedure \
   (e.g. PERCEIVE_DECIDE_ACT ≈ PERCEIVE_AND_ACT ≈ PERCEIVE_PLAN_AND_ACT).
2. Keep 10-25 canonical families — enough to be meaningful, few enough to be useful.
3. Choose the MOST descriptive name for each canonical family.
4. Output a JSON OBJECT mapping each raw label to its canonical family name.
   Example: {"PERCEIVE_DECIDE_ACT": "PERCEIVE_AND_ACT", "PERCEIVE_AND_ACT": "PERCEIVE_AND_ACT", ...}
5. EVERY input label must appear as a key. The value is the canonical name.
6. Output ONLY the JSON object, nothing else."""


def build_merge_prompt(raw_summary):
    """Build the merge prompt from raw labels."""
    lines = []
    for lbl, info in sorted(raw_summary["mega_skills"].items(),
                             key=lambda x: -x[1]["count"]):
        domains = "+".join(info["domains"])
        lines.append(
            f"- {lbl} ({info['count']} skills, {domains}): {info['procedure']}"
        )
    return "Merge these labels:\n\n" + "\n".join(lines)


def main():
    with open(RAW_FILE) as f:
        raw = json.load(f)

    labels = list(raw["mega_skills"].keys())
    print(f"Merging {len(labels)} raw labels...")

    # Split into batches if too many labels
    batch_size = 45
    canonical_map = {}

    for i in range(0, len(labels), batch_size):
        batch = labels[i:i+batch_size]
        batch_lines = []
        for lbl in batch:
            info = raw["mega_skills"][lbl]
            domains = "+".join(info["domains"])
            batch_lines.append(
                f"- {lbl} ({info['count']} skills, {domains}): {info['procedure']}"
            )
        prompt_text = "Merge these labels:\n\n" + "\n".join(batch_lines)

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            max_completion_tokens=2000,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\{[\s\S]+\}', text)
        if not match:
            print(f"ERROR parsing batch {i//batch_size}: {text[:200]}")
            for lbl in batch:
                canonical_map[lbl] = lbl
            continue
        batch_map = json.loads(match.group())
        canonical_map.update(batch_map)
        print(f"  batch {i//batch_size+1}: {len(batch_map)} labels mapped")

    # Iterative unification — merge canonical names down to 15-20 families
    TARGET_MIN, TARGET_MAX, MAX_ROUNDS = 15, 20, 5
    for rnd in range(1, MAX_ROUNDS + 1):
        canonical_names = sorted(set(canonical_map.values()))
        if len(canonical_names) <= TARGET_MAX:
            break
        print(f"  Unify round {rnd}: {len(canonical_names)} canonical names → target {TARGET_MIN}-{TARGET_MAX}...")

        canon_info = defaultdict(lambda: {"count": 0, "domains": set(), "procedure": ""})
        for raw_lbl, canon in canonical_map.items():
            info = raw["mega_skills"].get(raw_lbl)
            if info is None:
                continue
            canon_info[canon]["count"] += info["count"]
            canon_info[canon]["domains"].update(info["domains"])
            if not canon_info[canon]["procedure"]:
                canon_info[canon]["procedure"] = info["procedure"]

        lines = []
        for cn in sorted(canon_info, key=lambda k: -canon_info[k]["count"]):
            ci = canon_info[cn]
            doms = "+".join(sorted(ci["domains"]))
            lines.append(f"- {cn} ({ci['count']} skills, {doms}): {ci['procedure']}")

        unify_system = f"""\
You are a cognitive-science expert merging mega-skill families for
CROSS-DOMAIN transfer (GAME ↔ WEB ↔ VR).

GOAL: Reduce the families to {TARGET_MIN}-{TARGET_MAX} abstract families
that capture TRANSFERABLE cognitive procedures shared across domains.

Rules:
1. Aggressively merge families that share the same core cognitive loop
   (e.g. perceive→decide→act variants should become ONE family).
2. Prefer CROSS-DOMAIN families. Single-domain families should be merged
   into the closest cross-domain family whenever the core procedure matches.
3. Target {TARGET_MIN}-{TARGET_MAX} final families. No single family >25%.
4. Output a JSON object mapping EVERY input name to its final canonical name.
   You MUST map ALL {len(canonical_names)} input names — do NOT skip any.
5. Use SHORT_SNAKE_CASE names (2-4 words).
6. Output ONLY the JSON object, nothing else."""

        unify_prompt = "Merge these families:\n\n" + "\n".join(lines)

        # Split into batches if too many names
        batch_sz = 45
        if len(canonical_names) > batch_sz:
            full_unify_map = {}
            for bi in range(0, len(lines), batch_sz):
                batch_lines = lines[bi:bi + batch_sz]
                bp = f"Merge these families (batch {bi//batch_sz+1}, map EVERY name):\n\n" + "\n".join(batch_lines)
                resp2 = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": unify_system},
                        {"role": "user", "content": bp},
                    ],
                    temperature=0.0,
                    max_completion_tokens=8000,
                )
                text2 = resp2.choices[0].message.content.strip()
                match2 = re.search(r'\{[\s\S]+\}', text2)
                if match2:
                    bmap = json.loads(match2.group())
                    full_unify_map.update(bmap)
            unify_map = full_unify_map
        else:
            resp2 = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": unify_system},
                    {"role": "user", "content": unify_prompt},
                ],
                temperature=0.0,
                max_completion_tokens=8000,
            )
            text2 = resp2.choices[0].message.content.strip()
            match2 = re.search(r'\{[\s\S]+\}', text2)
            unify_map = json.loads(match2.group()) if match2 else {}

        if unify_map:
            mapped_count = 0
            for raw_lbl in canonical_map:
                old_canon = canonical_map[raw_lbl]
                new_canon = unify_map.get(old_canon)
                if new_canon and new_canon != old_canon:
                    canonical_map[raw_lbl] = new_canon
                    mapped_count += 1
            print(f"    Applied {mapped_count} remappings")
        else:
            print("    WARNING: unify pass returned empty map, stopping")
            break

    print(f"Final canonical families: {len(set(canonical_map.values()))}")

    # Verify all labels mapped
    unmapped = set(raw["mega_skills"].keys()) - set(canonical_map.keys())
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped labels: {unmapped}")
        for u in unmapped:
            canonical_map[u] = "UNCATEGORIZED"

    # Re-assign skills to canonical families
    canonical_clusters = defaultdict(lambda: {
        "skills": [], "domains": set(), "tasks": set(), "procedure": ""
    })

    for lbl, info in raw["mega_skills"].items():
        canon = canonical_map.get(lbl, "UNCATEGORIZED")
        cluster = canonical_clusters[canon]
        if not cluster["procedure"]:
            cluster["procedure"] = info["procedure"]
        for sk in info["skills"]:
            cluster["skills"].append({
                **sk,
                "raw_label": lbl,
            })
            cluster["domains"].add(sk["domain"])
            cluster["tasks"].add(sk["task"])

    # Build output
    output = {
        "total_skills": raw["total_skills"],
        "total_canonical_families": len(canonical_clusters),
        "families": {},
    }

    for canon in sorted(canonical_clusters,
                        key=lambda k: -len(canonical_clusters[k]["skills"])):
        c = canonical_clusters[canon]
        domains = sorted(c["domains"])
        tasks = sorted(c["tasks"])
        n_way = len(domains)
        output["families"][canon] = {
            "count": len(c["skills"]),
            "domains": domains,
            "tasks": tasks,
            "n_way": n_way,
            "is_cross_domain": n_way >= 2,
            "procedure": c["procedure"],
            "raw_labels_merged": sorted(set(
                sk["raw_label"] for sk in c["skills"]
            )),
            "skills": [
                {
                    "skill_id": s["skill_id"],
                    "task": s["task"],
                    "domain": s["domain"],
                    "raw_label": s["raw_label"],
                }
                for s in c["skills"]
            ],
        }

    three_way = sum(1 for v in output["families"].values() if v["n_way"] >= 3)
    two_way = sum(1 for v in output["families"].values() if v["n_way"] == 2)
    cross_domain = three_way + two_way
    single = sum(1 for v in output["families"].values() if v["n_way"] == 1)

    output["stats"] = {
        "three_way_GAME_WEB_VR": three_way,
        "two_way": two_way,
        "cross_domain_total": cross_domain,
        "single_domain": single,
    }

    # Print
    print(f"\n{'='*65}")
    print(f"CANONICAL MEGA-SKILL FAMILIES (after merging)")
    print(f"{'='*65}")
    print(f"Total families: {len(canonical_clusters)}")
    print(f"  Three-way (GAME+WEB+VR): {three_way}")
    print(f"  Two-way: {two_way}")
    print(f"  Single-domain: {single}")
    print()

    for canon, info in sorted(output["families"].items(),
                                key=lambda x: -x[1]["count"]):
        dom_str = "+".join(info["domains"])
        xd = " ★" if info["n_way"] >= 3 else (" ●" if info["n_way"] >= 2 else "")
        merged = len(info["raw_labels_merged"])
        print(f"  {canon:40s}  {info['count']:3d} skills  [{dom_str}]{xd}  (merged {merged} raw)")
        print(f"    → {info['procedure']}")

    # Non-game coverage
    nongame_covered = set()
    for info in output["families"].values():
        if info["is_cross_domain"]:
            for sk in info["skills"]:
                if sk["domain"] != "GAME":
                    nongame_covered.add(sk["skill_id"])
    all_nongame = set()
    for info in output["families"].values():
        for sk in info["skills"]:
            if sk["domain"] != "GAME":
                all_nongame.add(sk["skill_id"])

    print(f"\nNon-game skills in cross-domain families: "
          f"{len(nongame_covered)}/{len(all_nongame)} "
          f"({100*len(nongame_covered)/max(len(all_nongame),1):.1f}%)")

    with open(CLUSTERED_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {CLUSTERED_FILE}")

    # Build codebook for re-labeling pass (extract_mega_skills.py --relabel)
    codebook = {}
    for canon, info in output["families"].items():
        codebook[canon] = info["procedure"]
    with open(CODEBOOK_FILE, "w") as f:
        json.dump(codebook, f, indent=2)
    print(f"Codebook ({len(codebook)} families) → {CODEBOOK_FILE}")


if __name__ == "__main__":
    main()
