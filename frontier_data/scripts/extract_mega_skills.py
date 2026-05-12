#!/usr/bin/env python3
"""
Extract a mega-skill label for each skill via LLM, then cluster.

Bottom-up approach: instead of pairwise comparison (O(n²)),
ask the LLM to classify each skill's reasoning plan into a
short mega-skill category name + one-line procedure summary.
Then group by label for natural clustering.

Usage:
    python frontier_data/scripts/extract_mega_skills.py
"""

import json, os, sys, time, re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

import importlib.util
_keys_path = ROOT.parent / "keys.py"
_spec = importlib.util.spec_from_file_location("keys", _keys_path)
_keys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_keys)

from openai import OpenAI

client = OpenAI(api_key=_keys.openai_api_key)

BANKS_DIR = ROOT / "frontier_data" / "output" / "per_task_banks"
OUT_DIR = ROOT / "frontier_data" / "output"
OUT_FILE = OUT_DIR / "mega_skill_labels.json"

COHORT_MAP = {
    "candy_crush": "GAME", "super_mario": "GAME", "tetris": "GAME",
    "twenty_forty_eight": "GAME",
    "Temporal_Airstriker-v0": "GAME", "Temporal_AlteredBeast-v0": "GAME",
    "Temporal_Columns-v0": "GAME", "Temporal_DynamiteHeaddy-v0": "GAME",
    "Temporal_SpaceHarrierII-v0": "GAME", "Temporal_StreetsOfRage2-v0": "GAME",
    "Temporal_Strider-v0": "GAME", "Temporal_ThunderForceIII-v0": "GAME",
    "miniwob": "WEB", "webshop": "WEB",
    "siv_bench": "VR", "video_holmes": "VR",
    "tir_bench": "VR", "visual_toolbench": "VR",
}

SYSTEM = """\
You are a cognitive-science expert classifying agent skills into mega-skill families.

A mega-skill is the CORE cognitive procedure of a skill, described at a level \
that distinguishes meaningfully different reasoning strategies.

You MUST choose from EXACTLY these canonical families:

1. COMPARE_AND_RANK — Perceive multiple options → compare by priority/criterion → select best
2. NAVIGATE_AND_REACH — Plan path through environment → move to target → arrive
3. DODGE_AND_SURVIVE — Monitor threats → decide evasive action → execute avoidance
4. ENGAGE_AND_DEFEAT — Identify target → approach → attack/interact → verify elimination
5. COLLECT_AND_ACCUMULATE — Identify collectible → move to it → acquire → track count
6. SEQUENCE_AND_COMPLETE — Break goal into ordered sub-steps → execute each in order → verify
7. RECALL_MATCH_AND_SELECT — Retrieve goal/criteria → perceive options → match → select
8. FILTER_AND_NARROW — Apply criteria → eliminate non-matching → act on survivors
9. INPUT_AND_SUBMIT — Perceive form/field → enter value → submit → verify acceptance
10. COUNT_AND_REPORT — Identify target attribute → count/aggregate → output result
11. POSITION_AND_PLACE — Perceive target location → move/arrange item → verify placement
12. TIME_AND_REACT — Wait for trigger/timing window → execute precisely timed action
13. TRANSFORM_AND_VERIFY — Apply transformation to state → verify new state matches goal
14. EXPLORE_AND_DISCOVER — Probe unknown space → observe result → update knowledge
15. MONITOR_AND_SUSTAIN — Track ongoing process → maintain/adjust → ensure completion
16. INFER_AND_DECIDE — Perceive evidence → reason about causes → select explanation/action
17. RETRIEVE_AND_EXECUTE — Recall known procedure → execute it → confirm outcome
18. EVALUATE_AND_OPTIMIZE — Assess current state quality → try improvement → compare

Rules:
1. Output EXACTLY one JSON object: {"mega_skill": "LABEL", "procedure": "≤15 word summary"}
2. Use ONLY the 18 labels above. Pick the BEST match.
3. Focus on the REASONING PATTERN — what cognitive steps does the agent perform?
4. Output ONLY the JSON object, nothing else."""

USER_TEMPLATE = """\
Skill: {skill_id}
Task domain: {task} ({domain})
Description: {description}

Reasoning plan (Layer-C template):
{steps}

Collapsed signature: {collapsed_sig}

Classify this into a mega-skill family."""


def load_all_skills():
    """Load all skills from per-task banks."""
    skills = []
    for task_dir in sorted(BANKS_DIR.iterdir()):
        bank_file = task_dir / "skill_bank.jsonl"
        if not bank_file.exists():
            continue
        task = task_dir.name
        domain = COHORT_MAP.get(task, "UNKNOWN")
        with open(bank_file) as f:
            for line in f:
                rec = json.loads(line)
                sk = rec.get("skill", rec)
                sid = sk.get("skill_id", "?")
                desc = sk.get("strategic_description", sk.get("name", ""))
                proto = sk.get("protocol", {})
                steps = proto.get("steps", [])
                csig = sk.get("collapsed_signature", "")
                tsig = sk.get("template_signature", "")
                skills.append({
                    "task": task,
                    "domain": domain,
                    "skill_id": sid,
                    "description": desc,
                    "steps": steps,
                    "collapsed_signature": csig,
                    "template_signature": tsig,
                })
    return skills


def classify_skill(skill, retries=3):
    """Ask LLM to classify a single skill into a mega-skill family."""
    steps_text = "\n".join(
        f"  {i+1}. {s}" for i, s in enumerate(skill["steps"])
    ) if skill["steps"] else "(no explicit steps)"

    user_msg = USER_TEMPLATE.format(
        skill_id=skill["skill_id"],
        task=skill["task"],
        domain=skill["domain"],
        description=skill["description"],
        steps=steps_text,
        collapsed_sig=skill["collapsed_signature"],
    )

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=150,
            )
            text = resp.choices[0].message.content.strip()
            # Extract JSON from response
            match = re.search(r'\{[^}]+\}', text)
            if match:
                result = json.loads(match.group())
                return {
                    "mega_skill": result.get("mega_skill", "UNKNOWN"),
                    "procedure": result.get("procedure", ""),
                }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAIL {skill['skill_id']}: {e}")
    return {"mega_skill": "UNKNOWN", "procedure": "classification failed"}


def main():
    skills = load_all_skills()
    print(f"Loaded {len(skills)} skills from {len(COHORT_MAP)} tasks")

    results = []
    n_done = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(classify_skill, sk): sk for sk in skills
        }
        for future in as_completed(futures):
            sk = futures[future]
            label = future.result()
            results.append({
                **sk,
                "mega_skill": label["mega_skill"],
                "mega_procedure": label["procedure"],
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"  classified {n_done}/{len(skills)} ...")

    print(f"Classified all {len(results)} skills")

    # --- Cluster by mega_skill label ---
    clusters = defaultdict(lambda: {"skills": [], "domains": set(), "tasks": set()})
    for r in results:
        lbl = r["mega_skill"]
        clusters[lbl]["skills"].append(r)
        clusters[lbl]["domains"].add(r["domain"])
        clusters[lbl]["tasks"].add(r["task"])

    # Build summary
    summary = {
        "total_skills": len(results),
        "total_mega_skills": len(clusters),
        "mega_skills": {},
    }

    for lbl in sorted(clusters, key=lambda k: -len(clusters[k]["skills"])):
        c = clusters[lbl]
        domains = sorted(c["domains"])
        tasks = sorted(c["tasks"])
        procedure = c["skills"][0]["mega_procedure"]
        n_way = len(domains)
        summary["mega_skills"][lbl] = {
            "count": len(c["skills"]),
            "domains": domains,
            "tasks": tasks,
            "n_way": n_way,
            "is_cross_domain": n_way >= 2,
            "procedure": procedure,
            "skills": [
                {"skill_id": s["skill_id"], "task": s["task"], "domain": s["domain"]}
                for s in c["skills"]
            ],
        }

    three_way = sum(1 for v in summary["mega_skills"].values() if v["n_way"] >= 3)
    two_way = sum(1 for v in summary["mega_skills"].values() if v["n_way"] == 2)
    cross_domain = sum(1 for v in summary["mega_skills"].values() if v["is_cross_domain"])
    single = sum(1 for v in summary["mega_skills"].values() if v["n_way"] == 1)

    summary["stats"] = {
        "three_way": three_way,
        "two_way": two_way,
        "cross_domain_total": cross_domain,
        "single_domain": single,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"MEGA-SKILL CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total mega-skills: {len(clusters)}")
    print(f"  Three-way (GAME+WEB+VR): {three_way}")
    print(f"  Two-way: {two_way}")
    print(f"  Single-domain: {single}")
    print()

    for lbl, info in sorted(summary["mega_skills"].items(),
                             key=lambda x: -x[1]["count"]):
        dom_str = "+".join(info["domains"])
        xd = " ★" if info["n_way"] >= 3 else (" ●" if info["n_way"] >= 2 else "")
        print(f"  {lbl:35s}  {info['count']:3d} skills  [{dom_str}]{xd}")
        print(f"    → {info['procedure']}")

    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
