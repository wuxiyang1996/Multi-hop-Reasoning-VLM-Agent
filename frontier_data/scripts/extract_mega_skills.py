#!/usr/bin/env python3
"""
Extract a mega-skill label for each skill via LLM, then cluster.

Two-pass approach:
  Pass 1 (default):  Free-form labeling — LLM generates labels from
      reasoning steps with NO pre-defined categories.
  Pass 2 (--relabel): Re-label using a codebook of canonical families
      produced by cluster_mega_skills.py, guaranteeing consistency.

Usage:
    # Pass 1: free-form discovery
    python frontier_data/scripts/extract_mega_skills.py

    # (run cluster_mega_skills.py to build codebook)

    # Pass 2: consistent re-labeling
    python frontier_data/scripts/extract_mega_skills.py --relabel
"""

import argparse
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
CODEBOOK_FILE = OUT_DIR / "mega_skill_codebook.json"
RELABEL_FILE = OUT_DIR / "mega_skill_labels_final.json"

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
You are a cognitive-science expert analysing agent reasoning strategies.

Your task: read a skill's REASONING STEPS and produce a short label that \
captures the CORE cognitive procedure — the sequence of mental operations \
the agent performs, independent of domain objects.

Rules:
1. Output EXACTLY one JSON object:
   {"mega_skill": "LABEL", "procedure": "≤20 word step-by-step procedure"}
2. LABEL must be 2–4 words in UPPER_SNAKE_CASE (e.g. TRACE_CAUSAL_CHAIN, \
   MATCH_VISUAL_PATTERN, ELIMINATE_BY_CRITERIA).
3. Focus ONLY on the reasoning steps — what cognitive operations are performed \
   and in what order? Ignore domain-specific nouns (game names, benchmark names).
4. Two skills that follow the SAME sequence of cognitive operations should \
   get the SAME label, even if they are from completely different domains.
5. Two skills that follow DIFFERENT cognitive sequences MUST get DIFFERENT \
   labels, even if they share domain or topic.
6. The procedure field should describe the step-by-step reasoning in a \
   domain-agnostic way (e.g. "perceive options → compare by criterion → \
   select best" not "compare candy colours").
7. Output ONLY the JSON object, nothing else."""

USER_TEMPLATE = """\
Skill: {skill_id}
Task domain: {task} ({domain})
Description: {description}

Reasoning steps:
{steps}

Template signature: {collapsed_sig}

Analyse the reasoning steps above. What is the core cognitive procedure?"""


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


def _normalise_label(label: str) -> str:
    """Normalise a free-form label to UPPER_SNAKE_CASE for consistent grouping."""
    label = label.strip().upper()
    label = re.sub(r"[^A-Z0-9]+", "_", label)
    label = label.strip("_")
    return label


def classify_skill(skill, retries=3):
    """Ask LLM to analyse reasoning steps and produce a free-form mega-skill label."""
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
                max_completion_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r'\{[^}]+\}', text)
            if match:
                result = json.loads(match.group())
                raw_label = result.get("mega_skill", "UNKNOWN")
                return {
                    "mega_skill": _normalise_label(raw_label),
                    "mega_skill_raw": raw_label,
                    "procedure": result.get("procedure", ""),
                }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAIL {skill['skill_id']}: {e}")
    return {"mega_skill": "UNKNOWN", "mega_skill_raw": "UNKNOWN",
            "procedure": "classification failed"}


RELABEL_SYSTEM_TEMPLATE = """\
You are a cognitive-science expert classifying agent reasoning strategies.

Your task: read a skill's REASONING STEPS and classify it into EXACTLY one \
of the canonical mega-skill families below. Pick the family whose cognitive \
procedure best matches the skill's reasoning steps.

Canonical families:
{codebook_text}

Rules:
1. Output EXACTLY one JSON object:
   {{"mega_skill": "LABEL", "confidence": <1-5>}}
2. LABEL must be EXACTLY one of the canonical family names listed above.
3. Focus on the SEQUENCE of cognitive operations in the reasoning steps.
4. If no family fits well, pick the closest match and set confidence=1 or 2.
5. Output ONLY the JSON object, nothing else."""

RELABEL_USER_TEMPLATE = """\
Skill: {skill_id}
Task domain: {task} ({domain})
Description: {description}

Reasoning steps:
{steps}

Template signature: {collapsed_sig}

Which canonical mega-skill family does this belong to?"""


def _build_relabel_system(codebook: dict) -> str:
    """Build the relabel system prompt from the codebook."""
    lines = []
    for i, (label, procedure) in enumerate(sorted(codebook.items()), 1):
        lines.append(f"{i}. {label} — {procedure}")
    codebook_text = "\n".join(lines)
    return RELABEL_SYSTEM_TEMPLATE.format(codebook_text=codebook_text)


def relabel_skill(skill, system_prompt, codebook_labels, retries=3):
    """Pass 2: classify a skill using the canonical codebook."""
    steps_text = "\n".join(
        f"  {i+1}. {s}" for i, s in enumerate(skill["steps"])
    ) if skill["steps"] else "(no explicit steps)"

    user_msg = RELABEL_USER_TEMPLATE.format(
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=100,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r'\{[^}]+\}', text)
            if match:
                result = json.loads(match.group())
                label = _normalise_label(result.get("mega_skill", ""))
                if label in codebook_labels:
                    return {
                        "mega_skill": label,
                        "confidence": result.get("confidence", 3),
                    }
                best = _fuzzy_match(label, codebook_labels)
                return {
                    "mega_skill": best,
                    "confidence": max(1, result.get("confidence", 3) - 1),
                }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAIL {skill['skill_id']}: {e}")
    return {"mega_skill": "UNCATEGORIZED", "confidence": 0}


def _fuzzy_match(label: str, codebook_labels: set) -> str:
    """Find the codebook label with the most word overlap."""
    label_words = set(label.split("_"))
    best, best_score = "UNCATEGORIZED", 0
    for cl in codebook_labels:
        cl_words = set(cl.split("_"))
        overlap = len(label_words & cl_words)
        if overlap > best_score:
            best, best_score = cl, overlap
    return best


def run_relabel():
    """Pass 2: re-label all skills using the canonical codebook."""
    if not CODEBOOK_FILE.exists():
        print(f"ERROR: codebook not found at {CODEBOOK_FILE}")
        print("Run cluster_mega_skills.py first to build the codebook.")
        sys.exit(1)

    with open(CODEBOOK_FILE) as f:
        codebook = json.load(f)

    codebook_labels = {_normalise_label(k) for k in codebook}
    system_prompt = _build_relabel_system(codebook)
    print(f"Codebook: {len(codebook)} canonical families")

    skills = load_all_skills()
    print(f"Loaded {len(skills)} skills — re-labeling with codebook...")

    results = []
    n_done = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(relabel_skill, sk, system_prompt, codebook_labels): sk
            for sk in skills
        }
        for future in as_completed(futures):
            sk = futures[future]
            label = future.result()
            results.append({
                **sk,
                "mega_skill": label["mega_skill"],
                "confidence": label["confidence"],
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"  re-labeled {n_done}/{len(skills)} ...")

    print(f"Re-labeled all {len(results)} skills")

    clusters = defaultdict(lambda: {"skills": [], "domains": set(), "tasks": set()})
    for r in results:
        lbl = r["mega_skill"]
        clusters[lbl]["skills"].append(r)
        clusters[lbl]["domains"].add(r["domain"])
        clusters[lbl]["tasks"].add(r["task"])

    summary = {
        "total_skills": len(results),
        "total_mega_skills": len(clusters),
        "clustering_method": "two_pass_free_then_codebook",
        "codebook_families": len(codebook),
        "mega_skills": {},
    }

    avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)
    low_conf = sum(1 for r in results if r["confidence"] <= 2)

    for lbl in sorted(clusters, key=lambda k: -len(clusters[k]["skills"])):
        c = clusters[lbl]
        domains = sorted(c["domains"])
        tasks = sorted(c["tasks"])
        n_way = len(domains)
        confs = [s["confidence"] for s in c["skills"]]
        summary["mega_skills"][lbl] = {
            "count": len(c["skills"]),
            "domains": domains,
            "tasks": tasks,
            "n_way": n_way,
            "is_cross_domain": n_way >= 2,
            "procedure": codebook.get(lbl, ""),
            "avg_confidence": sum(confs) / max(len(confs), 1),
            "skills": [
                {"skill_id": s["skill_id"], "task": s["task"],
                 "domain": s["domain"], "confidence": s["confidence"]}
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
        "avg_confidence": round(avg_conf, 2),
        "low_confidence_count": low_conf,
    }

    print(f"\n{'='*60}")
    print(f"RE-LABELED MEGA-SKILL RESULTS (Pass 2)")
    print(f"{'='*60}")
    print(f"Codebook families: {len(codebook)}")
    print(f"Families used: {len(clusters)}")
    print(f"Avg confidence: {avg_conf:.2f}")
    print(f"Low confidence (<=2): {low_conf}/{len(results)}")
    print(f"  Three-way (GAME+WEB+VR): {three_way}")
    print(f"  Two-way: {two_way}")
    print(f"  Single-domain: {single}")
    print()

    for lbl, info in sorted(summary["mega_skills"].items(),
                             key=lambda x: -x[1]["count"]):
        dom_str = "+".join(info["domains"])
        xd = " ★" if info["n_way"] >= 3 else (" ●" if info["n_way"] >= 2 else "")
        print(f"  {lbl:35s}  {info['count']:3d} skills  [{dom_str}]{xd}"
              f"  conf={info['avg_confidence']:.1f}")
        print(f"    → {info['procedure']}")

    with open(RELABEL_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {RELABEL_FILE}")


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
                "mega_skill_raw": label.get("mega_skill_raw", label["mega_skill"]),
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
        "clustering_method": "free_form_reasoning_step_labels",
        "mega_skills": {},
    }

    for lbl in sorted(clusters, key=lambda k: -len(clusters[k]["skills"])):
        c = clusters[lbl]
        domains = sorted(c["domains"])
        tasks = sorted(c["tasks"])
        procedures = [s["mega_procedure"] for s in c["skills"] if s["mega_procedure"]]
        raw_labels = sorted(set(s.get("mega_skill_raw", lbl) for s in c["skills"]))
        n_way = len(domains)
        summary["mega_skills"][lbl] = {
            "count": len(c["skills"]),
            "domains": domains,
            "tasks": tasks,
            "n_way": n_way,
            "is_cross_domain": n_way >= 2,
            "procedure": procedures[0] if procedures else "",
            "raw_labels": raw_labels,
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relabel", action="store_true",
                        help="Pass 2: re-label using codebook from cluster_mega_skills.py")
    args = parser.parse_args()

    if args.relabel:
        run_relabel()
    else:
        main()
