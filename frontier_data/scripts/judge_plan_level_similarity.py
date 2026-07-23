#!/usr/bin/env python3
"""LLM-as-judge for PLAN-LEVEL cross-domain similarity.

Unlike `judge_plan_similarity.py` (which compares within collapsed-signature
groups), this script compares ACTUAL reasoning plan text across ALL
cross-domain skill pairs — finding matches that structural signatures miss.

Strategy: batch comparison.  For each non-game skill, present its full
plan alongside N representative game skills and ask the LLM which ones
(if any) share the same cognitive procedure.  Similarly for VR↔WEB.

Output: frontier_data/output/plan_level_similarity_judgments.json

Usage::

    python frontier_data/scripts/judge_plan_level_similarity.py
    python frontier_data/scripts/judge_plan_level_similarity.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("judge_plan_level")

ROOT = Path(__file__).resolve().parent.parent.parent
LAYER_C_DIR = ROOT / "frontier_data" / "output" / "layer_c_templates"
OUT_PATH = ROOT / "frontier_data" / "output" / "plan_level_similarity_judgments.json"
GAME_COHORTS = {"gymv_game", "env_wr_game"}
MAX_WORKERS = 8

BATCH_JUDGE_SYSTEM = """You are an expert reasoning-plan analyst. You will receive one TARGET skill's reasoning plan and a list of CANDIDATE skills from a different domain.

Your job: identify which candidates (if any) share the SAME transferable cognitive procedure as the target — meaning a human expert seeing ONLY the step-by-step reasoning (no task names) would recognize them as the same general strategy.

For EACH candidate, rate:
  5 = IDENTICAL procedure — same cognitive strategy, different domain objects
  4 = HIGHLY SIMILAR — same core strategy with minor variations
  3 = MODERATELY SIMILAR — overlapping reasoning but meaningful differences
  2 = WEAKLY SIMILAR — same structure, different cognitive challenge
  1 = DIFFERENT — fundamentally different procedures

You MUST respond with ONLY a JSON object (no markdown):
{
  "matches": [
    {
      "candidate_id": "<candidate letter A/B/C/...>",
      "score": <1-5>,
      "same_procedure": <true if score >= 4>,
      "shared_reasoning": "<1-sentence shared strategy description>",
      "transfer_value": "<high|medium|low|none>"
    }
  ],
  "best_match": "<candidate letter with highest score, or 'none'>",
  "target_reasoning_summary": "<1-sentence summary of target's cognitive strategy>"
}

Only include candidates with score >= 3 in the matches list. If no candidate scores >= 3, return an empty matches list."""

BATCH_JUDGE_USER = """## TARGET skill  [{target_domain}]  (task: {target_task})
Name: {target_name}

Reasoning plan:
{target_plan}

---

## CANDIDATE skills  [{candidate_domain}]

{candidates_text}

---

For each candidate scoring >= 3, provide a match entry. Respond with JSON only."""


def load_all_skills() -> Dict[str, List[Dict[str, Any]]]:
    """Return {domain: [skill_record, ...]}."""
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for cohort in sorted(os.listdir(LAYER_C_DIR)):
        cpath = LAYER_C_DIR / cohort
        if not cpath.is_dir():
            continue
        domain = ("GAME" if cohort in GAME_COHORTS
                  else "WEB" if cohort == "web" else "VR")
        for task in sorted(os.listdir(cpath)):
            tb = cpath / task / "template_bank.jsonl"
            if not tb.is_file():
                continue
            with open(tb) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    steps = r.get("template_steps", [])
                    plan_text = "\n".join(
                        f"  {i+1}. [{s.get('op','')}] {s.get('predicate','')}"
                        for i, s in enumerate(steps)
                    )
                    by_domain[domain].append({
                        "skill_id": r["skill_id"],
                        "name": r.get("skill_name", r["skill_id"]),
                        "task": task,
                        "csig": r.get("collapsed_signature", ""),
                        "raw_sig": r.get("template_signature", ""),
                        "plan_text": plan_text,
                        "n_steps": len(steps),
                    })

    return dict(by_domain)


def select_diverse_representatives(
    skills: List[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    """Pick N diverse skills covering different tasks and signatures."""
    by_task_sig: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for s in skills:
        key = (s["task"], s["csig"])
        by_task_sig[key].append(s)

    selected = []
    keys = list(by_task_sig.keys())
    random.seed(42)
    random.shuffle(keys)

    for key in keys:
        if len(selected) >= n:
            break
        selected.append(by_task_sig[key][0])

    if len(selected) < n:
        remaining = [s for s in skills if s not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


def format_candidates(skills: List[Dict[str, Any]]) -> str:
    """Format candidate skills with letter IDs."""
    lines = []
    for i, s in enumerate(skills):
        letter = chr(65 + i)  # A, B, C, ...
        lines.append(f"### Candidate {letter}  (task: {s['task']})")
        lines.append(f"Name: {s['name']}")
        lines.append(f"Plan:")
        lines.append(s["plan_text"])
        lines.append("")
    return "\n".join(lines)


def build_batch_queries(
    by_domain: Dict[str, List[Dict[str, Any]]],
    n_game_reps: int = 20,
) -> List[Dict[str, Any]]:
    """Build batch queries: each non-game skill vs game representatives,
    plus VR vs WEB."""
    queries = []

    game_reps = select_diverse_representatives(by_domain.get("GAME", []), n_game_reps)
    logger.info("Selected %d diverse game representatives from %d tasks",
                len(game_reps), len({s["task"] for s in game_reps}))

    for domain in ["WEB", "VR"]:
        for skill in by_domain.get(domain, []):
            queries.append({
                "target": skill,
                "target_domain": domain,
                "candidates": game_reps,
                "candidate_domain": "GAME",
                "query_type": f"{domain}→GAME",
            })

    vr_skills = by_domain.get("VR", [])
    web_skills = by_domain.get("WEB", [])
    if vr_skills and web_skills:
        web_reps = select_diverse_representatives(web_skills, min(15, len(web_skills)))
        for skill in vr_skills:
            queries.append({
                "target": skill,
                "target_domain": "VR",
                "candidates": web_reps,
                "candidate_domain": "WEB",
                "query_type": "VR→WEB",
            })

    return queries


def call_batch_judge(
    query: Dict[str, Any],
    api_key: str,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """Call LLM for one batch comparison."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    target = query["target"]
    candidates = query["candidates"]

    user_msg = BATCH_JUDGE_USER.format(
        target_domain=query["target_domain"],
        target_task=target["task"],
        target_name=target["name"],
        target_plan=target["plan_text"],
        candidate_domain=query["candidate_domain"],
        candidates_text=format_candidates(candidates),
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": BATCH_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=1000,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            for m in result.get("matches", []):
                cid = m.get("candidate_id", "")
                if cid and len(cid) == 1:
                    idx = ord(cid) - 65
                    if 0 <= idx < len(candidates):
                        m["candidate_skill_id"] = candidates[idx]["skill_id"]
                        m["candidate_task"] = candidates[idx]["task"]
                        m["candidate_name"] = candidates[idx]["name"]
                        m["candidate_csig"] = candidates[idx]["csig"]

            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("  attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))

    return {"matches": [], "best_match": "none",
            "target_reasoning_summary": "JUDGE_FAILED"}


def run_all(
    queries: List[Dict[str, Any]],
    api_key: str,
    model: str = "gpt-4.1-mini",
    max_workers: int = MAX_WORKERS,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run all batch queries."""
    results = []

    if dry_run:
        logger.info("DRY-RUN: would run %d batch queries", len(queries))
        for q in queries:
            results.append({
                "target_skill_id": q["target"]["skill_id"],
                "target_task": q["target"]["task"],
                "target_domain": q["target_domain"],
                "target_csig": q["target"]["csig"],
                "candidate_domain": q["candidate_domain"],
                "query_type": q["query_type"],
                "judgment": {"matches": [], "best_match": "none", "_dry_run": True},
            })
        return results

    logger.info("Running %d batch queries with %s (%d workers)",
                len(queries), model, max_workers)

    def _run_one(idx_query):
        idx, query = idx_query
        judgment = call_batch_judge(query, api_key, model)
        return idx, query, judgment

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, (i, q)): i for i, q in enumerate(queries)}
        done = 0
        for fut in as_completed(futures):
            idx, query, judgment = fut.result()
            n_matches = len(judgment.get("matches", []))
            best = judgment.get("best_match", "none")
            target = query["target"]

            results.append({
                "target_skill_id": target["skill_id"],
                "target_task": target["task"],
                "target_name": target["name"],
                "target_domain": query["target_domain"],
                "target_csig": target["csig"],
                "candidate_domain": query["candidate_domain"],
                "query_type": query["query_type"],
                "judgment": judgment,
            })

            done += 1
            best_name = ""
            if best != "none" and judgment.get("matches"):
                top = max(judgment["matches"], key=lambda m: m.get("score", 0))
                best_name = top.get("candidate_name", "")[:25]
            logger.info("  [%d/%d] %s %s/%s → %d matches, best=%s (%s)",
                        done, len(queries), query["query_type"],
                        target["task"][:12], target["name"][:20],
                        n_matches, best, best_name)

    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary statistics."""
    all_matches = []
    for r in results:
        for m in r["judgment"].get("matches", []):
            all_matches.append({
                "target_skill_id": r["target_skill_id"],
                "target_task": r["target_task"],
                "target_name": r["target_name"],
                "target_domain": r["target_domain"],
                "target_csig": r["target_csig"],
                "candidate_skill_id": m.get("candidate_skill_id", ""),
                "candidate_task": m.get("candidate_task", ""),
                "candidate_name": m.get("candidate_name", ""),
                "candidate_csig": m.get("candidate_csig", ""),
                "score": m.get("score", 0),
                "same_procedure": m.get("same_procedure", False),
                "shared_reasoning": m.get("shared_reasoning", ""),
                "transfer_value": m.get("transfer_value", ""),
                "same_collapsed_sig": r["target_csig"] == m.get("candidate_csig", ""),
            })

    n_total_targets = len(results)
    n_with_matches = sum(1 for r in results if r["judgment"].get("matches"))
    n_high_matches = sum(1 for m in all_matches if m["score"] >= 4)
    n_new_discoveries = sum(1 for m in all_matches
                           if m["score"] >= 4 and not m["same_collapsed_sig"])

    by_query_type = defaultdict(lambda: {"n": 0, "matches": 0, "high": 0, "new": 0})
    for r in results:
        qt = r["query_type"]
        by_query_type[qt]["n"] += 1
        matches = r["judgment"].get("matches", [])
        by_query_type[qt]["matches"] += len(matches)
        by_query_type[qt]["high"] += sum(1 for m in matches if m.get("score", 0) >= 4)
        for m in matches:
            if m.get("score", 0) >= 4 and r["target_csig"] != m.get("candidate_csig", ""):
                by_query_type[qt]["new"] += 1

    return {
        "n_targets": n_total_targets,
        "n_with_any_match": n_with_matches,
        "n_total_matches_gte3": len(all_matches),
        "n_high_confidence_gte4": n_high_matches,
        "n_NEW_discoveries": n_new_discoveries,
        "by_query_type": dict(by_query_type),
        "top_new_discoveries": sorted(
            [m for m in all_matches if m["score"] >= 4 and not m["same_collapsed_sig"]],
            key=lambda m: -m["score"],
        )[:20],
        "all_high_matches": sorted(
            [m for m in all_matches if m["score"] >= 4],
            key=lambda m: -m["score"],
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--n-game-reps", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-o", "--output", default=str(OUT_PATH))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    keys_path = ROOT.parent / "keys.py"
    if not keys_path.exists():
        keys_path = ROOT / "keys.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("keys", str(keys_path))
    keys_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(keys_mod)
    openai_api_key = keys_mod.openai_api_key

    logger.info("Loading Layer-C templates from %s", LAYER_C_DIR)
    by_domain = load_all_skills()
    for d in ["GAME", "WEB", "VR"]:
        logger.info("  %s: %d skills", d, len(by_domain.get(d, [])))

    queries = build_batch_queries(by_domain, n_game_reps=args.n_game_reps)
    logger.info("Built %d batch queries", len(queries))

    results = run_all(queries, openai_api_key,
                      model=args.model, max_workers=args.max_workers,
                      dry_run=args.dry_run)

    summary = summarize(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "results": results,
        "config": {
            "model": args.model,
            "n_game_reps": args.n_game_reps,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }, indent=2, ensure_ascii=False))

    logger.info("=" * 60)
    logger.info("SUMMARY: %d targets, %d with matches, %d high-confidence (≥4)",
                summary["n_targets"], summary["n_with_any_match"],
                summary["n_high_confidence_gte4"])
    logger.info("  NEW discoveries (≥4, different collapsed sig): %d",
                summary["n_NEW_discoveries"])

    for qt, stats in summary["by_query_type"].items():
        logger.info("  %s: %d targets, %d matches(≥3), %d high(≥4), %d NEW",
                     qt, stats["n"], stats["matches"], stats["high"], stats["new"])

    if summary["top_new_discoveries"]:
        logger.info("\nTOP NEW CROSS-DOMAIN DISCOVERIES:")
        for m in summary["top_new_discoveries"][:10]:
            logger.info("  [%d] %s/%s (%s) ↔ %s/%s (%s)",
                        m["score"],
                        m["target_task"][:15], m["target_name"][:20], m["target_csig"],
                        m["candidate_task"][:15], m["candidate_name"][:20], m["candidate_csig"])
            logger.info("       → %s", m["shared_reasoning"])

    logger.info("\nOutput: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
