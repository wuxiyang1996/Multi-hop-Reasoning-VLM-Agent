#!/usr/bin/env python3
"""LLM-as-judge for cross-domain reasoning plan similarity.

For each collapsed-signature group that spans ≥ 2 domains, samples
representative skills from each domain and asks GPT-5.4 to judge whether
their full plan context (predicate text) represents the SAME transferable
reasoning procedure — not just structural similarity.

Output:  frontier_data/output/plan_similarity_judgments.json

Usage::

    python frontier_data/scripts/judge_plan_similarity.py
    python frontier_data/scripts/judge_plan_similarity.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("judge_plan_similarity")

ROOT = Path(__file__).resolve().parent.parent.parent
LAYER_C_DIR = ROOT / "frontier_data" / "output" / "layer_c_templates"
OUT_PATH = ROOT / "frontier_data" / "output" / "plan_similarity_judgments.json"
GAME_COHORTS = {"gymv_game", "env_wr_game"}

SAMPLES_PER_DOMAIN = 3
MAX_WORKERS = 8

JUDGE_SYSTEM = """You are an expert reasoning-plan analyst. Your job is to judge whether two multi-step reasoning plans from DIFFERENT task domains represent the SAME transferable cognitive procedure.

Two plans are "same procedure" if a human expert, seeing ONLY the step-by-step reasoning text (no task names), would recognize them as instances of the same general cognitive strategy — even though they operate on different objects/domains.

Two plans are "different procedure" if the reasoning steps solve fundamentally different cognitive challenges, even if they share the same high-level structure (perceive→decide→act).

Rate on this scale:
  5 = IDENTICAL procedure — same cognitive strategy applied to different domains
  4 = HIGHLY SIMILAR — same core strategy with minor domain-specific variations
  3 = MODERATELY SIMILAR — overlapping reasoning patterns but with meaningful differences
  2 = WEAKLY SIMILAR — same structure but different cognitive challenges
  1 = DIFFERENT — fundamentally different procedures despite structural overlap

You MUST respond with ONLY a JSON object (no markdown, no extra text):
{
  "score": <1-5>,
  "same_procedure": <true if score >= 4, false otherwise>,
  "shared_reasoning": "<1-sentence description of the shared cognitive strategy, or 'none' if score < 3>",
  "key_difference": "<1-sentence description of the main difference>",
  "transfer_value": "<'high' | 'medium' | 'low' | 'none'> — would knowing plan A help an agent learn plan B faster?"
}"""

JUDGE_USER_TEMPLATE = """Compare these two reasoning plans from different domains.
Do they represent the SAME transferable cognitive procedure?

## Plan A  [{domain_a}]  (task: {task_a})
Skill: {name_a}

Steps:
{steps_a}

## Plan B  [{domain_b}]  (task: {task_b})
Skill: {name_b}

Steps:
{steps_b}

Judge whether these are the SAME reasoning procedure. Respond with JSON only."""


def load_all_templates() -> Dict[str, List[Dict[str, Any]]]:
    """Return {collapsed_signature: [template_record_with_domain, ...]}."""
    by_csig: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

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
                    r["_domain"] = domain
                    r["_task"] = task
                    csig = r.get("collapsed_signature", "")
                    if csig:
                        by_csig[csig].append(r)

    return dict(by_csig)


def format_steps(template: Dict[str, Any]) -> str:
    """Render template_steps as numbered NL lines."""
    lines = []
    for i, s in enumerate(template.get("template_steps", []), 1):
        op = s.get("op", "?")
        pred = s.get("predicate", "")
        lines.append(f"  {i}. [{op}] {pred}")
    return "\n".join(lines)


def build_cross_domain_pairs(
    by_csig: Dict[str, List[Dict[str, Any]]],
    samples_per_domain: int = SAMPLES_PER_DOMAIN,
) -> List[Dict[str, Any]]:
    """For each cross-domain collapsed sig, build all domain-pair comparisons."""
    pairs = []

    for csig, templates in sorted(by_csig.items()):
        by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in templates:
            by_domain[t["_domain"]].append(t)

        if len(by_domain) < 2:
            continue

        domains = sorted(by_domain.keys())
        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                sample_a = by_domain[d1][:samples_per_domain]
                sample_b = by_domain[d2][:samples_per_domain]

                for a in sample_a:
                    for b in sample_b:
                        pairs.append({
                            "collapsed_sig": csig,
                            "domain_a": d1,
                            "domain_b": d2,
                            "task_a": a["_task"],
                            "task_b": b["_task"],
                            "name_a": a.get("skill_name", a["skill_id"]),
                            "name_b": b.get("skill_name", b["skill_id"]),
                            "steps_a": format_steps(a),
                            "steps_b": format_steps(b),
                            "skill_id_a": a["skill_id"],
                            "skill_id_b": b["skill_id"],
                        })

    return pairs


def call_judge(
    pair: Dict[str, Any],
    api_key: str,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """Call the LLM judge for one pair."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    user_msg = JUDGE_USER_TEMPLATE.format(**pair)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            result["_raw"] = text
            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("  attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.0 * (attempt + 1))

    return {"score": 0, "same_procedure": False,
            "shared_reasoning": "JUDGE_FAILED",
            "key_difference": "JUDGE_FAILED",
            "transfer_value": "none", "_raw": "FAILED"}


def run_judgments(
    pairs: List[Dict[str, Any]],
    api_key: str,
    model: str = "gpt-4.1-mini",
    max_workers: int = MAX_WORKERS,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run all pairwise judgments."""
    results = []

    if dry_run:
        logger.info("DRY-RUN: would judge %d pairs", len(pairs))
        for p in pairs:
            results.append({**p, "judgment": {"score": 0, "_dry_run": True}})
        return results

    logger.info("Judging %d cross-domain pairs with %s (%d workers)",
                len(pairs), model, max_workers)

    def _judge_one(idx_pair):
        idx, pair = idx_pair
        judgment = call_judge(pair, api_key, model)
        return idx, pair, judgment

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, (i, p)): i
                   for i, p in enumerate(pairs)}
        done = 0
        for fut in as_completed(futures):
            idx, pair, judgment = fut.result()
            results.append({**pair, "judgment": judgment})
            done += 1
            score = judgment.get("score", 0)
            tv = judgment.get("transfer_value", "?")
            logger.info("  [%d/%d] %s vs %s: score=%d transfer=%s",
                        done, len(pairs),
                        pair["task_a"][:15], pair["task_b"][:15],
                        score, tv)

    results.sort(key=lambda r: -r["judgment"].get("score", 0))
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate judgments per collapsed_sig."""
    by_csig: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_csig[r["collapsed_sig"]].append(r)

    sig_summaries = []
    for csig, group in sorted(by_csig.items()):
        scores = [r["judgment"].get("score", 0) for r in group]
        n_same = sum(1 for r in group
                     if r["judgment"].get("same_procedure", False))
        transfer_counts = defaultdict(int)
        for r in group:
            tv = r["judgment"].get("transfer_value", "none")
            transfer_counts[tv] += 1

        domains = sorted({r["domain_a"] for r in group} |
                         {r["domain_b"] for r in group})
        domain_pairs = sorted({(r["domain_a"], r["domain_b"]) for r in group})

        avg_score = sum(scores) / len(scores) if scores else 0
        sig_summaries.append({
            "collapsed_sig": csig,
            "domains": domains,
            "domain_pairs": [f"{a}↔{b}" for a, b in domain_pairs],
            "n_pairs_judged": len(group),
            "avg_score": round(avg_score, 2),
            "n_same_procedure": n_same,
            "pct_same": round(n_same * 100 / len(group), 1) if group else 0,
            "transfer_value_dist": dict(transfer_counts),
            "verdict": (
                "STRONG_TRANSFER" if avg_score >= 4.0 else
                "MODERATE_TRANSFER" if avg_score >= 3.0 else
                "WEAK_TRANSFER" if avg_score >= 2.0 else
                "NO_TRANSFER"
            ),
            "representative_shared_reasoning": next(
                (r["judgment"].get("shared_reasoning", "")
                 for r in group if r["judgment"].get("score", 0) >= 4),
                next(
                    (r["judgment"].get("shared_reasoning", "")
                     for r in group if r["judgment"].get("score", 0) >= 3),
                    "none"
                ),
            ),
        })

    sig_summaries.sort(key=lambda s: -s["avg_score"])

    total_pairs = len(results)
    total_same = sum(1 for r in results
                     if r["judgment"].get("same_procedure", False))
    all_scores = [r["judgment"].get("score", 0) for r in results]
    avg_all = sum(all_scores) / len(all_scores) if all_scores else 0

    return {
        "total_pairs_judged": total_pairs,
        "total_same_procedure": total_same,
        "pct_same_overall": round(total_same * 100 / total_pairs, 1) if total_pairs else 0,
        "avg_score_overall": round(avg_all, 2),
        "per_signature": sig_summaries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4.1-mini",
                    help="Judge model (default: gpt-4.1-mini)")
    ap.add_argument("--samples-per-domain", type=int, default=SAMPLES_PER_DOMAIN)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-o", "--output", default=str(OUT_PATH))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    keys_path = ROOT.parent / "keys.py"
    if not keys_path.exists():
        keys_path = ROOT / "keys.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("keys", str(keys_path))
    keys_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(keys_mod)
    openai_api_key = keys_mod.openai_api_key

    logger.info("Loading Layer-C templates from %s", LAYER_C_DIR)
    by_csig = load_all_templates()
    n_cross = sum(1 for cs, ts in by_csig.items()
                  if len({t["_domain"] for t in ts}) >= 2)
    logger.info("  %d collapsed sigs, %d cross-domain", len(by_csig), n_cross)

    pairs = build_cross_domain_pairs(by_csig, args.samples_per_domain)
    logger.info("  %d cross-domain pairs to judge", len(pairs))

    results = run_judgments(
        pairs, openai_api_key,
        model=args.model,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )

    summary = summarize(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "judgments": results,
        "config": {
            "model": args.model,
            "samples_per_domain": args.samples_per_domain,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }, indent=2, ensure_ascii=False))

    logger.info("="*60)
    logger.info("SUMMARY: %d pairs judged, avg_score=%.2f, %d%% same_procedure",
                summary["total_pairs_judged"],
                summary["avg_score_overall"],
                summary["pct_same_overall"])
    for s in summary["per_signature"]:
        logger.info("  %-40s avg=%.1f  same=%d/%d (%d%%)  verdict=%s",
                     s["collapsed_sig"], s["avg_score"],
                     s["n_same_procedure"], s["n_pairs_judged"],
                     s["pct_same"],
                     s["verdict"])
        if s["representative_shared_reasoning"] != "none":
            logger.info("    → %s", s["representative_shared_reasoning"])

    logger.info("Output: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
