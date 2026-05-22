#!/usr/bin/env python
"""Stage 3 evaluation on held-out test splits.

Runs inference (no GRPO) on the 80% test split of each non-game benchmark,
comparing three conditions:

  1. **Baseline**: Raw Qwen3.5-9B, no seeds, no GRPO adapters
  2. **Seeds only**: Raw Qwen3.5-9B + mega-skill seed bank (no GRPO)
  3. **Seeds + GRPO**: Raw Qwen3.5-9B + mega-skill seeds + trained adapters

Metrics per task:
  - VR/Video (QA): answer accuracy (exact match + fuzzy/MCQ)
  - Web (MiniWoB/WebShop): task success rate, reward

Usage::

    python scripts/run_stage3_eval.py \\
        --task visual_toolbench \\
        --condition seeds_grpo \\
        --adapter-dir runs/stage3_adapters/visual_toolbench

    # All conditions for one task:
    python scripts/run_stage3_eval.py --task visual_toolbench --condition all

    # All tasks, all conditions:
    python scripts/run_stage3_eval.py --task all --condition all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("stage3_eval")


# ── Task registry (mirrors run_stage3_grpo.py) ──────────────────────

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "visual_toolbench": {"domain": "qa", "max_episode_steps": 15, "env_type": "vr"},
    "tir_bench":        {"domain": "qa", "max_episode_steps": 15, "env_type": "vr"},
    "video_holmes":     {"domain": "qa", "max_episode_steps": 15, "env_type": "vr"},
    "siv_bench":        {"domain": "qa", "max_episode_steps": 15, "env_type": "vr"},
    "webshop":          {"domain": "web", "max_episode_steps": 20, "env_type": "browsergym"},
    "miniwob":          {"domain": "web", "max_episode_steps": 10, "env_type": "browsergym"},
}

CONDITIONS = ["baseline", "seeds_only", "seeds_grpo"]


# ── Test split loading ───────────────────────────────────────────────

def load_test_ids(task: str) -> List[str]:
    split_path = REPO_ROOT / "cold_start" / "task_samples" / "stage3_splits" / f"{task}_test.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Test split not found: {split_path}")
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    logger.info("[%s] loaded %d test sample IDs", task, len(ids))
    return ids


def load_seed_bank(task: str) -> List[dict]:
    bank_path = (
        REPO_ROOT / "frontier_data" / "output" / "stage3_seed_banks"
        / task / "skill_bank.jsonl"
    )
    if not bank_path.exists():
        return []
    skills = []
    with open(bank_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    skills.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return skills


# ── Answer matching utilities ────────────────────────────────────────

def _normalize(s: str) -> str:
    """Normalize answer string for comparison."""
    import re
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".")
    return s


def exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def fuzzy_match(pred: str, gold: str) -> float:
    """Compute a fuzzy match score (0-1) between pred and gold answers."""
    np, ng = _normalize(pred), _normalize(gold)
    if np == ng:
        return 1.0
    if ng in np or np in ng:
        return 0.8

    # Token-level F1
    pred_tokens = set(np.split())
    gold_tokens = set(ng.split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def mcq_match(pred: str, gold: str) -> bool:
    """Match MCQ-style answers (e.g. 'A', 'B', 'C', 'D')."""
    import re
    np = _normalize(pred)
    ng = _normalize(gold)
    # Try to extract option letters
    pred_opt = re.search(r"\b([a-d])\b", np)
    gold_opt = re.search(r"\b([a-d])\b", ng)
    if pred_opt and gold_opt:
        return pred_opt.group(1) == gold_opt.group(1)
    return np == ng


# ── Environment factories (reused from grpo script) ─────────────────

def _build_vr_eval_configs(task: str, sample_ids: List[str]) -> List[dict]:
    """Build env configs for VR/video eval samples."""
    configs = []

    if task == "visual_toolbench":
        from visual_reasoning_wrapper.benchmarks import iter_visual_toolbench_samples
        id_set = set(sample_ids)
        for sample in iter_visual_toolbench_samples():
            if sample.sample_id in id_set:
                configs.append({
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "ground_truth": sample.gold_answer or "",
                    "image_source": sample.image_cell,
                })
                if len(configs) >= len(sample_ids):
                    break

    elif task == "tir_bench":
        from visual_reasoning_wrapper.benchmarks import iter_tir_bench_samples
        id_set = set(sample_ids)
        for sample in iter_tir_bench_samples():
            if sample.sample_id in id_set:
                configs.append({
                    "sample_id": sample.sample_id,
                    "question": sample.prompt,
                    "ground_truth": sample.answer or "",
                    "image_source": sample.image_1,
                })
                if len(configs) >= len(sample_ids):
                    break

    elif task == "video_holmes":
        from visual_reasoning_wrapper.benchmarks import iter_video_holmes_samples
        id_set = set(sample_ids)
        for sample in iter_video_holmes_samples():
            if sample.sample_id in id_set:
                configs.append({
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth or "",
                    "image_source": sample.frames,
                })
                if len(configs) >= len(sample_ids):
                    break

    elif task == "siv_bench":
        from visual_reasoning_wrapper.benchmarks import iter_siv_bench_samples
        id_set = set(sample_ids)
        for sample in iter_siv_bench_samples():
            if sample.sample_id in id_set:
                configs.append({
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "ground_truth": sample.answer or "",
                    "image_source": sample.frames,
                })
                if len(configs) >= len(sample_ids):
                    break

    logger.info("[%s] matched %d / %d test samples", task, len(configs), len(sample_ids))
    return configs


# ── Eval runner ──────────────────────────────────────────────────────

async def eval_task_condition(
    task: str,
    condition: str,
    adapter_dir: Optional[Path],
    model_name: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a single (task, condition) pair and return metrics."""
    from decision_agents.skill_decision_core import DOMAIN_QA, DOMAIN_WEB
    from trainer.coevolution.unified_episode_runner import run_unified_episode
    from trainer.coevolution.vllm_client import AsyncVLLMClient

    cfg = TASK_CONFIGS[task]
    domain = DOMAIN_QA if cfg["domain"] == "qa" else DOMAIN_WEB
    max_ep_steps = cfg["max_episode_steps"]

    test_ids = load_test_ids(task)
    if max_samples and max_samples < len(test_ids):
        import random
        random.seed(42)
        test_ids = random.sample(test_ids, max_samples)

    # Condition-specific setup
    use_seeds = condition in ("seeds_only", "seeds_grpo")
    use_adapters = condition == "seeds_grpo"

    seed_skills = load_seed_bank(task) if use_seeds else []

    # For baseline and seeds_only, no adapter loading
    # For seeds_grpo, adapters should already be loaded into the model
    vllm_client = AsyncVLLMClient(model_name=model_name)

    # Build env configs
    if cfg["env_type"] == "vr":
        env_configs = _build_vr_eval_configs(task, test_ids)
    else:
        env_configs = [{"sample_id": sid, "task_id": sid} for sid in test_ids]

    if not env_configs:
        logger.warning("[%s:%s] no eval configs; skipping", task, condition)
        return {"task": task, "condition": condition, "error": "no_configs"}

    # Run episodes
    results: List[dict] = []
    for i, ecfg in enumerate(env_configs):
        try:
            if cfg["env_type"] == "vr":
                from env_wrappers.vr_reasoning_env import make_vr_env
                env = make_vr_env(
                    question=ecfg["question"],
                    image_source=ecfg["image_source"],
                    ground_truth=ecfg["ground_truth"],
                    max_steps=max_ep_steps,
                )
            else:
                from scripts.run_stage3_grpo import make_web_env
                env = make_web_env(task, ecfg["task_id"])

            ep_result = await run_unified_episode(
                env=env,
                task_name=task,
                max_steps=max_ep_steps,
                vllm_client=vllm_client,
                domain=domain,
                skill_bank=seed_skills if seed_skills else None,
                temperature=0.0,
            )

            # Compute metrics
            sample_metrics: Dict[str, Any] = {
                "sample_id": ecfg.get("sample_id", "?"),
                "steps": ep_result.steps,
                "reward": ep_result.total_reward,
                "terminated": ep_result.terminated,
                "final_answer": ep_result.final_answer,
            }

            # For QA tasks, compute answer accuracy
            if cfg["domain"] == "qa" and "ground_truth" in ecfg:
                gold = ecfg["ground_truth"]
                pred = ep_result.final_answer or ""
                sample_metrics["exact_match"] = exact_match(pred, gold)
                sample_metrics["fuzzy_score"] = fuzzy_match(pred, gold)
                if task == "siv_bench":
                    sample_metrics["mcq_match"] = mcq_match(pred, gold)

            # For web tasks, reward IS the metric
            if cfg["domain"] == "web":
                sample_metrics["success"] = ep_result.total_reward > 0.5

            results.append(sample_metrics)

            if (i + 1) % 10 == 0:
                logger.info(
                    "[%s:%s] %d / %d evaluated",
                    task, condition, i + 1, len(env_configs),
                )

        except Exception as e:
            logger.warning(
                "[%s:%s] sample %s failed: %s",
                task, condition, ecfg.get("sample_id", "?"), e,
            )
            results.append({
                "sample_id": ecfg.get("sample_id", "?"),
                "error": str(e),
            })

    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    agg: Dict[str, Any] = {
        "task": task,
        "condition": condition,
        "n_total": len(results),
        "n_valid": len(valid_results),
        "n_errors": len(results) - len(valid_results),
    }

    if cfg["domain"] == "qa":
        em_scores = [r["exact_match"] for r in valid_results if "exact_match" in r]
        fuzzy_scores = [r["fuzzy_score"] for r in valid_results if "fuzzy_score" in r]
        agg["exact_match_accuracy"] = (
            sum(em_scores) / len(em_scores) if em_scores else 0.0
        )
        agg["fuzzy_match_mean"] = (
            sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0
        )
        if task == "siv_bench":
            mcq_scores = [r["mcq_match"] for r in valid_results if "mcq_match" in r]
            agg["mcq_accuracy"] = (
                sum(mcq_scores) / len(mcq_scores) if mcq_scores else 0.0
            )

    if cfg["domain"] == "web":
        rewards = [r["reward"] for r in valid_results if "reward" in r]
        successes = [r["success"] for r in valid_results if "success" in r]
        agg["mean_reward"] = sum(rewards) / len(rewards) if rewards else 0.0
        agg["success_rate"] = sum(successes) / len(successes) if successes else 0.0

    agg["mean_steps"] = (
        sum(r["steps"] for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"{task}_{condition}_results.json"
        results_path.write_text(json.dumps({
            "aggregate": agg,
            "per_sample": results,
        }, indent=2))
        logger.info("results → %s", results_path)

    return agg


async def run_comparison(
    task: str,
    conditions: List[str],
    adapter_dir: Optional[Path],
    model_name: str,
    max_samples: Optional[int],
    output_dir: Path,
):
    """Run all conditions for a task and produce a comparison table."""
    all_results = {}
    for condition in conditions:
        logger.info("=" * 60)
        logger.info("[%s] evaluating condition: %s", task, condition)
        logger.info("=" * 60)
        result = await eval_task_condition(
            task=task,
            condition=condition,
            adapter_dir=adapter_dir,
            model_name=model_name,
            max_samples=max_samples,
            output_dir=output_dir,
        )
        all_results[condition] = result

    # Save comparison
    comparison_path = output_dir / f"{task}_comparison.json"
    comparison_path.write_text(json.dumps(all_results, indent=2))

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("[%s] COMPARISON TABLE", task)
    logger.info("=" * 60)

    cfg = TASK_CONFIGS[task]
    if cfg["domain"] == "qa":
        logger.info("%-15s | %10s | %10s | %8s", "Condition", "Exact Match", "Fuzzy Mean", "Steps")
        logger.info("-" * 55)
        for cond, res in all_results.items():
            logger.info(
                "%-15s | %10.4f | %10.4f | %8.1f",
                cond,
                res.get("exact_match_accuracy", 0),
                res.get("fuzzy_match_mean", 0),
                res.get("mean_steps", 0),
            )
    else:
        logger.info("%-15s | %10s | %10s | %8s", "Condition", "Success", "Mean Reward", "Steps")
        logger.info("-" * 55)
        for cond, res in all_results.items():
            logger.info(
                "%-15s | %10.4f | %10.4f | %8.1f",
                cond,
                res.get("success_rate", 0),
                res.get("mean_reward", 0),
                res.get("mean_steps", 0),
            )

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", required=True,
                    help="Task name or 'all'")
    ap.add_argument("--condition", default="all",
                    help="Condition (baseline, seeds_only, seeds_grpo) or 'all'")
    ap.add_argument("--adapter-dir", type=str,
                    default=str(REPO_ROOT / "runs" / "stage3_adapters"))
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit test samples per task (for debugging)")
    ap.add_argument("--output-dir", type=str,
                    default=str(REPO_ROOT / "runs" / "stage3_eval"))
    return ap.parse_args()


def main():
    args = parse_args()
    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]
    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    output_dir = Path(args.output_dir)

    all_comparisons = {}
    for task in tasks:
        if task not in TASK_CONFIGS:
            logger.error("Unknown task: %s", task)
            continue
        adapter_dir = Path(args.adapter_dir) / task
        results = asyncio.run(run_comparison(
            task=task,
            conditions=conditions,
            adapter_dir=adapter_dir,
            model_name=args.model_name,
            max_samples=args.max_samples,
            output_dir=output_dir,
        ))
        all_comparisons[task] = results

    # Save overall summary
    if len(tasks) > 1:
        summary_path = output_dir / "stage3_eval_summary.json"
        summary_path.write_text(json.dumps(all_comparisons, indent=2))
        logger.info("Overall summary → %s", summary_path)

        # Print overall table
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 3 OVERALL EVALUATION SUMMARY")
        logger.info("=" * 70)
        for task, results in all_comparisons.items():
            cfg = TASK_CONFIGS[task]
            primary_metric = (
                "exact_match_accuracy" if cfg["domain"] == "qa" else "success_rate"
            )
            logger.info("\n[%s] (%s):", task, primary_metric)
            for cond, res in results.items():
                val = res.get(primary_metric, res.get("error", "N/A"))
                logger.info("  %-15s: %s", cond, val)


if __name__ == "__main__":
    main()
