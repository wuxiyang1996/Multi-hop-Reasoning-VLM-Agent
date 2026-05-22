#!/usr/bin/env python
"""Stage 3 GRPO domain adaptation for non-game benchmarks.

Runs GRPO on the 20% train split of each non-game benchmark:

  1. Loads train sample IDs from ``stage3_splits/<task>_train.txt``
  2. Constructs environments (VRReasoningEnv for QA, BrowserGym for web)
  3. Runs rollout episodes via ``unified_episode_runner``
  4. Converts results to GRPORecords
  5. Trains decision LoRAs (skill_selection + action_taking) via FSDP GRPO

The base model is raw Qwen3.5-9B with Gaussian-initialized LoRA (no
game-trained weights).  Mega-skill seed banks (from
``stage3_seeds_from_megaskills.py``) are loaded to bootstrap the skill bank.

Usage::

    python scripts/run_stage3_grpo.py \\
        --task visual_toolbench \\
        --total-steps 10 \\
        --seed-bank-dir frontier_data/output/stage3_seed_banks/visual_toolbench

    # Or run all tasks sequentially:
    python scripts/run_stage3_grpo.py --task all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
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
logger = logging.getLogger("stage3_grpo")


# ── Task registry ────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "visual_toolbench": {
        "domain": "qa",
        "default_steps": 10,
        "max_episode_steps": 15,
        "env_type": "vr",
    },
    "tir_bench": {
        "domain": "qa",
        "default_steps": 10,
        "max_episode_steps": 15,
        "env_type": "vr",
    },
    "video_holmes": {
        "domain": "qa",
        "default_steps": 10,
        "max_episode_steps": 15,
        "env_type": "vr",
    },
    "siv_bench": {
        "domain": "qa",
        "default_steps": 10,
        "max_episode_steps": 15,
        "env_type": "vr",
    },
    "webshop": {
        "domain": "web",
        "default_steps": 25,
        "max_episode_steps": 20,
        "env_type": "browsergym",
    },
    "miniwob": {
        "domain": "web",
        "default_steps": 15,
        "max_episode_steps": 10,
        "env_type": "browsergym",
    },
}


# ── Train split loading ──────────────────────────────────────────────

def load_train_ids(task: str) -> List[str]:
    split_path = REPO_ROOT / "cold_start" / "task_samples" / "stage3_splits" / f"{task}_train.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Train split not found: {split_path}")
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    logger.info("[%s] loaded %d train sample IDs", task, len(ids))
    return ids


# ── Seed bank loading ────────────────────────────────────────────────

def load_seed_bank(seed_bank_dir: Path) -> List[dict]:
    bank_path = seed_bank_dir / "skill_bank.jsonl"
    if not bank_path.exists():
        logger.warning("No seed bank at %s", bank_path)
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
    logger.info("loaded %d seed skills from %s", len(skills), bank_path)
    return skills


# ── Environment factories ────────────────────────────────────────────

def _build_vr_envs(task: str, sample_ids: List[str], max_steps: int) -> List[dict]:
    """Build env configs for visual/video reasoning tasks.

    Returns a list of dicts with keys: sample_id, env_factory, question, ground_truth.
    The env_factory is a callable that creates the env on demand.
    """
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
                    "task": task,
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
                    "task": task,
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
                    "task": task,
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
                    "task": task,
                })
                if len(configs) >= len(sample_ids):
                    break

    logger.info("[%s] matched %d / %d train samples to benchmark data",
                task, len(configs), len(sample_ids))
    return configs


def make_vr_env_from_config(cfg: dict, max_steps: int):
    """Create a VRReasoningEnv from a sample config dict."""
    from env_wrappers.vr_reasoning_env import make_vr_env
    return make_vr_env(
        question=cfg["question"],
        image_source=cfg["image_source"],
        ground_truth=cfg["ground_truth"],
        max_steps=max_steps,
    )


def _build_web_envs(task: str, sample_ids: List[str]) -> List[dict]:
    """Build env configs for web tasks (MiniWoB/WebShop).

    Returns a list of dicts with keys: sample_id, task_id.
    BrowserGym envs are created on-demand during rollout.
    """
    configs = []
    for sid in sample_ids:
        configs.append({
            "sample_id": sid,
            "task_id": sid,
            "task": task,
        })
    return configs


def make_web_env(task: str, task_id: str):
    """Create a BrowserGym environment for a web task."""
    import gymnasium as gym
    try:
        import browsergym.core  # noqa: F401
    except ImportError:
        logger.error("browsergym not installed; web tasks unavailable")
        raise

    if task == "miniwob":
        try:
            import browsergym.miniwob  # noqa: F401
        except ImportError:
            pass
        env_id = f"browsergym/{task_id}"
    elif task == "webshop":
        try:
            import browsergym.webshop  # noqa: F401
        except ImportError:
            pass
        env_id = f"browsergym/webshop.{task_id}"
    else:
        env_id = f"browsergym/{task_id}"

    env = gym.make(env_id)
    return env


# ── Rollout + GRPO loop ─────────────────────────────────────────────

async def run_grpo_for_task(
    task: str,
    total_steps: int,
    seed_bank_dir: Optional[Path],
    adapter_dir: Path,
    model_name: str,
    devices: List[int],
    checkpoint_every: int = 5,
    episodes_per_step: int = 8,
    output_dir: Optional[Path] = None,
):
    """Main GRPO loop for a single non-game task."""
    from trainer.coevolution.episode_runner import GRPORecord
    from trainer.coevolution.grpo_training import (
        DecisionGRPOTrainer,
        _collect_grpo_records,
        _MIN_SAMPLES,
    )
    from trainer.coevolution.unified_episode_runner import (
        UnifiedEpisodeResult,
        run_unified_episode,
        unified_result_to_grpo_records,
    )
    from trainer.coevolution.vllm_client import AsyncVLLMClient
    from decision_agents.skill_decision_core import DOMAIN_QA, DOMAIN_WEB

    cfg = TASK_CONFIGS[task]
    domain = DOMAIN_QA if cfg["domain"] == "qa" else DOMAIN_WEB
    max_ep_steps = cfg["max_episode_steps"]

    # Load train IDs
    train_ids = load_train_ids(task)

    # Load seed bank
    seed_skills = []
    if seed_bank_dir and seed_bank_dir.exists():
        seed_skills = load_seed_bank(seed_bank_dir)

    # Build env configs
    if cfg["env_type"] == "vr":
        env_configs = _build_vr_envs(task, train_ids, max_ep_steps)
    else:
        env_configs = _build_web_envs(task, train_ids)

    if not env_configs:
        logger.error("[%s] no env configs built; aborting", task)
        return

    # Output directory for checkpoints and logs
    if output_dir is None:
        output_dir = REPO_ROOT / "runs" / "stage3_grpo" / task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize vLLM client
    vllm_client = AsyncVLLMClient(model_name=model_name)

    # Initialize GRPO trainer
    grpo_trainer = DecisionGRPOTrainer(
        model_name=model_name,
        adapter_dir=str(adapter_dir),
        devices=devices,
        lr=5e-5,
        kl_coeff=0.05,
        clip_ratio=0.2,
        max_epochs=4,
    )

    metrics_log: List[dict] = []

    for step in range(total_steps):
        step_t0 = time.monotonic()
        logger.info("=" * 60)
        logger.info("[%s] GRPO step %d / %d", task, step + 1, total_steps)
        logger.info("=" * 60)

        # ── Phase A: Rollout collection ──
        import random
        sample_batch = random.sample(
            env_configs, min(episodes_per_step, len(env_configs))
        )

        episode_results: List[UnifiedEpisodeResult] = []
        all_grpo_records: Dict[str, List[GRPORecord]] = {
            "action_taking": [],
            "skill_selection": [],
        }

        for i, ecfg in enumerate(sample_batch):
            try:
                if cfg["env_type"] == "vr":
                    env = make_vr_env_from_config(ecfg, max_ep_steps)
                else:
                    env = make_web_env(task, ecfg["task_id"])

                result = await run_unified_episode(
                    env=env,
                    task_name=task,
                    max_steps=max_ep_steps,
                    vllm_client=vllm_client,
                    domain=domain,
                    skill_bank=seed_skills if seed_skills else None,
                    temperature=0.3,
                )
                episode_results.append(result)

                records = unified_result_to_grpo_records(
                    result,
                    episode_reward=result.total_reward,
                )
                for rec in records:
                    if rec.adapter in all_grpo_records:
                        all_grpo_records[rec.adapter].append(rec)

                logger.info(
                    "  episode %d/%d [%s] steps=%d reward=%.3f",
                    i + 1, len(sample_batch),
                    ecfg.get("sample_id", "?"),
                    result.steps,
                    result.total_reward,
                )
            except Exception as e:
                logger.warning("  episode %d failed: %s", i + 1, e)
                continue

        total_records = sum(len(v) for v in all_grpo_records.values())
        logger.info(
            "[%s] step %d rollout: %d episodes, %d GRPO records",
            task, step + 1, len(episode_results), total_records,
        )

        # ── Phase C: GRPO training ──
        if total_records < 16:
            logger.warning(
                "[%s] step %d: only %d records (< 16 minimum), skipping GRPO",
                task, step + 1, total_records,
            )
        else:
            try:
                train_stats = grpo_trainer.train_step(all_grpo_records, step=step)
                for adapter, stats in train_stats.items():
                    logger.info(
                        "  GRPO [%s]: %d samples, loss=%.4f, epochs=%d, %.1fs",
                        adapter, stats.n_samples, stats.mean_loss,
                        stats.epochs, stats.wall_time_s,
                    )
            except Exception as e:
                logger.error("[%s] step %d GRPO failed: %s", task, step + 1, e)
                train_stats = {}

        step_wall = time.monotonic() - step_t0

        # ── Metrics ──
        ep_rewards = [r.total_reward for r in episode_results]
        avg_reward = sum(ep_rewards) / max(len(ep_rewards), 1)
        step_metrics = {
            "task": task,
            "step": step + 1,
            "n_episodes": len(episode_results),
            "n_grpo_records": total_records,
            "avg_reward": round(avg_reward, 4),
            "wall_time_s": round(step_wall, 1),
        }
        metrics_log.append(step_metrics)
        logger.info("[%s] step %d metrics: %s", task, step + 1, json.dumps(step_metrics))

        # ── Checkpoint ──
        if (step + 1) % checkpoint_every == 0 or (step + 1) == total_steps:
            ckpt_path = output_dir / f"step_{step + 1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            (ckpt_path / "metrics.json").write_text(
                json.dumps(metrics_log, indent=2)
            )
            logger.info("[%s] checkpoint saved to %s", task, ckpt_path)

    # Final summary
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps({
        "task": task,
        "total_steps": total_steps,
        "model_name": model_name,
        "seed_bank_dir": str(seed_bank_dir) if seed_bank_dir else None,
        "adapter_dir": str(adapter_dir),
        "metrics": metrics_log,
    }, indent=2))
    logger.info("[%s] training complete. Summary → %s", task, summary_path)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--task", required=True,
        help="Task name (visual_toolbench, tir_bench, video_holmes, "
             "siv_bench, webshop, miniwob) or 'all'",
    )
    ap.add_argument("--total-steps", type=int, default=None,
                    help="GRPO steps (default: per-task default)")
    ap.add_argument("--seed-bank-dir", type=str, default=None,
                    help="Path to seed bank directory")
    ap.add_argument("--adapter-dir", type=str,
                    default=str(REPO_ROOT / "runs" / "stage3_adapters"))
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-9B")
    ap.add_argument("--devices", type=str, default="4,5,6,7",
                    help="GPU IDs for GRPO training")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--episodes-per-step", type=int, default=8)
    ap.add_argument("--output-dir", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    devices = [int(d) for d in args.devices.split(",")]

    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    for task in tasks:
        if task not in TASK_CONFIGS:
            logger.error("Unknown task: %s", task)
            continue

        total_steps = args.total_steps or TASK_CONFIGS[task]["default_steps"]
        seed_bank_dir = (
            Path(args.seed_bank_dir) if args.seed_bank_dir
            else REPO_ROOT / "frontier_data" / "output" / "stage3_seed_banks" / task
        )
        output_dir = Path(args.output_dir) if args.output_dir else None

        # Ensure adapter dir exists with gaussian-init LoRAs
        adapter_dir = Path(args.adapter_dir) / task
        adapter_dir.mkdir(parents=True, exist_ok=True)

        asyncio.run(run_grpo_for_task(
            task=task,
            total_steps=total_steps,
            seed_bank_dir=seed_bank_dir,
            adapter_dir=adapter_dir,
            model_name=args.model_name,
            devices=devices,
            checkpoint_every=args.checkpoint_every,
            episodes_per_step=args.episodes_per_step,
            output_dir=output_dir,
        ))


if __name__ == "__main__":
    main()
