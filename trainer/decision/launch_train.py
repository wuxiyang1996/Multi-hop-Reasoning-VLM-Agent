"""
CLI entry point for Decision Agent GRPO training.

Usage:
    python -m trainer.decision.launch_train --config trainer/common/configs/decision_grpo.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from trainer.common.eval_harness import run_decision_eval
from trainer.common.logging import TrainLogger
from trainer.common.metrics import aggregate_decision_metrics
from trainer.common.seeds import SeedManager, set_global_seed
from trainer.decision.grpo_trainer import GRPOConfig, GRPOTrainer
from trainer.decision.policy_interface import LLMPolicy
from trainer.decision.replay_buffer import ReplayBuffer
from trainer.decision.rollout_collector import collect_batch

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grpo_config(cfg: Dict[str, Any]) -> GRPOConfig:
    grpo = cfg.get("grpo", {})
    return GRPOConfig(
        group_size=grpo.get("group_size", 8),
        clip_ratio=grpo.get("clip_ratio", 0.2),
        kl_coeff=grpo.get("kl_coeff", 0.01),
        lr=grpo.get("lr", 1e-5),
        epochs_per_batch=grpo.get("epochs_per_batch", 4),
        max_grad_norm=grpo.get("max_grad_norm", 1.0),
        gamma=grpo.get("gamma", 0.99),
        gae_lambda=grpo.get("gae_lambda", 0.95),
        normalize_advantages=grpo.get("normalize_advantages", True),
        entropy_coeff=grpo.get("entropy_coeff", 0.01),
    )


def train_decision_agent(
    cfg: Dict[str, Any],
    env_factory: Any = None,
    skill_bank: Any = None,
    memory: Any = None,
) -> None:
    """Main training loop for the Decision Agent.

    Args:
        cfg: parsed YAML config dict
        env_factory: callable(seed) -> env; if None, uses a dummy
        skill_bank: SkillBankMVP instance
        memory: EpisodicMemoryStore instance
    """
    set_global_seed(42)

    grpo_cfg = build_grpo_config(cfg)
    model_cfg = cfg.get("model", {})
    rollout_cfg = cfg.get("rollout", {})
    replay_cfg = cfg.get("replay", {})
    eval_cfg = cfg.get("eval", {})
    log_cfg = cfg.get("logging", {})
    sched_cfg = cfg.get("schedule", {})

    policy = LLMPolicy(
        model_name=model_cfg.get("name", "gpt-4o-mini"),
        lr=grpo_cfg.lr,
    )

    buffer = ReplayBuffer(
        capacity=replay_cfg.get("capacity", 10000),
        priority_alpha=replay_cfg.get("priority_alpha", 0.6),
        priority_beta=replay_cfg.get("priority_beta", 0.4),
        min_episodes=replay_cfg.get("min_episodes", 64),
    )

    trainer = GRPOTrainer(policy=policy, config=grpo_cfg, replay_buffer=buffer)

    seed_manager = SeedManager(
        base_seed=42,
        eval_seeds=eval_cfg.get("seeds", [42, 137, 256, 512]),
    )

    train_logger = TrainLogger(
        log_dir=log_cfg.get("log_dir", "runs/decision_grpo"),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project"),
    )

    from decision_agents.reward_func import RewardConfig
    costs = cfg.get("costs", {})
    follow = cfg.get("follow_shaping", {})
    reward_config = RewardConfig(
        w_follow=follow.get("w_follow", 0.1),
        query_mem_cost=costs.get("c_mem", -0.05),
        query_skill_cost=costs.get("c_skill", -0.05),
        call_skill_cost=costs.get("c_call", -0.02),
        skill_switch_cost=costs.get("c_switch", -0.10),
        follow_predicate_bonus=follow.get("predicate_bonus", 0.05),
        follow_completion_bonus=follow.get("completion_bonus", 0.20),
        follow_no_progress_penalty=follow.get("no_progress_penalty", -0.01),
    )

    total_episodes = sched_cfg.get("total_episodes", 50000)
    batch_size = rollout_cfg.get("batch_size", 32)
    max_steps = rollout_cfg.get("max_steps", 500)
    eval_interval = eval_cfg.get("interval_episodes", 50)
    log_interval = log_cfg.get("log_interval", 10)
    save_interval = log_cfg.get("save_interval", 100)

    episode_count = 0
    bank_version = 0

    logger.info("Starting Decision Agent GRPO training (total=%d)", total_episodes)

    while episode_count < total_episodes:
        rollouts = collect_batch(
            env_factory=env_factory,
            policy=policy,
            skill_bank=skill_bank,
            memory=memory,
            reward_config=reward_config,
            seed_manager=seed_manager,
            batch_size=batch_size,
            max_steps=max_steps,
        )

        stats = trainer.train_step(rollouts)
        episode_count += len(rollouts)

        if episode_count % log_interval < batch_size:
            metrics = aggregate_decision_metrics(rollouts)
            train_logger.log_decision_metrics(
                metrics, episode=episode_count,
                extra=stats.to_dict(),
            )

        if episode_count % eval_interval < batch_size:
            eval_result = run_decision_eval(
                env_factory=env_factory,
                agent=None,
                seed_manager=seed_manager,
                num_episodes=eval_cfg.get("num_eval_episodes", 10),
                max_steps=eval_cfg.get("timeout_steps", 1000),
                bank_version=bank_version,
            )
            train_logger.log_eval(
                eval_result.metrics,
                episode=episode_count,
                bank_version=bank_version,
                seeds_used=eval_result.seeds_used,
            )

        if episode_count % save_interval < batch_size:
            checkpoint_dir = Path(log_cfg.get("log_dir", "runs/decision_grpo")) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            train_logger.log_event("checkpoint", {
                "episode": episode_count,
                "iteration": trainer.iteration,
            })

    train_logger.log_event("training_complete", {"total_episodes": episode_count})
    train_logger.close()
    logger.info("Decision Agent training complete (%d episodes)", episode_count)


def main():
    parser = argparse.ArgumentParser(description="Decision Agent GRPO Training")
    parser.add_argument(
        "--config", type=str,
        default="trainer/common/configs/decision_grpo.yaml",
        help="Path to GRPO config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    train_decision_agent(cfg)


if __name__ == "__main__":
    main()
