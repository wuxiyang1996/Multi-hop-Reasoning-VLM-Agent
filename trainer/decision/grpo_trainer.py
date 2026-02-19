"""
GRPO (Group Relative Policy Optimization) trainer for the VLM Decision Agent.

Core loop:
  1. Sample G rollouts per prompt group with current policy
  2. Compute returns/advantages from r_total
  3. Rank within group → compute GRPO objective
  4. Update policy parameters
  5. Log metrics
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from trainer.common.metrics import (
    DecisionMetrics,
    RolloutRecord,
    aggregate_decision_metrics,
)
from trainer.decision.policy_interface import PolicyInterface
from trainer.decision.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO hyperparameters."""

    group_size: int = 8
    clip_ratio: float = 0.2
    kl_coeff: float = 0.01
    lr: float = 1e-5
    epochs_per_batch: int = 4
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coeff: float = 0.01


@dataclass
class GRPOTrainStats:
    """Statistics from one GRPO training iteration."""

    loss: float = 0.0
    policy_loss: float = 0.0
    kl_loss: float = 0.0
    entropy: float = 0.0
    mean_advantage: float = 0.0
    mean_return: float = 0.0
    grad_norm: float = 0.0
    n_episodes: int = 0
    n_steps: int = 0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


class GRPOTrainer:
    """GRPO trainer for the VLM Decision Agent.

    Implements the group-relative ranking objective: for each group of G
    rollouts from the same prompt/scenario, rank by return, compute
    relative advantages, and update the policy to increase probability of
    higher-ranked actions.
    """

    def __init__(
        self,
        policy: PolicyInterface,
        config: Optional[GRPOConfig] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        self.policy = policy
        self.cfg = config or GRPOConfig()
        self.buffer = replay_buffer or ReplayBuffer()
        self._iteration = 0
        self._total_episodes = 0

    def train_step(self, rollouts: List[RolloutRecord]) -> GRPOTrainStats:
        """Execute one GRPO training step on a batch of rollouts.

        Args:
            rollouts: list of RolloutRecords (ideally grouped by scenario)

        Returns:
            GRPOTrainStats with loss and diagnostic info.
        """
        self._iteration += 1
        self._total_episodes += len(rollouts)

        self.buffer.add_batch(rollouts)

        groups = self._form_groups(rollouts)
        all_advantages: List[float] = []
        all_logprobs: List[float] = []
        all_old_logprobs: List[float] = []
        all_returns: List[float] = []

        for group in groups:
            returns = [r.total_reward for r in group]
            advantages = self._compute_group_advantages(returns)
            all_returns.extend(returns)
            all_advantages.extend(advantages)

            for record, adv in zip(group, advantages):
                for step in record.steps:
                    if step.logprob is not None:
                        all_old_logprobs.append(step.logprob)
                        new_lp = self.policy.logprob(
                            observation=step.obs_id,
                            action=step.action,
                        )
                        all_logprobs.append(new_lp)

        if not all_logprobs:
            return GRPOTrainStats(n_episodes=len(rollouts))

        loss, stats = self._compute_grpo_loss(
            all_logprobs, all_old_logprobs, all_advantages
        )

        update_info = self.policy.update(loss)
        stats.n_episodes = len(rollouts)
        stats.n_steps = sum(r.episode_length for r in rollouts)
        stats.mean_return = float(np.mean(all_returns)) if all_returns else 0.0
        stats.grad_norm = update_info.get("grad_norm", 0.0)

        return stats

    def _form_groups(self, rollouts: List[RolloutRecord]) -> List[List[RolloutRecord]]:
        """Partition rollouts into groups of size G for relative ranking.

        If rollouts don't divide evenly, the last group may be smaller.
        Groups with < 2 members are dropped.
        """
        g = self.cfg.group_size
        groups = [rollouts[i:i + g] for i in range(0, len(rollouts), g)]
        return [grp for grp in groups if len(grp) >= 2]

    def _compute_group_advantages(self, returns: List[float]) -> List[float]:
        """Compute relative advantages within a group by ranking returns."""
        n = len(returns)
        if n <= 1:
            return [0.0] * n

        arr = np.array(returns, dtype=np.float64)

        if self.cfg.normalize_advantages:
            mean = arr.mean()
            std = arr.std() + 1e-8
            advantages = ((arr - mean) / std).tolist()
        else:
            ranks = np.argsort(np.argsort(arr)).astype(np.float64)
            advantages = (2.0 * ranks / (n - 1) - 1.0).tolist()

        return advantages

    def _compute_grpo_loss(
        self,
        logprobs: List[float],
        old_logprobs: List[float],
        advantages: List[float],
    ) -> tuple:
        """Compute the GRPO clipped objective + KL penalty.

        Returns (loss, GRPOTrainStats).
        """
        lp = np.array(logprobs, dtype=np.float64)
        old_lp = np.array(old_logprobs, dtype=np.float64)

        n = min(len(lp), len(old_lp), len(advantages))
        if n == 0:
            return 0.0, GRPOTrainStats()

        lp = lp[:n]
        old_lp = old_lp[:n]
        adv = np.array(advantages[:n], dtype=np.float64)

        if self.cfg.normalize_advantages and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ratio = np.exp(lp - old_lp)
        clipped = np.clip(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
        policy_loss = -np.mean(np.minimum(ratio * adv, clipped * adv))

        kl = np.mean(old_lp - lp)
        kl_loss = self.cfg.kl_coeff * kl

        entropy = -np.mean(lp)
        entropy_bonus = -self.cfg.entropy_coeff * entropy

        loss = float(policy_loss + kl_loss + entropy_bonus)

        stats = GRPOTrainStats(
            loss=loss,
            policy_loss=float(policy_loss),
            kl_loss=float(kl_loss),
            entropy=float(entropy),
            mean_advantage=float(adv.mean()),
        )
        return loss, stats

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> List[float]:
        """Compute Generalized Advantage Estimation for a single episode."""
        T = len(rewards)
        advantages = [0.0] * T
        gae = 0.0
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            if dones[t]:
                next_val = 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * gae * (0.0 if dones[t] else 1.0)
            advantages[t] = gae

        return advantages

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def total_episodes(self) -> int:
        return self._total_episodes
