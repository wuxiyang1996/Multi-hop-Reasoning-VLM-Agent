"""
Reward shaping for the Decision Agent GRPO trainer.

Wraps decision_agents.reward_func.RewardComputer with bank-state-aware
shaping and provides the compute_reward(prev, action, next, bank_state)
contract specified in the training plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from decision_agents.reward_func import (
    RewardComputer,
    RewardConfig,
    RewardResult,
)


@dataclass
class RewardBreakdown:
    """Extended reward breakdown for training logs."""

    r_env: float = 0.0
    r_follow: float = 0.0
    r_cost: float = 0.0
    r_total: float = 0.0
    action_type: str = "primitive"
    active_skill_id: Optional[str] = None
    newly_satisfied: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_env": self.r_env,
            "r_follow": self.r_follow,
            "r_cost": self.r_cost,
            "r_total": self.r_total,
            "action_type": self.action_type,
            "active_skill_id": self.active_skill_id,
            "newly_satisfied": self.newly_satisfied,
        }


class TrainRewardShaper:
    """Reward shaper for GRPO training.

    Extends the base RewardComputer with:
      - Bank-state-aware contract lookup
      - Progress-to-late-states shaping when active_skill != None
      - Full breakdown output for logging
    """

    def __init__(self, config: Optional[RewardConfig] = None, skill_bank: Any = None):
        self.config = config or RewardConfig()
        self.computer = RewardComputer(self.config)
        self.skill_bank = skill_bank

    def reset(self) -> None:
        self.computer.reset()

    def compute_reward(
        self,
        r_env: float,
        action_type: str,
        observation: str,
        active_skill_id: Optional[str] = None,
        skill_contract: Any = None,
        bank_state: Optional[Any] = None,
    ) -> RewardBreakdown:
        """Compute reward with full breakdown.

        Args:
            r_env: raw environment reward
            action_type: "primitive" | "QUERY_MEM" | "QUERY_SKILL" | "CALL_SKILL"
            observation: current observation text
            active_skill_id: skill currently being followed
            skill_contract: SkillEffectsContract (if known)
            bank_state: current skill bank (for contract lookup)

        Returns:
            RewardBreakdown with all components.
        """
        bank = bank_state or self.skill_bank
        contract = skill_contract
        if contract is None and active_skill_id and bank:
            try:
                contract = bank.get_contract(active_skill_id)
            except Exception:
                contract = None

        prev_satisfied = len(self.computer.satisfied_predicates)

        rr = self.computer.compute_reward(
            r_env=r_env,
            action_type=action_type,
            observation=observation,
            active_skill_id=active_skill_id,
            skill_contract=contract,
        )

        new_satisfied = len(self.computer.satisfied_predicates) - prev_satisfied

        return RewardBreakdown(
            r_env=rr.r_env,
            r_follow=rr.r_follow,
            r_cost=rr.r_cost,
            r_total=rr.r_total,
            action_type=action_type,
            active_skill_id=active_skill_id,
            newly_satisfied=max(new_satisfied, 0),
        )

    def update_bank(self, new_bank: Any) -> None:
        """Update the skill bank reference (after co-evolution update)."""
        self.skill_bank = new_bank

    @property
    def cumulative(self) -> RewardResult:
        return self.computer.cumulative

    @property
    def history(self) -> list:
        return self.computer.history
