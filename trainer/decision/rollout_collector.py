"""
Rollout collector for the Decision Agent GRPO trainer.

Collects complete episode rollouts using the EnvWrapper, recording actions,
observations, reward breakdowns, predicates, embeddings, and policy logprobs.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from trainer.common.metrics import RolloutRecord, RolloutStep
from trainer.common.seeds import SeedManager
from trainer.decision.env_wrapper import EnvWrapper
from trainer.decision.reward_shaping import TrainRewardShaper

logger = logging.getLogger(__name__)


def classify_action_type(action: str) -> str:
    """Determine whether an action is primitive or a retrieval action."""
    if not action:
        return "primitive"
    upper = action.upper()
    for prefix in ("QUERY_MEM", "QUERY_SKILL", "CALL_SKILL"):
        if prefix in upper:
            return prefix
    return "primitive"


def collect_rollout(
    env_wrapper: EnvWrapper,
    policy: Any,
    reward_shaper: TrainRewardShaper,
    seed: int,
    max_steps: int = 500,
    episode_id: Optional[str] = None,
    extract_predicates: Optional[Callable[[str], Dict[str, float]]] = None,
    extract_embedding: Optional[Callable[[str], List[float]]] = None,
) -> RolloutRecord:
    """Collect one complete episode rollout.

    Args:
        env_wrapper: EnvWrapper instance (handles retrieval-as-action)
        policy: PolicyInterface for action sampling
        reward_shaper: TrainRewardShaper for reward computation
        seed: environment seed
        max_steps: per-episode step limit
        episode_id: unique episode identifier
        extract_predicates: optional callable(obs) -> dict of predicate probs
        extract_embedding: optional callable(obs) -> embedding vector

    Returns:
        RolloutRecord with complete step-by-step data.
    """
    episode_id = episode_id or str(uuid.uuid4())[:8]
    reward_shaper.reset()

    obs, info = env_wrapper.reset(seed=seed)

    steps: List[RolloutStep] = []
    done = False
    t = 0

    while t < max_steps and not done:
        context = {
            "skill_cards": env_wrapper.state.retrieved_cards,
            "active_skill": env_wrapper.active_skill_id,
            "step": t,
        }

        policy_out = policy.sample(obs, context=context)
        action = policy_out.action
        logprob = policy_out.logprob

        action_type = classify_action_type(action)
        step_result = env_wrapper.step(action)

        r_env = step_result["r_env"]
        next_obs = step_result["obs"]
        done = step_result["done"]

        breakdown = reward_shaper.compute_reward(
            r_env=r_env,
            action_type=action_type,
            observation=next_obs,
            active_skill_id=step_result.get("active_skill_id"),
            skill_contract=step_result.get("active_skill_contract"),
        )

        predicates = {}
        if extract_predicates:
            try:
                predicates = extract_predicates(next_obs)
            except Exception:
                pass

        embedding = None
        if extract_embedding:
            try:
                embedding = extract_embedding(next_obs)
            except Exception:
                pass

        step = RolloutStep(
            step=t,
            obs_id=f"{episode_id}_obs_{t}",
            action=action,
            action_type=action_type,
            ui_events=[],
            predicates=predicates,
            embedding=embedding,
            r_env=breakdown.r_env,
            r_follow=breakdown.r_follow,
            r_cost=breakdown.r_cost,
            r_total=breakdown.r_total,
            done=done,
            episode_id=episode_id,
            traj_id=episode_id,
            seed=seed,
            active_skill_id=step_result.get("active_skill_id"),
            query_key=step_result.get("query_key"),
            logprob=logprob,
            value=policy_out.value,
        )
        steps.append(step)
        obs = next_obs
        t += 1

    record = RolloutRecord(
        episode_id=episode_id,
        traj_id=episode_id,
        seed=seed,
        steps=steps,
    )
    record.finalize()
    record.won = done
    record.score = record.total_r_env
    return record


def collect_batch(
    env_factory: Callable[[int], Any],
    policy: Any,
    skill_bank: Any,
    memory: Any,
    reward_config: Any,
    seed_manager: SeedManager,
    batch_size: int = 32,
    max_steps: int = 500,
    **kwargs,
) -> List[RolloutRecord]:
    """Collect a batch of rollouts with different seeds.

    Args:
        env_factory: callable(seed) -> game env instance
        policy: PolicyInterface
        skill_bank: current SkillBankMVP
        memory: EpisodicMemoryStore
        reward_config: RewardConfig
        seed_manager: SeedManager for deterministic seeds
        batch_size: number of episodes
        max_steps: per-episode step limit

    Returns:
        list of RolloutRecords
    """
    records: List[RolloutRecord] = []

    for i in range(batch_size):
        seed = seed_manager.next_train_seed()
        env = env_factory(seed)
        wrapper = EnvWrapper(env=env, skill_bank=skill_bank, memory=memory)
        shaper = TrainRewardShaper(config=reward_config, skill_bank=skill_bank)

        try:
            record = collect_rollout(
                env_wrapper=wrapper,
                policy=policy,
                reward_shaper=shaper,
                seed=seed,
                max_steps=max_steps,
                **kwargs,
            )
            records.append(record)
        except Exception as exc:
            logger.warning("Rollout %d (seed=%d) failed: %s", i, seed, exc)

    return records
