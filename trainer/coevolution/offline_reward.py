"""Offline reward relabeling for skill_selection GRPO records.

This module reconciles the fundamental reward-density difference
between domains:

    * **Game**: per-step env reward is available during rollout.
    * **QA / Web**: only episode-end reward (binary correct/incorrect
      or task-completion score).

The SkillDecisionCore pipeline emits identical ``SkillSelectionRecord``
objects for all domains.  This module then relabels their rewards
post-hoc from full trajectory information, providing a unified GRPO
training signal that does not depend on per-step reward availability.

Why offline?
    Online skill_selection rewards for non-game domains would be either
    zero (no per-step signal) or heuristic (noisy intrinsic bonuses).
    Offline relabeling gives us access to the full trajectory outcome
    and can assign credit to each skill decision based on what actually
    happened afterward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySegment:
    """One skill-selection decision and its downstream trajectory."""

    skill_id: Optional[str]
    skill_name: str
    chosen_idx: int
    decision: str  # CONTINUE | SWITCH
    step_start: int
    step_end: int
    n_candidates: int
    hop_history: List[str]
    env_rewards: List[float]
    step_progress_ratio: float = 0.0
    reselect_reason: str = ""


def segment_trajectory(
    records: List[Dict[str, Any]],
) -> List[TrajectorySegment]:
    """Split a trajectory into segments, one per skill-selection decision.

    Each segment covers the steps from one skill selection to the next.
    ``records`` is the list of per-step experience dicts from the
    episode runner (same format for all domains).
    """
    segments: List[TrajectorySegment] = []
    current_skill_id: Optional[str] = None
    current_start = 0
    current_rewards: List[float] = []
    current_hops: List[str] = []

    for i, rec in enumerate(records):
        skill_id = rec.get("skill_id")
        hop_type = rec.get("hop_type", "")

        if skill_id != current_skill_id and i > 0:
            segments.append(TrajectorySegment(
                skill_id=current_skill_id,
                skill_name=rec.get("skill_name", ""),
                chosen_idx=rec.get("chosen_idx", 0),
                decision="SWITCH" if len(segments) > 0 else "SWITCH",
                step_start=current_start,
                step_end=i - 1,
                n_candidates=rec.get("n_candidates", 0),
                hop_history=list(current_hops),
                env_rewards=list(current_rewards),
                step_progress_ratio=rec.get("step_progress_ratio", 0.0),
                reselect_reason=rec.get("reselect_reason", ""),
            ))
            current_start = i
            current_rewards = []
            current_hops = []

        current_skill_id = skill_id
        current_rewards.append(float(rec.get("reward", 0.0)))
        if hop_type:
            current_hops.append(hop_type)

    if current_rewards:
        last = records[-1] if records else {}
        segments.append(TrajectorySegment(
            skill_id=current_skill_id,
            skill_name=last.get("skill_name", ""),
            chosen_idx=last.get("chosen_idx", 0),
            decision="CONTINUE",
            step_start=current_start,
            step_end=len(records) - 1,
            n_candidates=last.get("n_candidates", 0),
            hop_history=list(current_hops),
            env_rewards=list(current_rewards),
            step_progress_ratio=last.get("step_progress_ratio", 0.0),
            reselect_reason=last.get("reselect_reason", ""),
        ))

    return segments


def relabel_game_rewards(
    segments: List[TrajectorySegment],
    total_reward: float,
    gamma: float = 0.95,
) -> List[float]:
    """Relabel skill_selection rewards for game trajectories.

    Games already have per-step env reward, so we use discounted
    return from each decision point.
    """
    n = len(segments)
    rewards = [0.0] * n

    cumulative_from = [0.0] * n
    running = 0.0
    for i in range(n - 1, -1, -1):
        seg_reward = sum(segments[i].env_rewards)
        running = seg_reward + gamma * running
        cumulative_from[i] = running

    max_ret = max(abs(cumulative_from[0]), 1.0)
    for i in range(n):
        base = cumulative_from[i] / max_ret

        progress_bonus = segments[i].step_progress_ratio * 0.15

        if segments[i].reselect_reason.startswith("success:"):
            base += 0.2
        elif segments[i].reselect_reason.startswith("abort:"):
            base -= 0.1

        rewards[i] = max(0.0, min(1.0, base + progress_bonus))

    return rewards


def relabel_sparse_rewards(
    segments: List[TrajectorySegment],
    episode_reward: float,
    episode_success: bool = False,
    gamma: float = 0.95,
) -> List[float]:
    """Relabel skill_selection rewards for sparse-reward domains (QA/web).

    Since there is no per-step env reward, all skill decisions in the
    episode share the episode outcome, weighted by recency (later
    decisions get more credit — they are closer to the final answer).
    """
    n = len(segments)
    if n == 0:
        return []

    rewards = [0.0] * n
    for i in range(n):
        recency = gamma ** (n - i - 1)
        base = episode_reward * recency

        if episode_success:
            base += 0.2 * recency

        progress_bonus = segments[i].step_progress_ratio * 0.15

        if segments[i].reselect_reason.startswith("success:"):
            base += 0.15
        elif segments[i].reselect_reason.startswith("abort:"):
            base -= 0.1

        duration = segments[i].step_end - segments[i].step_start + 1
        if duration >= 1:
            efficiency = min(1.0, len(segments[i].hop_history) / max(duration, 1))
            base += efficiency * 0.05

        rewards[i] = max(0.0, min(1.0, base + progress_bonus))

    return rewards


def relabel_episode(
    segments: List[TrajectorySegment],
    total_reward: float,
    episode_success: bool = False,
    domain: str = "game",
    gamma: float = 0.95,
) -> List[float]:
    """Unified entry point: relabel rewards based on domain.

    All domains produce the same ``TrajectorySegment`` list (from
    identical ``SkillDecisionCore`` pipeline), but use domain-appropriate
    reward assignment.
    """
    if domain == "game":
        return relabel_game_rewards(segments, total_reward, gamma=gamma)
    else:
        return relabel_sparse_rewards(
            segments, total_reward,
            episode_success=episode_success, gamma=gamma,
        )
