"""Unified episode runner for non-game domains (QA + Web).

Uses the SAME ``SkillDecisionCore`` pipeline as the game episode
runner in ``episode_runner.py``, but operates on any gym-like env
that provides ``reset()`` / ``step()`` — e.g.:

    * ``VRReasoningEnv``   (visual/video QA multi-hop)
    * ``BrowserGym``       (MiniWoB / WebShop / WebArena)

The pipeline is IDENTICAL to games:
    1. summary_state (deterministic)
    2. skill_selection (LoRA) — same prompt format, same DECISION output
    3. intention (base model)
    4. action_taking (LoRA)
    5. env.step()
    6. GRPO record emission

The ONLY difference: reward handling.
    - Games:     per-step env reward is used inline AND relabeled offline
    - QA / Web:  per-step reward = 0.0; offline relabeling fills in
                 episode-level reward post-hoc
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from decision_agents.skill_decision_core import (
    DOMAIN_QA,
    DOMAIN_WEB,
    SkillSelectionRecord,
    StepTracker,
    build_skill_selection_prompt,
    parse_skill_selection,
)

logger = logging.getLogger(__name__)


@dataclass
class UnifiedEpisodeResult:
    """Result of one non-game episode."""

    domain: str
    task: str
    episode_id: str
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    skill_switches: int
    skill_records: List[SkillSelectionRecord] = field(default_factory=list)
    action_records: List[Dict[str, Any]] = field(default_factory=list)
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    final_answer: Optional[str] = None
    hop_history: List[str] = field(default_factory=list)


async def run_unified_episode(
    env: Any,
    task_name: str,
    max_steps: int,
    vllm_client: Any,
    *,
    domain: str = DOMAIN_QA,
    skill_bank: Any = None,
    temperature: float = 0.3,
    strip_think_tags: Optional[Callable] = None,
) -> UnifiedEpisodeResult:
    """Run one episode using the unified decision-agent pipeline.

    This function mirrors ``run_episode_async`` in ``episode_runner.py``
    but is domain-agnostic.  The env must expose:
        - ``reset() -> (obs_nl: str, info: dict)``
        - ``step(action) -> (obs_nl, reward, terminated, truncated, info)``
    """
    episode_id = f"{domain}_{uuid.uuid4().hex[:8]}"
    t0 = time.monotonic()

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])

    tracker = StepTracker(domain=domain, game_name=task_name)
    skill_records: List[SkillSelectionRecord] = []
    action_records: List[Dict[str, Any]] = []
    experiences: List[Dict[str, Any]] = []

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    current_intention = f"Complete the {task_name} task"
    last_guidance: Optional[Dict[str, Any]] = None

    bank_available = skill_bank is not None and (
        hasattr(skill_bank, "__len__") and len(skill_bank) > 0
        or hasattr(skill_bank, "skill_ids")
        and len(list(skill_bank.skill_ids)) > 0
    )

    while step_count < max_steps:
        state_text = obs_nl[:3500] if isinstance(obs_nl, str) else str(obs_nl)[:3500]

        # ── Skill selection (same pipeline as games) ──────────────
        need_reselect = tracker.should_reselect(
            last_guidance, state_text=state_text,
        )
        skill_prompt: Optional[str] = None
        sk_result_text: Optional[str] = None

        if bank_available and (need_reselect or last_guidance is None):
            from scripts.qwen3_decision_agent import get_top_k_skill_candidates

            candidates = get_top_k_skill_candidates(
                skill_bank,
                state_text,
                game_name=task_name,
                intention=current_intention,
                top_k=3,
            )

            if candidates and len(candidates) >= 2:
                skill_prompt = build_skill_selection_prompt(
                    state_text=state_text,
                    intention=current_intention,
                    candidates=candidates,
                    tracker=tracker,
                )

                sk_result = await vllm_client.generate_chat(
                    [{"role": "user", "content": skill_prompt}],
                    adapter="skill_selection",
                    temperature=temperature,
                    max_tokens=128,
                    stop=["\n\nAvailable", "\n\nGame state", "\n\n---"],
                )
                sk_result_text = sk_result.text

                chosen_idx, lora_effects, decision = parse_skill_selection(
                    sk_result_text,
                    len(candidates),
                    candidates,
                    strip_think_tags=strip_think_tags,
                )

                if lora_effects:
                    tracker.receive_lora_effects(lora_effects)

                if decision == "CONTINUE" and last_guidance is not None:
                    guidance = last_guidance
                else:
                    guidance = candidates[chosen_idx]
                    tracker.set_protocol(guidance.get("protocol"))

                last_guidance = guidance

                skill_records.append(SkillSelectionRecord(
                    domain=domain,
                    task=task_name,
                    episode_id=episode_id,
                    step=step_count,
                    prompt=skill_prompt,
                    completion=sk_result_text or "",
                    reward=0.0,
                    candidates=[c.get("skill_id", "") for c in candidates],
                    chosen_skill_id=guidance.get("skill_id"),
                    chosen_idx=chosen_idx,
                    decision=decision,
                    effects=lora_effects,
                    reselect_reason=tracker._reselect_reason,
                    hop_history=list(tracker.hop_history),
                ))
            elif candidates:
                last_guidance = candidates[0]
                tracker.set_protocol(last_guidance.get("protocol"))

        guidance = last_guidance

        # ── Action selection (same pipeline as games) ─────────────
        skill_context = ""
        if guidance and guidance.get("skill_id"):
            sk_name = guidance.get("skill_name", guidance["skill_id"])
            sk_hint = guidance.get("execution_hint", "")
            skill_context = f"Active skill: {sk_name}"
            if sk_hint:
                skill_context += f" — {sk_hint[:100]}"
            skill_context += "\n"

        progress = tracker.get_progress_summary(state_text)
        progress_line = f"Progress: {progress}\n" if progress else ""

        action_prompt = (
            f"Current state:\n{state_text}\n\n"
            f"Task: {current_intention}\n"
            f"{skill_context}{progress_line}"
            f"Available actions: {', '.join(action_names)}\n\n"
            f"Choose the best action.\n"
            f"REASONING: <why this action>\n"
            f"ACTION: <action>"
        )

        action_result = await vllm_client.generate_chat(
            [{"role": "user", "content": action_prompt}],
            adapter="action_taking",
            temperature=temperature,
            max_tokens=128,
        )

        action = _parse_action(action_result.text, action_names)

        # ── env.step() ────────────────────────────────────────────
        try:
            next_obs, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            logger.warning("env.step failed at step %d: %s", step_count, e)
            break

        total_reward += reward
        hop_type = next_info.get("hop_type", action if domain == DOMAIN_QA else "")

        skill_id = guidance.get("skill_id") if guidance else None
        skill_name = guidance.get("skill_name", "") if guidance else ""
        tracker.update(skill_id, skill_name, float(reward),
                       state_text=state_text, hop_type=hop_type or None)

        action_records.append({
            "step": step_count,
            "prompt": action_prompt,
            "completion": action_result.text,
            "action": action,
            "reward": float(reward),
            "skill_id": skill_id,
            "hop_type": hop_type,
        })

        experiences.append({
            "step": step_count,
            "state": obs_nl,
            "action": action,
            "reward": float(reward),
            "next_state": next_obs if isinstance(next_obs, str) else str(next_obs),
            "done": terminated or truncated,
            "intention": current_intention,
            "skill_id": skill_id,
            "hop_type": hop_type,
        })

        obs_nl = next_obs
        action_names = next_info.get("action_names", action_names)
        step_count += 1

        if terminated or truncated:
            break

    try:
        env.close()
    except Exception:
        pass

    wall_time = time.monotonic() - t0

    return UnifiedEpisodeResult(
        domain=domain,
        task=task_name,
        episode_id=episode_id,
        steps=step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        skill_switches=tracker.skill_switches,
        skill_records=skill_records,
        action_records=action_records,
        experiences=experiences,
        wall_time_s=wall_time,
        final_answer=info.get("final_answer") if info else None,
        hop_history=list(tracker.hop_history),
    )


def unified_result_to_grpo_records(
    result: UnifiedEpisodeResult,
    episode_reward: Optional[float] = None,
) -> List["GRPORecord"]:
    """Convert a UnifiedEpisodeResult into a list of GRPORecord objects.

    The unified runner stores action_records as plain dicts and
    skill_records as SkillSelectionRecord.  The GRPO trainer needs
    GRPORecord dataclass instances.  This function bridges the gap.

    If *episode_reward* is given, it overrides per-step rewards with a
    uniform redistribution of the episode-level reward across all
    action records — appropriate for sparse-reward tasks (QA, web).
    """
    from trainer.coevolution.episode_runner import GRPORecord

    records: List[GRPORecord] = []
    ep_len = max(result.steps, 1)
    final_reward = episode_reward if episode_reward is not None else result.total_reward

    for ar in result.action_records:
        step_reward = ar.get("reward", 0.0)
        if episode_reward is not None:
            step_reward = final_reward / max(len(result.action_records), 1)
        records.append(GRPORecord(
            adapter="action_taking",
            game=result.task,
            episode_id=result.episode_id,
            step=ar.get("step", 0),
            prompt=ar.get("prompt", ""),
            completion=ar.get("completion", ""),
            reward=step_reward,
            episode_length=ep_len,
            metadata={
                "action": ar.get("action", ""),
                "skill_id": ar.get("skill_id"),
                "hop_type": ar.get("hop_type", ""),
                "domain": result.domain,
            },
        ))

    for sr in result.skill_records:
        step_reward = sr.reward
        if episode_reward is not None:
            step_reward = final_reward / max(len(result.skill_records), 1)
        records.append(GRPORecord(
            adapter="skill_selection",
            game=result.task,
            episode_id=result.episode_id,
            step=sr.step,
            prompt=sr.prompt,
            completion=sr.completion,
            reward=step_reward,
            episode_length=ep_len,
            metadata={
                "candidates": sr.candidates,
                "chosen_skill_id": sr.chosen_skill_id,
                "decision": sr.decision,
                "domain": result.domain,
            },
        ))

    return records


def _parse_action(reply: str, valid_actions: List[str]) -> str:
    """Parse action from LLM response."""
    if not reply:
        return valid_actions[0] if valid_actions else ""

    import re
    m = re.search(r"ACTION\s*:\s*(.+)", reply, re.IGNORECASE)
    if m:
        action_text = m.group(1).strip()
        for va in valid_actions:
            if va.lower() == action_text.lower():
                return va
        for va in valid_actions:
            if va.lower() in action_text.lower():
                return va

    reply_lower = reply.lower()
    for va in valid_actions:
        if va.lower() in reply_lower:
            return va

    return valid_actions[0] if valid_actions else reply.strip()[:50]
