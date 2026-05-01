"""Skill Harness — per-invocation runtime for skill execution and verification.

Spec: PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION §4 (Phase A).

The harness's role:
  1.  Filter the Bank's candidate set to a domain- and contract-eligible
      sub-set (`select_eligible_skills`).
  2.  Execute one chosen skill via the right adapter (`run_skill`).
  3.  Record everything as a SkillEpisode + reward log entry.
  4.  Provide replay validation for the gate (Stage 1, PLAN-UNIFIED-SKILL-GATE §7).

It does NOT:
  - Choose which skill to commit to (that is the Actor's job).
  - Mutate `SkillRecord.status` (only `SkillLifecycleManager` may).
  - Read/write a "memory" buffer.
  - Train policies / call the teacher.

Public surface (what `decision_agents/` and `orchestrator/` import):

    from harness import (
        SkillHarness,
        AdapterRegistry,
        SkillAdapter,
        EligibleSkill,
        ReplayValidator,
        RewardLogger,
    )
"""

from harness.adapter_registry import AdapterRegistry
from harness.eligibility import EligibleSkill, EligibilityFilter, task_id_from_state
from harness.few_shot_adapter import (
    AdaptResult,
    FewShotAdapter,
    FewShotAdapterError,
    FewShotDemo,
    default_success_fn,
)
from harness.gymv_executor import (
    ACTION_ALIAS_MAP,
    GymvExecutorState,
    initial_state_from_env,
    make_gymv_executor,
)
from harness.gymv_success import (
    EFFECT_PREDICATE_TYPES,
    HopEffectResult,
    PredicateResult,
    evaluate_episode_effects,
    evaluate_hop_effects,
    evaluate_predicate,
    make_per_step_success_fn,
)
from harness.replay_validator import ReplayValidator, ReplayResult
from harness.reward_logger import RewardLogger
from harness.skill_adapter import SkillAdapter, AdapterRunContext, AdapterRunResult
from harness.skill_harness import HarnessConfig, SkillHarness

__all__ = [
    "ACTION_ALIAS_MAP",
    "AdaptResult",
    "AdapterRegistry",
    "AdapterRunContext",
    "AdapterRunResult",
    "EFFECT_PREDICATE_TYPES",
    "EligibilityFilter",
    "EligibleSkill",
    "FewShotAdapter",
    "FewShotAdapterError",
    "FewShotDemo",
    "GymvExecutorState",
    "HarnessConfig",
    "HopEffectResult",
    "PredicateResult",
    "ReplayResult",
    "ReplayValidator",
    "RewardLogger",
    "SkillAdapter",
    "SkillHarness",
    "default_success_fn",
    "evaluate_episode_effects",
    "evaluate_hop_effects",
    "evaluate_predicate",
    "initial_state_from_env",
    "make_gymv_executor",
    "make_per_step_success_fn",
    "task_id_from_state",
]
