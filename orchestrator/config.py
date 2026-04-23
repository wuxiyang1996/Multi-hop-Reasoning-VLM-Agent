"""Orchestrator runtime configuration (PLAN-PIPELINE-ORCHESTRATOR §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common.models import BACKBONE_JUDGE_MODEL, BACKBONE_MODEL, BACKBONE_TEACHER_MODEL


@dataclass
class FewShotConfig:
    """K-shot adaptation budget for the Stage 3a transfer gate.

    PLAN-UNIFIED-SKILL-GATE §7 Stage 3a + §9. The few-shot adapter is
    given a *small* budget of target-domain demonstrations; if the skill
    cannot reach `target_domain_pass_rate_min` within `k_shot_max`
    shots, the target binding fails and the skill cannot earn a
    `verified_domains` entry for that domain.
    """

    k_shot_default: int = 5
    k_shot_max: int = 16
    target_domain_pass_rate_min: float = 0.5
    adaptation_cost_max_tokens: int = 8_000


@dataclass
class GateThresholds:
    replay_pass_rate: float = 0.8
    shadow_pass_rate: float = 0.7
    # Legacy (PLAN-UNIFIED-SKILL-GATE pre-asymmetry). Kept so callers
    # that still inspect it keep working; the real gate now consults
    # `transfer_min_target_domains_verified` + `few_shot` below.
    transfer_min_domains: int = 2
    # New: a skill must clear the few-shot adapter on at least this
    # many target domains (one of TRANSFER_TARGET_DOMAINS) to be ACTIVE.
    transfer_min_target_domains_verified: int = 1
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    non_regression_max_delta: float = 0.02


@dataclass
class BudgetLimits:
    max_outer_steps: int = 32
    max_inner_steps: int = 64
    max_skill_invocations: int = 32
    max_tokens: int = 32_000
    max_ms: float = 5 * 60_000.0
    max_grounding_escalations: int = 4
    max_teacher_calls: int = 1


@dataclass
class TeacherConfig:
    """All teacher models must be frozen — no fine-tuning is allowed
    against them inside the loop (PLAN-SKILL-CRAFTER §3).

    The current phase pins the teacher to GPT-4o (see `common.models`).
    The 32B / 72B Qwen tracks remain reachable by setting
    `model_name="Qwen/Qwen2.5-72B"` (or the `VLM_AGENT_BACKBONE_TEACHER_MODEL`
    env var) once that track is enabled.
    """

    model_name: str = BACKBONE_TEACHER_MODEL
    frozen: bool = True
    max_calls_per_episode: int = 1


@dataclass
class JudgeConfig:
    """Default LLM judge for the eval driver (E0 / E1 / E2)."""

    model_name: str = BACKBONE_JUDGE_MODEL
    frozen: bool = True


@dataclass
class OrchestratorConfig:
    artifact_root: str = "./_artifacts"
    bank_root: str = "./_bank"
    gate_thresholds: GateThresholds = field(default_factory=GateThresholds)
    budget: BudgetLimits = field(default_factory=BudgetLimits)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    backbone_model: str = BACKBONE_MODEL  # actor / policy default
    enabled_domains: List[str] = field(default_factory=lambda: ["gymv", "browser"])
    seed: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)


__all__ = [
    "BudgetLimits",
    "FewShotConfig",
    "GateThresholds",
    "JudgeConfig",
    "OrchestratorConfig",
    "TeacherConfig",
]
