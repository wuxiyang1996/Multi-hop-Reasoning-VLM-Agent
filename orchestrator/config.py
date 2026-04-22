"""Orchestrator runtime configuration (PLAN-PIPELINE-ORCHESTRATOR §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common.models import BACKBONE_JUDGE_MODEL, BACKBONE_MODEL, BACKBONE_TEACHER_MODEL


@dataclass
class GateThresholds:
    replay_pass_rate: float = 0.8
    shadow_pass_rate: float = 0.7
    transfer_min_domains: int = 2
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
    "GateThresholds",
    "JudgeConfig",
    "OrchestratorConfig",
    "TeacherConfig",
]
