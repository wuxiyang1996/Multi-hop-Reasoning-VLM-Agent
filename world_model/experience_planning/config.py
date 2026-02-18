"""
Configuration for experience planning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .schemas import PlanFormat


def _resolve_model() -> Optional[str]:
    return os.environ.get("WORLD_MODEL_PLANNING_MODEL") or None


@dataclass
class PlanningConfig:
    """Configuration for the experience planner."""

    model: Optional[str] = field(default_factory=_resolve_model)
    ask_model_fn: Optional[object] = None
    temperature: float = 0.5
    max_tokens: int = 2000

    # Output format: natural_language or action_language
    plan_format: PlanFormat = PlanFormat.NATURAL_LANGUAGE

    # Action language format when using skill bank (pddl, strips, sas, compact)
    action_language_format: str = "compact"
