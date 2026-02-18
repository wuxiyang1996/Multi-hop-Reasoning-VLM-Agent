"""
Configuration for the agentic textual world model (experience synthesis via LLM).

Uses ask_model (GPT/Claude/Gemini) to generate synthetic experience sequences
from state, historical summaries, and action/goal state/skills.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _resolve_model() -> Optional[str]:
    """Resolve model from env or None (ask_model default)."""
    return os.environ.get("WORLD_MODEL_TEXTUAL_MODEL") or None


@dataclass
class TextInferenceConfig:
    """Inference parameters for LLM calls."""

    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class TextWorldModelConfig:
    """Top-level configuration for the agentic textual world model."""

    # Model name for ask_model (gpt-4o, gpt-4o-mini, claude-*, gemini-*, etc.)
    # None = use ask_model default
    model: Optional[str] = field(default_factory=_resolve_model)

    # ask_model_fn override (callable(prompt, model=..., temperature=..., max_tokens=...) -> str)
    ask_model_fn: Optional[object] = None

    # Inference parameters
    inference: TextInferenceConfig = field(default_factory=TextInferenceConfig)
