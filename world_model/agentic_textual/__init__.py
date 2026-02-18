"""
Agentic textual world model for experience synthesis via LLM.

Uses ask_model (GPT/Claude/Gemini) to generate synthetic experience sequences
from state, historical summaries, and action/goal state/skills. Mirrors the
multi_modal API: synthesize_step and synthesize_sequence.
"""

from .config import TextWorldModelConfig, TextInferenceConfig
from .schemas import (
    TextSynthesisInput,
    TextSynthesisStepOutput,
    TextSyntheticExperienceSequence,
)
from .world_model import TextWorldModel, SYNTHESIS_PROMPT_TEMPLATE

__all__ = [
    "TextWorldModel",
    "TextWorldModelConfig",
    "TextInferenceConfig",
    "TextSynthesisInput",
    "TextSynthesisStepOutput",
    "TextSyntheticExperienceSequence",
    "SYNTHESIS_PROMPT_TEMPLATE",
]
