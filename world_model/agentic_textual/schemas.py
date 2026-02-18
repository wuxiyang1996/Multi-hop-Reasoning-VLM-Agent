"""
Data schemas for agentic textual world model inputs and outputs.

The world model takes the current state (text), historical summaries (text),
and instructions (action/goal state/skills), and produces synthetic experience
sequences as text (next_state, action, reward, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TextSynthesisInput:
    """Input to the textual world model for a single synthesis step."""

    # Current state (text description)
    state: str

    # Summary of historical states (context from prior steps)
    historical_summary: str

    # Instructions: action to take, goal state, or skill to apply
    instructions: str

    # Optional: sub_task, intentions, task
    extra_context: Optional[dict] = None


@dataclass
class TextSynthesisStepOutput:
    """Output of one textual synthesis step."""

    # Predicted next state (text)
    next_state: str

    # Action taken (if inferred)
    action: Optional[str] = None

    # Reward (if inferred, typically 0 for intermediate)
    reward: Optional[float] = None

    # Raw prompt used
    prompt: Optional[str] = None

    # Raw model response (before parsing)
    raw_response: Optional[str] = None

    metadata: Optional[dict] = None


@dataclass
class TextSyntheticExperienceSequence:
    """A sequence of synthetic experience steps (multi-step rollouts)."""

    steps: List[TextSynthesisStepOutput]
    sequence_instructions: str
