"""
Data schemas for multi-modal world model inputs and outputs.

The world model takes the current frame (image), historical state summary (text),
and instructions (action/intended state/skills), and produces synthetic experience
sequences as edited images representing intended next states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

# Type alias: image can be PIL.Image or path
ImageLike = Union[Any, str]


# Prompt template for experience synthesis (BAGEL, LongCat, Qwen-Edit)
# Combines historical context and edit instruction per BAGEL deployment guide.
BAGEL_EDIT_PROMPT_TEMPLATE = (
    "Context: {historical_summary}\nEdit instruction: {instructions}"
)


@dataclass
class SynthesisInput:
    """Input to the world model for a single edit step."""

    # Current frame from the experience (image)
    current_frame: ImageLike

    # Summary of historical state in text (context from prior steps)
    historical_summary: str

    # Instructions: action to take, intended state, or skill to apply
    instructions: str

    # Optional: extra context (e.g. sub_task, intentions)
    extra_context: Optional[dict] = None


@dataclass
class SynthesisStepOutput:
    """Output of one synthesis step: edited image as intended next state."""

    # Generated image (intended next state)
    next_frame: Any  # PIL.Image.Image

    # Prompt used for this edit
    edit_prompt: str

    # Optional metadata from the model
    metadata: Optional[dict] = None


@dataclass
class SyntheticExperienceSequence:
    """A sequence of synthetic experience steps (multi-step rollouts)."""

    # List of (current_frame, next_frame, edit_prompt) per step
    steps: List[SynthesisStepOutput]

    # Combined instructions/skills used for the sequence
    sequence_instructions: str
