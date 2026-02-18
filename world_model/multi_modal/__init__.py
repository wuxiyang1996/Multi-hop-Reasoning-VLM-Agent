"""
Multi-modal world model for experience synthesis via image editing.

Supports LongCat, Qwen-Edit, and BAGEL (see Readme.md). Initialize with config or
pass model_name to use a specific model. The world model takes current frame,
historical state summary, and action/intended-state/skill instructions to produce
synthetic experience sequences.
"""

from .config import (
    WorldModelConfig,
    InferenceConfig,
    MODEL_LONGCAT,
    MODEL_QWEN_EDIT,
    MODEL_BAGEL,
    SUPPORTED_MODELS,
    MODEL_IDS,
)
from .schemas import (
    SynthesisInput,
    SynthesisStepOutput,
    SyntheticExperienceSequence,
    BAGEL_EDIT_PROMPT_TEMPLATE,
)
from .world_model import WorldModel

__all__ = [
    "WorldModel",
    "WorldModelConfig",
    "InferenceConfig",
    "SynthesisInput",
    "SynthesisStepOutput",
    "SyntheticExperienceSequence",
    "BAGEL_EDIT_PROMPT_TEMPLATE",
    "MODEL_LONGCAT",
    "MODEL_QWEN_EDIT",
    "MODEL_BAGEL",
    "SUPPORTED_MODELS",
    "MODEL_IDS",
]
