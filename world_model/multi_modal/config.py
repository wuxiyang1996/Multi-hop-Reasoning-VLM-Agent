"""
Configuration for the multi-modal world model (experience synthesis via image editing).

Supports model selection among LongCat, Qwen-Edit, and BAGEL as referenced in
world_model/Readme.md. Users can declare the model via model_name or MODEL_NAME env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


# Supported model identifiers (align with Readme.md)
MODEL_LONGCAT = "longcat"
MODEL_QWEN_EDIT = "qwen_edit"
MODEL_BAGEL = "bagel"

SUPPORTED_MODELS = (MODEL_LONGCAT, MODEL_QWEN_EDIT, MODEL_BAGEL)

# HuggingFace model IDs
MODEL_IDS = {
    MODEL_LONGCAT: "meituan-longcat/LongCat-Image-Edit",
    MODEL_QWEN_EDIT: "Qwen/Qwen-Image-Edit-2511",
    MODEL_BAGEL: "ByteDance-Seed/BAGEL-7B-MoT",
}


def _resolve_model_name() -> str:
    """Resolve model name from env MODEL_NAME or default to LongCat."""
    return os.environ.get("WORLD_MODEL_IMAGE_EDIT_MODEL", MODEL_LONGCAT).lower()


@dataclass
class InferenceConfig:
    """Inference parameters for image editing pipelines."""

    guidance_scale: float = 4.5
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    negative_prompt: str = ""
    seed: Optional[int] = None
    # For Qwen-Edit
    true_cfg_scale: float = 4.0


@dataclass
class WorldModelConfig:
    """Top-level configuration for the multi-modal world model."""

    # Model selection: "longcat" | "qwen_edit" | "bagel"
    model_name: str = field(default_factory=_resolve_model_name)

    # HuggingFace model ID override (if None, uses MODEL_IDS[model_name])
    model_id: Optional[str] = None

    # Device: "cuda", "cpu", or "auto" (cpu_offload when available)
    device: str = "auto"

    # dtype for model weights
    torch_dtype: str = "bfloat16"

    # Inference parameters
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def get_model_id(self) -> str:
        """Return the HuggingFace model ID to load."""
        if self.model_id is not None:
            return self.model_id
        name = self.model_name.lower()
        if name not in MODEL_IDS:
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. "
                f"Supported: {SUPPORTED_MODELS}. "
                f"Or set model_id to a custom HuggingFace repo."
            )
        return MODEL_IDS[name]
