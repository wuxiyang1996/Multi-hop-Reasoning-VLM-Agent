"""
Multi-modal world model for experience synthesis via image editing.

Takes the image frame from current experience, historical state summary in text,
and instructions (action/intended state/skills) to produce synthetic experience
sequences. Supports LongCat, Qwen-Edit, and BAGEL as referenced in Readme.md.

Usage:
    from world_model.multi_modal import WorldModel, SynthesisInput

    config = WorldModelConfig(model_name="longcat")
    model = WorldModel(config)
    out = model.synthesize_step(SynthesisInput(
        current_frame=frame,
        historical_summary="Agent at (2,1), pot has 2 onions.",
        instructions="Agent picks up onion from dispenser."
    ))
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any, List, Optional, Union

from .config import (
    WorldModelConfig,
    MODEL_LONGCAT,
    MODEL_QWEN_EDIT,
    MODEL_BAGEL,
    SUPPORTED_MODELS,
)
from .schemas import (
    SynthesisInput,
    SynthesisStepOutput,
    SyntheticExperienceSequence,
    BAGEL_EDIT_PROMPT_TEMPLATE,
)


def _load_image(img: Union["PIL.Image.Image", str]) -> "PIL.Image.Image":
    """Load image from path or return as-is if already PIL Image."""
    from PIL import Image

    if isinstance(img, str):
        path = Path(img)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        return Image.open(path).convert("RGB")
    if hasattr(img, "convert"):
        return img.convert("RGB") if img.mode != "RGB" else img
    raise TypeError(f"current_frame must be PIL.Image or path str, got {type(img)}")


def _build_edit_prompt(historical_summary: str, instructions: str) -> str:
    """Build the edit prompt from historical context and instructions.

    Uses BAGEL deployment format (see BAGEL_DEPLOYMENT.md):
    Context: {historical_summary}
    Edit instruction: {instructions}
    """
    hist = (historical_summary or "").strip()
    instr = (instructions or "Apply the intended state change.").strip()
    return BAGEL_EDIT_PROMPT_TEMPLATE.format(
        historical_summary=hist or "No prior context.",
        instructions=instr,
    )


class WorldModel:
    """
    Multi-modal world model that uses image editing models to synthesize
    experience sequences. Initializes the chosen generative model and exposes
    synthesize_step / synthesize_sequence.
    """

    def __init__(self, config: Optional[WorldModelConfig] = None):
        config = config or WorldModelConfig()
        self.config = config
        self._pipeline: Any = None
        self._backend: str = config.model_name.lower()

    def _get_pipeline(self):
        """Lazy-load the pipeline for the selected model."""
        if self._pipeline is not None:
            return self._pipeline

        model_id = self.config.get_model_id()
        dtype_name = self.config.torch_dtype.lower()
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16

        if self._backend == MODEL_LONGCAT:
            self._pipeline = self._load_longcat(model_id, dtype)
        elif self._backend == MODEL_QWEN_EDIT:
            self._pipeline = self._load_qwen_edit(model_id, dtype)
        elif self._backend == MODEL_BAGEL:
            self._pipeline = self._load_bagel(model_id, dtype)
        else:
            raise ValueError(
                f"Unknown backend '{self._backend}'. Supported: {SUPPORTED_MODELS}"
            )
        return self._pipeline

    def _load_longcat(self, model_id: str, dtype: torch.dtype):
        from diffusers import LongCatImageEditPipeline

        pipe = LongCatImageEditPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if self.config.device == "auto":
            pipe.enable_model_cpu_offload()
        elif self.config.device == "cuda":
            pipe = pipe.to("cuda", dtype)
        return pipe

    def _load_qwen_edit(self, model_id: str, dtype: torch.dtype):
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if self.config.device == "auto":
            pipe.enable_model_cpu_offload()
        elif self.config.device == "cuda":
            pipe = pipe.to("cuda", dtype)
        return pipe

    def _load_bagel(self, model_id: str, dtype: torch.dtype):
        # BAGEL uses bagel-mot library with a different API. Provide a clear
        # NotImplementedError and suggest using LongCat or Qwen-Edit, or
        # integrating BAGEL via https://github.com/ByteDance-Seed/BAGEL
        raise NotImplementedError(
            "BAGEL backend requires the bagel-mot library and a different API. "
            "See https://github.com/ByteDance-Seed/BAGEL for integration. "
            "Use model_name='longcat' or 'qwen_edit' for now."
        )

    def synthesize_step(
        self,
        inp: SynthesisInput,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SynthesisStepOutput:
        """
        Produce a single synthetic next frame from current frame + context + instructions.

        Args:
            inp: SynthesisInput with current_frame, historical_summary, instructions.
            negative_prompt: Override default negative prompt.
            guidance_scale: Override config.
            num_inference_steps: Override config.
            seed: Random seed for reproducibility.

        Returns:
            SynthesisStepOutput with next_frame (PIL.Image) and edit_prompt.
        """
        pipe = self._get_pipeline()
        img = _load_image(inp.current_frame)
        edit_prompt = _build_edit_prompt(inp.historical_summary, inp.instructions)

        cfg = self.config.inference
        neg = negative_prompt if negative_prompt is not None else cfg.negative_prompt
        gs = guidance_scale if guidance_scale is not None else cfg.guidance_scale
        steps = num_inference_steps if num_inference_steps is not None else cfg.num_inference_steps
        s = seed if seed is not None else cfg.seed

        generator = None
        if s is not None:
            generator = torch.Generator("cpu").manual_seed(s)

        if self._backend == MODEL_LONGCAT:
            out = pipe(
                img,
                edit_prompt,
                negative_prompt=neg,
                guidance_scale=gs,
                num_inference_steps=steps,
                num_images_per_prompt=cfg.num_images_per_prompt,
                generator=generator,
            )
            next_frame = out.images[0]
        elif self._backend == MODEL_QWEN_EDIT:
            out = pipe(
                image=img,
                prompt=edit_prompt,
                generator=generator,
                true_cfg_scale=cfg.true_cfg_scale,
                negative_prompt=neg or " ",
                num_inference_steps=steps,
                guidance_scale=1.0,
                num_images_per_prompt=cfg.num_images_per_prompt,
            )
            next_frame = out.images[0]
        else:
            raise ValueError(f"Backend {self._backend} not implemented in synthesize_step")

        return SynthesisStepOutput(next_frame=next_frame, edit_prompt=edit_prompt)

    def synthesize_sequence(
        self,
        initial_frame: Union["PIL.Image.Image", str],
        instructions_per_step: List[str],
        historical_summaries: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> SyntheticExperienceSequence:
        """
        Produce a multi-step synthetic experience sequence by chaining edits.

        Args:
            initial_frame: Starting image.
            instructions_per_step: Edit instruction for each step.
            historical_summaries: Optional historical summary per step. If shorter
                than instructions_per_step, later steps use empty string.
            seed: Base seed; each step uses seed + step_idx.

        Returns:
            SyntheticExperienceSequence with steps (next_frame, edit_prompt).
        """
        steps_out: List[SynthesisStepOutput] = []
        frame = _load_image(initial_frame)
        summaries = historical_summaries or []
        base_seed = seed if seed is not None else self.config.inference.seed

        for i, instr in enumerate(instructions_per_step):
            hist = summaries[i] if i < len(summaries) else ""
            inp = SynthesisInput(
                current_frame=frame,
                historical_summary=hist,
                instructions=instr,
            )
            step_seed = (base_seed + i) if base_seed is not None else None
            out = self.synthesize_step(inp, seed=step_seed)
            steps_out.append(out)
            frame = out.next_frame

        return SyntheticExperienceSequence(
            steps=steps_out,
            sequence_instructions="; ".join(instructions_per_step),
        )
