#!/usr/bin/env python
"""
Shared ALFWorld-7B policy wrapper (HF checkpoints).

This module exposes a simple text-in / action-list-out interface so we can
reuse the same ALFWorld-7B checkpoint as a baseline across different cold_start
scripts (LMGame-Bench, AgentEvolver, Orak, Pokemon Red).

It does NOT depend on any specific environment; callers are responsible for:
  - Constructing a textual observation string.
  - Providing a list of admissible action strings for the current step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass
class Alfworld7BConfig:
    model_path: str
    checkpoint_type: str = "sft"  # "sft" or "rl"
    temperature: float = 0.8
    max_new_tokens: int = 64


class Alfworld7BPolicy:
    """
    Lightweight HF policy that selects exactly one action from a discrete set.
    """

    def __init__(self, cfg: Alfworld7BConfig, device: Optional[str] = None):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is not installed. Install transformers to use "
                "ALFWORLD-7B checkpoints as a baseline."
            )
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto" if device is None else device,
            torch_dtype="auto",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, obs: str, action_names: List[str]) -> str:
        actions_str = ", ".join(action_names)
        return (
            "You are a game-playing agent powered by an ALFWorld-7B checkpoint. "
            "You must choose EXACTLY one next action from the allowed action list.\n\n"
            f"Observation:\n{obs}\n\n"
            f"Allowed actions: {actions_str}\n\n"
            "Respond with ONLY the chosen action string, nothing else.\n"
        )

    def choose_action(self, obs: str, action_names: List[str]) -> str:
        if not action_names:
            # Fallback to a generic 'noop' if nothing is provided.
            return "noop"

        prompt = self._build_prompt(obs, action_names)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]

        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        generated = full_text[len(prompt) :].strip()

        # First, try to match any allowed action that appears in the generated text.
        for a in action_names:
            if a.lower() in generated.lower():
                return a

        # Second, try to interpret the first line as an exact action.
        first_line = generated.splitlines()[0].strip()
        for a in action_names:
            if first_line.lower() == a.lower():
                return a

        # Fallback: default to the first action.
        return action_names[0]

