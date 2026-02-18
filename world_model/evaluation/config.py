"""
Configuration for the Synthetic Experience Evaluation module.

All quality judgements are produced by LLM-as-a-judge calls.
This config controls LLM call parameters, which dimensions to run,
verdict thresholds, and output routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class LLMJudgeConfig:
    """Parameters for the LLM judge used across all evaluation dimensions."""

    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2048

    # Max characters per state string injected into the prompt
    max_state_chars: int = 400

    # Max steps from a trajectory to include in the prompt
    max_steps_in_prompt: int = 15

    # Whether to request chain-of-thought in the response
    chain_of_thought: bool = True

    # Custom ask_model callable; if None, imports from API_func
    ask_model_fn: Optional[Callable] = None


@dataclass
class ExperienceEvaluationConfig:
    """Top-level configuration for the experience evaluation pipeline."""

    llm: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)

    # Per-dimension weights for overall score aggregation
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "fidelity": 1.2,
        "consistency": 1.2,
        "instruction_adherence": 1.0,
        "diversity": 0.8,
        "informativeness": 0.8,
    })

    # Subset of dimensions to evaluate (all five by default)
    enabled_dimensions: List[str] = field(default_factory=lambda: [
        "fidelity",
        "consistency",
        "instruction_adherence",
        "diversity",
        "informativeness",
    ])

    # Run a holistic synthesis pass after per-dimension scoring
    run_holistic_pass: bool = True

    # Score thresholds for automatic verdict assignment
    # (holistic pass can override these)
    accept_threshold: float = 0.7
    refine_threshold: float = 0.4
    discard_threshold: float = 0.2

    # Path for saving the evaluation report JSON
    report_path: Optional[str] = None
