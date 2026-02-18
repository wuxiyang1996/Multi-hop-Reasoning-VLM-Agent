"""
Agentic textual world model for experience synthesis via LLM.

Takes the current state, historical summaries, and instructions (action/goal state/skills)
to produce synthetic experience sequences using GPT/Claude/Gemini (ask_model).

Usage:
    from world_model.agentic_textual import TextWorldModel, TextSynthesisInput

    config = TextWorldModelConfig(model="gpt-4o-mini")
    model = TextWorldModel(config)
    out = model.synthesize_step(TextSynthesisInput(
        state="Agent at (2,1). Pot has 2 onions.",
        historical_summary="Agent moved from dispenser to pot.",
        instructions="Agent picks up onion from dispenser.",
    ))
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Union

from .config import TextWorldModelConfig, TextInferenceConfig
from .schemas import (
    TextSynthesisInput,
    TextSynthesisStepOutput,
    TextSyntheticExperienceSequence,
)

# Prompt template for single-step synthesis
SYNTHESIS_PROMPT_TEMPLATE = """You are a world model that predicts the outcome of agent actions in a game or environment.

Given:
- **Current state**: {state}
- **Historical context**: {historical_summary}
- **Instructions** (action / goal state / skill): {instructions}

Predict the next state after the agent follows the instructions. Be concise but complete.

Format your response as:
Next state: <description of the resulting state>
Action: <the action that was taken, if clear>
Reward: <numeric reward, use 0 for intermediate steps, or estimate if task-related>
"""


def _get_ask_model(config: TextWorldModelConfig) -> Callable[..., str]:
    """Resolve the LLM call function."""
    if config.ask_model_fn is not None:
        return config.ask_model_fn
    from API_func import ask_model
    return ask_model


def _parse_synthesis_response(raw: str) -> tuple[str, Optional[str], Optional[float]]:
    """Parse next_state, action, reward from structured LLM response."""
    next_state = raw.strip()
    action = None
    reward = None

    # Try to extract labeled sections
    next_match = re.search(r"Next state:\s*(.+?)(?=\n\w+\s*:|\Z)", raw, re.DOTALL | re.I)
    if next_match:
        next_state = next_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+?)(?=\n\w+\s*:|\Z)", raw, re.DOTALL | re.I)
    if action_match:
        action = action_match.group(1).strip()

    reward_match = re.search(r"Reward:\s*(-?\d+\.?\d*)", raw, re.I)
    if reward_match:
        try:
            reward = float(reward_match.group(1))
        except ValueError:
            pass

    return next_state, action, reward


class TextWorldModel:
    """
    Agentic textual world model that uses an LLM (via ask_model) to synthesize
    experience sequences. Takes state, historical summaries, and instructions
    to predict next state, action, and reward.
    """

    def __init__(self, config: Optional[TextWorldModelConfig] = None):
        config = config or TextWorldModelConfig()
        self.config = config

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        ask = _get_ask_model(self.config)
        cfg = self.config.inference
        kwargs: dict = {
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if self.config.model is not None:
            kwargs["model"] = self.config.model
        return ask(prompt, **kwargs)

    def synthesize_step(
        self,
        inp: TextSynthesisInput,
        parse_response: bool = True,
    ) -> TextSynthesisStepOutput:
        """
        Produce a single synthetic next state from current state + context + instructions.

        Args:
            inp: TextSynthesisInput with state, historical_summary, instructions.
            parse_response: If True, parse Next state/Action/Reward from output.

        Returns:
            TextSynthesisStepOutput with next_state, optional action and reward.
        """
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            state=inp.state,
            historical_summary=inp.historical_summary or "None.",
            instructions=inp.instructions,
        )
        raw = self._call_llm(prompt)

        if parse_response:
            next_state, action, reward = _parse_synthesis_response(raw)
        else:
            next_state = raw.strip()
            action = None
            reward = None

        return TextSynthesisStepOutput(
            next_state=next_state,
            action=action,
            reward=reward,
            prompt=prompt,
            raw_response=raw,
        )

    def synthesize_sequence(
        self,
        initial_state: str,
        instructions_per_step: List[str],
        historical_summaries: Optional[List[str]] = None,
    ) -> TextSyntheticExperienceSequence:
        """
        Produce a multi-step synthetic experience sequence by chaining synthesis.

        Args:
            initial_state: Starting state (text).
            instructions_per_step: Instruction (action/goal/skill) for each step.
            historical_summaries: Optional historical summary per step.

        Returns:
            TextSyntheticExperienceSequence with steps.
        """
        steps_out: List[TextSynthesisStepOutput] = []
        state = initial_state
        summaries = historical_summaries or []

        for i, instr in enumerate(instructions_per_step):
            hist = summaries[i] if i < len(summaries) else ""
            inp = TextSynthesisInput(
                state=state,
                historical_summary=hist,
                instructions=instr,
            )
            out = self.synthesize_step(inp)
            steps_out.append(out)
            state = out.next_state

        return TextSyntheticExperienceSequence(
            steps=steps_out,
            sequence_instructions="; ".join(instructions_per_step),
        )

    def to_experiences(
        self,
        seq: TextSyntheticExperienceSequence,
        initial_state: str,
        done: bool = False,
    ) -> List[dict]:
        """
        Convert a TextSyntheticExperienceSequence to Experience-like dicts.

        Returns list of dicts with state, action, reward, next_state, done, is_synthetic.
        """
        experiences = []
        state = initial_state
        for i, step in enumerate(seq.steps):
            exp = {
                "state": state,
                "action": step.action or "",
                "reward": step.reward if step.reward is not None else 0.0,
                "next_state": step.next_state,
                "done": done and (i == len(seq.steps) - 1),
                "is_synthetic": True,
            }
            experiences.append(exp)
            state = step.next_state
        return experiences
