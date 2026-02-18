"""
Experience planner: convert NL descriptions of episodes/skills into plans
(natural language or action language) for world model synthesis.

Uses skill_agents (skill bank, action language) when available. Output plans
drive the multi_modal or agentic_textual world models.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from .config import PlanningConfig
from .schemas import (
    EpisodeDescription,
    SkillDescription,
    SynthesisPlan,
    PlanStep,
    PlanFormat,
)

if TYPE_CHECKING:
    from skill_agents.stage3_mvp.schemas import SkillEffectsContract
    from skill_agents.skill_bank.bank import SkillBankMVP


EPISODE_PLAN_PROMPT = """You are a planner for generating synthetic game experiences.

Given a natural language description of a desired episode, produce a step-by-step plan.
Each step should be a single, concrete instruction that a world model can use to generate the next state.

Input description:
{description}

Constraints:
- Maximum {max_steps} steps.
- Each step: one clear action or sub-goal.
- Be concrete (e.g. "Agent picks up onion from dispenser", not "do something").

Output format (one instruction per line, numbered):
1. <instruction>
2. <instruction>
...
"""

SKILL_PLAN_PROMPT = """You are a planner for generating synthetic skill executions.

Given a natural language description of a desired skill or sub-task, produce a step-by-step plan.
Each step should be a single, concrete instruction for the world model.

Skill description: {description}
Goal (optional): {goal}

Output format (one instruction per line, numbered):
1. <instruction>
2. <instruction>
...
"""

GROUNDING_PROMPT = """You have a skill bank (action language) and a natural language plan.

Skill bank (contracts):
{skill_bank_text}

NL plan:
{nl_plan}

For each step in the NL plan, pick the best matching skill from the bank and rewrite the instruction using that skill's pre/effects if helpful. Keep instructions concise. Output the same numbered format.
"""


def _get_ask_model(config: PlanningConfig) -> Callable[..., str]:
    if config.ask_model_fn is not None:
        return config.ask_model_fn
    from API_func import ask_model
    return ask_model


def _parse_numbered_plan(raw: str) -> List[str]:
    """Parse numbered list (1. ... 2. ...) into list of instructions."""
    lines = raw.strip().splitlines()
    instructions: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)]\s*(.+)", line)
        if m:
            instructions.append(m.group(1).strip())
        else:
            instructions.append(line)
    return instructions


def _skill_bank_to_text(bank: Any, fmt: str = "compact") -> str:
    """Get skill bank as text for grounding. Uses contract_verification or stage3_mvp."""
    # contract_verification SkillBank has to_action_language
    if hasattr(bank, "to_action_language"):
        try:
            return bank.to_action_language(fmt=fmt)
        except Exception:
            pass
    # stage3_mvp SkillBankMVP: manual dump of eff_add/eff_del
    try:
        parts = []
        for sid in getattr(bank, "skill_ids", []):
            c = bank.get_contract(sid)
            if c:
                add_str = ",".join(sorted(getattr(c, "eff_add", set()) or []))
                del_str = ",".join(sorted(getattr(c, "eff_del", set()) or []))
                parts.append(f"{sid}: +({add_str}) -({del_str})")
        return "\n".join(parts) if parts else "(no skills)"
    except Exception:
        return "(skill bank unavailable)"


class ExperiencePlanner:
    """
    Plans experience synthesis from NL descriptions. Produces step-by-step plans
    in natural language or action language for the world model.
    """

    def __init__(self, config: Optional[PlanningConfig] = None):
        self.config = config or PlanningConfig()

    def _call_llm(self, prompt: str) -> str:
        ask = _get_ask_model(self.config)
        kwargs = {"temperature": self.config.temperature, "max_tokens": self.config.max_tokens}
        if self.config.model:
            kwargs["model"] = self.config.model
        return ask(prompt, **kwargs)

    def plan_episode(
        self,
        desc: EpisodeDescription,
        skill_bank: Optional[Any] = None,
    ) -> SynthesisPlan:
        """
        Create a synthesis plan from an episode description.

        Args:
            desc: Natural language description of the desired episode.
            skill_bank: Optional skill bank (SkillBank, SkillBankMVP) for grounding.

        Returns:
            SynthesisPlan with steps usable by the world model.
        """
        max_steps = desc.max_steps or 10
        prompt = EPISODE_PLAN_PROMPT.format(
            description=desc.description,
            max_steps=max_steps,
        )
        raw = self._call_llm(prompt)
        if skill_bank:
            bank_text = _skill_bank_to_text(skill_bank, self.config.action_language_format)
            raw = self._call_llm(GROUNDING_PROMPT.format(
                skill_bank_text=bank_text,
                nl_plan=raw,
            ))

        instructions = _parse_numbered_plan(raw)
        steps = [
            PlanStep(step_idx=i + 1, instruction=instr)
            for i, instr in enumerate(instructions)
        ]

        action_block = None
        if skill_bank and self.config.plan_format == PlanFormat.ACTION_LANGUAGE:
            action_block = _skill_bank_to_text(skill_bank, self.config.action_language_format)

        return SynthesisPlan(
            steps=steps,
            format=self.config.plan_format,
            source_description=desc.description,
            action_language_block=action_block,
        )

    def plan_skill(
        self,
        desc: SkillDescription,
        skill_bank: Optional[Any] = None,
    ) -> SynthesisPlan:
        """
        Create a synthesis plan from a skill description.

        Args:
            desc: Natural language description of the desired skill.
            skill_bank: Optional skill bank for grounding.

        Returns:
            SynthesisPlan with steps.
        """
        prompt = SKILL_PLAN_PROMPT.format(
            description=desc.description,
            goal=desc.goal or "(none)",
        )
        raw = self._call_llm(prompt)
        if skill_bank:
            bank_text = _skill_bank_to_text(skill_bank, self.config.action_language_format)
            raw = self._call_llm(GROUNDING_PROMPT.format(
                skill_bank_text=bank_text,
                nl_plan=raw,
            ))

        instructions = _parse_numbered_plan(raw)
        steps = [
            PlanStep(step_idx=i + 1, instruction=instr, skill_id=desc.skill_id)
            for i, instr in enumerate(instructions)
        ]

        return SynthesisPlan(
            steps=steps,
            format=self.config.plan_format,
            source_description=desc.description,
        )

    def to_world_model_inputs(
        self,
        plan: SynthesisPlan,
        initial_state: str = "",
        initial_historical: str = "",
    ) -> List[dict]:
        """
        Convert a SynthesisPlan to a list of inputs for the world model.

        Each element has: state, historical_summary, instructions (for agentic_textual)
        or current_frame, historical_summary, instructions (for multi_modal).

        For textual world model, state is updated each step; for multi-modal,
        the caller provides frames.
        """
        inputs = []
        state = initial_state
        hist = initial_historical
        for step in plan.steps:
            inputs.append({
                "state": state,
                "historical_summary": hist,
                "instructions": step.instruction,
            })
            # Next step: state becomes "to be predicted", hist accumulates
            hist = (hist + " " + step.instruction).strip() if hist else step.instruction
        return inputs
