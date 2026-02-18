"""
Schemas for experience planning: convert NL descriptions of episodes/skills
into plans (natural language or action language) for world model synthesis.

References: skill_agents (contract_verification, stage3_mvp, skill_bank),
data_structure.experience (Episode, SubTask_Experience, Experience).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


class PlanFormat(str, Enum):
    """Output format for the synthesis plan."""

    NATURAL_LANGUAGE = "natural_language"
    ACTION_LANGUAGE = "action_language"


@dataclass
class EpisodeDescription:
    """Natural language description of a desired episode."""

    description: str
    task: Optional[str] = None
    max_steps: Optional[int] = None
    env_hint: Optional[str] = None  # e.g. "overcooked", "avalon"


@dataclass
class SkillDescription:
    """Natural language description of a desired skill/sub-task."""

    description: str
    skill_id: Optional[str] = None
    goal: Optional[str] = None


@dataclass
class PlanStep:
    """One step in the synthesis plan."""

    step_idx: int
    instruction: str  # NL or action-language instruction
    skill_id: Optional[str] = None
    preconditions: Optional[List[str]] = None
    effects: Optional[List[str]] = None
    metadata: Optional[dict] = None


@dataclass
class SynthesisPlan:
    """
    Plan to drive world model synthesis. Can be natural language (step-by-step
    instructions) or action language (PDDL/STRIPS/compact from skill bank).
    """

    steps: List[PlanStep]
    format: PlanFormat = PlanFormat.NATURAL_LANGUAGE
    source_description: Optional[str] = None
    action_language_block: Optional[str] = None  # Raw PDDL/STRIPS/etc if used
    metadata: Optional[dict] = None

    def to_instructions(self) -> List[str]:
        """Extract instruction strings for each step (for world model inputs)."""
        return [s.instruction for s in self.steps]

    def to_instructions_with_context(self) -> List[tuple[str, str]]:
        """Return (instruction, preconditions_str) per step."""
        out = []
        for s in self.steps:
            ctx = ""
            if s.preconditions:
                ctx = f"Preconditions: {', '.join(s.preconditions)}."
            out.append((s.instruction, ctx))
        return out
