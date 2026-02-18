"""
Example: plan from NL description, then generate synthetic experiences.

Run:
  python -m world_model.experience_planning.example_usage

Requires: API_func.ask_model, optionally skill bank.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from world_model.experience_planning import (
    ExperiencePlanner,
    EpisodeDescription,
    SkillDescription,
)


def example_episode_plan():
    """Plan an episode from NL description."""
    planner = ExperiencePlanner()
    desc = EpisodeDescription(
        description="In Overcooked, the agent should pick up an onion, carry it to the pot, and add it.",
        max_steps=5,
        env_hint="overcooked",
    )
    plan = planner.plan_episode(desc)
    print("Plan steps:")
    for s in plan.steps:
        print(f"  {s.step_idx}. {s.instruction}")
    return plan


def example_skill_plan():
    """Plan a skill from NL description."""
    planner = ExperiencePlanner()
    desc = SkillDescription(
        description="Place an ingredient into the cooking pot.",
        skill_id="place_in_pot",
        goal="Pot contains the ingredient",
    )
    plan = planner.plan_skill(desc)
    print("Skill plan:")
    for s in plan.steps:
        print(f"  {s.step_idx}. {s.instruction}")
    return plan


def example_plan_to_world_model():
    """Use plan to drive agentic textual world model."""
    planner = ExperiencePlanner()
    desc = EpisodeDescription(
        description="Agent picks up onion and places it in the pot.",
        max_steps=3,
    )
    plan = planner.plan_episode(desc)
    inputs = planner.to_world_model_inputs(
        plan,
        initial_state="Agent at (2,1). Pot empty. Onion dispenser nearby.",
        initial_historical="",
    )
    print("World model inputs:")
    for i, inp in enumerate(inputs):
        print(f"  Step {i+1}: instructions={inp['instructions'][:60]}...")

    # Feed to TextWorldModel
    from world_model.agentic_textual import TextWorldModel, TextSynthesisInput
    model = TextWorldModel()
    state = "Agent at (2,1). Pot empty."
    for inp in inputs:
        out = model.synthesize_step(TextSynthesisInput(
            state=inp["state"] or state,
            historical_summary=inp["historical_summary"],
            instructions=inp["instructions"],
        ))
        state = out.next_state
        print(f"  -> next_state: {state[:80]}...")


if __name__ == "__main__":
    print("=== Episode plan ===")
    example_episode_plan()
    print("\n=== Skill plan ===")
    example_skill_plan()
    # print("\n=== Plan -> World Model ===")
    # example_plan_to_world_model()
