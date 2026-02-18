"""
Example usage of the agentic textual world model.

Run with:
    python -m world_model.agentic_textual.example_usage

Requires: API_func.ask_model (and API keys) for real LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from world_model.agentic_textual import (
    TextWorldModel,
    TextWorldModelConfig,
    TextSynthesisInput,
    TextSyntheticExperienceSequence,
)


def example_single_step():
    """Single-step synthesis."""
    config = TextWorldModelConfig(model="gpt-4o-mini")
    model = TextWorldModel(config)

    inp = TextSynthesisInput(
        state="Agent at (2,1) facing east. Pot has 2 onions. No orders pending.",
        historical_summary="Agent moved from dispenser toward the pot.",
        instructions="Agent picks up an onion from the dispenser.",
    )
    out = model.synthesize_step(inp)
    print("Next state:", out.next_state[:200] + "..." if len(out.next_state) > 200 else out.next_state)
    print("Action:", out.action)
    print("Reward:", out.reward)


def example_sequence():
    """Multi-step synthesis."""
    config = TextWorldModelConfig(model="gpt-4o-mini")
    model = TextWorldModel(config)

    seq = model.synthesize_sequence(
        initial_state="Agent at (3,0). Pot empty. Onion dispenser nearby.",
        instructions_per_step=[
            "Agent picks up onion from dispenser.",
            "Agent carries onion to pot.",
            "Agent places onion in pot.",
        ],
        historical_summaries=["", "Agent picked up onion.", "Agent walked to pot."],
    )
    for i, step in enumerate(seq.steps):
        print(f"Step {i+1}: next_state = {step.next_state[:100]}...")


def example_to_experiences():
    """Convert sequence to Experience-like dicts."""
    config = TextWorldModelConfig(model="gpt-4o-mini")
    model = TextWorldModel(config)

    seq = model.synthesize_sequence(
        initial_state="Agent at (2,1). Pot has 1 onion.",
        instructions_per_step=["Agent places onion in pot."],
    )
    exps = model.to_experiences(seq, initial_state="Agent at (2,1). Pot has 1 onion.")
    for exp in exps:
        print(exp)


if __name__ == "__main__":
    print("Running agentic textual world model examples...")
    example_single_step()
    # example_sequence()
    # example_to_experiences()
