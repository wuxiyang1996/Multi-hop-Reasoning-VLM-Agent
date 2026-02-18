"""
Example usage of the multi-modal world model for experience synthesis.

Run with:
    python -m world_model.multi_modal.example_usage

Requires: pip install -r world_model/multi_modal/requirements.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on path
_repo = Path(__file__).resolve().parent.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from world_model.multi_modal import (
    WorldModel,
    WorldModelConfig,
    SynthesisInput,
    MODEL_LONGCAT,
    MODEL_QWEN_EDIT,
)


def example_single_step():
    """Single-step synthesis: current frame + instructions -> next frame."""
    config = WorldModelConfig(
        model_name=MODEL_LONGCAT,
        device="auto",
    )
    model = WorldModel(config)

    # In practice, load your game frame
    # frame = Image.open("current_state.png")
    # For demo, we just show the API
    print("Single-step synthesis API:")
    print("  inp = SynthesisInput(")
    print('    current_frame=<PIL.Image or path>,')
    print('    historical_summary="Agent at (2,1). Pot has 2 onions.",')
    print('    instructions="Agent picks up onion from dispenser.",')
    print("  )")
    print("  out = model.synthesize_step(inp, seed=42)")
    print("  next_frame = out.next_frame  # PIL.Image")
    print()


def example_multi_step():
    """Multi-step synthesis: chain edits for a sequence."""
    config = WorldModelConfig(model_name=MODEL_QWEN_EDIT)
    model = WorldModel(config)

    print("Multi-step synthesis API:")
    print("  seq = model.synthesize_sequence(")
    print("    initial_frame=<PIL.Image or path>,")
    print('    instructions_per_step=[')
    print('      "Agent moves to onion dispenser.",')
    print('      "Agent picks up onion.",')
    print('      "Agent carries onion to pot.",')
    print("    ],")
    print('    historical_summaries=["...", "..."]  # optional')
    print("  )")
    print("  for step in seq.steps:")
    print("      frame = step.next_frame")
    print()


def example_model_selection():
    """Show how to declare the model used."""
    print("Model selection options:")
    print("  1. Config: WorldModelConfig(model_name='longcat')")
    print("  2. Config: WorldModelConfig(model_name='qwen_edit')")
    print("  3. Env:   WORLD_MODEL_IMAGE_EDIT_MODEL=qwen_edit")
    print("  4. Custom: WorldModelConfig(model_id='org/custom-model')")
    print()


if __name__ == "__main__":
    example_model_selection()
    example_single_step()
    example_multi_step()
    print("To actually run synthesis, provide a real image path and GPU.")
