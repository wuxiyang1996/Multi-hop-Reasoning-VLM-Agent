"""VLM structured-state wrappers for Gym-V and BrowserGym.

Two heads produce the same <state>…</state> schema (TODO-VLM §12a):

**Head 1 — Heuristic (text-in → schema-out)**
  Fast, free, deterministic.  Parses native text state (obs.text,
  AXTree/DOM) into the schema with regex/tree-walking.
  Good for: real-time RL rollouts, cheap baselines, validation.

**Head 2 — Vision (image-in → schema-out)**
  Sends the screenshot to GPT-4o (or any vision LLM) and receives
  the schema.  The image is the primary input; native text is
  optional grounding context.
  Good for: training-label generation, Qwen3-VL-8B distillation.

Gym-V examples::

    # Heuristic head
    from vlm_wrapper import gymv_heuristic_schema
    schema = gymv_heuristic_schema(obs_text="...", description="...", task_id="Game2048-v0")

    # Vision head
    from vlm_wrapper import gymv_generate_label
    result = gymv_generate_label(frame, goal="Reach 2048", task_id="Game2048-v0")

BrowserGym examples::

    # Heuristic head
    from vlm_wrapper import browser_heuristic_schema
    schema = browser_heuristic_schema(obs, step=3, task_id="webarena.shopping.143")

    # Vision head
    from vlm_wrapper import browser_obs_to_schema
    result = browser_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
"""

# ── Shared utilities ──────────────────────────────────────────────────
from vlm_wrapper.schema import (
    SCHEMA_VERSION,
    build_system_prompt,
    encode_image_b64,
    parse_schema_output,
    validate_schema,
)

# ── Head 1: Heuristic (text-in → schema-out) ─────────────────────────
from vlm_wrapper.gymv_heuristic import text_to_schema as gymv_heuristic_schema
from vlm_wrapper.browser_heuristic import obs_to_schema as browser_heuristic_schema

# ── Head 2: Vision (image-in → schema-out via GPT-4o) ────────────────
from vlm_wrapper.gymv_adapter import generate_label as gymv_generate_label
from vlm_wrapper.browser_adapter import generate_label as browser_generate_label
from vlm_wrapper.browser_adapter import browser_obs_to_schema

__all__ = [
    # shared
    "SCHEMA_VERSION",
    "build_system_prompt",
    "encode_image_b64",
    "parse_schema_output",
    "validate_schema",
    # head 1: heuristic
    "gymv_heuristic_schema",
    "browser_heuristic_schema",
    # head 2: vision
    "gymv_generate_label",
    "browser_generate_label",
    "browser_obs_to_schema",
]

try:
    from vlm_wrapper.gymv_adapter import GymVSchemaWrapper
    __all__.append("GymVSchemaWrapper")
except (ImportError, TypeError):
    pass
