"""VLM structured-state wrappers for Gym-V and BrowserGym.

Two heads produce the same <state>…</state> schema (see plans/PLAN-VISUAL-GROUNDING.md §3):

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

# ── Unified grounding pipeline ────────────────────────────────────────
from vlm_wrapper.ground import (
    GroundingRequest,
    GroundingResult,
    HopTrace,
    ground,
)

# ── Shared utilities ──────────────────────────────────────────────────
from vlm_wrapper.schema import (
    SCHEMA_VERSION,
    build_adaptive_system_prompt,
    build_system_prompt,
    encode_image_b64,
    parse_answer_block,
    parse_answer_from_schema,
    parse_evidence_from_schema,
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
    # unified pipeline
    "ground",
    "GroundingRequest",
    "GroundingResult",
    "HopTrace",
    # shared
    "SCHEMA_VERSION",
    "build_adaptive_system_prompt",
    "build_system_prompt",
    "encode_image_b64",
    "parse_answer_block",
    "parse_answer_from_schema",
    "parse_evidence_from_schema",
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

# ── Head 3: Grounding (image-in → OmniParser-v2 → schema-out) ────────
try:
    from vlm_wrapper.grounding import parse_screen, parse_screen_annotated, ScreenElement, BBox
    from vlm_wrapper.grounding_browsergym import (
        grounding_image_to_schema,
        grounding_obs_to_schema,
        grounding_osworld_obs_to_schema,
    )
    __all__ += [
        "parse_screen",
        "parse_screen_annotated",
        "ScreenElement",
        "BBox",
        "grounding_image_to_schema",
        "grounding_obs_to_schema",
        "grounding_osworld_obs_to_schema",
    ]
except ImportError:
    pass

# ── Tool-calling infrastructure ──────────────────────────────────────
from vlm_wrapper.tools import ToolRegistry, ToolDef, ToolResult

# Domain-specific registries
from vlm_wrapper.tools_gymv import build_gymv_registry
from vlm_wrapper.tools_browser import build_browser_registry, build_osworld_registry
from vlm_wrapper.tools_video import build_video_registry
from vlm_wrapper.tools_visual import build_visual_registry
from vlm_wrapper.tools_video_visual import build_video_visual_registry

# Tool-calling loop + convenience wrappers
from vlm_wrapper.tool_loop import (
    run_tool_loop,
    gymv_generate_label_with_tools,
    browser_generate_label_with_tools,
    video_generate_label_with_tools,
    visual_generate_label_with_tools,
    video_visual_generate_label_with_tools,
)

__all__ += [
    # tool infrastructure
    "ToolRegistry",
    "ToolDef",
    "ToolResult",
    # domain registries
    "build_gymv_registry",
    "build_browser_registry",
    "build_osworld_registry",
    "build_video_registry",
    "build_visual_registry",
    "build_video_visual_registry",
    # tool-calling loop
    "run_tool_loop",
    "gymv_generate_label_with_tools",
    "browser_generate_label_with_tools",
    "video_generate_label_with_tools",
    "visual_generate_label_with_tools",
    "video_visual_generate_label_with_tools",
]
