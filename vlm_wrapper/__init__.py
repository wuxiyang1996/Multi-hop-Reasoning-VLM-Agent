"""VLM structured-state wrappers for Gym-V and BrowserGym.

Two heads produce the same <state>…</state> schema (see plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md §3):

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
    cascaded_ground,
    ground,
)

# ── Shared utilities ──────────────────────────────────────────────────
from vlm_wrapper.schema import (
    SCHEMA_VERSION,
    ValidationResult,
    build_adaptive_system_prompt,
    build_system_prompt,
    encode_image_b64,
    parse_answer_block,
    parse_answer_from_schema,
    parse_evidence_from_schema,
    parse_schema_output,
    semantic_validate,
    validate_schema,
)

# ── Head 1: Heuristic (text-in → schema-out) ─────────────────────────
# NOTE: ``gymv_*`` symbols are exposed lazily via ``__getattr__`` (PEP 562)
# below, so importing ``gymv_wrapper.adapter`` first does not create a
# circular import through this package's eager re-exports.
from vlm_wrapper.browser_heuristic import obs_to_schema as browser_heuristic_schema

# ── Head 2: Vision (image-in → schema-out via GPT-5.5/GPT-4o) ────────
from vlm_wrapper.browser_adapter import generate_label as browser_generate_label
from vlm_wrapper.browser_adapter import browser_obs_to_schema
from vlm_wrapper.osworld_adapter import generate_label as osworld_generate_label
from vlm_wrapper.osworld_adapter import osworld_obs_to_schema

__all__ = [
    # unified pipeline
    "ground",
    "cascaded_ground",
    "GroundingRequest",
    "GroundingResult",
    "HopTrace",
    # shared
    "SCHEMA_VERSION",
    "ValidationResult",
    "build_adaptive_system_prompt",
    "build_system_prompt",
    "encode_image_b64",
    "parse_answer_block",
    "parse_answer_from_schema",
    "parse_evidence_from_schema",
    "parse_schema_output",
    "semantic_validate",
    "validate_schema",
    # head 1: heuristic
    "gymv_heuristic_schema",
    "browser_heuristic_schema",
    # head 2: vision
    "gymv_generate_label",
    "browser_generate_label",
    "browser_obs_to_schema",
    "osworld_generate_label",
    "osworld_obs_to_schema",
]

__all__.append("GymVSchemaWrapper")


# PEP 562 lazy re-exports for the gymv_wrapper symbols. Defining these here
# (instead of eager top-level imports) breaks the circular dependency that
# would otherwise occur when ``gymv_wrapper.adapter`` is imported first and
# its ``from vlm_wrapper.schema import ...`` triggers this module before
# ``gymv_wrapper.adapter`` has finished initialising.
def __getattr__(name):  # noqa: D401
    if name == "gymv_heuristic_schema":
        from gymv_wrapper.heuristic import text_to_schema as _f
        return _f
    if name == "gymv_generate_label":
        from gymv_wrapper.adapter import generate_label as _f
        return _f
    if name == "GymVSchemaWrapper":
        from gymv_wrapper.adapter import GymVSchemaWrapper as _cls
        return _cls
    if name == "build_gymv_registry":
        from gymv_wrapper.tools import build_gymv_registry as _f
        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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

# Domain-specific registries (gymv lazily re-exported via __getattr__)
from vlm_wrapper.tools_browser import build_browser_registry, build_osworld_registry
from vlm_wrapper.visual_reasoning_wrapper.tools_video import build_video_registry
from vlm_wrapper.visual_reasoning_wrapper.tools_visual import build_visual_registry
from vlm_wrapper.visual_reasoning_wrapper.tools_video_visual import (
    build_video_visual_registry,
)

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

# ── Benchmark loaders + parsers (TIR-Bench / VTB / Video-Holmes) ──────
try:
    from vlm_wrapper.visual_reasoning_wrapper.benchmarks import (
        TIRBenchSample,
        VideoHolmesSample,
        VisualToolBenchSample,
        default_tir_bench_root,
        default_video_holmes_root,
        default_visual_toolbench_root,
        iter_tir_bench_samples,
        iter_video_holmes_samples,
        iter_visual_toolbench_samples,
        load_tir_bench_image,
        load_video_holmes_questions,
        load_visual_toolbench_image,
        parse_tir_bench_sample,
        parse_video_holmes_sample,
        parse_visual_toolbench_sample,
        sample_video_frames,
    )
    __all__ += [
        "TIRBenchSample",
        "VideoHolmesSample",
        "VisualToolBenchSample",
        "default_tir_bench_root",
        "default_video_holmes_root",
        "default_visual_toolbench_root",
        "iter_tir_bench_samples",
        "iter_video_holmes_samples",
        "iter_visual_toolbench_samples",
        "load_tir_bench_image",
        "load_video_holmes_questions",
        "load_visual_toolbench_image",
        "parse_tir_bench_sample",
        "parse_video_holmes_sample",
        "parse_visual_toolbench_sample",
        "sample_video_frames",
    ]
except ImportError:
    pass

# ── Visual reasoning (design notes + standard 2×2 benchmark matrix) ─
from vlm_wrapper.visual_reasoning_wrapper import (
    PRIMARY_VISUAL_REASONING_BENCHMARKS,
    VisualReasoningBenchmark,
)

__all__ += [
    "PRIMARY_VISUAL_REASONING_BENCHMARKS",
    "VisualReasoningBenchmark",
]
