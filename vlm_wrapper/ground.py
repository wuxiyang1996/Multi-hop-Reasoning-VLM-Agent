"""Unified visual grounding pipeline.

Single entry point for visual grounding + structured schema generation
across ALL task domains: video games, web agents, OS agents, image QA,
and video understanding.

**All tasks are interactive.**  The multi-hop tool-calling loop IS the
interaction — whether the VLM is reasoning about a CLEVR scene, playing
2048, or answering a video question.  Every task follows the same
interactive pattern:

  1. Input: image(s) + goal/question + optional domain context
  2. Process: VLM sees image → calls tools (multi-hop) → gathers
     evidence → produces structured schema
  3. Output: structured <state> with entities, relations, evidence,
     and a terminal section (actions for env tasks, answer for QA)

The only thing that varies per domain is:
  - Which tools are available (auto-composed from domain)
  - What context is injected (game rules, AXTree, question, etc.)
  - The terminal section: ``<actions>`` vs ``<answer>`` (or both)

Usage::

    from vlm_wrapper.ground import ground, GroundingRequest

    # Image QA (CLEVR, GQA) — interactive multi-hop reasoning
    result = ground(GroundingRequest(
        images=pil_image,
        goal="How many red spheres are left of the blue cube?",
        domain="image_qa",
    ))

    # Video game — interactive multi-hop grounding
    result = ground(GroundingRequest(
        images=frame,
        goal="Reach 2048",
        domain="gymv",
        context={"obs_text": obs.text, "description": env.description},
    ))

    # Browser — interactive multi-hop grounding
    result = ground(GroundingRequest(
        images=screenshot,
        goal="Find cheapest laptop",
        domain="browser",
        context={"obs": browsergym_obs},
    ))

    # Video understanding — interactive multi-hop temporal reasoning
    result = ground(GroundingRequest(
        images=frames,          # list of PIL Images
        goal="When does the character first appear?",
        domain="video_qa",
        context={"fps": 24.0},
    ))

    # All results share the same structure:
    result.schema        # <state>...</state>
    result.answer        # "3" (QA) or None (env tasks)
    result.evidence      # list of hop traces (always populated)
    result.tool_trace    # for SFT data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from .schema import (
    build_adaptive_system_prompt,
    parse_answer_from_schema,
    parse_evidence_from_schema,
    parse_schema_output,
    validate_schema,
)
from .tool_loop import run_tool_loop
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

DomainType = Literal[
    "gymv", "browser", "desktop", "image_qa", "video_qa", "video", "auto",
]

# All tasks are interactive.  The "output_mode" controls which terminal
# section(s) the schema includes: actions (env), answer (QA), or both.
OutputMode = Literal["actions", "answer", "both"]

_DOMAIN_TO_OUTPUT_MODE: dict[str, OutputMode] = {
    "gymv": "actions",
    "browser": "actions",
    "desktop": "actions",
    "image_qa": "answer",
    "video_qa": "answer",
    "video": "answer",
}

# Shared core sections — every task gets these.  The multi-hop evidence
# chain is always present because every task is interactive.
_CORE_SECTIONS: list[str] = [
    "entities", "attributes", "relations",
    "state_flags", "targets", "uncertainty", "evidence",
]

# Legacy alias kept for backward compat with schema.py defaults
TASK_TYPE_SECTIONS: dict[str, list[str]] = {
    "interactive": _CORE_SECTIONS + ["actions"],
    "qa": _CORE_SECTIONS + ["answer"],
    "temporal": _CORE_SECTIONS + ["answer"],
}


def _sections_for_output_mode(mode: OutputMode) -> list[str]:
    """Return the full section list for a given output mode."""
    if mode == "actions":
        return _CORE_SECTIONS + ["actions"]
    elif mode == "answer":
        return _CORE_SECTIONS + ["answer"]
    else:
        return _CORE_SECTIONS + ["actions", "answer"]


@dataclass
class GroundingRequest:
    """Universal input for the grounding pipeline.

    All tasks are interactive multi-hop reasoning.  The ``output_mode``
    controls the terminal section of the schema (actions vs answer).

    Parameters
    ----------
    images : Image or list[Image]
        Single frame (games, browser, desktop, image QA) or frame
        sequence (video QA, video understanding).
    goal : str
        Task goal or question to answer.
    domain : str
        ``"gymv"`` | ``"browser"`` | ``"desktop"`` | ``"image_qa"``
        | ``"video_qa"`` | ``"video"`` | ``"auto"``.
    output_mode : str
        ``"actions"`` (env tasks) | ``"answer"`` (QA tasks) |
        ``"both"`` | ``"auto"`` (inferred from domain).
    task_id : str
        Environment or benchmark identifier.
    step : int
        Current step number.
    context : dict
        Domain-specific context. Known keys:

        - ``obs_text`` (gymv): raw ``obs.text``
        - ``description`` (gymv): ``env.description`` (game rules)
        - ``obs`` (browser): full BrowserGym observation dict
        - ``axtree_text`` (browser): truncated AXTree string
        - ``a11y_tree_xml`` (desktop): OS accessibility tree XML
        - ``instruction`` (desktop): task instruction
        - ``fps`` (video): frames per second
        - ``current_index`` (video): current frame index
        - ``extra_tools`` (any): additional ``ToolRegistry`` to merge
    max_entities : int
        Entity cap for schema output.
    max_rounds : int
        Maximum tool-call rounds (each round = 1 VLM inference).
    model : str or None
        VLM model override. Defaults to env ``VLM_LABEL_MODEL``.
    api_key : str or None
        OpenAI API key override.
    base_url : str or None
        OpenAI base URL override.
    temperature : float or None
        Generation temperature override.
    """
    images: Image.Image | np.ndarray | list[Image.Image | np.ndarray] = field(
        default=None  # type: ignore[assignment]
    )
    goal: str = ""
    domain: DomainType = "auto"
    output_mode: OutputMode | Literal["auto"] = "auto"
    task_id: str = ""
    step: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    max_entities: int = 25
    max_rounds: int = 5
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None


@dataclass
class HopTrace:
    """One reasoning hop in the evidence chain."""
    hop_id: int
    tool: str | None = None
    result_ref: str | None = None
    frame: int | None = None
    timestamp: float | None = None
    confidence: str | None = None
    raw: str = ""


@dataclass
class GroundingResult:
    """Universal output from the grounding pipeline.

    Every domain produces the same structure.  ``evidence`` is always
    populated (multi-hop reasoning traces).  ``answer`` is populated for
    QA domains; ``None`` for env-action domains.
    """
    schema: str | None = None
    answer: str | None = None
    evidence: list[HopTrace] = field(default_factory=list)
    raw: str = ""
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    domain: str = ""
    output_mode: str = ""


def _infer_domain(req: GroundingRequest) -> str:
    """Best-effort domain detection from the request contents."""
    ctx = req.context
    if ctx.get("obs") and isinstance(ctx["obs"], dict):
        if ctx["obs"].get("axtree_object") or ctx["obs"].get("screenshot") is not None:
            return "browser"
    if ctx.get("a11y_tree_xml"):
        return "desktop"
    if ctx.get("obs_text") or ctx.get("description"):
        return "gymv"
    if isinstance(req.images, list) and len(req.images) > 1:
        return "video_qa"
    return "image_qa"


def _resolve_output_mode(req: GroundingRequest, domain: str) -> OutputMode:
    """Determine output mode from explicit setting or domain default."""
    if req.output_mode != "auto":
        return req.output_mode  # type: ignore[return-value]
    return _DOMAIN_TO_OUTPUT_MODE.get(domain, "answer")


def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img


_NATURAL_IMAGE_DOMAINS = {"image_qa", "video_qa", "video"}


def _build_registry(
    req: GroundingRequest,
    domain: str,
    primary_image: Image.Image,
) -> ToolRegistry:
    """Auto-compose the right tool registry from domain + context."""
    from .tools_visual import build_visual_registry

    use_gdino = domain in _NATURAL_IMAGE_DOMAINS
    visual_reg = build_visual_registry(primary_image, prefer_gdino=use_gdino)
    registries: list[ToolRegistry] = [visual_reg]

    if domain == "gymv":
        from .tools_gymv import build_gymv_registry
        gymv_reg = build_gymv_registry(
            obs_text=req.context.get("obs_text", ""),
            description=req.context.get("description", ""),
            step=req.step,
        )
        registries.append(gymv_reg)

    elif domain == "browser":
        obs = req.context.get("obs")
        if obs and isinstance(obs, dict):
            from .tools_browser import build_browser_registry
            registries.append(build_browser_registry(obs))

    elif domain == "desktop":
        from .tools_browser import build_osworld_registry
        registries.append(build_osworld_registry(
            a11y_tree_xml=req.context.get("a11y_tree_xml", ""),
            instruction=req.context.get("instruction", req.goal),
            terminal_output=req.context.get("terminal_output", ""),
        ))

    elif domain in ("video_qa", "video") and isinstance(req.images, list):
        from .tools_video import build_video_registry
        from .tools_video_visual import build_video_visual_registry
        vv_reg = build_video_visual_registry(
            frames=req.images,
            fps=req.context.get("fps", 1.0),
            current_index=req.context.get("current_index", 0),
        )
        registries = [vv_reg]

    extra = req.context.get("extra_tools")
    if isinstance(extra, ToolRegistry):
        registries.append(extra)

    combined = registries[0]
    for reg in registries[1:]:
        combined = combined.merge(reg)
    return combined


def _build_extra_context(req: GroundingRequest, domain: str) -> str:
    """Assemble domain-specific text context for the user message."""
    parts: list[str] = []

    if domain == "gymv":
        desc = req.context.get("description", "")
        obs_text = req.context.get("obs_text", "")
        if desc:
            parts.append(f"Game rules:\n{desc}")
        if obs_text:
            parts.append(f"Environment text state (for reference):\n{obs_text}")

    elif domain == "browser":
        obs = req.context.get("obs", {})
        url = obs.get("url", "") if isinstance(obs, dict) else ""
        axtree = req.context.get("axtree_text", "")
        if url:
            parts.append(f"URL: {url}")
        if axtree:
            parts.append(f"AXTree (truncated):\n{axtree[:3000]}")

    elif domain == "desktop":
        instruction = req.context.get("instruction", "")
        if instruction:
            parts.append(f"Task instruction: {instruction}")

    elif domain in ("video_qa", "video") and isinstance(req.images, list):
        fps = req.context.get("fps", 1.0)
        cur = req.context.get("current_index", 0)
        parts.append(
            f"Video: {len(req.images)} frames at {fps} FPS, "
            f"currently at frame {cur}.\n"
            f"Full tool suite available: temporal navigation, "
            f"vision-model detection, cross-frame tracking."
        )

    elif domain == "image_qa":
        parts.append(
            "Vision-model tools available for precise element detection "
            "and spatial reasoning. Use tools to gather evidence before "
            "answering."
        )

    return "\n\n".join(parts)


def ground(req: GroundingRequest) -> GroundingResult:
    """Unified visual grounding entry point.

    All tasks are interactive multi-hop reasoning.  The pipeline:
    auto-detects domain if needed, composes the right tool registry,
    builds a system prompt with the universal schema (core + terminal),
    runs the VLM tool-calling loop, and parses the result.
    """
    if req.images is None:
        return GroundingResult(
            warnings=["no images provided"],
            domain=req.domain,
        )

    # 1. Resolve domain and output mode
    domain = req.domain if req.domain != "auto" else _infer_domain(req)
    output_mode = _resolve_output_mode(req, domain)
    sections = _sections_for_output_mode(output_mode)

    # 2. Get primary image for the VLM
    if isinstance(req.images, list):
        if not req.images:
            return GroundingResult(
                warnings=["empty frame list"],
                domain=domain,
                output_mode=output_mode,
            )
        idx = req.context.get("current_index", 0)
        idx = max(0, min(idx, len(req.images) - 1))
        primary_image = _to_pil(req.images[idx])
    else:
        primary_image = _to_pil(req.images)

    # 3. Auto-compose registry
    registry = _build_registry(req, domain, primary_image)

    # 4. Build extra context
    extra_context = _build_extra_context(req, domain)

    # 5. Run the tool loop — same loop for all tasks
    loop_result = run_tool_loop(
        primary_image,
        domain=domain,
        registry=registry,
        goal=req.goal,
        task_id=req.task_id,
        step=req.step,
        extra_context=extra_context,
        max_entities=req.max_entities,
        max_rounds=req.max_rounds,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        temperature=req.temperature,
        sections=sections,
        task_type="interactive",
    )

    # 6. Parse into universal result
    schema = loop_result.get("schema")
    raw = loop_result.get("raw", "")

    answer = None
    evidence: list[HopTrace] = []
    if schema:
        answer = parse_answer_from_schema(schema)
        evidence = _parse_evidence_hops(schema)

    return GroundingResult(
        schema=schema,
        answer=answer,
        evidence=evidence,
        raw=raw,
        warnings=loop_result.get("warnings", []),
        model=loop_result.get("model", ""),
        tool_trace=loop_result.get("tool_trace", []),
        rounds=loop_result.get("rounds", 0),
        messages=loop_result.get("messages", []),
        domain=domain,
        output_mode=output_mode,
    )


def _parse_evidence_hops(schema: str) -> list[HopTrace]:
    """Extract HopTrace objects from the <evidence> section."""
    raw_evidence = parse_evidence_from_schema(schema)
    if not raw_evidence:
        return []

    hops: list[HopTrace] = []
    current_hop_id = 0
    current: dict[str, Any] = {}

    for line in raw_evidence.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("hop") and "." in line:
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            key_part, value = parts[0].strip(), parts[1].strip()
            dot_parts = key_part.split(".")
            if len(dot_parts) != 2:
                continue

            hop_tag, field_name = dot_parts
            try:
                hop_num = int(hop_tag.replace("hop", ""))
            except ValueError:
                continue

            if hop_num != current_hop_id and current:
                hops.append(HopTrace(
                    hop_id=current_hop_id,
                    raw=str(current),
                    **{k: v for k, v in current.items() if k != "hop_id"},
                ))
                current = {}

            current_hop_id = hop_num

            if field_name == "tool":
                current["tool"] = value
            elif field_name == "result_ref":
                current["result_ref"] = value
            elif field_name == "frame":
                current["frame"] = None if value == "null" else int(value)
            elif field_name == "timestamp":
                current["timestamp"] = None if value == "null" else float(value)
            elif field_name == "confidence":
                current["confidence"] = value

    if current:
        hops.append(HopTrace(
            hop_id=current_hop_id,
            raw=str(current),
            **{k: v for k, v in current.items() if k != "hop_id"},
        ))

    return hops
