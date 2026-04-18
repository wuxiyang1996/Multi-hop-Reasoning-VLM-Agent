"""Shared schema definition, system prompts, and image-encoding helpers.

This is the single source of truth for the structured-state schema
(see plans/PLAN-VISUAL-GROUNDING.md §3).  Both the Gym-V and BrowserGym adapters import from here.
The adaptive prompt builder supports all domains (game, browser, desktop,
image QA, video QA) through configurable schema sections.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

import numpy as np
from PIL import Image

# ── Schema version (bumped when fields change) ───────────────────────
SCHEMA_VERSION = "0.2"

# ── Per-section schema snippets (composed by the adaptive builder) ────

_SECTION_HEADER = """\
<state>
domain={domain}
task={task_id}
goal={goal}
step={step}
"""

_SECTION_ENTITIES = """\
<entities>
e1[type={element|object|region|text}, label={short_label}, pos={x,y,w,h or null}]
e2[...]
(list every visually distinct entity you can identify — ≤{max_entities})
"""

_SECTION_ATTRIBUTES = """\
<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={value or null}
(one line per entity that has a notable attribute)
"""

_SECTION_RELATIONS = """\
<relations>
contains(eA,eB)
adjacent(eA,eB)
blocks(eA,eB)
grouped(eA,eB,eC)
(spatial and semantic relations you observe)
"""

_SECTION_STATE_FLAGS = """\
<state_flags>
progress={0.0–1.0 or null}
phase={early|mid|late|null}
error={description or null}
dialog_open={true|false}
input_pending={true|false}
"""

_SECTION_TARGETS = """\
<targets>
target={eid of the most task-relevant entity}
blocker={eid or null}
constraint={short constraint or null}
candidate_set=[eid,eid,...]
"""

_SECTION_UNCERTAINTY = """\
<uncertainty>
{eid}.{field}={high|medium|low}
(only for entities where you are unsure)
"""

_SECTION_ACTIONS = """\
<actions>
a1={action_string}
a2={action_string}
(top 3–5 plausible next actions)
"""

_SECTION_EVIDENCE = """\
<evidence>
hop1.tool={tool_name}
hop1.result_ref={e1,e3}
hop1.frame={frame_index or null}
hop1.timestamp={seconds or null}
hop1.confidence={high|medium|low}
hop2.tool={tool_name}
hop2.result_ref={e2}
hop2.confidence={high|medium|low}
(one group per reasoning hop — record every tool you called and which
entities it grounded.  This is the trace of your interactive reasoning.
For video/temporal tasks include frame/timestamp.)
"""

_SECTION_ANSWER = """\
<answer>
answer={predicted answer}
grounding=[e1,e3]
evidence_chain=[hop1,hop2]
confidence={high|medium|low}
"""

_SECTION_FOOTER = "</state>"

_SECTION_MAP: dict[str, str] = {
    "entities": _SECTION_ENTITIES,
    "attributes": _SECTION_ATTRIBUTES,
    "relations": _SECTION_RELATIONS,
    "state_flags": _SECTION_STATE_FLAGS,
    "targets": _SECTION_TARGETS,
    "uncertainty": _SECTION_UNCERTAINTY,
    "actions": _SECTION_ACTIONS,
    "evidence": _SECTION_EVIDENCE,
    "answer": _SECTION_ANSWER,
}

_SCHEMA_RULES = """\
Rules:
- pos= uses pixel coordinates (x,y,w,h) for browser/desktop, grid coords (r,c,1,1) for games.
- Keep labels short (≤5 words).
- Entity IDs are sequential: e1, e2, e3 …
- Reuse entity IDs across sections — never repeat the full label.
- Output ≤{max_entities} entities.  Prefer interactive/important ones.
- If you cannot determine a field, write null.
"""

# ── Legacy full spec (for backward compat with existing adapters) ─────
SCHEMA_SPEC = """\
You are a visual-state parser.  Given a screenshot, output a structured
summary using EXACTLY the tagged format below.  Do NOT output anything
outside the <state>…</state> block.

<state>
domain={domain}
task={task_id}
goal={goal}
step={step}

<entities>
e1[type={element|object|region|text}, label={short_label}, pos={x,y,w,h or null}]
e2[...]
(list every visually distinct entity you can identify — ≤{max_entities})

<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={value or null}
(one line per entity that has a notable attribute)

<relations>
contains(eA,eB)
adjacent(eA,eB)
blocks(eA,eB)
grouped(eA,eB,eC)
(spatial and semantic relations you observe)

<state_flags>
progress={0.0–1.0 or null}
phase={early|mid|late|null}
error={description or null}
dialog_open={true|false}
input_pending={true|false}

<targets>
target={eid of the most task-relevant entity}
blocker={eid or null}
constraint={short constraint or null}
candidate_set=[eid,eid,...]

<uncertainty>
{eid}.{field}={high|medium|low}
(only for entities where you are unsure)

<actions>
a1={action_string}
a2={action_string}
(top 3–5 plausible next actions)
</state>

Rules:
- pos= uses pixel coordinates (x,y,w,h) for browser, grid coords (r,c,1,1) for games.
- Keep labels short (≤5 words).
- Entity IDs are sequential: e1, e2, e3 …
- Reuse entity IDs across sections — never repeat the full label.
- Output ≤{max_entities} entities.  Prefer interactive/important ones.
- If you cannot determine a field, write null.
"""

# ── Domain-specific context injected after the schema spec ────────────

GYMV_CONTEXT = """\
Domain: gymv (video game).
The screenshot is a rendered game frame.  Entities are game objects
(tiles, pieces, player, walls, targets, etc.).  Positions are grid
coordinates (row, col, 1, 1).  Actions are the valid game moves.
"""

BROWSER_CONTEXT = """\
Domain: browser (web page).
The screenshot is a browser viewport.  Entities are UI elements
(buttons, links, inputs, text blocks, images, etc.).  Positions are
pixel coordinates (x, y, width, height).  Actions are browser
commands: click(bid), fill(bid, "text"), scroll(direction), etc.
If element IDs (bid) are visible as overlays, include them.
"""

DESKTOP_CONTEXT = """\
Domain: desktop (OS-level application).
The screenshot is a desktop or application window.  Entities are UI
elements (buttons, menus, text fields, icons, windows, dialogs).
Positions are pixel coordinates (x, y, width, height).  Actions
are mouse/keyboard operations: click(x, y), type("text"), etc.
"""

IMAGE_QA_CONTEXT = """\
Domain: image_qa (visual reasoning / question answering).
The screenshot is a static image.  This is an interactive multi-hop
reasoning task: call tools to detect objects, verify spatial relations,
count elements, describe regions, and gather grounded evidence.
Record each reasoning hop in <evidence> with the tool you called and
which entities it referenced.  Build your understanding step by step
before producing the final answer in <answer>.
"""

VIDEO_QA_CONTEXT = """\
Domain: video_qa (video understanding / question answering).
You have access to a video via temporal navigation tools.  This is an
interactive multi-hop reasoning task: call tools to navigate frames,
detect objects, track elements across time, find moments, and gather
grounded evidence.  Record each reasoning hop in <evidence> — include
frame index and timestamp for temporal grounding.  Build your
understanding step by step before producing the final answer in
<answer>.
"""

_DOMAIN_CONTEXT: dict[str, str] = {
    "gymv": GYMV_CONTEXT,
    "browser": BROWSER_CONTEXT,
    "desktop": DESKTOP_CONTEXT,
    "image_qa": IMAGE_QA_CONTEXT,
    "video_qa": VIDEO_QA_CONTEXT,
    "video": VIDEO_QA_CONTEXT,
}


def build_system_prompt(
    domain: str,
    max_entities: int = 20,
) -> str:
    """Build the full system prompt for GPT-4o vision calls.

    Legacy interface — returns the full interactive-task schema with
    all sections.  Use ``build_adaptive_system_prompt`` for the new
    unified pipeline.
    """
    spec = SCHEMA_SPEC.replace("{max_entities}", str(max_entities))
    ctx = GYMV_CONTEXT if domain == "gymv" else BROWSER_CONTEXT
    return f"schema_version={SCHEMA_VERSION}\n\n{spec}\n{ctx}"


def build_adaptive_system_prompt(
    domain: str,
    *,
    sections: list[str] | None = None,
    task_type: str = "interactive",
    max_entities: int = 20,
) -> str:
    """Build a system prompt that includes only the requested schema sections.

    All tasks are interactive multi-hop reasoning.  The *sections*
    parameter controls which parts of the schema to include.

    Parameters
    ----------
    domain : str
        One of ``"gymv"``, ``"browser"``, ``"desktop"``, ``"image_qa"``,
        ``"video_qa"``, ``"video"``.
    sections : list[str] or None
        Which schema sections to include.  If None, defaults to the
        full interactive set (core + actions).  Valid names: entities,
        attributes, relations, state_flags, targets, uncertainty,
        actions, evidence, answer.
    task_type : str
        Ignored (kept for backward compat).  All tasks are interactive.
    max_entities : int
        Entity cap.
    """
    if sections is None:
        sections = [
            "entities", "attributes", "relations",
            "state_flags", "targets", "uncertainty",
            "evidence", "actions",
        ]

    preamble = (
        "You are a visual-state parser performing interactive multi-hop "
        "reasoning.  Given a screenshot and tools, call tools to gather "
        "grounded evidence, then output a structured summary using "
        "EXACTLY the tagged format below.  Record every reasoning hop "
        "in the <evidence> section.  Do NOT output anything outside the "
        "<state>…</state> block.\n\n"
    )

    header = _SECTION_HEADER.replace("{max_entities}", str(max_entities))
    body_parts = [header]
    for sec_name in sections:
        snippet = _SECTION_MAP.get(sec_name)
        if snippet:
            body_parts.append(snippet.replace("{max_entities}", str(max_entities)))
    body_parts.append(_SECTION_FOOTER)

    schema_template = "\n".join(body_parts)

    rules = _SCHEMA_RULES.replace("{max_entities}", str(max_entities))
    ctx = _DOMAIN_CONTEXT.get(domain, IMAGE_QA_CONTEXT)

    return (
        f"schema_version={SCHEMA_VERSION}\n\n"
        f"{preamble}{schema_template}\n\n{rules}\n{ctx}"
    )


def build_user_message(
    image: Image.Image | np.ndarray,
    *,
    domain: str,
    task_id: str = "",
    goal: str = "",
    step: int = 0,
    extra_context: str = "",
) -> list[dict[str, Any]]:
    """Build the multimodal user message (image + text context).

    Returns an OpenAI-compatible ``content`` list for the user role.
    """
    b64 = encode_image_b64(image)

    text_parts = [
        f"domain={domain}",
        f"task={task_id}",
        f"goal={goal}",
        f"step={step}",
    ]
    if extra_context:
        text_parts.append(f"\nAdditional context:\n{extra_context}")

    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        {"type": "text", "text": "\n".join(text_parts)},
    ]


# ── Image helpers ─────────────────────────────────────────────────────

def encode_image_b64(
    image: Image.Image | np.ndarray,
    max_side: int = 1024,
    quality: int = 90,
) -> str:
    """Encode a PIL Image or numpy array to a base64 PNG string.

    Down-scales to *max_side* on the longest edge to keep API costs
    reasonable while retaining enough detail for GPT-4o.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Output parsing / validation ───────────────────────────────────────

_STATE_BLOCK_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL)
_ENTITY_RE = re.compile(r"^(e\d+)\[", re.MULTILINE)


def parse_schema_output(raw: str) -> str | None:
    """Extract the <state>…</state> block from raw GPT output.

    Returns the block (including tags) or None if not found.
    """
    m = _STATE_BLOCK_RE.search(raw)
    if m:
        return f"<state>{m.group(1)}</state>"
    return None


def count_entities(schema_text: str) -> int:
    """Count entity lines in a schema string."""
    return len(_ENTITY_RE.findall(schema_text))


def validate_schema(
    schema_text: str,
    *,
    required_sections: list[str] | None = None,
) -> list[str]:
    """Quick structural checks.  Returns a list of warnings (empty = OK).

    Parameters
    ----------
    schema_text : str
        The ``<state>…</state>`` block.
    required_sections : list[str] or None
        Which sections to check for.  If None, uses the legacy default
        (entities, attributes, relations, state_flags, targets).
    """
    warnings: list[str] = []
    if not schema_text:
        warnings.append("empty schema")
        return warnings
    if "<state>" not in schema_text:
        warnings.append("missing <state> tag")
    if "</state>" not in schema_text:
        warnings.append("missing </state> tag")

    if required_sections is None:
        required_sections = ["entities", "attributes", "relations", "state_flags", "targets"]
    for section in required_sections:
        if f"<{section}>" not in schema_text:
            warnings.append(f"missing <{section}> section")

    n = count_entities(schema_text)
    if n == 0:
        warnings.append("no entities found")
    return warnings


# ── Evidence / answer extraction ──────────────────────────────────────

_EVIDENCE_RE = re.compile(r"<evidence>(.*?)(?=<\w|</state>)", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)(?=<\w|</state>)", re.DOTALL)
_ANSWER_FIELD_RE = re.compile(r"^answer=(.+)$", re.MULTILINE)
_GROUNDING_RE = re.compile(r"^grounding=\[([^\]]*)\]", re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^confidence=(.+)$", re.MULTILINE)


def parse_evidence_from_schema(schema_text: str) -> str | None:
    """Extract the raw <evidence> section content from a schema string."""
    m = _EVIDENCE_RE.search(schema_text)
    return m.group(1).strip() if m else None


def parse_answer_from_schema(schema_text: str) -> str | None:
    """Extract the answer value from the <answer> section."""
    block = _ANSWER_BLOCK_RE.search(schema_text)
    if not block:
        return None
    m = _ANSWER_FIELD_RE.search(block.group(1))
    return m.group(1).strip() if m else None


def parse_answer_block(schema_text: str) -> dict[str, str | None]:
    """Extract the full answer block: answer, grounding, confidence."""
    block = _ANSWER_BLOCK_RE.search(schema_text)
    if not block:
        return {"answer": None, "grounding": None, "confidence": None}
    content = block.group(1)

    answer_m = _ANSWER_FIELD_RE.search(content)
    grounding_m = _GROUNDING_RE.search(content)
    confidence_m = _CONFIDENCE_RE.search(content)

    return {
        "answer": answer_m.group(1).strip() if answer_m else None,
        "grounding": grounding_m.group(1).strip() if grounding_m else None,
        "confidence": confidence_m.group(1).strip() if confidence_m else None,
    }
