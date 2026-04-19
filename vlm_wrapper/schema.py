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
from dataclasses import dataclass, field
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
e1[type={element|object|region|text}, label={short_label}, pos={x,y,w,h or null}, ontology={ontology_type}]
e2[...]
(list every visually distinct entity you can identify — ≤{max_entities})
(ontology must be one of: selectable_entity | interactive_entity | container_entity |
 textual_anchor | navigable_region | tracked_entity | goal_indicator | blocking_entity.
 This is the cross-domain bridge that lets skills transfer between games / browser /
 desktop / image_qa / video — pick the type that best matches the entity's role.)
"""

_SECTION_ATTRIBUTES = """\
<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={value or null}
(one line per entity that has a notable attribute)
"""

_SECTION_AFFORDANCES = """\
<affordances>
e1.affords=[focus, select, inspect]
e2.affords=[open, read]
(per-entity list of *abstract* operators it supports — drawn from:
 focus | approach | inspect | select | open | close | read | track |
 compare | wait_until | toggle | enter_text | navigate_to.
 These are skill-level verbs, NOT the concrete environment action.
 Skip entities that are purely decorative.)
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
scene_type={main_menu|landing_page|form_entry|modal_dialog|loading|results_view|game_play|game_over|video_segment|null}
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
history_anchor={eid of an entity carried over from the previous step, or null}
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
hop1.abstract_op={GROUND|CHECK|RETRIEVE|CONCLUDE}
hop1.tool={tool_name}
hop1.result_ref={e1,e3}
hop1.frame={frame_index or null}
hop1.timestamp={seconds or null}
hop1.confidence={high|medium|low}
hop2.abstract_op={GROUND|CHECK|RETRIEVE|CONCLUDE}
hop2.tool={tool_name}
hop2.result_ref={e2}
hop2.confidence={high|medium|low}
(one group per reasoning hop — record every tool you ACTUALLY called and
which entities its result grounded.  Do NOT invent hops for tools you did
not call.  abstract_op is the inner-MDP verb: GROUND (locate/bind), CHECK
(verify a relation/attribute), RETRIEVE (lookup memory or skill), CONCLUDE
(commit an intermediate result).  For video/temporal tasks include
frame/timestamp.)
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
    "affordances": _SECTION_AFFORDANCES,
    "relations": _SECTION_RELATIONS,
    "state_flags": _SECTION_STATE_FLAGS,
    "targets": _SECTION_TARGETS,
    "uncertainty": _SECTION_UNCERTAINTY,
    "actions": _SECTION_ACTIONS,
    "evidence": _SECTION_EVIDENCE,
    "answer": _SECTION_ANSWER,
}

# Cross-domain entity ontology (PLAN-VISUAL-SKILLS §5).
ONTOLOGY_TYPES: tuple[str, ...] = (
    "selectable_entity",
    "interactive_entity",
    "container_entity",
    "textual_anchor",
    "navigable_region",
    "tracked_entity",
    "goal_indicator",
    "blocking_entity",
)

# Abstract operators recognised in <affordances> (PLAN-VISUAL-SKILLS §3d).
ABSTRACT_OPERATORS: tuple[str, ...] = (
    "focus", "approach", "inspect", "select", "open", "close",
    "read", "track", "compare", "wait_until", "toggle",
    "enter_text", "navigate_to",
)

# Inner-MDP actions used in <evidence> hop.abstract_op
# (PLAN-SKILL-BANK §1.5 / PLAN-ACTION-AGENT §5).
INNER_MDP_OPS: tuple[str, ...] = (
    "GROUND", "CHECK", "RETRIEVE", "CONCLUDE", "VERIFY",
)

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
e1[type={element|object|region|text}, label={short_label}, pos={x,y,w,h or null}, ontology={ontology_type}]
e2[...]
(list every visually distinct entity you can identify — ≤{max_entities})
(ontology must be one of: selectable_entity | interactive_entity | container_entity |
 textual_anchor | navigable_region | tracked_entity | goal_indicator | blocking_entity.)

<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={value or null}
(one line per entity that has a notable attribute)

<affordances>
e1.affords=[focus, select, inspect]
(per-entity list of abstract operators it supports — focus/approach/inspect/
 select/open/close/read/track/compare/wait_until/toggle/enter_text/navigate_to.)

<relations>
contains(eA,eB)
adjacent(eA,eB)
blocks(eA,eB)
grouped(eA,eB,eC)
(spatial and semantic relations you observe)

<state_flags>
progress={0.0–1.0 or null}
phase={early|mid|late|null}
scene_type={main_menu|landing_page|form_entry|modal_dialog|loading|results_view|game_play|game_over|video_segment|null}
error={description or null}
dialog_open={true|false}
input_pending={true|false}

<targets>
target={eid of the most task-relevant entity}
blocker={eid or null}
constraint={short constraint or null}
candidate_set=[eid,eid,...]
history_anchor={eid carried over from previous step, or null}

<uncertainty>
{eid}.{field}={high|medium|low}
(only for entities where you are unsure)

<actions>
a1={action_string}
a2={action_string}
(top 3–5 plausible next actions — copy verbatim from the env's valid action list)
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
            "entities", "attributes", "affordances", "relations",
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


# ── Semantic validator (PLAN-VISUAL-GROUNDING §12 Layer 1) ────────────
#
# The legacy ``validate_schema`` only checks that required *tags* are
# present.  The semantic validator goes further: it inspects the *content*
# of each section and decides whether the schema is strong enough for the
# reasoning layer to consume, or whether grounding should escalate to a
# stronger head / tool repair (Path B) / offline teacher (Path C).

_SECTION_RE = re.compile(
    r"<(?P<name>entities|attributes|affordances|relations|state_flags|targets|"
    r"uncertainty|actions|evidence|answer)>(?P<body>.*?)"
    r"(?=<\w+>|</state>)",
    re.DOTALL,
)

_ENTITY_LINE_RE = re.compile(
    r"^(e\d+)\s*\[(.*?)\]\s*$",
    re.MULTILINE,
)

_TARGET_FIELD_RE = re.compile(r"^target\s*=\s*(.+?)\s*$", re.MULTILINE)
_BLOCKER_FIELD_RE = re.compile(r"^blocker\s*=\s*(.+?)\s*$", re.MULTILINE)
_CONSTRAINT_FIELD_RE = re.compile(r"^constraint\s*=\s*(.+?)\s*$", re.MULTILINE)
_CANDIDATE_SET_RE = re.compile(r"^candidate_set\s*=\s*\[([^\]]*)\]", re.MULTILINE)
_HISTORY_ANCHOR_RE = re.compile(r"^history_anchor\s*=\s*(.+?)\s*$", re.MULTILINE)

# Per-entity inline fields inside ``e1[...]`` (ontology key=value pairs).
_ENTITY_FIELD_RE = re.compile(r"(\w+)\s*=\s*([^,\]]+)")

_ONTOLOGY_FIELD_RE = re.compile(r"ontology\s*=\s*([a-zA-Z_]+)")
_AFFORDS_LINE_RE = re.compile(
    r"^(e\d+)\.affords\s*=\s*\[([^\]]*)\]\s*$",
    re.MULTILINE,
)
_HOP_ABSTRACT_OP_RE = re.compile(
    r"^hop(\d+)\.abstract_op\s*=\s*([A-Z_]+)\s*$",
    re.MULTILINE,
)
_HOP_TOOL_RE = re.compile(
    r"^hop(\d+)\.tool\s*=\s*([\w./-]+)\s*$",
    re.MULTILINE,
)
_HOP_RESULT_REF_RE = re.compile(
    r"^hop(\d+)\.result_ref\s*=\s*\{?([^}\n]*)\}?\s*$",
    re.MULTILINE,
)
_SCENE_TYPE_RE = re.compile(r"^scene_type\s*=\s*([\w_]+)\s*$", re.MULTILINE)

_UNCERTAINTY_LINE_RE = re.compile(
    r"^(e\d+)\.(\w+)\s*=\s*(high|medium|low)\s*$",
    re.MULTILINE,
)

_RELATION_LINE_RE = re.compile(
    r"^(\w+)\(\s*(e\d+(?:\s*,\s*e\d+)*)\s*\)\s*$",
    re.MULTILINE,
)

_POS_FIELD_RE = re.compile(
    r"pos\s*=\s*(?P<val>null|\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)",
)

_EID_REF_RE = re.compile(r"\be(\d+)\b")

# Per-domain minimum entity counts (PLAN-VISUAL-GROUNDING §12 Layer 1).
_ENTITY_MIN_BY_DOMAIN: dict[str, int] = {
    "gymv": 3,
    "game": 3,
    "browser": 5,
    "desktop": 5,
    "image_qa": 1,
    "video_qa": 1,
    "video": 1,
}

_ENV_DOMAINS: set[str] = {"gymv", "game", "browser", "desktop"}

# Sections that must exist AND have at least one content line (not just
# the tag).  Keyed by the task type inferred from whether ``<actions>`` or
# ``<answer>`` is present in the schema.
_REQUIRED_SECTIONS_CORE: list[str] = [
    "entities", "attributes", "relations", "state_flags", "targets",
]


@dataclass
class ValidationResult:
    """Outcome of ``semantic_validate`` — tells the caller whether the
    schema is consumable and, if not, whether to escalate the grounding
    head (see PLAN-VISUAL-GROUNDING §12 Layer 2 and Milestones §7).

    Attributes
    ----------
    valid : bool
        True iff there are no hard errors.  A ``valid=True`` result can
        still carry warnings.
    warnings : list[str]
        Soft issues (e.g. no relations when ≥3 entities) — the schema is
        usable but the reasoning layer should be cautious.
    errors : list[str]
        Hard failures (missing tag, unresolved entity reference, missing
        ``target`` for env task).  Triggers escalation.
    missing_slots : list[str]
        Names of required slots that are unpopulated, e.g.
        ``["target", "candidate_set"]``.
    escalation_recommended : bool
        True iff the caller should try the next head in the escalation
        chain (PLAN-VISUAL-GROUNDING-MILESTONES §7).
    entity_count : int
        Number of distinct entity IDs parsed from ``<entities>``.
    high_uncertainty_frac : float
        Fraction of entities flagged with ``uncertainty=high`` (0–1).
    """

    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    escalation_recommended: bool = False
    entity_count: int = 0
    high_uncertainty_frac: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "missing_slots": list(self.missing_slots),
            "escalation_recommended": self.escalation_recommended,
            "entity_count": self.entity_count,
            "high_uncertainty_frac": round(self.high_uncertainty_frac, 3),
        }


def _split_sections(schema_text: str) -> dict[str, str]:
    """Parse ``<section>…</section>`` bodies into a name→body dict.

    Bodies are stripped of leading/trailing whitespace.  Missing sections
    are simply absent from the returned dict.
    """
    out: dict[str, str] = {}
    for m in _SECTION_RE.finditer(schema_text):
        out[m.group("name")] = m.group("body").strip()
    return out


def _extract_entity_ids(entities_body: str) -> list[str]:
    """Return the ordered list of entity IDs declared in ``<entities>``."""
    ids = []
    seen: set[str] = set()
    for m in _ENTITY_LINE_RE.finditer(entities_body):
        eid = m.group(1)
        if eid not in seen:
            ids.append(eid)
            seen.add(eid)
    return ids


def _content_lines(body: str) -> list[str]:
    """Lines that look like real content (not empty, not a comment).

    Comments in the schema templates are written as ``(one line per …)``
    — we strip those so placeholder prompt text doesn't count as content.
    """
    lines: list[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("(") and s.endswith(")"):
            continue
        lines.append(s)
    return lines


def _referenced_eids(body: str) -> set[str]:
    """Return all ``eN`` tokens referenced in a section body."""
    return {f"e{n}" for n in _EID_REF_RE.findall(body)}


def semantic_validate(
    schema_text: str | None,
    domain: str = "image_qa",
    *,
    image_size: tuple[int, int] | None = None,
    required_sections: list[str] | None = None,
) -> ValidationResult:
    """Semantic schema validator — implements PLAN-VISUAL-GROUNDING §12 Layer 1.

    Runs seven checks and returns a ``ValidationResult`` summarising
    whether the schema is consumable by the reasoning layer and, if not,
    whether the caller should escalate to the next grounding head
    (PLAN-VISUAL-GROUNDING-MILESTONES §7).

    Parameters
    ----------
    schema_text : str or None
        A full ``<state>…</state>`` block (what ``parse_schema_output``
        returns).  ``None`` or empty strings produce a hard error.
    domain : str
        Task domain — controls the entity-minimum threshold and whether
        ``target`` is required.  One of ``gymv``, ``browser``, ``desktop``,
        ``image_qa``, ``video_qa``, ``video``.
    image_size : (width, height), optional
        If given, ``pos=`` values are checked for being inside the image.
        Otherwise that check is skipped.
    required_sections : list[str], optional
        Override the default list of required sections.  Defaults to
        ``["entities", "attributes", "relations", "state_flags", "targets"]``.

    Checks (see PLAN-VISUAL-GROUNDING-MILESTONES §6)
    ------
    1. Slot population — ``<targets> target=`` is set (not ``null``) for
       env tasks.  Missing target on env task → error, escalate.
    2. Entity minimum — ≥3 (gymv), ≥5 (browser/desktop), ≥1 (image/video).
       Below threshold → error, escalate.
    3. Uncertainty budget — ≤50 % of entities flagged ``uncertainty=high``.
       Over budget → error, escalate.
    4. Section content — every required section has ≥1 content line.
       Empty section → error, escalate.
    5. Relation coverage — ≥1 relation if ≥3 entities present.  Violation
       is a warning, not an error (soft).
    6. Coordinate consistency — ``pos=`` values inside image bounds.
       Out-of-bounds is a warning.
    7. Entity reference integrity — every ``eN`` in relations/targets/
       attributes/uncertainty exists in ``<entities>``.  Unknown eid →
       error, escalate.
    """
    if required_sections is None:
        required_sections = list(_REQUIRED_SECTIONS_CORE)

    result = ValidationResult()

    if not schema_text:
        result.errors.append("empty schema")
        result.valid = False
        result.escalation_recommended = True
        return result

    if "<state>" not in schema_text or "</state>" not in schema_text:
        result.errors.append("missing <state>…</state> delimiters")

    sections = _split_sections(schema_text)
    missing_sections = [s for s in required_sections if s not in sections]
    for name in missing_sections:
        result.errors.append(f"missing <{name}> section")

    entities_body = sections.get("entities", "")
    entity_ids = _extract_entity_ids(entities_body)
    result.entity_count = len(entity_ids)
    known_eids = set(entity_ids)

    # Check 2 — entity minimum (domain-aware)
    entity_min = _ENTITY_MIN_BY_DOMAIN.get(domain, 1)
    if result.entity_count < entity_min:
        result.errors.append(
            f"entity_count={result.entity_count} below minimum "
            f"{entity_min} for domain={domain}"
        )

    # Check 4 — every required section has ≥1 content line
    for name in required_sections:
        body = sections.get(name)
        if body is None:
            continue  # already reported above
        if not _content_lines(body):
            result.errors.append(f"<{name}> section is empty")

    # Check 1 — slot population (target for env tasks)
    targets_body = sections.get("targets", "")
    target_match = _TARGET_FIELD_RE.search(targets_body)
    target_val = target_match.group(1).strip() if target_match else None
    is_missing_target = target_val is None or target_val.lower() in {
        "null", "none", "", "{eid}",
    }
    if domain in _ENV_DOMAINS and is_missing_target:
        result.errors.append("target= unset in <targets> (required for env task)")
        result.missing_slots.append("target")

    # Also surface missing soft slots so callers can react (warnings only
    # unless the skill that runs next actually requires them).
    blocker_match = _BLOCKER_FIELD_RE.search(targets_body)
    if blocker_match is None:
        result.missing_slots.append("blocker")
    constraint_match = _CONSTRAINT_FIELD_RE.search(targets_body)
    if constraint_match is None:
        result.missing_slots.append("constraint")
    candidate_match = _CANDIDATE_SET_RE.search(targets_body)
    if candidate_match is None or not candidate_match.group(1).strip():
        result.missing_slots.append("candidate_set")

    # Check 3 — uncertainty budget
    uncertainty_body = sections.get("uncertainty", "")
    high_count = 0
    for m in _UNCERTAINTY_LINE_RE.finditer(uncertainty_body):
        if m.group(3).lower() == "high":
            high_count += 1
    if result.entity_count > 0:
        frac = high_count / result.entity_count
        result.high_uncertainty_frac = frac
        if frac > 0.5:
            result.errors.append(
                f"{high_count}/{result.entity_count} entities flagged "
                f"uncertainty=high (>50%)"
            )

    # Check 7 — entity reference integrity (hard error)
    cross_ref_sections = ["attributes", "relations", "targets", "uncertainty",
                           "evidence", "answer"]
    for name in cross_ref_sections:
        body = sections.get(name, "")
        if not body:
            continue
        refs = _referenced_eids(body)
        unknown = refs - known_eids
        if unknown:
            result.errors.append(
                f"<{name}> references unknown entity ids: "
                f"{','.join(sorted(unknown))}"
            )

    # Check 5 — relation coverage (soft / warning only)
    relations_body = sections.get("relations", "")
    rel_count = len(_RELATION_LINE_RE.findall(relations_body))
    if result.entity_count >= 3 and rel_count == 0:
        result.warnings.append(
            "no relations declared despite ≥3 entities "
            "(schema may be under-described)"
        )

    # Check 6 — coordinate consistency (soft / warning only)
    if image_size is not None and entities_body:
        img_w, img_h = image_size
        for m in _POS_FIELD_RE.finditer(entities_body):
            val = m.group("val")
            if val == "null":
                continue
            try:
                x, y, w, h = [int(p.strip()) for p in val.split(",")]
            except ValueError:
                continue
            if x < 0 or y < 0 or w <= 0 or h <= 0 \
                    or x + w > img_w or y + h > img_h:
                result.warnings.append(
                    f"pos={val} is outside image bounds "
                    f"({img_w}x{img_h})"
                )

    # ── Skill-context completeness checks (PLAN-SKILL-BANK §3 / ─────
    # PLAN-VISUAL-SKILLS §3-§5).  These produce *warnings*, not errors —
    # the schema is still consumable by the existing reasoning layer, but
    # downstream skill discovery / contract learning will be weaker.

    # Ontology coverage — every entity should declare an ontology type.
    if entities_body:
        ontology_count = 0
        bad_ontology: list[str] = []
        for line in _content_lines(entities_body):
            m = _ENTITY_LINE_RE.match(line)
            if not m:
                continue
            inline = m.group(2)
            o = _ONTOLOGY_FIELD_RE.search(inline)
            if o:
                ontology_count += 1
                val = o.group(1).strip()
                if val not in ONTOLOGY_TYPES and val.lower() != "null":
                    bad_ontology.append(f"{m.group(1)}={val}")
        if ontology_count < result.entity_count:
            result.warnings.append(
                f"only {ontology_count}/{result.entity_count} entities "
                f"declare an ontology= type (skill transfer signal missing)"
            )
        if bad_ontology:
            result.warnings.append(
                "entities with non-canonical ontology types: "
                + ", ".join(bad_ontology[:5])
            )

    # Affordance coverage — interactive/selectable entities should declare
    # at least one abstract operator.
    affordances_body = sections.get("affordances", "")
    afforded_eids: set[str] = set()
    bad_affords: list[str] = []
    for m in _AFFORDS_LINE_RE.finditer(affordances_body):
        afforded_eids.add(m.group(1))
        ops = [o.strip() for o in m.group(2).split(",") if o.strip()]
        for op in ops:
            if op not in ABSTRACT_OPERATORS:
                bad_affords.append(f"{m.group(1)}:{op}")
    if domain in _ENV_DOMAINS and result.entity_count > 0 and not afforded_eids:
        result.warnings.append(
            "<affordances> empty — no abstract operators recorded "
            "(skill applicability scoring will be weak)"
        )
    if bad_affords:
        result.warnings.append(
            "non-canonical abstract operators in <affordances>: "
            + ", ".join(bad_affords[:5])
        )
    unknown_afford_eids = afforded_eids - known_eids
    if unknown_afford_eids:
        result.errors.append(
            "<affordances> references unknown entity ids: "
            + ",".join(sorted(unknown_afford_eids))
        )

    # scene_type — soft signal but very useful for skill retrieval.
    state_flags_body = sections.get("state_flags", "")
    if state_flags_body and not _SCENE_TYPE_RE.search(state_flags_body):
        result.warnings.append(
            "scene_type missing from <state_flags> "
            "(skill retrieval index will be coarser)"
        )

    # history_anchor — only flag for env tasks where multi-step tracking
    # actually matters (gymv / browser / desktop).
    if domain in _ENV_DOMAINS and targets_body \
            and not _HISTORY_ANCHOR_RE.search(targets_body):
        result.warnings.append(
            "history_anchor missing from <targets> "
            "(tracking-family skills cannot bind across steps)"
        )

    # Evidence: every hop with a result_ref should also declare an
    # abstract_op.  Otherwise we cannot mine reasoning-skill protocols.
    evidence_body = sections.get("evidence", "")
    if evidence_body:
        op_hops = {h for h, _ in _HOP_ABSTRACT_OP_RE.findall(evidence_body)}
        result_hops = {h for h, _ in _HOP_RESULT_REF_RE.findall(evidence_body)}
        if result_hops and not op_hops:
            result.warnings.append(
                "<evidence> hops have no abstract_op= "
                "(reasoning-skill protocol cannot be extracted)"
            )
        bad_ops = [
            f"hop{h}={op}" for h, op in _HOP_ABSTRACT_OP_RE.findall(evidence_body)
            if op not in INNER_MDP_OPS
        ]
        if bad_ops:
            result.warnings.append(
                "non-canonical abstract_op in <evidence>: "
                + ", ".join(bad_ops[:5])
            )

    # Final verdict + escalation recommendation.
    result.valid = not result.errors
    result.escalation_recommended = bool(result.errors)
    return result


# ── Tool-trace reconciliation (catches fabricated evidence) ──────────


def reconcile_evidence_with_tool_trace(
    schema_text: str | None,
    tool_trace: list[dict[str, Any]] | None,
) -> list[str]:
    """Cross-check ``<evidence>`` hops against the actual ``tool_trace``.

    The VLM sometimes records a hop with a confident ``result_ref`` even
    when the tool itself returned no detections (e.g. GroundingDINO
    failed, Florence-2 missing ``timm``).  The schema looks fine on its
    own but is not actually grounded — which poisons downstream skill
    contract learning.

    Returns a list of warning strings.  Empty list ⇒ all hops are
    grounded by real tool outputs.

    Parameters
    ----------
    schema_text : str
        Full ``<state>…</state>`` block.
    tool_trace : list[dict]
        Each entry should have at minimum ``"tool"`` (str) and
        ``"result"`` (the raw tool return value, dict/list/str).
        ``None`` or empty trace skips reconciliation entirely.
    """
    warnings: list[str] = []
    if not schema_text or not tool_trace:
        return warnings

    sections = _split_sections(schema_text)
    evidence_body = sections.get("evidence", "")
    if not evidence_body:
        return warnings

    # Build lookup: tool_name -> list of trace records.  ``tool_loop.py``
    # writes records as ``{"call": {"name": ..., "arguments": ...},
    # "result": ..., "reobserved": bool}`` while ad-hoc test traces use
    # ``{"tool": ..., "result": ...}``.  Accept both shapes.
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for rec in tool_trace:
        tool_name = ""
        call = rec.get("call")
        if isinstance(call, dict):
            tool_name = str(call.get("name", "")).strip()
        if not tool_name:
            tool_name = str(rec.get("tool", "")).strip()
        if not tool_name:
            tool_name = str(rec.get("name", "")).strip()
        if not tool_name:
            continue
        by_tool.setdefault(tool_name, []).append(rec)

    # Helper — does a tool record contain ANY positive results?
    def _has_positive_result(rec: dict[str, Any]) -> bool:
        res = rec.get("result")
        if res is None:
            return False
        if isinstance(res, (list, tuple)):
            return len(res) > 0
        if isinstance(res, dict):
            for k in ("elements", "detections", "objects", "matches",
                      "frames", "moments", "boxes", "regions"):
                v = res.get(k)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return True
            count = res.get("count")
            if isinstance(count, int) and count > 0:
                return True
            text = res.get("text")
            if isinstance(text, str) and text.strip():
                return True
            return any(_has_positive_result({"result": v})
                       for v in res.values() if isinstance(v, (list, dict)))
        if isinstance(res, str):
            return bool(res.strip())
        return True

    # Scan each hop.
    hop_tools = dict(_HOP_TOOL_RE.findall(evidence_body))   # {hop_idx: tool}
    hop_refs: dict[str, str] = {
        h: refs for h, refs in _HOP_RESULT_REF_RE.findall(evidence_body)
    }

    for hop_idx, tool_name in hop_tools.items():
        records = by_tool.get(tool_name, [])
        if not records:
            warnings.append(
                f"hop{hop_idx} claims tool={tool_name} but no such call "
                f"appears in the tool_trace (fabricated)"
            )
            continue
        # If hop declares grounded entities but every recorded call to
        # this tool returned nothing, flag it.
        refs = hop_refs.get(hop_idx, "").strip()
        has_refs = bool(re.search(r"\be\d+\b", refs))
        if has_refs and not any(_has_positive_result(r) for r in records):
            warnings.append(
                f"hop{hop_idx} ({tool_name}) declares result_ref={refs} "
                f"but every tool call returned no detections — likely "
                f"fabricated grounding"
            )

    # Catch the inverse too: tools that were called but ignored.
    declared_tools = set(hop_tools.values())
    for tool_name, records in by_tool.items():
        if tool_name in declared_tools:
            continue
        if any(_has_positive_result(r) for r in records):
            warnings.append(
                f"tool {tool_name} was called {len(records)}x with positive "
                f"results but is not referenced in any hop (evidence gap)"
            )

    return warnings
