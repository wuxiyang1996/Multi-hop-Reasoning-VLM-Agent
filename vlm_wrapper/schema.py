"""Shared schema definition, system prompts, and image-encoding helpers.

This is the single source of truth for the structured-state schema
(TODO-VLM §12a).  Both the Gym-V and BrowserGym adapters import from here.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

import numpy as np
from PIL import Image

# ── Schema version (bumped when fields change) ───────────────────────
SCHEMA_VERSION = "0.1"

# ── The schema spec that goes into every GPT system prompt ────────────
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


def build_system_prompt(
    domain: str,
    max_entities: int = 20,
) -> str:
    """Build the full system prompt for GPT-4o vision calls."""
    spec = SCHEMA_SPEC.replace("{max_entities}", str(max_entities))
    ctx = GYMV_CONTEXT if domain == "gymv" else BROWSER_CONTEXT
    return f"schema_version={SCHEMA_VERSION}\n\n{spec}\n{ctx}"


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


def validate_schema(schema_text: str) -> list[str]:
    """Quick structural checks.  Returns a list of warnings (empty = OK)."""
    warnings: list[str] = []
    if not schema_text:
        warnings.append("empty schema")
        return warnings
    if "<state>" not in schema_text:
        warnings.append("missing <state> tag")
    if "</state>" not in schema_text:
        warnings.append("missing </state> tag")
    for section in ("entities", "attributes", "relations", "state_flags", "targets"):
        if f"<{section}>" not in schema_text:
            warnings.append(f"missing <{section}> section")
    n = count_entities(schema_text)
    if n == 0:
        warnings.append("no entities found")
    return warnings
