"""BrowserGym / OSWorld grounding adapter: screenshot → OmniParser-v2 → schema.

Head 3 adapter for browser and desktop environments.  Takes a screenshot
(from BrowserGym or OSWorld), runs OmniParser-v2 to detect UI elements,
and maps the results into the canonical ``<state>…</state>`` schema.

This adapter produces the same schema format as:
  - Head 1 (browser_heuristic.py) — from AXTree
  - Head 2 (browser_adapter.py)   — from GPT-4o vision

but without requiring an AXTree, DOM, or API calls.  It's the **local
vision-only** option.

Usage::

    from vlm_wrapper.grounding_browsergym import grounding_obs_to_schema

    # From a BrowserGym observation dict:
    schema_str = grounding_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")

    # From a raw screenshot:
    from vlm_wrapper.grounding_browsergym import grounding_image_to_schema
    schema_str = grounding_image_to_schema(
        pil_image, goal="Find cheapest laptop", step=2,
    )

    # For OSWorld observations:
    from vlm_wrapper.grounding_browsergym import grounding_osworld_obs_to_schema
    schema_str = grounding_osworld_obs_to_schema(
        osworld_obs, step=1, task_id="install-spotify",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .grounding import BBox, ScreenElement, parse_screen

logger = logging.getLogger(__name__)


# ── Element → entity mapping ─────────────────────────────────────────

def _element_to_entity_type(el: ScreenElement) -> str:
    """Map a ScreenElement type to the schema entity type."""
    if el.element_type == "text":
        return "text"
    if el.interactable:
        return "element"
    return "region"


def _infer_role(el: ScreenElement) -> str:
    """Infer a UI role from the element label + type."""
    label = el.label.lower()
    if el.element_type == "text":
        return "text"
    for keyword, role in [
        ("button", "button"), ("btn", "button"),
        ("search", "searchbox"), ("input", "textbox"),
        ("link", "link"), ("menu", "menu"), ("tab", "tab"),
        ("checkbox", "checkbox"), ("check", "checkbox"),
        ("radio", "radio"), ("slider", "slider"),
        ("dropdown", "combobox"), ("select", "combobox"),
        ("icon", "icon"), ("image", "image"), ("logo", "image"),
        ("close", "button"), ("submit", "button"),
        ("arrow", "icon"), ("nav", "navigation"),
    ]:
        if keyword in label:
            return role
    return "icon" if el.interactable else "region"


def _suggest_actions_from_elements(
    elements: List[ScreenElement],
    target_idx: int,
) -> List[str]:
    """Generate plausible next actions from detected elements."""
    actions: List[str] = []
    if target_idx >= 0 and target_idx < len(elements):
        te = elements[target_idx]
        cx, cy = te.bbox.center
        role = _infer_role(te)
        if role in ("textbox", "searchbox", "combobox"):
            actions.append(f'click({cx}, {cy}) then type "..."')
        else:
            actions.append(f"click({cx}, {cy})")

    clickable = [
        (i, el) for i, el in enumerate(elements)
        if el.interactable and i != target_idx
    ]
    for i, el in clickable[:4]:
        cx, cy = el.bbox.center
        actions.append(f"click({cx}, {cy})")

    if not actions:
        actions.append("scroll(down)")
    return actions[:5]


# ── Schema builders ───────────────────────────────────────────────────

def _elements_to_schema(
    elements: List[ScreenElement],
    *,
    domain: str = "browser",
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    url: str = "",
    max_entities: int = 25,
) -> str:
    """Convert a list of ScreenElements into the canonical <state> schema."""
    capped = elements[:max_entities]

    # Pick target: interactable element whose label best matches the goal
    target_idx = -1
    blocker_idx = -1
    if goal:
        goal_words = set(goal.lower().split())
        best_score = 0
        for i, el in enumerate(capped):
            if not el.interactable:
                continue
            label_words = set(el.label.lower().split())
            overlap = len(goal_words & label_words)
            if overlap > best_score:
                best_score = overlap
                target_idx = i
    if target_idx < 0:
        for i, el in enumerate(capped):
            if el.interactable:
                target_idx = i
                break

    dialog_open = any("dialog" in el.label.lower() or "modal" in el.label.lower() for el in capped)
    input_pending = any(
        _infer_role(el) in ("textbox", "searchbox", "combobox") for el in capped
    )

    lines: List[str] = ["<state>"]
    lines.append(f"domain={domain}")
    lines.append(f"task={task_id}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")

    # Entities
    lines.append("<entities>")
    for i, el in enumerate(capped):
        eid = f"e{i + 1}"
        etype = _element_to_entity_type(el)
        role = _infer_role(el)
        pos = f"{el.bbox.x},{el.bbox.y},{el.bbox.w},{el.bbox.h}"
        parts = [f"type={etype}", f"label={role} '{el.label}'", f"pos={pos}"]
        if el.source:
            parts.append(f"src={el.source}")
        lines.append(f"{eid}[{', '.join(parts)}]")
    lines.append("")

    # Attributes
    lines.append("<attributes>")
    for i, el in enumerate(capped):
        eid = f"e{i + 1}"
        states = ["visible"]
        if el.interactable:
            states.append("clickable")
        lines.append(f"{eid}.state={','.join(states)}")
        if el.element_type == "text":
            lines.append(f"{eid}.value={el.label}")
    lines.append("")

    # Relations: adjacency based on spatial proximity
    lines.append("<relations>")
    for i, ei in enumerate(capped):
        for j, ej in enumerate(capped):
            if j <= i:
                continue
            ci = ei.bbox.center
            cj = ej.bbox.center
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist < 80:
                lines.append(f"adjacent(e{i+1},e{j+1})")
    lines.append("")

    # State flags
    lines.append("<state_flags>")
    lines.append("progress=null")
    lines.append("phase=null")
    lines.append("error=null")
    lines.append(f"dialog_open={'true' if dialog_open else 'false'}")
    lines.append(f"input_pending={'true' if input_pending else 'false'}")
    if url:
        lines.append(f"url={url}")
    lines.append(f"grounding_model=omniparser-v2")
    lines.append(f"total_detections={len(elements)}")
    lines.append("")

    # Targets
    lines.append("<targets>")
    target_eid = f"e{target_idx + 1}" if target_idx >= 0 else "null"
    blocker_eid = f"e{blocker_idx + 1}" if blocker_idx >= 0 else "null"
    lines.append(f"target={target_eid}")
    lines.append(f"blocker={blocker_eid}")
    clickable_eids = [f"e{i+1}" for i, el in enumerate(capped) if el.interactable][:8]
    lines.append(f"candidate_set=[{','.join(clickable_eids)}]")
    lines.append("")

    # Actions
    actions = _suggest_actions_from_elements(capped, target_idx)
    if actions:
        lines.append("<actions>")
        for ai, act in enumerate(actions, 1):
            lines.append(f"a{ai}={act}")
        lines.append("")

    lines.append("</state>")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────

def grounding_image_to_schema(
    image: Union[Image.Image, np.ndarray],
    *,
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    url: str = "",
    domain: str = "browser",
    max_entities: int = 25,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = False,
    caption_icons: bool = True,
) -> Dict[str, Any]:
    """Run OmniParser-v2 on a screenshot and return the structured schema.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The screenshot to parse.
    goal, task_id, step, url : str/int
        Context for the schema header.
    domain : str
        ``"browser"`` for BrowserGym, ``"desktop"`` for OSWorld.
    max_entities : int
        Cap on entities in the schema output.
    box_threshold, iou_threshold : float
        Detection thresholds forwarded to OmniParser.
    use_paddleocr : bool
        Use PaddleOCR instead of EasyOCR.
    caption_icons : bool
        Whether to run Florence-2 on icon crops.

    Returns
    -------
    dict with keys:
        ``"schema"``   – the ``<state>…</state>`` string
        ``"elements"`` – list of ScreenElement objects
        ``"warnings"`` – list of validation warnings
        ``"model"``    – ``"omniparser-v2"``
    """
    from .schema import validate_schema

    elements = parse_screen(
        image,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        caption_icons=caption_icons,
        max_elements=max_entities * 2,
    )

    schema = _elements_to_schema(
        elements,
        domain=domain,
        goal=goal,
        task_id=task_id,
        step=step,
        url=url,
        max_entities=max_entities,
    )

    warnings = validate_schema(schema)
    return {
        "schema": schema,
        "elements": elements,
        "warnings": warnings,
        "model": "omniparser-v2",
    }


def grounding_obs_to_schema(
    obs: Dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    max_entities: int = 25,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = False,
    caption_icons: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper that unpacks a BrowserGym observation dict.

    Parameters
    ----------
    obs : dict
        BrowserGym observation with ``screenshot``, ``goal``, ``url``, etc.

    Returns
    -------
    dict — same as ``grounding_image_to_schema``.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return {
            "schema": None, "elements": [],
            "warnings": ["no screenshot"], "model": "omniparser-v2",
        }

    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return {
            "schema": None, "elements": [],
            "warnings": ["unknown screenshot type"], "model": "omniparser-v2",
        }

    goal = obs.get("goal", "")
    if not goal:
        goal_obj = obs.get("goal_object", ())
        texts = [m.get("text", "") for m in goal_obj if m.get("type") == "text"]
        goal = " ".join(texts)

    return grounding_image_to_schema(
        image,
        goal=goal,
        task_id=task_id,
        step=step,
        url=obs.get("url", ""),
        domain="browser",
        max_entities=max_entities,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        caption_icons=caption_icons,
    )


def grounding_osworld_obs_to_schema(
    obs: Dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    max_entities: int = 25,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = False,
    caption_icons: bool = True,
) -> Dict[str, Any]:
    """Adapter for OSWorld observations (from OSWorldGymWrapper).

    Parameters
    ----------
    obs : dict
        OSWorld observation with ``screenshot`` (np.ndarray),
        ``instruction``, ``accessibility_tree``, ``terminal``.

    Returns
    -------
    dict — same as ``grounding_image_to_schema``.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return {
            "schema": None, "elements": [],
            "warnings": ["no screenshot"], "model": "omniparser-v2",
        }

    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return {
            "schema": None, "elements": [],
            "warnings": ["unknown screenshot type"], "model": "omniparser-v2",
        }

    goal = obs.get("instruction", "")

    return grounding_image_to_schema(
        image,
        goal=goal,
        task_id=task_id,
        step=step,
        domain="desktop",
        max_entities=max_entities,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        caption_icons=caption_icons,
    )
