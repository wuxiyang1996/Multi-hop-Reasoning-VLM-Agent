"""OSWorld tool implementations for multi-turn visual grounding.

Tool handlers that query the OS-level accessibility tree (AT-SPI on
Linux, UI Automation on Windows) to give the VLM ground-truth element
positions, sizes, and state flags. The VLM identifies elements visually,
then calls these tools to convert the visual identification into pixel
coordinates suitable for ``pyautogui.click(x, y)``.

Usage::

    from osworld_wrapper.tools import build_osworld_registry

    registry = build_osworld_registry(
        a11y_tree_xml=obs["accessibility_tree"],
        instruction=obs["instruction"],
        terminal_output=obs.get("terminal", ""),
    )
    # pass registry to tool_loop for multi-turn grounding
"""

from __future__ import annotations

import re
from typing import Any

from vlm_wrapper.tools import (
    TOOL_GET_STATE_FLAGS,
    TOOL_QUERY_ENTITY_POS,
    ToolDef,
    ToolRegistry,
)


# ── OSWorld-specific tool definitions ────────────────────────────────

TOOL_QUERY_OS_ELEMENT = ToolDef(
    name="query_os_element",
    description=(
        "Look up a desktop UI element by name or role from the OS "
        "accessibility tree (AT-SPI on Linux, UI Automation on Windows). "
        "Returns screen coordinates, size, and state flags."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Element name or text to search for.",
            },
            "role": {
                "type": "string",
                "description": "Accessibility role filter (e.g. 'push button', 'text', 'menu item').",
            },
        },
        "required": ["name"],
    },
    domain="osworld",
)


# ── Handler implementation ───────────────────────────────────────────

_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _h_query_os_element(a11y_tree_xml: str, *, name: str, role: str = "") -> dict:
    """Search OS accessibility tree XML for matching elements."""
    name_lower = name.lower()
    role_lower = role.lower() if role else ""

    matches = []
    for line in a11y_tree_xml.splitlines():
        attrs = dict(_ATTR_RE.findall(line))
        el_name = attrs.get("name", "")
        el_role = attrs.get("roleName", attrs.get("role", ""))
        el_text = attrs.get("text", "")

        if name_lower not in el_name.lower() and name_lower not in el_text.lower():
            continue
        if role_lower and role_lower not in el_role.lower():
            continue

        coord = attrs.get("screencoord", "")
        size = attrs.get("size", "")
        entry: dict[str, Any] = {
            "name": el_name,
            "role": el_role,
            "text": el_text[:80] if el_text else None,
        }
        if coord:
            parts = coord.strip("()").split(",")
            if len(parts) == 2:
                entry["x"], entry["y"] = int(parts[0].strip()), int(parts[1].strip())
        if size:
            parts = size.strip("()").split(",")
            if len(parts) == 2:
                entry["width"], entry["height"] = int(parts[0].strip()), int(parts[1].strip())

        states = []
        for flag in ("showing", "visible", "enabled", "editable", "expandable", "checkable"):
            if attrs.get(flag) == "True":
                states.append(flag)
        entry["states"] = states
        matches.append(entry)
        if len(matches) >= 10:
            break

    if not matches:
        return {"found": False, "message": f"No OS element matching name='{name}'"}
    return {"found": True, "matches": matches}


# ── Public: build registry ───────────────────────────────────────────

def build_osworld_registry(
    a11y_tree_xml: str = "",
    instruction: str = "",
    terminal_output: str = "",
) -> ToolRegistry:
    """Create a ToolRegistry with OSWorld tools bound to the current state.

    Parameters
    ----------
    a11y_tree_xml : str
        Raw XML accessibility tree from the OS.
    instruction : str
        Task instruction.
    terminal_output : str
        Latest terminal output.

    Returns
    -------
    ToolRegistry
    """
    reg = ToolRegistry(domain="osworld")

    reg.register(TOOL_QUERY_OS_ELEMENT, lambda **kw: _h_query_os_element(a11y_tree_xml, **kw))

    reg.register(TOOL_QUERY_ENTITY_POS, lambda **kw: _h_query_os_element(a11y_tree_xml, **kw))

    def _os_state_flags(**kw: Any) -> dict:
        return {
            "instruction": instruction,
            "has_terminal": bool(terminal_output),
            "terminal_last_line": terminal_output.strip().splitlines()[-1] if terminal_output.strip() else None,
        }
    reg.register(TOOL_GET_STATE_FLAGS, _os_state_flags)

    return reg


__all__ = [
    "TOOL_QUERY_OS_ELEMENT",
    "build_osworld_registry",
]
