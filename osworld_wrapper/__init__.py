"""OSWorld-specific visual grounding, VLM adapter, OmniParser glue, and tools.

Code that only concerns [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)
(``pyautogui``-style actions, OS-level accessibility trees, terminal
context) lives here so the cross-domain code in :mod:`vlm_wrapper` can
stay environment-agnostic.

**Two heads** (no AXTree-based heuristic — see the wrapper-level README
for why)

- :func:`osworld_wrapper.adapter.generate_label` /
  :func:`osworld_wrapper.adapter.osworld_obs_to_schema` — desktop
  screenshot → vision LLM → ``<state>`` schema (Head 2; the canonical
  default for ``cascaded_ground(domain="desktop")``).
- :func:`osworld_wrapper.grounding.grounding_osworld_obs_to_schema` —
  desktop screenshot → OmniParser-v2 → ``<state>`` schema (Head 3;
  delegates to :mod:`browsergym_wrapper.grounding` with
  ``domain="desktop"`` since the OmniParser pipeline is domain-agnostic).

**Tools** — :func:`osworld_wrapper.tools.build_osworld_registry`
exposes a :class:`vlm_wrapper.tools.ToolRegistry` of OS-level
accessibility-tree helpers (``query_os_element``,
``get_state_flags``) for ``vlm_wrapper.tool_loop`` to use during
multi-turn grounding.

The legacy modules ``vlm_wrapper.osworld_adapter`` and the OSWorld parts
of ``vlm_wrapper.tools_browser`` / ``vlm_wrapper.grounding_browsergym``
remain as thin re-export shims, so existing imports keep working.
"""

from __future__ import annotations

from osworld_wrapper.adapter import (
    generate_label,
    osworld_obs_to_schema,
)
from osworld_wrapper.tools import build_osworld_registry

__all__ = [
    "generate_label",
    "osworld_obs_to_schema",
    "build_osworld_registry",
]

# OmniParser-backed grounding pulls in the heavyweight ``vlm_wrapper.grounding``
# (torch / transformers / OmniParser-v2). Make it optional so a thin install
# without the vision extras can still use the API-only Head 2.
try:
    from osworld_wrapper.grounding import grounding_osworld_obs_to_schema

    __all__ += ["grounding_osworld_obs_to_schema"]
except ImportError:
    pass
