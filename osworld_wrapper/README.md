# `osworld_wrapper`

OSWorld-specific visual grounding: VLM adapter, OmniParser-v2 glue, and a
tool registry — all in one place.

This package centralises everything that depends on
[xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) (`pyautogui`
actions, OS-level accessibility trees, terminal context) so the
cross-domain code in `vlm_wrapper/` can stay environment-agnostic.

> **TL;DR** — give it an OSWorld observation dict (or just a desktop
> screenshot) and get the canonical `<state>…</state>` schema back, via
> the vision-LLM head or the local OmniParser-v2 head.

---

## What's inside

| File | What it does |
|------|--------------|
| `__init__.py`    | Re-exports the public API (`generate_label`, `osworld_obs_to_schema`, `grounding_osworld_obs_to_schema`, `build_osworld_registry`). |
| `adapter.py`     | Vision head: `generate_label(image, …)` calls a VLM (GPT-4o by default) and returns the `<state>` schema for the `desktop` domain. Also exposes `osworld_obs_to_schema(obs, …)` which unpacks an OSWorld `obs` dict (`screenshot`, `accessibility_tree`, `instruction`, `terminal`, …). |
| `grounding.py`   | OmniParser-v2 head: `grounding_osworld_obs_to_schema(obs, …)` runs YOLO + OCR + Florence-2 locally (delegates to `browsergym_wrapper.grounding.grounding_image_to_schema(domain="desktop", …)` since the OmniParser pipeline is domain-agnostic). |
| `tools.py`       | Tool-calling registry for multi-turn visual reasoning over the OS accessibility tree (`query_os_element`, `get_state_flags`). Build with `build_osworld_registry(a11y_tree_xml=…, instruction=…, terminal_output=…)`. |

The legacy modules under `vlm_wrapper/` (`osworld_adapter.py`, the
OSWorld parts of `grounding_browsergym.py` and `tools_browser.py`) are
now thin compatibility shims that re-export from this package, so
existing imports keep working.

---

## Why no AXTree-based heuristic head?

OSWorld accessibility trees (AT-SPI / UI Automation XML) are noisy and
inconsistent across applications, so we deliberately do **not** ship a
text-only deterministic head for `desktop`. The default cascade for
`desktop` is `omniparser → vlm → tool_loop` (see
`vlm_wrapper.ground._ESCALATION_CHAINS`); the AXTree XML is still
available to the tool loop and to the VLM as grounding context.

---

## Quick start

### Head 2 — Vision (screenshot → VLM → schema)

```python
from osworld_wrapper import osworld_obs_to_schema

result = osworld_obs_to_schema(
    obs,                                    # OSWorld observation dict
    step=3,
    task_id="osworld.install-spotify",
    model="gpt-4o",                         # or rely on $VLM_LABEL_MODEL
    api_key="sk-...",                       # or rely on the OpenAI client default
)
print(result["schema"])
```

### Head 3 — OmniParser-v2 (screenshot → local models → schema)

```python
from osworld_wrapper import grounding_osworld_obs_to_schema

result = grounding_osworld_obs_to_schema(
    obs, step=1, task_id="install-spotify",
)
print(result["schema"])
```

### Tool-calling registry (multi-turn grounding)

```python
from osworld_wrapper import build_osworld_registry
from vlm_wrapper.tool_loop import run_tool_loop

registry = build_osworld_registry(
    a11y_tree_xml=obs["accessibility_tree"],
    instruction=obs["instruction"],
    terminal_output=obs.get("terminal", ""),
)
# `registry` plugs into `run_tool_loop` exactly like the BrowserGym one.
```
