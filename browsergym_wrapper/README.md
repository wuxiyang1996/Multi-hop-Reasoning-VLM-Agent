# `browsergym_wrapper`

BrowserGym-specific visual grounding: heuristics, VLM adapter,
OmniParser-v2 glue, and a tool registry — all in one place.

This package centralises everything that depends on
[ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym) (AXTree,
`extra_element_properties`, `browsergym_id`, set-of-marks, etc.) so the
cross-domain code in `vlm_wrapper/` can stay environment-agnostic.

> **TL;DR** — give it a BrowserGym observation dict (or just a screenshot)
> and get the canonical `<state>…</state>` schema back, via any of the three
> heads (heuristic / vision LLM / local OmniParser-v2).

---

## What's inside

| File | What it does |
|------|--------------|
| `__init__.py`    | Re-exports the public API (`generate_label`, `browser_obs_to_schema`, `obs_to_schema`, `build_browser_registry`, `grounding_image_to_schema`, `grounding_obs_to_schema`, …). |
| `adapter.py`     | Vision head: `generate_label(image, …)` calls a VLM (GPT-4o by default) and returns the `<state>` schema. Also exposes `browser_obs_to_schema(obs, …)` which unpacks a BrowserGym `obs` dict. |
| `heuristic.py`   | Text head: `obs_to_schema(obs, …)` walks the AXTree + `extra_element_properties` deterministically into a schema. Fast, free, no API. |
| `grounding.py`   | OmniParser-v2 head: `grounding_image_to_schema(image, …)` and `grounding_obs_to_schema(obs, …)` run YOLO + OCR + Florence-2 locally (via `vlm_wrapper.grounding`) and emit the schema. Hosts the shared `_elements_to_schema` helper that `osworld_wrapper.grounding` reuses with `domain="desktop"`. |
| `tools.py`       | Tool-calling registry for multi-turn visual reasoning (query bbox, search elements, get page info, get SoM elements, etc.). Build with `build_browser_registry(obs)`. |
| `example.py`     | `python -m browsergym_wrapper.example` — synthetic shopping-page obs + screenshot for end-to-end smoke tests. |

The legacy modules under `vlm_wrapper/` (`browser_adapter.py`,
`browser_heuristic.py`, `grounding_browsergym.py`, `tools_browser.py`,
`example_browsergym.py`) are now thin compatibility shims that re-export
from this package, so existing imports keep working.

---

## Quick start

### Head 1 — Heuristic (AXTree → schema)

```python
from browsergym_wrapper import obs_to_schema

schema_str = obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
```

### Head 2 — Vision (screenshot → VLM → schema)

```python
from browsergym_wrapper import browser_obs_to_schema

result = browser_obs_to_schema(
    obs,
    step=3,
    task_id="webarena.shopping.143",
    model="gpt-4o",                # or rely on $VLM_LABEL_MODEL
    api_key="sk-...",              # or rely on the OpenAI client default
)
print(result["schema"])
```

### Head 3 — OmniParser-v2 (screenshot → local models → schema)

```python
from browsergym_wrapper import grounding_obs_to_schema, grounding_image_to_schema

# From a BrowserGym observation:
result = grounding_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")

# From a raw screenshot:
result = grounding_image_to_schema(pil_image, goal="Find cheapest laptop", step=2)
```

### Tool-calling registry (multi-turn grounding)

```python
from browsergym_wrapper import build_browser_registry
from vlm_wrapper.tool_loop import browser_generate_label_with_tools

registry = build_browser_registry(obs)
result = browser_generate_label_with_tools(
    obs["screenshot"], goal=obs["goal"], task_id="…",
    registry=registry, model="gpt-4o", api_key="sk-...",
)
print(result["schema"])      # <state>…</state>
print(result["tool_trace"])  # [{call, result}, …] — SFT training data
```

---

## Run the synthetic example

No BrowserGym install required — `example.py` builds a fake obs dict
(AXTree + bounding boxes + screenshot) for a shopping page so you can
exercise both Head 1 and Head 2 end-to-end:

```bash
# Head 1 only
python -m browsergym_wrapper.example

# Head 1 + Head 2 (needs OPENAI_API_KEY)
python -m browsergym_wrapper.example --vision

# Use OpenRouter
python -m browsergym_wrapper.example --vision \
    --api-key "$OPENROUTER_KEY" \
    --base-url https://openrouter.ai/api/v1 \
    --model openai/gpt-4.1
```
