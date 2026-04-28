# `osworld_wrapper`

OSWorld-specific adapters that turn a `DesktopEnv` observation into the
canonical `<state>…</state>` schema. Three heads (XML walker, vision LLM,
OmniParser-v2) plus a tool-calling registry, all in one place so the
cross-domain code in `vlm_wrapper/` can stay environment-agnostic.

This package is the **desktop** counterpart to `browsergym_wrapper`
(web) and `gymv_wrapper` (text adventures). It targets
[xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) — `pyautogui`
actions, OS-level AT-SPI / UI-Automation accessibility trees, terminal
context.

> **TL;DR** — give it an OSWorld observation dict (or just a desktop
> screenshot) and get back the canonical schema. Pick the head you can
> afford: free deterministic XML walker, GPT-4o-class vision LLM, or
> local OmniParser-v2.

---

## Contents

| File | What it does |
|------|--------------|
| `__init__.py`    | Re-exports the public API. |
| `heuristic.py`   | **Head 1.** `obs_to_schema(obs, …)` / `xml_to_schema(xml, …)` — deterministic walker over the namespaced AT-SPI / UI-Automation XML accessibility tree. Free, no LLM. Verified against the real Ubuntu VM (25 entities, ~6 ms). |
| `adapter.py`     | **Head 2.** `generate_label(image, …)` calls a vision LLM (GPT-4o by default) and returns the `<state>` schema for the `desktop` domain. `osworld_obs_to_schema(obs, …)` unpacks an OSWorld `obs` dict (`screenshot`, `accessibility_tree`, `instruction`, `terminal`). |
| `grounding.py`   | **Head 3.** `grounding_osworld_obs_to_schema(obs, …)` runs OmniParser-v2 (YOLO + OCR + Florence-2) locally. Delegates to `browsergym_wrapper.grounding.grounding_image_to_schema(domain="desktop", …)` since the OmniParser pipeline is domain-agnostic. |
| `tools.py`       | Tool registry for multi-turn grounding over the AT-SPI tree (`query_os_element`, `query_entity_pos`, `get_state_flags`). Build with `build_osworld_registry(a11y_tree_xml=…, instruction=…, terminal_output=…)`. Uses `xml.etree.ElementTree` so it handles real namespaced XML (`cp:screencoord`, `st:visible`, …). |

The legacy modules under `vlm_wrapper/` (`osworld_adapter.py`, the
OSWorld parts of `tools_browser.py` / `grounding_browsergym.py`) are now
thin compatibility shims that re-export from this package, so existing
imports keep working.

---

## Install

OSWorld has two layers: the Python SDK (`desktop_env`) and the VM
backend (Docker container running an Ubuntu KVM guest).

### 1. Python env — clone + conda

From the repo root:

```bash
bash install/install_osworld.sh
```

This creates the `osworld` conda env from
`install/osworld.environment.yml`, clones `xlang-ai/OSWorld` to
`/fs/gamma-projects/vlm-robot/OSWorld` (override with the first
positional arg), `pip install -e`'s it, and runs `osworld_smoke.py` to
confirm `desktop_env`, `DesktopEnv`, `docker`, `pyautogui` etc. are all
importable.

### 2. VM backend — Docker + qcow2

You need two binary blobs: a tiny coordinator image and a fat Ubuntu
disk image. Roughly **35 GB free** is needed during install (12 GB zip
+ 23 GB unzipped); the zip can be deleted after extraction.

```bash
docker pull happysixd/osworld-docker                   # ~360 MB
mkdir -p docker_vm_data && cd docker_vm_data
curl -L -C - -o Ubuntu.qcow2.zip \
    https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip -o Ubuntu.qcow2.zip                              # produces Ubuntu.qcow2 (23 GB)
rm Ubuntu.qcow2.zip                                    # reclaim 12 GB
```

**Where the qcow2 must live.** OSWorld's
`desktop_env.providers.docker.manager.VMS_DIR` is `./docker_vm_data`
*relative to your current working directory*, so always launch from
the repo root (or `cd` to wherever the qcow2 is).

If you skip the manual download, the first `OSWorldGymWrapper.reset()`
will fall through to `DockerVMManager.get_vm_path()` and pull the zip
itself — it just gives you a much louder progress bar to watch.

### 3. KVM acceleration (recommended)

The Docker provider auto-detects `/dev/kvm`. With KVM the cold-boot
path is **~28 s** on a modern CPU; without it, expect 5-10× slower.
Make sure the host kernel has KVM modules loaded and `/dev/kvm` is
world-readable inside the container.

### 4. Smoke test the whole stack

```bash
conda activate osworld
python install/osworld_smoke.py        # imports only, no VM boot
```

For an end-to-end render test see
[Render the live VM](#render-the-live-vm) below.

---

## Heads

### Head 1 — Heuristic (XML walker, no LLM)

```python
from osworld_wrapper import obs_to_schema   # alias for heuristic.obs_to_schema

result = obs_to_schema(
    obs,                                   # OSWorld observation dict
    step=3,
    task_id="osworld.install-spotify",
    max_entities=25,
)
print(result)                              # str  — the canonical <state> block
```

Walks the namespaced XML returned by
`controller.get_accessibility_tree()`, classifies each node into
`window` / `container` / `control` / `text` / `element`, lifts roles
from element tags (`<push-button>` → `push-button`), bboxes from
`cp:screencoord` + `cp:size`, and boolean states from `st:*="true"`.
Handles all three OS namespace flavours (Ubuntu / Windows / macOS) by
mapping known namespaces to a single canonical prefix.

Real Ubuntu output (full a11y tree, 151 KB XML):

```
e1[type=window, label=@!0,0;BDHF, pos=0,0,1920,1080, role=frame]
e3[type=control, label=Show Applications, role=toggle-button]
e14[type=control, label=Google Chrome, pos=0,33,70,64, role=push-button]
e16[type=control, label=Visual Studio Code, pos=0,169,70,64, role=push-button]
e17[type=control, label=VLC media player, pos=0,237,70,64, role=push-button]
…
```

### Head 2 — Vision (screenshot → VLM → schema)

```python
from osworld_wrapper import osworld_obs_to_schema

result = osworld_obs_to_schema(
    obs,
    step=3,
    task_id="osworld.install-spotify",
    model="gpt-4o",                        # or rely on $VLM_LABEL_MODEL
    api_key="sk-...",                      # or rely on the OpenAI client default
)
print(result["schema"])
```

### Head 3 — OmniParser-v2 (local CV stack)

```python
from osworld_wrapper import grounding_osworld_obs_to_schema

result = grounding_osworld_obs_to_schema(obs, step=1, task_id="install-spotify")
print(result["schema"])
```

Optional — only available if the heavyweight `vlm_wrapper.grounding`
extras (torch / transformers / OmniParser-v2) are installed. The
`__init__` swallows the `ImportError` so a thin install can still use
Head 1 and Head 2.

### Tool registry — multi-turn grounding via AT-SPI

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

Exposes `query_os_element(name, role=…)`, `query_entity_pos(name)`,
`get_state_flags(name)`. Backed by `xml.etree.ElementTree` over the
namespaced XML — handles `cp:screencoord`, `cp:size`, `st:visible`,
`st:enabled`, etc. without any regex hacks.

---

## Render the live VM

End-to-end check that the Docker provider boots, the wrapper decodes
the screenshot, and a real schema comes out. From the repo root with
the `osworld` env active:

```python
from env_wrappers.osworld_wrapper import OSWorldGymWrapper

DEFAULT_TASK = {
    "id": "smoke_idle",
    "instruction": "Look at the desktop. Identify the visible applications.",
    "config": [],
    "evaluator": {"func": "exact_match",
                  "result":   {"type": "rule", "rules": {"expected": "true"}},
                  "expected": {"type": "rule", "rules": {"expected": "true"}}},
    "proxy": False, "fixed_ip": False,
}

env = OSWorldGymWrapper(
    provider_name="docker", headless=True, max_steps=2,
    require_a11y_tree=True, require_terminal=True,
    screen_size=(1280, 800), task_catalog=[DEFAULT_TASK],
)
obs, info = env.reset()
print(obs["screenshot"].shape)              # (1080, 1920, 3) uint8
print(len(obs["accessibility_tree"]))       # ~150_000 chars of namespaced XML
env.close()
```

Or use the catalogued entry-points (boot the VM themselves, save
PNG + a11y XML + JSONL records under
`visual_grounding_tests/output/osworld_{text,image}/`):

```bash
# Heuristic + text-LLM (XML in prompt, no image)
python visual_grounding_tests/generate_osworld_text_schema.py \
    --task_catalog /path/to/OSWorld/evaluation_examples/test_small.json \
    --task_limit 1 --provider docker --max_steps 1 -v

# Heuristic + image-LLM (screenshot in prompt, optional OmniParser)
python visual_grounding_tests/generate_osworld_image_schema.py \
    --task_catalog /path/to/OSWorld/evaluation_examples/test_small.json \
    --task_limit 1 --provider docker --max_steps 1 -v
```

Both scripts honour `--synthetic --dry_run` for offline iteration when
you don't want to pay the boot cost.

### Live measurements (Ubuntu 22 guest, 1920×1080, KVM on)

| Stage | Time | Output |
|---|---:|---|
| `docker pull happysixd/osworld-docker` | once, ~5 s | 359 MB image |
| `Ubuntu.qcow2.zip` download (HF, 50 MB/s) | once, ~3.5 min | 12 GB zip → 23 GB qcow2 |
| `OSWorldGymWrapper(provider="docker").reset()` | ~28 s | first observation |
| `obs["screenshot"]` | — | `(1080, 1920, 3)` uint8 |
| `obs["accessibility_tree"]` | — | ~150 KB namespaced AT-SPI XML |
| Heuristic head (Head 1) | ~6 ms | 25 entities |
| Image-LLM head, `gpt-4.1` (Head 2) | ~7.5 s | 11 entities |
| Text-LLM head, `gpt-4.1` (XML in prompt) | ~8.5 s | 10 entities |
| OmniParser head (Head 3) | ~1-3 s on GPU | varies |

---

## Why expose all three heads?

They produce **complementary** entity sets, which is exactly what
`vlm_wrapper.cascaded_ground` wants for cross-validation:

- The **heuristic** enumerates everything the AT-SPI tree exposes
  (`Google Chrome`, `Visual Studio Code`, `VLC media player`, `Trash`,
  `System` menu) with exact pixel rects. Cheapest, but only knows what
  the OS itself advertises — invisible-but-rendered overlays often
  leak through, and elements drawn by non-AT-SPI surfaces (Electron
  apps, custom canvases, games) are missing.
- The **image-LLM** sees what a human would (`Activities`, `Chrome
  icon`, `Top bar date/time`, `Home shortcut`) but tends to use
  screenshot-style names that don't match the accessible names.
- The **OmniParser** head is reproducible, GPU-friendly, and produces
  bounding boxes you can trust even when no a11y tree is available
  (e.g. games, screen recordings).

The default cascade for `desktop` in
`vlm_wrapper.ground._ESCALATION_CHAINS` is
`heuristic → omniparser → vlm → tool_loop`; the AXTree XML is also
available to the tool loop and to the vision LLM as grounding context.
