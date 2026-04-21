# vlm_wrapper — Multi-hop Visual Reasoning for Games, Browsers, Desktop, Images, and Video

Converts screenshots, video frames, and environment observations into a shared structured text schema (`<state>…</state>`), which plugs into the COS-PLAY pipeline for skill retrieval and decision-making. Supports multi-hop tool-calling reasoning where a VLM gathers grounded evidence from specialised vision models before producing the final schema.

**Full plan:** [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md) • Skill-context schema fields come from [`plans/01-visual-grounding/PLAN-VISUAL-SKILLS.md`](../plans/01-visual-grounding/PLAN-VISUAL-SKILLS.md) and [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](../plans/03-skill-bank/PLAN-SKILL-BANK.md).

---

## Quick setup

`vlm_wrapper` itself runs inside a single **grounding** env that bundles torch / transformers / OmniParser-v2 / OCR / GroundingDINO / decord plus the CLEVR and Video-Holmes benchmark loaders. For **live data collection** from the three interactive runtimes (Gym-V, BrowserGym, OSWorld), each runtime has its own env because their transitive pins conflict — see [`install/INSTALL_BENCHMARKS.md`](../install/INSTALL_BENCHMARKS.md) for the full incompatibility matrix.

### The four conda envs

| Env | YAML | Role | What's in it |
|-----|------|------|--------------|
| `vlm_benchmarks` | [`install/vlm_benchmarks.environment.yml`](../install/vlm_benchmarks.environment.yml) | **Grounding pipeline + offline benchmarks.** Use this to run `cascaded_ground`, Head 1/2/3, the tool loop, CLEVR, Video-Holmes, and all `scripts/` entry points. | torch 2.4.1+cu121, transformers 4.51–4.56, timm, ultralytics, easyocr, decord, supervision, openai, anthropic, google-genai, datasets, Pillow, playwright |
| `gymv` | [`install/gymv.environment.yml`](../install/gymv.environment.yml) | **Gym-V env runtime** — drive [ModalMinds/gym-v](https://github.com/ModalMinds/gym-v) to capture `(frame, obs.text, info)` tuples. | gymnasium ≥1.2.2, gym-v editable install with `[games,spatial]`, textarena, pettingzoo, minigrid, miniworld |
| `browsergym` | [`install/browsergym.environment.yml`](../install/browsergym.environment.yml) | **BrowserGym env runtime** — drive [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym) to capture screenshots + AXTrees. | playwright 1.44, browsergym-core + miniwob + webarena + visualwebarena + assistantbench + experiments, libwebarena, libvisualwebarena |
| `osworld` | [`install/osworld.environment.yml`](../install/osworld.environment.yml) | **OSWorld desktop runtime** — drive [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) to capture screenshots + a11y trees. | gymnasium ~0.28.1, transformers ~4.35.2, desktop-env 1.0.x, docker SDK, pyautogui |

**Why four?** gym-v needs `gymnasium>=1.2.2`; OSWorld hard-pins `gymnasium~=0.28.1` (API-breaking difference); BrowserGym hard-pins `playwright==1.44`; OSWorld hard-pins `transformers~=4.35.2` which conflicts with the `>=4.51` that GroundingDINO in `vlm_benchmarks` requires. No single resolver solution exists.

### Build the grounding env (this is the one you need to use `vlm_wrapper`)

```bash
# Create the unified grounding + benchmarks env (~10–15 min, downloads ~6 GB)
conda env create -f install/vlm_benchmarks.environment.yml
conda activate vlm_benchmarks

# Make `vlm_wrapper` importable without pulling in the training-time vllm deps
pip install -e . --no-deps

# Offline sanity check (no API calls)
pytest vlm_wrapper/tests -q

# Env-wide smoke test (torch, transformers, OmniParser weights, etc.)
python install/vlm_benchmarks_smoke.py

# Live end-to-end parse across all five domains (requires OPENAI_API_KEY)
python scripts/test_vlm_parsers.py --cases gymv browser desktop clevr video_holmes
```

### Build the runtime envs (only if you need to *step* real environments)

One-liner installers clone the upstream repos, create the env, install in editable mode, and run a smoke test. Run only the ones you plan to use:

```bash
# Gym-V — 179 procedurally generated envs
bash install/install_gymv.sh          # creates conda env 'gymv'

# BrowserGym — MiniWoB++ / WebArena / VisualWebArena / AssistantBench
bash install/install_browsergym.sh    # creates conda env 'browsergym'

# OSWorld — desktop tasks over Office / Daily / Professional suites
bash install/install_osworld.sh       # creates conda env 'osworld'
```

Each script honours an optional argument for the clone directory (defaults to `/fs/gamma-projects/vlm-robot/{gym-v,BrowserGym,OSWorld}`). See [`install/INSTALL_BENCHMARKS.md`](../install/INSTALL_BENCHMARKS.md) for VM-backend setup (Docker / VMware / AWS) that OSWorld additionally requires to execute tasks.

### End-to-end data-collection flow across envs

The grounding pipeline never calls `env.step()` itself — it only consumes observation dicts, screenshots, or frame stacks. That's the whole reason the envs can stay isolated: step in the runtime env, dump to disk, parse in the grounding env.

```bash
# ── 1. Collect from Gym-V ────────────────────────────────────────────────
conda activate gymv
python scripts/collect_gymv_rollouts.py \
    --env-id Games/Game2048-v0 --episodes 50 \
    --out data/rollouts/gymv_2048.jsonl           # writes (frame_path, obs_text, info) per step

# ── 2. Collect from BrowserGym ───────────────────────────────────────────
conda activate browsergym
python scripts/collect_browsergym_rollouts.py \
    --task-id miniwob.click-menu --episodes 20 \
    --out data/rollouts/bgym_click_menu.jsonl

# ── 3. Collect from OSWorld (requires Docker/VMware backend) ─────────────
conda activate osworld
python scripts/collect_osworld_rollouts.py \
    --task-id osworld.install-spotify --episodes 5 \
    --out data/rollouts/osw_install_spotify.jsonl

# ── 4. Parse every rollout into <state> schemas (grounding env) ──────────
conda activate vlm_benchmarks
python scripts/label_rollouts.py \
    --inputs data/rollouts/*.jsonl \
    --out data/schemas/
```

The `scripts/collect_*_rollouts.py` data-collection helpers are Phase-0 TODOs in the milestones plan — see [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) §7. While they're pending, you can exercise every part of `vlm_wrapper` from the bundled fixtures under `vlm_wrapper/real_*.png` without any runtime env:

```bash
conda activate vlm_benchmarks
python scripts/test_vlm_parsers.py --cases gymv browser desktop
```

### Legacy minimal env (`vlm_wrapper`)

A slimmer env is shipped at the project root ([`environment.yml`](../environment.yml)) with only the grounding stack — no benchmark loaders, no multi-provider API clients. Use it if you only need Head 1/2/3 on your own image inputs and want a smaller install:

```bash
conda env create -f environment.yml           # env name: vlm_wrapper
conda activate vlm_wrapper
pip install -e . --no-deps
```

Everything documented below works in either `vlm_benchmarks` or `vlm_wrapper`; the CLEVR / Video-Holmes examples only work in `vlm_benchmarks`.

If you just want the vision back-ends on top of an already-existing env, the same stack is exposed as the `[vision]` pyproject extra: `pip install -e ".[vision]"` (pulls `timm`, `easyocr`, `ultralytics`, `opencv-python`, `decord`, `supervision`).

---

## Grounding heads

The **main pipeline is VLM-first**: `cascaded_ground()` drives every domain through the vision-based heads and falls back to the multi-turn tool-calling loop when a schema misses the validator bar.  The `obs.text` / AXTree heuristic still exists as an **opt-in alternative** — useful for text-only smoke tests, regression fixtures, and real-time RL baselines — but it is **NOT** on the default cascade, because letting a regex silently satisfy the schema would mask real VLM grounding bugs.

| | Vision (Head 1, default) | OmniParser-v2 (Head 2, browser/desktop) | Tool Loop (Head 3) | Heuristic (opt-in only) |
|---|---|---|---|---|
| Input | Screenshot / frame pixels | Screenshot (pixels) | Screenshot + tool defs | `obs.text` / AXTree (text) |
| Method | GPT-4o / Qwen3 vision API | YOLO + OCR + Florence-2 (local) | VLM calls tools iteratively | Regex + tree-walking |
| Cost | ~$0.01/call | Free (local GPU) | ~$0.05–0.10/schema | Free |
| Latency | ~1–3 s | ~0.6 s (GPU) | ~5–15 s | <1 ms |
| On default path | ✅ | ✅ (browser/desktop) | ✅ (on demand / image_qa, video_qa) | ❌ — pass `chain=["heuristic", …]` or `--*-head heuristic` |
| Use case | Training-label generation | UI screenshots, precise bbox | Complex reasoning with evidence chains | Real-time RL baselines, offline text tests |

Default chains (`vlm_wrapper.ground._ESCALATION_CHAINS`):

```python
"gymv":     ["vlm", "tool_loop"]
"browser":  ["vlm", "omniparser", "tool_loop"]
"desktop":  ["omniparser", "vlm", "tool_loop"]
"image_qa": ["vlm", "tool_loop"]
"video_qa": ["tool_loop"]
```

Opt-in legacy chains that start with the heuristic are exposed via `vlm_wrapper.ground._HEURISTIC_CHAINS` for callers that want them explicitly.

### Five-domain feasibility snapshot

This table is the quick answer to *"does the inference pipeline work end-to-end for every domain in the milestones plan?"*  Each row is wired into `cascaded_ground()` today, validated by `semantic_validate` with domain-specific rules, and covered by a live test in `vlm_wrapper/tests/test_gpt4o_parsers.py`.

| Domain (plan name) | `cascaded_ground` chain | Tool registry composed | Required schema sections | Entity min | Benchmark data on disk | Live test |
|---|---|---|---|---|---|---|
| `gymv` (Gym-V) | `vlm → tool_loop` | `tools_visual` + `tools_gymv` | `entities, attributes, state_flags, targets, actions` | 3 | N/A — env-provided frames | `test_live_gymv_schema`, `test_live_gymv_tool_loop_schema` |
| `browser` (BrowserGym) | `vlm → omniparser → tool_loop` | `tools_visual` + `tools_browser` | `entities, attributes, state_flags, targets, actions` | 5 | N/A — env-provided obs | `test_live_browser_schema` |
| `desktop` (OSWorld) | `omniparser → vlm → tool_loop` | `tools_visual` + `tools_osworld` | `entities, attributes, state_flags, targets, actions` | 5 | N/A — env-provided obs | `test_live_desktop_schema` |
| `image_qa` (CLEVR) | `vlm → tool_loop` | `tools_visual` (GroundingDINO-preferred) | `entities, attributes, state_flags, targets, evidence, answer` | 1 | `data/CLEVR/CLEVR_v1.0/` ✅ | `test_live_clevr_schema` |
| `video_qa` (Video-Holmes) | `tool_loop` only | `tools_video_visual` (temporal + visual + cross-frame) | `entities, state_flags, targets, evidence, answer` | 1 | `data/Video-Holmes/Benchmark/` ✅ | `test_live_video_holmes_schema` |

**Intentional design choices (not bugs):**

- **No text-heuristic head for `desktop`** — OSWorld a11y trees are noisy and the OmniParser image path is both more reliable and faster.
- **`video_qa` starts at `tool_loop`** — video QA is temporal reasoning; a single-shot VLM over a frame grid wastes context and can't call `sample_frames` / `find_moment` / `track_object`.  The `vlm` head is still reachable via `chain=["vlm", "tool_loop"]` if you want it.
- **`browser` OmniParser fallback is observable** — when `context["obs"]` is missing (e.g. raw screenshot only), `_attempt_omniparser` falls back to image-only grounding and appends an explicit warning to the `GroundingResult.warnings` list so the cascade telemetry records the degraded mode.
- **Head 2 is a single code path** — `_attempt_vlm` for `gymv` / `browser` / `desktop` delegates to the same `generate_label` adapter that data-collection scripts call directly. No prompt drift between "Head 2 inside cascade" and "Head 2 at labeling time".

**Not yet wired (Phase-0 TODOs, not feasibility blockers):**

- GQA and SIV-Bench loaders under `benchmarks/` — follow the `clevr.py` / `video_holmes.py` pattern and they drop straight into the `image_qa` / `video_qa` chains.
- Dual-teacher data collection scripts (`labeling/teachers.py`, `labeling/collect_*.py`).
- Schema-vs-teacher cross-validation harness and ablation-study evaluation harness.

See [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md §7`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) for the canonical status list.

### Adapter vs. `cascaded_ground` — when to use which

There are two shapes of Head-2 entry points.  They now share a single implementation; the difference is the call site and what you get back:

| Entry point | Return type | Good for |
|---|---|---|
| `gymv_adapter.generate_label(image, …)` <br> `browser_adapter.generate_label(image, …)` <br> `osworld_adapter.generate_label(image, …)` | `dict` with `schema / raw / warnings / model / validation` | Data-collection loops that step through an env and log one schema per step, or pytest fixtures that want a deterministic single call. |
| `cascaded_ground(GroundingRequest(...))` | `GroundingResult` with `schema / evidence / tool_trace / validation / head_used / escalation_trace` | Runtime inference where you want the cascade to escalate to OmniParser / the tool loop if the single-shot schema fails validation.  This is the path the decision-agent pipeline calls. |

Internally, `cascaded_ground → _attempt_vlm → generate_label` for `gymv` / `browser` / `desktop`, so there is one prompt + one validator on both paths.

---

## Schema structure

Every head emits the same `<state>…</state>` block. Beyond the classic `<entities>` / `<attributes>` / `<relations>` / `<state_flags>` / `<targets>` / `<uncertainty>` / `<actions>` / `<evidence>` / `<answer>` sections, the schema now carries **skill-context fields** that let downstream skill discovery (see [`PLAN-SKILL-BANK`](../plans/03-skill-bank/PLAN-SKILL-BANK.md)) mine reusable reasoning-and-control programs:

| Field | Where | Purpose |
|---|---|---|
| `ontology=<type>` | inside each `e_i[...]` line in `<entities>` | Cross-domain bridge. One of `selectable_entity`, `interactive_entity`, `container_entity`, `textual_anchor`, `navigable_region`, `tracked_entity`, `goal_indicator`, `blocking_entity`. Lets a skill learned on one env transfer to another. |
| `<affordances>` section | its own block | Per-entity list of abstract operators the entity supports: `focus`, `approach`, `inspect`, `select`, `open`, `close`, `read`, `track`, `compare`, `wait_until`, `toggle`, `enter_text`, `navigate_to`. Drives skill-applicability matching. |
| `hop.abstract_op=` | each hop in `<evidence>` | Inner-MDP verb: `GROUND`, `CHECK`, `RETRIEVE`, `CONCLUDE`, `VERIFY`. Protocol primitives used by skill extraction. |
| `scene_type=` | in `<state_flags>` | Coarse scene tag (`main_menu`, `form_entry`, `modal_dialog`, `results_view`, `game_play`, `video_segment`, …) for skill retrieval. |
| `history_anchor=` | in `<targets>` | Entity ID carried over from the previous step so tracking-family skills can bind across time. |

Canonical value lists are exported from `vlm_wrapper.schema`:

```python
from vlm_wrapper.schema import (
    ONTOLOGY_TYPES, ABSTRACT_OPERATORS, INNER_MDP_OPS,
)
```

Missing skill-context fields produce *warnings* (not errors) from `semantic_validate`, so older schemas remain consumable.

---

## Tool registries

Three tool registries provide structured APIs for multi-hop visual reasoning:

### Visual tools (single frame) — `build_visual_registry(image)`

| Tool | Purpose |
|------|---------|
| `detect_objects` | OmniParser-v2 element detection with bboxes, labels, types, confidence |
| `grounded_detect` | GroundingDINO open-vocabulary detection (natural-image queries) |
| `describe_region` | Florence-2 caption for a specific rectangular crop |
| `zoom_region` | **Option-B re-observation** — crop + upscale and re-feed the image to the VLM on the next turn |
| `visual_search` | Text-query search over detected elements, ranked by relevance |
| `count_objects` | Count elements by type (`icon`/`text`/`all`) or description |
| `classify_scene` | Scene type classification (game, browser, form, dialog, etc.) |
| `spatial_query` | Spatial relations between two elements (distance, direction, overlap, containment) |
| `measure_distance` | Pixel distance between two points with direction |
| `extract_colors` | Dominant colours in a region (K-means clustering) |
| `read_text_region` | OCR on a region with line structure and reading order |

### Video tools (temporal navigation) — `build_video_registry(frames, fps)`

| Tool | Purpose |
|------|---------|
| `get_frame` | Retrieve frame by index or timestamp |
| `sample_frames` | Uniformly sample N frames across a range |
| `compare_frames` | Pixel diff between two frames (quadrant analysis) |
| `detect_scene_changes` | Find scene-change boundaries by visual delta |
| `get_video_info` | Video metadata (frames, duration, FPS, resolution) |
| `read_text_in_frame` | OCR on a specific frame |
| `temporal_navigate` | Move to a different point (absolute, relative, named) |
| `list_valid_actions` | Available navigation actions |

### Cross-frame tools (video + visual) — `build_video_visual_registry(frames, fps)`

Combines all video + visual tools, plus 6 cross-frame tools:

| Tool | Purpose |
|------|---------|
| `track_object` | Track element across frames by label (motion summary, per-frame bbox) |
| `summarize_clip` | Timeline of visual changes across sampled frames |
| `find_moment` | Find frame where a visual event occurs (appears/disappears/changes) |
| `detect_activity` | Classify activity in a frame range (idle, scrolling, navigation, game action) |
| `compare_elements` | Semantic diff: added/removed/moved elements between two frames |
| `detect_objects_at_frame` | Run detection on any frame, not just current |

---

## File layout

```
vlm_wrapper/
├── __init__.py                # exports all heads, tools, registries
├── schema.py                  # shared schema spec, system prompt, image encoding, parsing, validation
│
│  ── Head 1: Heuristic (text-in → schema-out) ──
├── gymv_heuristic.py          # Gym-V: obs.text → schema
├── browser_heuristic.py       # BrowserGym: AXTree/DOM → schema
│
│  ── Head 2: Vision (image-in → VLM API → schema-out) ──
├── gymv_adapter.py            # Gym-V: screenshot → GPT-4o → schema
├── browser_adapter.py         # BrowserGym: screenshot → GPT-4o → schema
├── osworld_adapter.py         # OSWorld: desktop screenshot → GPT-4o → schema
│
│  ── Benchmark loaders + GPT-4o parsers ──
├── benchmarks/
│   ├── clevr.py               # CLEVR v1.0 image-QA → GPT-4o → answer
│   └── video_holmes.py        # Video-Holmes video-QA → GPT-4o → A–F letter
│
│  ── Head 3: OmniParser-v2 Grounding (image-in → local models → schema-out) ──
├── grounding.py               # YOLO icon detector + Florence-2 captioner + OCR → ScreenElement list
├── grounding_browsergym.py    # BrowserGym/OSWorld adapter: ScreenElements → <state> schema
│
│  ── Tool infrastructure ──
├── tools.py                   # ToolDef, ToolRegistry, ToolResult (shared base)
├── tools_gymv.py              # Gym-V tool implementations (list_entities, query_entity_pos, etc.)
├── tools_browser.py           # BrowserGym + OSWorld tool implementations
├── tools_visual.py            # Vision-model-backed tools (detect_objects, spatial_query, etc.)
├── tools_video.py             # Video temporal navigation tools (get_frame, compare_frames, etc.)
├── tools_video_visual.py      # Cross-frame tools (track_object, find_moment, etc.)
│
│  ── Tool-calling loop ──
├── tool_loop.py               # Multi-turn VLM tool-calling loop + convenience wrappers
│
│  ── Examples and demos ──
├── example_browsergym.py      # Synthetic BrowserGym observation for testing
├── example_grounding.py       # Example: OmniParser grounding pipeline
├── demo_visual_grounding.py   # Demo: Mode A (direct), Mode B (VLM+tools), Mode C (comparison)
│
│  ── Other ──
├── test_schema_gen.py         # Schema generation tests
└── PLAN_GROUNDING.md          # Legacy grounding design doc
```

---

## Quick start

See also: [**EXAMPLES.md**](EXAMPLES.md) — fully worked schemas and tool-calling traces for all five domains (gymv, browser, desktop, CLEVR, Video-Holmes).

### Head 1 — Vision (screenshot → VLM → schema)  *(default)*

```python
from vlm_wrapper import gymv_generate_label, browser_obs_to_schema

# Gym-V — pass `valid_actions` so the VLM copies the env's action
# strings verbatim into <actions> instead of inventing English verbs.
result = gymv_generate_label(
    frame, goal="Reach 2048", task_id="Game2048-v0", step=5,
    valid_actions=["[Up]", "[Down]", "[Left]", "[Right]"],
)
print(result["schema"])      # <state>…</state> or None
print(result["warnings"])    # structural + skill-context warnings
print(result["validation"])  # dict: {"valid": bool, "entity_count": int, ...}

# BrowserGym
result = browser_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
```

### Head 2 — OmniParser-v2 Grounding (screenshot → local models → schema)

```python
from vlm_wrapper import parse_screen, parse_screen_annotated
from vlm_wrapper.grounding_browsergym import grounding_image_to_schema

# Raw element detection
elements = parse_screen(pil_image)
for el in elements:
    print(el.label, el.bbox, el.element_type, el.interactable)

# Full schema from screenshot
result = grounding_image_to_schema(pil_image, goal="Find cheapest laptop", step=2)
print(result["schema"])

# Annotated image with detected elements overlaid
elements, annotated_img = parse_screen_annotated(pil_image)
annotated_img.save("annotated.png")
```

### Head 3 — Tool-calling loop (multi-hop visual reasoning)

```python
from vlm_wrapper import visual_generate_label_with_tools, video_visual_generate_label_with_tools

# Single image — VLM calls detect_objects, spatial_query, etc.
result = visual_generate_label_with_tools(
    pil_image, goal="Find the cheapest red jacket", task_id="demo.1",
    model="gpt-4o", api_key="sk-...",
)
print(result["schema"])      # <state>…</state>
print(result["tool_trace"])  # list of {call, result} dicts — SFT training data

# Video — VLM navigates time + detects objects + tracks across frames
result = video_visual_generate_label_with_tools(
    frames, goal="When does the person enter?", fps=2.0,
    model="gpt-4o", api_key="sk-...",
)
print(result["tool_trace"])  # temporal + visual evidence chain
```

### Heuristic (opt-in only — text state → schema)

Useful for text-only regression fixtures, real-time RL baselines, and wiring a new environment before the VLM-first pipeline is tuned.  It is NOT on the default cascade; reach it explicitly:

```python
from vlm_wrapper import gymv_heuristic_schema, browser_heuristic_schema

schema = gymv_heuristic_schema(
    obs_text="| 2 | 4 | 0 | 0 |\n| 0 | 16 | 8 | 0 |",
    description="You are playing 2048. Valid moves: [Up], [Down], [Left], [Right].",
    task_id="Game2048-v0", step=5,
)

schema = browser_heuristic_schema(obs, step=3, task_id="webarena.shopping.143")

# Or as part of a cascade escalation (opt-in legacy chain):
from vlm_wrapper.ground import cascaded_ground, _HEURISTIC_CHAINS, GroundingRequest
result = cascaded_ground(
    GroundingRequest(images=frame, goal=goal, domain="gymv", context={...}),
    chain=_HEURISTIC_CHAINS["gymv"],   # ["heuristic", "vlm", "tool_loop"]
)
```

### OSWorld (desktop screenshots)

Two paths to a ``<state>`` schema. Head 2 (OmniParser) runs locally:

```python
from vlm_wrapper.grounding_browsergym import grounding_osworld_obs_to_schema

result = grounding_osworld_obs_to_schema(osworld_obs, step=1, task_id="install-spotify")
```

Head 1 (GPT-4o vision) — a single-shot adapter parallel to `gymv_adapter` / `browser_adapter`:

```python
from vlm_wrapper import osworld_generate_label, osworld_obs_to_schema

# From a raw screenshot
result = osworld_generate_label(
    pil_screenshot,
    instruction="Install Spotify from the Ubuntu Software store",
    task_id="osworld.install-spotify", step=1,
    a11y_tree_xml=a11y_xml_str,           # optional grounding context
    terminal_output=terminal_tail_str,    # optional grounding context
)

# Or directly from an OSWorldGymWrapper observation dict
result = osworld_obs_to_schema(obs, step=1, task_id="osworld.install-spotify")
```

### Image-QA benchmark — CLEVR

Streams questions from `data/CLEVR/CLEVR_v1.0`, runs each through the `image_qa` tool loop (GPT-4o + visual tools), and returns the predicted answer alongside the ground truth.

```python
from vlm_wrapper.benchmarks.clevr import (
    iter_clevr_samples, parse_clevr_sample, parse_clevr_batch,
)

for sample in iter_clevr_samples(split="val", limit=5):
    out = parse_clevr_sample(sample, model="gpt-4o", api_key=KEY)
    print(sample.question, "→", out["answer"], "gt:", out["ground_truth"])

# Batch run with resumable JSONL output
results = parse_clevr_batch(
    iter_clevr_samples(split="val", limit=100),
    output_jsonl="out/clevr_val_100.jsonl",
    api_key=KEY,
)
```

### Video-QA benchmark — Video-Holmes

Uniformly samples frames from `data/Video-Holmes/Benchmark/videos/videos_cropped/<video_id>.mp4` and routes them through the `video_qa` tool loop. Answers are normalised to the benchmark's A–F letter so they can be scored against `Answer` from `test_Video-Holmes.json`.

```python
from vlm_wrapper.benchmarks.video_holmes import (
    iter_video_holmes_samples, parse_video_holmes_sample,
)

for sample in iter_video_holmes_samples(split="test", limit=3, question_types=["SR"]):
    out = parse_video_holmes_sample(
        sample, num_frames=8, model="gpt-4o", api_key=KEY,
    )
    print(sample.question_type, out["answer"], "gt:", sample.answer,
          "correct:", out["correct"])
```

Requires either `decord` (recommended) or `opencv-python` for video decoding.

### Unified CLI — `scripts/run_vlm_parser.py`

A single entry point for all five domains. Reads `OPENAI_API_KEY` from `.env` when present.

```bash
# One gym-v game frame
python scripts/run_vlm_parser.py gymv \
    --image vlm_wrapper/real_Games_Game2048-v0_step0.png \
    --goal "Reach 2048" --task-id Game2048-v0

# One browser screenshot + AXTree
python scripts/run_vlm_parser.py browser \
    --image screenshot.png --axtree axtree.txt \
    --goal "Find cheapest laptop"

# One OSWorld screenshot
python scripts/run_vlm_parser.py desktop \
    --image desktop.png --goal "Install Spotify"

# List 5 CLEVR val samples (no API call)
python scripts/run_vlm_parser.py clevr --split val --limit 5 --dry-run

# Parse 3 Video-Holmes SR questions, saving JSONL
python scripts/run_vlm_parser.py video_holmes \
    --split test --limit 3 --question-types SR \
    --output out/vh_sr.jsonl
```

### Schema completeness guarantee — validator + cascaded escalation + reconciliation

Implements [`PLAN-VISUAL-GROUNDING` §12 Layers 1 & 2](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md#12-schema-completeness-guarantee-grounding--reasoning-contract). `semantic_validate` goes beyond tag-presence checks — it verifies slot population, entity minima, uncertainty budget, section content, relation coverage, coordinate bounds, entity-reference integrity, **and the skill-context fields above** (ontology coverage, canonical affordance operators, inner-MDP `abstract_op` on every hop, `scene_type`, `history_anchor`). Missing skill-context fields emit warnings so pre-existing callers keep working.

`cascaded_ground` runs the domain's VLM-first escalation chain (e.g. `vlm → tool_loop` for gymv, `vlm → omniparser → tool_loop` for browser) and returns the first schema that passes validation, with `ValidationResult` and `escalation_trace` attached for telemetry. The obs-text / AXTree heuristic is opt-in and not on the default path. The same validation now also runs inside `ground()` and every adapter (`gymv_adapter.generate_label`, `browser_adapter.generate_label`, `osworld_adapter.generate_label`), which all return a `validation` dict.

`reconcile_evidence_with_tool_trace` cross-checks the `<evidence>` block against the actual tool calls. It catches three failure modes the validator alone cannot:

- the schema names a tool that was never called (fabricated hop);
- the schema records `result_ref={e1,e2}` on a hop whose tool call returned no detections (fabricated grounding — e.g. GroundingDINO silently failed);
- a tool was called and returned positive results but no hop references it (evidence gap).

```python
from vlm_wrapper import (
    cascaded_ground, semantic_validate, GroundingRequest, ValidationResult,
)
from vlm_wrapper.schema import reconcile_evidence_with_tool_trace

vr: ValidationResult = semantic_validate(schema_text, domain="browser",
                                         image_size=(1280, 720))
if vr.escalation_recommended:
    print("errors:", vr.errors)
    print("missing_slots:", vr.missing_slots)
    print("warnings:", vr.warnings)   # skill-context + soft checks

result = cascaded_ground(GroundingRequest(
    images=screenshot, goal="Find cheapest laptop",
    domain="browser", context={"obs": browsergym_obs},
))
print("head used:", result.head_used)
print("valid?   :", result.validation.valid)
print("trace    :", result.escalation_trace)

fab_warnings = reconcile_evidence_with_tool_trace(
    result.schema, result.tool_trace,
)
for w in fab_warnings:
    print("evidence warning:", w)
```

### End-to-end smoke test — `scripts/test_vlm_parsers.py`

Exercises every parser on a real (or synthetic-fallback) input and prints a **validation scorecard** per case, so you can see at a glance which schemas pass, how many entities they found, which skill-context fields are missing, and which evidence hops look fabricated:

```bash
python scripts/test_vlm_parsers.py                  # all five domains
python scripts/test_vlm_parsers.py --cases gymv     # one domain only
python scripts/test_vlm_parsers.py --max-rounds 4   # cap tool-loop rounds

# Force a specific grounding head (overrides the domain's default cascade):
python scripts/test_vlm_parsers.py --cases gymv --gymv-head vlm        # pure-vision gymv
python scripts/test_vlm_parsers.py --cases gymv --gymv-head heuristic  # text-only gymv
python scripts/test_vlm_parsers.py --head vlm                          # all cases single-shot VLM
python scripts/test_vlm_parsers.py --browser-head omniparser           # local OmniParser for browser
```

`--gymv-head vlm` zeroes out the synthetic `obs_text` before the call so GPT-4o is genuinely parsing the rendered game frame (not paraphrasing the text grid), while the game rules and the env's valid-action vocabulary still travel along in the system/user prompts — so the `<actions>` block comes back as `[Up]`/`[Down]`/`[Left]`/`[Right]` instead of invented names like `slide_left`. Programmatic equivalent:

```python
from vlm_wrapper.ground import GroundingRequest, cascaded_ground

result = cascaded_ground(
    GroundingRequest(
        images=frame,
        goal="Reach 2048",
        domain="gymv",
        context={"description": game_rules,
                 "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"]},
    ),
    chain=["vlm", "tool_loop"],   # skip the heuristic, fall back to tool loop
)
print(result.head_used)           # "vlm"
print(result.schema)              # valid <state> parsed from pixels
```

**VLM + tool-calling for Gym-V** &nbsp;(`--gymv-head tool_loop`). Runs the multi-turn tool loop against the full `gymv` tool registry (`list_entities`, `query_entity_pos`, `get_grid_state`, `check_relation`, `count_merge_candidates`, `check_deadlock`, `spatial_analysis`, `get_state_flags`, `list_valid_actions`). Internally the runner keeps `obs_text` wired into the **tool handlers** (so they return ground-truth positions/grids) but sets `show_obs_text=False` in the request context, which hides the text grid from the VLM's user prompt — forcing GPT-4o to actually call tools instead of paraphrasing the grid:

```bash
python scripts/test_vlm_parsers.py --cases gymv --gymv-head tool_loop
```

The scorecard now surfaces the tool trace, e.g. `tool_trace: 2 call(s) [get_grid_state×1, count_merge_candidates×1]`, and the resulting `<evidence>` block cites each hop with its `abstract_op` (GROUND/CHECK/…) and the `functions.<tool>` it called. The pytest mirror of this case lives at `vlm_wrapper/tests/test_gpt4o_parsers.py::test_live_gymv_tool_loop_schema` and asserts head, tool-trace non-emptiness, valid-action verbatim copy, and `<evidence>` presence:

```bash
pytest vlm_wrapper/tests/test_gpt4o_parsers.py::test_live_gymv_tool_loop_schema -m live
```

Each case writes `<state>` text to `out/schemas/<case>.schema.txt` and the full adapter result dict to `out/schemas/<case>.raw.json`.

### Re-observation (Option B) — zoom into a region between hops

Implements [`PLAN-VISUAL-GROUNDING` §4 Option B](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md#4-three-grounding-heads). The VLM can call `zoom_region(x, y, w, h, zoom)` — the harness crops and upscales the frame, appends the new image as a user-side message on the next turn, and the VLM re-perceives the region with fresh visual focus. Defaults to ON for `image_qa` / `video_qa`, OFF for `gymv` / `browser` / `desktop` (Option A, schema-only updates between hops). Override with `GroundingRequest.allow_reobservation=True|False`.

```python
from vlm_wrapper import ground, GroundingRequest

# Force re-observation on a cluttered web page (usually Option A).
result = ground(GroundingRequest(
    images=screenshot, goal="Read the price in the small badge",
    domain="browser", context={"obs": browsergym_obs},
    allow_reobservation=True,
))
# Any hop where the VLM called zoom_region is marked in the trace:
for hop in result.tool_trace:
    if hop.get("reobserved"):
        print("zoomed at", hop["call"]["arguments"])
```

---

## Demo script

```bash
# Mode A — direct grounding only (no API key needed)
python -m vlm_wrapper.demo_visual_grounding

# Mode B — VLM tool-calling loop (multi-hop reasoning)
python -m vlm_wrapper.demo_visual_grounding --vlm-tools --api-key sk-... --model gpt-4o

# Mode C — compare all heads side-by-side
python -m vlm_wrapper.demo_visual_grounding --compare-all --api-key sk-...

# Custom screenshot + save annotated output
python -m vlm_wrapper.demo_visual_grounding --image screenshot.png --save-annotated out.png
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLM_LABEL_MODEL` | `gpt-4o` | Vision model for Head 2 and tool loop |
| `VLM_LABEL_MAX_TOKENS` | `1200` | Max output tokens per call |
| `VLM_LABEL_TEMPERATURE` | `0.2` | Low temp for consistent structured output |
| `OMNIPARSER_CACHE_DIR` | `~/.cache/omniparser-v2` | OmniParser-v2 weight cache directory |

Or pass `model=`, `api_key=`, `base_url=` directly to any `generate_label` call.

---

## Requirements

The tested combination for the **grounding env** ([`install/vlm_benchmarks.environment.yml`](../install/vlm_benchmarks.environment.yml)):

| Layer | Pinned package(s) |
|---|---|
| Python | `3.11` |
| Core ML | `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121` |
| HuggingFace | `transformers>=4.51,<4.56` (handles both the old `box_threshold=` and new `threshold=` GroundingDINO post-process API), `accelerate`, `safetensors`, `sentence-transformers`, `datasets`, `huggingface_hub` |
| Head 3 vision stack | `timm` (required by Florence-2's DaViT backbone), `ultralytics` (YOLO icon detector), `easyocr`, `opencv-python`, `supervision` |
| Video tools | `decord`, `av` |
| API clients | `openai`, `anthropic`, `google-genai` |
| Utils | `Pillow`, `numpy<2`, `omegaconf`, `pyyaml`, `loguru`, `python-dotenv`, `nvidia-ml-py`, `playwright` |
| Tests | `pytest`, `pytest-mock` |

For the **runtime envs**, the tightest pins are:

| Runtime env | Critical pins | Why |
|---|---|---|
| `gymv`       | `gymnasium>=1.2.2`, `pettingzoo[classic]>=1.25.0`, `minigrid>=2.5.0`, `textarena>=0.7.4` | ModalMinds/gym-v pyproject requirements |
| `browsergym` | `playwright==1.44`, `gymnasium>=0.27`, `numpy<2`, `libwebarena==0.0.5`, `libvisualwebarena==0.0.15` | BrowserGym-core's own pins; playwright 1.44 is the version the sub-packages ship against |
| `osworld`    | `gymnasium~=0.28.1`, `transformers~=4.35.2`, `torch~=2.5.0`, `desktop-env>=1.0.2`, `docker>=7.0` | OSWorld's `requirements.txt` — conflicts with the grounding env on `gymnasium` + `transformers` |

Running `python install/{gymv,browsergym,osworld,vlm_benchmarks}_smoke.py` inside the matching env prints the resolved versions plus one `[OK]/[FAIL]/[WARN]` line per module check.

If you prefer pip-only on an existing env, the same vision stack is exposed as the `[vision]` extra in `pyproject.toml`:

```bash
pip install -e ".[vision]"
```

### Common install gotchas (now handled automatically)

- **GroundingDINO API rename** (`box_threshold=` → `threshold=` in `transformers 4.50+`). `tools_visual.py` now picks the right keyword at call time via `inspect.signature`, so both old and new wheels work.
- **Florence-2 missing `timm`.** `grounding.py::_load_caption` now raises an actionable `ImportError` telling you to install the `vision` extra — instead of dying deep inside `modeling_florence2.py`.
- **GPU compute-capability mismatches.** If `detect_objects_at_frame` fails with `CUDA error: no kernel image is available for execution on the device`, your GPU is newer than torch 2.4.1+cu121 supports (common on RTX 50-series). Upgrade by editing `install/vlm_benchmarks.environment.yml` to `torch==2.5.x` + `--extra-index-url …/cu124`.
- **Wrong env activated.** Running `pytest vlm_wrapper/tests` or `scripts/test_vlm_parsers.py` from the `gymv` / `browsergym` / `osworld` env will fail with `ModuleNotFoundError: vlm_wrapper` or `No module named 'transformers.models.grounding_dino'`. Activate `vlm_benchmarks` (or the legacy `vlm_wrapper` env) first — the runtime envs intentionally don't include the grounding stack.
- **OSWorld env and GroundingDINO.** `osworld` pins `transformers~=4.35.2`, which predates the `GroundingDinoForObjectDetection` class. Run OSWorld observation capture in the `osworld` env, but always schema-parse in `vlm_benchmarks`.

OmniParser-v2 weights are downloaded automatically from HuggingFace (`microsoft/OmniParser-v2.0`) on first use into `~/.cache/omniparser-v2/` (override via `OMNIPARSER_CACHE_DIR`). Falls back to OCR-only mode if weights are unavailable.

---

## End goal

Train Qwen3-VL-8B via SFT distillation from GPT-4o labels so that at inference time the 8B model sees **only a screenshot** and produces the structured schema. The tool-calling traces from the multi-hop loop become additional training data: the model learns *when* to call tools and *how* to chain evidence.

**Training sequence:**

1. **Gym-V games first** (2048, Sokoban, Minesweeper) — grid layouts, limited entities, clean labels.
2. **Validate with heuristic head** — flag GPT-4o hallucinations before they enter training data.
3. **Add tool-use training** — teach the model to emit tool calls for position queries and relation checks.
4. **Browser: MiniWoB++ → WebArena** — simple pages, then complex real-web pages.
5. **Benchmark evaluation** — CLEVR, GQA (image), SIV-Bench, Video-Holmes (video).  Selected for transferable visual reasoning skills.

**Expected data budget:** ~3-5K labeled examples per domain. At ~$0.01/example with GPT-4o, that's $30-50 per domain.

---

## Challenges and mitigations

| Challenge | Mitigation |
|-----------|------------|
| Entity position accuracy from pixels | Tool-use delegation: VLM identifies entities, tools return exact coordinates |
| Entity coverage on cluttered web pages | Cascaded approach: 8B model first, escalate to API on low coverage (<10% escalation rate) |
| Format compliance at 8B scale | Flat tagged format + constrained decoding (vLLM) → ~98% compliance |
| Relations require game semantics | Game rules in prompt context + tool delegation for non-visual logic |
| Video temporal reasoning | Multi-hop tool loop: navigate time → detect per-frame → chain evidence |
