# vlm_wrapper — Multi-hop Visual Reasoning for Games, Browsers, Desktop, Images, and Video

Converts screenshots, video frames, and environment observations into a shared structured text schema (`<state>…</state>`), which plugs into the COS-PLAY pipeline for skill retrieval and decision-making. Supports multi-hop tool-calling reasoning where a VLM gathers grounded evidence from specialised vision models before producing the final schema.

**Full plan:** [`plans/PLAN-VISUAL-GROUNDING.md`](../plans/PLAN-VISUAL-GROUNDING.md)

---

## Four grounding heads

| | Head 1 — Heuristic | Head 2 — Vision | Head 3 — OmniParser-v2 | Tool Loop |
|---|---|---|---|---|
| Input | `obs.text` / AXTree (text) | Screenshot (pixels) | Screenshot (pixels) | Screenshot + tool defs |
| Method | Regex + tree-walking | GPT-4o / Qwen3 vision API | YOLO + OCR + Florence-2 (local) | VLM calls tools iteratively |
| Cost | Free | ~$0.01/call | Free (local GPU) | ~$0.05-0.10/schema |
| Latency | <1 ms | ~1–3 s | ~0.6 s (GPU) | ~5–15 s |
| Use case | Real-time RL, baselines | Training-label generation | UI screenshots, precise bbox | Complex reasoning with evidence chains |

---

## Tool registries

Three tool registries provide structured APIs for multi-hop visual reasoning:

### Visual tools (single frame) — `build_visual_registry(image)`

| Tool | Purpose |
|------|---------|
| `detect_objects` | OmniParser-v2 element detection with bboxes, labels, types, confidence |
| `describe_region` | Florence-2 caption for a specific rectangular crop |
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

### Head 1 — Heuristic (text state → schema)

```python
from vlm_wrapper import gymv_heuristic_schema, browser_heuristic_schema

# Gym-V
schema = gymv_heuristic_schema(
    obs_text="| 2 | 4 | 0 | 0 |\n| 0 | 16 | 8 | 0 |",
    description="You are playing 2048. Valid moves: [Up], [Down], [Left], [Right].",
    task_id="Game2048-v0", step=5,
)

# BrowserGym
schema = browser_heuristic_schema(obs, step=3, task_id="webarena.shopping.143")
```

### Head 2 — Vision (screenshot → VLM → schema)

```python
from vlm_wrapper import gymv_generate_label, browser_obs_to_schema

# Gym-V
result = gymv_generate_label(frame, goal="Reach 2048", task_id="Game2048-v0", step=5)
print(result["schema"])    # <state>…</state> or None
print(result["warnings"])  # validation issues

# BrowserGym
result = browser_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
```

### Head 3 — OmniParser-v2 Grounding (screenshot → local models → schema)

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

### Tool-calling loop (multi-hop visual reasoning)

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

### OSWorld (desktop screenshots)

```python
from vlm_wrapper.grounding_browsergym import grounding_osworld_obs_to_schema

result = grounding_osworld_obs_to_schema(osworld_obs, step=1, task_id="install-spotify")
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

**Core (Heads 1 & 2):**
```
openai
pillow
numpy
```

**Head 3 (OmniParser-v2 grounding):**
```
ultralytics          # YOLO icon detector
transformers         # Florence-2 icon captioner
easyocr              # OCR (or paddleocr as alternative)
torch torchvision    # model inference
```

**Visual/video tools (optional, for tool-calling loop):**
```
scikit-learn         # color extraction (KMeans)
opencv-python        # video decoding (or imageio[pyav])
```

OmniParser-v2 weights are downloaded automatically from HuggingFace (`microsoft/OmniParser-v2.0`) on first use. Falls back to OCR-only mode if weights are unavailable.

---

## End goal

Train Qwen3-VL-8B via SFT distillation from GPT-4o labels so that at inference time the 8B model sees **only a screenshot** and produces the structured schema. The tool-calling traces from the multi-hop loop become additional training data: the model learns *when* to call tools and *how* to chain evidence.

**Training sequence:**

1. **Gym-V games first** (2048, Sokoban, Minesweeper) — grid layouts, limited entities, clean labels.
2. **Validate with heuristic head** — flag GPT-4o hallucinations before they enter training data.
3. **Add tool-use training** — teach the model to emit tool calls for position queries and relation checks.
4. **Browser: MiniWoB++ → WebArena** — simple pages, then complex real-web pages.
5. **Benchmark evaluation** — CLEVR, GQA, ToolVQA (image), SIV-Bench, Video-Holmes (video).

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
