# PLAN: Visual Grounding

**Scope:** VLM visual parser — pixels → structured `<state>` schema — across games, browser, desktop, images, and video. Includes grounding heads, observation schema, adapters, training pipeline, benchmark evaluation, and multi-hop tool-calling reasoning.

**Upstream:** Raw observations from Gym-V, BrowserGym, OSWorld, image/video benchmarks.
**Downstream:** [Action Agent](PLAN-ACTION-AGENT.md) consumes the structured schema; [Skill Bank](PLAN-SKILL-BANK.md) uses schemas for contract learning and retrieval.

---

## 1. Goal

Train a VLM to produce structured scene summaries from visual observations (pixels only), using the native text state that Gym-V and BrowserGym already provide as free supervision. The trained VLM either grounds visually or calls env APIs for information not observable in pixels. Output plugs directly into the existing skill retrieval → reasoning → action pipeline.

**Key insight:** Both domains already emit rich text state (Gym-V wrapper captions/rules, BrowserGym AXTree/DOM). This text state is the training label — no manual annotation needed. The VLM learns to see what the text state already knows.

---

## 2. Collect paired training data from environments

Every episode step produces a free (visual input, text state) pair.

**Gym-V**

- Visual input: raw frame / short frame stack.
- Text state: wrapper captions, rules, state descriptions, procedural object/position metadata.
- Start with the richest wrapper environments; expand as quality allows.

**BrowserGym**

- Visual input: screenshot.
- Text state: AXTree / DOM (element types, labels, form state, validation, hierarchy).
- Available across all BrowserGym benchmarks (MiniWoB++, WebArena, VisualWebArena, WorkArena).

No labeling pipeline needed — run episodes, collect pairs.

---

## 3. Canonical schema

Map heterogeneous text state formats into one shared target format before using them as training labels.

**This is the main design work.** The schema should be derived from what the environments actually emit, not designed in the abstract.

Candidate fields (refined empirically during normalization):

- `domain`, `task`, `goal`
- `entities` (with type, attributes, source)
- `relations` (spatial, functional, grouping)
- `state_flags` (dialog_open, input_missing, page_changed, ...)
- `salient_targets` (target, blocker, constraint)
- `uncertainty` (per-field confidence where grounding is ambiguous)

Shared slot names across domains (for downstream skill transfer): `target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`.

Schema will iterate — expect 3-4 revisions as real data reveals missing or useless fields.

### 3a. Shared output schema (both domains emit this)

```text
<state>
domain={browser|gymv}
task={task_id or env_id}
goal={goal string, ≤60 tokens}
step={int}

<entities>
e1[type={element|object|region|text}, label={str}, bid={str|null}, pos={x,y,w,h|null}]
e2[...]
...

<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={str|null}
e2.state=...
...

<relations>
contains(e1,e2)
adjacent(e3,e4)
blocks(e5,e1)
grouped(e2,e3,e4)
...

<state_flags>
progress={float 0-1 or null}
phase={early|mid|late|null}
error={str|null}
dialog_open={bool}
input_pending={bool}

<targets>
target={eid}
blocker={eid|null}
constraint={str|null}
candidate_set=[eid,eid,...]

<uncertainty>
e1.label={high|medium|low}
e3.pos={medium}

<actions>
a1={action_string}
a2={action_string}
...
</state>
```

**Why this format:**
- Each `<section>` tag is a clear generation boundary — the model learns section ordering.
- Entity references (`e1`, `e2`) are short tokens the model can reuse across sections without repeating long names.
- No nested braces/brackets beyond one level — critical for 8B model reliability.
- Total token count for a typical web page: ~400–600 tokens. For a game frame: ~200–400 tokens.

**Schema as inner MDP state:** Under the two-level MDP (see [Action Agent §5](PLAN-ACTION-AGENT.md#5-two-level-mdp-long-horizon-reasoning)), this schema is the state representation for the inner reasoning MDP. Each GROUND/CHECK hop updates entities, relations, or uncertainty. The `<targets>` and `<uncertainty>` sections drive the agent's decision to continue reasoning (more hops) or act (EXECUTE). Shared slot names (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`) are the vocabulary that makes reasoning skills transferable across domains.

### 3b. Design constraints for small VLM (Qwen3-VL-8B)

- Target structured summary: **300–800 tokens** (output side).
- Input side: 1 image + short system prompt + optional history ≈ 1–2K text tokens + image tokens.
- Flat tagged format (not deeply-nested JSON) — easier for 8B models to produce reliably.
- Every field has a fixed tag name — parse with simple regex, no JSON decoder needed at inference.
- Fields are **ordered** (spatial → semantic → action) so the model can generate left-to-right without backtracking.

### 3c. Schema design decisions for 8B model

| Decision | Choice | Rationale |
|---|---|---|
| Format | Tagged text, not JSON | 8B models produce fewer bracket-matching errors with flat tags |
| Entity limit | ≤ 25 (browser), ≤ 20 (game) | Keeps output < 600 tokens; 8B accuracy degrades with long structured output |
| Position format | `x,y,w,h` integers | Pixel coords (browser) or grid coords (game); no floats to reduce token count |
| Relation format | `verb(eid,eid)` | Lisp-like prefix is shorter and more regular than natural language |
| Action format | Raw env action strings | No abstraction layer — model outputs what the environment accepts |
| Section ordering | spatial → semantic → action | Matches human scan order; model can attend to earlier sections when generating later ones |
| Uncertainty | Per-entity, 3 levels | `high/medium/low` — finer granularity isn't reliable at 8B |
| Candidate actions | Top 3–5 | Keeps action section short; full action space in system prompt |

### 3d. Schema versioning

- **v0.1** — initial schema, tested on 2048 + Sokoban + MiniWoB++
- **v0.2** — revised after first training run (expect field additions/removals)
- **v1.0** — stable schema validated across ≥ 3 Gym-V categories + ≥ 1 BrowserGym benchmark

Schema version goes in the system prompt so the model can be trained on mixed versions during transition.

---

## 4. Three grounding heads

### Head 1 — Heuristic (text-in → schema-out)

Fast, free, deterministic. Parses native text state (obs.text, AXTree/DOM) into the schema with regex/tree-walking. Good for: real-time RL rollouts, cheap baselines, validation.

- **Implementation:** `vlm_wrapper/gymv_heuristic.py`, `vlm_wrapper/browser_heuristic.py`

### Head 2 — Vision (image-in → schema-out)

Sends the screenshot to GPT-4o (or any vision LLM) and receives the schema. The image is the primary input; native text is optional grounding context. Good for: training-label generation, Qwen3-VL-8B distillation.

- **Implementation:** `vlm_wrapper/gymv_adapter.py`, `vlm_wrapper/browser_adapter.py`

### Head 3 — OmniParser-v2 (image-in → local models → schema-out)

Uses Microsoft's OmniParser-v2 (YOLO icon detector + Florence-2 icon captioner + OCR) to parse a GUI screenshot into structured elements with bounding boxes — all locally, no API calls. ~0.6s/frame on GPU.

- **Implementation:** `vlm_wrapper/grounding.py`, `vlm_wrapper/grounding_browsergym.py`

### Tool-calling loop (multi-hop visual reasoning / inner MDP)

VLM sees screenshot + system prompt + tool definitions → calls tools to gather ground-truth data → produces final schema. The trace becomes SFT data.

Under the two-level MDP (see [Action Agent §5](PLAN-ACTION-AGENT.md#5-two-level-mdp-long-horizon-reasoning)), each tool call in the loop maps to an inner MDP action:

| Tool call pattern | Inner MDP action | Schema update |
|-------------------|-----------------|---------------|
| `detect_objects`, `visual_search` | GROUND(query) | Add/update entities |
| `spatial_query`, `check_relation` | CHECK(predicate) | Update relations, state_flags |
| `describe_region`, `classify_scene` | GROUND(detail) | Refine attributes, reduce uncertainty |
| `get_frame`, `sample_frames` | GROUND(temporal) | Add temporal entities/events |
| Final schema output | CONCLUDE or EXECUTE | Complete schema for downstream |

**Design decision — re-observation between hops:**
- **Option A (default):** Hops operate on the same `<state>` schema, only updating an internal scratchpad. Cheaper, faster. Used for games/web.
- **Option B (selective):** GROUND actions can trigger re-rendering or zooming into a region. More expensive but handles fine-grained visual detail. Used for visual QA and video benchmarks.

- **Implementation:** `vlm_wrapper/tool_loop.py`

| Head | Input | Grounding method | Latency | Best for |
|------|-------|------------------|---------|----------|
| Head 1 | Text state | Regex/tree-walking | ~ms | Interactive envs with text state |
| Head 2 | Screenshot → VLM | VLM directly produces schema | ~2-4s | General images, static QA |
| Head 3 | Screenshot → YOLO+OCR+Florence-2 | Local model ensemble | ~0.6s GPU | UI screenshots, precise bbox |
| Tool loop | Screenshot → VLM calls tools | Multi-hop tool calling | ~5-15s | Complex reasoning with evidence chains |

---

## 5. Domain adapters

### 5a. BrowserGym adapter

Source observation fields from `BrowserEnv._get_obs()`:

| BrowserGym field | Type | Maps to schema |
|---|---|---|
| `obs["axtree_object"]` | merged AXTree dict | `<entities>`, `<attributes>`, `<relations>` |
| `obs["dom_object"]` | DOM snapshot dict | backup for AXTree gaps |
| `obs["extra_element_properties"]` | dict[bid → {visibility, bbox, clickable, set_of_marks}] | `e*.pos`, `e*.state` |
| `obs["screenshot"]` | np.array (H,W,3) | VLM visual input |
| `obs["focused_element_bid"]` | str | `<state_flags> focused_element=` |
| `obs["url"]` | str | `task=` context |
| `obs["goal"]` / `obs["goal_object"]` | str / list[dict] | `goal=` |
| `obs["last_action_error"]` | str | `<state_flags> error=` |

### 5b. Gym-V adapter

Source observation fields from `gym_v.core.Observation`:

| Gym-V field | Type | Maps to schema |
|---|---|---|
| `obs.image` | PIL Image | VLM visual input |
| `obs.text` | str | parse into `<entities>`, `<attributes>`, `<state_flags>` |
| `obs.metadata` | dict | additional context for label enrichment |
| `env.description` | str (property) | `goal=`, game rules → training system prompt |
| `info["history"]` | list[{obs, action}] | step count, prior context |
| `info["invalid_action"]` | bool | `<state_flags> error=` |

### 5c. OSWorld adapter

Uses Head 3 (OmniParser-v2) directly on desktop screenshots. Domain = `"desktop"`.

- **Implementation:** `vlm_wrapper/grounding_browsergym.py → grounding_osworld_obs_to_schema()`

---

## 6. VLM training pipeline

### Stage A — Pure visual grounding

Supervised distillation: VLM sees only the visual input, produces the normalized text state.

- **Training signal:** field-level match against normalized text state labels.
- **Metrics:** field accuracy, relation accuracy, target-slot accuracy, format compliance.
- **No action prediction.** The VLM is a parser.

Start with Qwen-VL or similar and fine-tune on the paired data.

### Stage B — Learn when to call APIs vs. ground visually

Some text state fields are not visually observable. The VLM needs to learn what it can see (produce from pixels) vs. what it can't see (emit an API call).

1. First train pure visual grounding (Stage A) — no API calls.
2. Then add tool-use training — teach the model to request non-visual info via env API.

### Training label generation pipeline

```
For each episode step in {Gym-V, BrowserGym}:
  1. Collect native observation
  2. Run domain adapter → raw structured summary
  3. Apply truncation/pruning to fit token budget
  4. Validate format: regex check all tags present, entity refs consistent
  5. Store as training pair: input = image + system_prompt; output = schema text
```

### Recommended training sequence

1. **Gym-V games first** (2048, Sokoban, Minesweeper) — grid layouts, limited entities, clean GPT-4o labels. Expect >90% field accuracy within ~3-5K examples per game.
2. **Validate with heuristic head** — for every GPT-4o label, run the heuristic head on the same observation. Flag disagreements to catch GPT-4o hallucinations before they enter training data.
3. **Add tool-use training** — after basic SFT, add a second stage where the model learns to emit tool calls for position queries and relation checks.
4. **Browser: MiniWoB++ → WebArena** — simple pages first, validate schema and entity prioritization, then scale to complex real-web pages. Use API escalation for the hard tail during data collection.
5. **Benchmark evaluation** — CLEVR, GQA, ToolVQA (image), SIV-Bench, Video-Holmes (video).
6. **Expected data budget:** ~3-5K labeled examples per domain. At ~$0.01/example with GPT-4o, that's $30-50 per domain.

### Can Qwen3-VL-8B learn this task?

**Yes.** The task is structured template generation (not open-ended reasoning). Gym-V benchmarks show Qwen3-VL-8B scores 16.5 average zero-shot on harder tasks (logic, algorithms, multi-turn strategy). Structured scene parsing is strictly easier, and we fine-tune rather than zero-shot. Distillation from GPT-4o to 8B is well-established for structured output.

---

## 7. Rollout order

**Phase 1 — Gym-V (controlled lab)**
1. Collect paired data from richest wrapper environments.
2. Normalize wrapper text into schema (this designs the schema).
3. Train VLM on (frame → normalized summary).
4. Plug into existing skill pipeline. Measure: does structured format improve retrieval?

**Phase 2 — BrowserGym (noisy deployment target)**
1. Collect paired data across benchmarks.
2. Normalize AXTree/DOM into the same schema.
3. Fine-tune or adapt the Gym-V-trained VLM.
4. Test whether game-learned visual grounding transfers to web UIs.

**Phase 3 — API calling (non-visual grounding)**
1. Add tool-use training on both domains.
2. Measure: does API fallback recover the information lost by visual-only grounding?

---

## 8. HallusionBench — canonical schema for visual reasoning QA

The canonical schema applies to HallusionBench without structural changes. HallusionBench is static image-QA (~1,100 samples, yes/no answers). The schema is a scene representation, not an environment interface.

**Hop trace** maps directly to hallucination reasoning:

```text
trigger=question asks whether the image contains triangles
hop1=locate relevant entities [e1, e3]
hop2=ground entity attributes from pixels [e1.shape → ambiguous]
hop3=check text overlay against visual evidence
hop4=resolve conflict: visual grounding vs language prior
output=answer=no
grounding=[e1, e2, e3]
```

Each hop forces explicit grounding with source attribution. Hallucination failure modes become auditable.

---

## 9. Tool registries for multi-hop reasoning

### Visual tools (single frame)

`build_visual_registry(image)` → 9 tools:

| Tool | Purpose |
|------|---------|
| `detect_objects` | OmniParser-v2 element detection with bboxes |
| `describe_region` | Florence-2 caption for a crop |
| `visual_search` | Text-query search over detected elements |
| `count_objects` | Count elements by type or description |
| `classify_scene` | Scene type classification |
| `spatial_query` | Spatial relations between two elements |
| `measure_distance` | Pixel distance between points |
| `extract_colors` | Dominant colors in a region |
| `read_text_region` | OCR on a region |

### Video tools (temporal navigation)

`build_video_registry(frames, fps)` → 8 tools:

| Tool | Purpose |
|------|---------|
| `get_frame` | Retrieve frame by index/timestamp |
| `sample_frames` | Uniformly sample N frames |
| `compare_frames` | Pixel diff between two frames |
| `detect_scene_changes` | Find scene boundaries |
| `get_video_info` | Video metadata |
| `read_text_in_frame` | OCR on a specific frame |
| `temporal_navigate` | Move to a different point in time |
| `list_valid_actions` | Available navigation actions |

### Cross-frame tools (video + visual)

`build_video_visual_registry(frames, fps)` → combines all above + 6 cross-frame tools:

| Tool | Purpose |
|------|---------|
| `track_object` | Track element across frames |
| `summarize_clip` | Timeline of visual changes |
| `find_moment` | Find frame where event occurs |
| `detect_activity` | Classify activity in frame range |
| `compare_elements` | Semantic diff between two frames |
| `detect_objects_at_frame` | Detection on a specific frame |

---

## 10. Chosen benchmarks

The following benchmarks are selected for multi-step visual reasoning evaluation. Mixed image + video.

### Image-based

**CLEVR** — Synthetic, compositional visual reasoning with functional programs.
- Download: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
- Official page: https://cs.stanford.edu/people/jcjohns/clevr/

**GQA** — Real-image visual reasoning with scene graphs.
- Downloads: https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip, questions1.2.zip, images.zip
- Official page: https://cs.stanford.edu/people/dorarad/gqa/download.html

**ToolVQA** — Multi-step visual reasoning with tool use.
- Download: https://drive.google.com/drive/folders/1diRjF2jK0aHoAMximnT7jNg4eN96ppCp?usp=sharing
- Repo: https://github.com/Fugtemypt123/ToolVQA-release

### Video-based

**SIV-Bench** — Social interaction understanding. 2,792 clips, 8,792 QAs.
- Dataset: https://huggingface.co/datasets/Fancylalala/SIV-Bench
- Code: https://github.com/kfq20/SIV-Bench

**Video-Holmes** — Multi-step clue-chaining. 270 films, 1,837 questions.
- Repo: https://github.com/TencentARC/Video-Holmes
- Dataset: https://huggingface.co/datasets/TencentARC/Video-Holmes

### Per-benchmark grounding strategy

| Benchmark | Modality | vlm_wrapper entry point | Key tools |
|-----------|----------|------------------------|-----------|
| CLEVR | Image | `visual_generate_label_with_tools()` | detect_objects, spatial_query, extract_colors |
| GQA | Image | `visual_generate_label_with_tools()` | detect_objects, spatial_query + gold scene-graph IoU eval |
| ToolVQA | Image | `visual_generate_label_with_tools()` + custom tools | detect_objects + ToolVQA-defined tools via ToolRegistry |
| SIV-Bench | Video | `video_visual_generate_label_with_tools()` | track_object, find_moment, detect_activity |
| Video-Holmes | Video | `video_visual_generate_label_with_tools()` | detect_scene_changes, track_object, describe_region |

---

## 11. Grounding-aware schema extensions for benchmarks

```text
<evidence>
hop1.tool=detect_objects
hop1.result_ref=e1,e3
hop1.frame=null                  # null for image benchmarks
hop1.confidence=high

<answer>
answer={predicted answer}
grounding=[e1,e3]
evidence_chain=[hop1,hop2]
confidence=high
```

For video benchmarks, add temporal grounding: `hop1.frame=42`, `hop1.timestamp=14.0`.

---

## 12. Validation checkpoints

- **After Phase 1:** Does the VLM produce structured summaries that the skill pipeline can use? Does structured format beat raw wrapper text for retrieval?
- **After Phase 2:** Does the same schema work for web pages? Does the VLM generalize from games to web?
- **After Phase 3:** Does API calling recover meaningful information?

---

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Gym-V wrapper quality is uneven across 179 envs | Start with richest environments; expand selectively |
| Schema designed from Gym-V doesn't fit BrowserGym | Keep shared slots minimal; allow domain-specific extensions |
| VLM structured output drifts or breaks format | Output validation layer; constrained decoding (vLLM) → ~98% compliance |
| API-calling adds latency and complexity | Measure visual-only ceiling first |
| Structured summaries don't actually help downstream | Test early (Phase 1 checkpoint) |
| Entity position accuracy from pixels | Tool-use delegation: VLM identifies entities, tools return exact coordinates from env APIs |
| Entity coverage on cluttered web pages | Cascaded approach: 8B model first, escalate to API on low coverage (<10% escalation rate expected) |
| Relations require game semantics, not just vision | Game rules in prompt context + tool delegation for non-visual logic |
| Video temporal reasoning is expensive | Multi-hop tool loop with frame sampling; detect_scene_changes narrows search before per-frame analysis |

---

## 14. Implementation status

| Component | Module | Status |
|-----------|--------|--------|
| Visual tool registry | `vlm_wrapper/tools_visual.py` | Done (9 tools) |
| Video tool registry | `vlm_wrapper/tools_video.py` | Done (8 tools) |
| Cross-frame registry | `vlm_wrapper/tools_video_visual.py` | Done (6 tools) |
| OmniParser grounding | `vlm_wrapper/grounding.py` | Done |
| BrowserGym grounding adapter | `vlm_wrapper/grounding_browsergym.py` | Done |
| Tool-calling loop | `vlm_wrapper/tool_loop.py` | Done |
| Heuristic adapters | `vlm_wrapper/gymv_heuristic.py`, `browser_heuristic.py` | Done |
| Vision adapters (Head 2) | `vlm_wrapper/gymv_adapter.py`, `browser_adapter.py` | Done |
| Schema utilities | `vlm_wrapper/schema.py` | Done |
| Demo script | `vlm_wrapper/demo_visual_grounding.py` | Done |
| Benchmark loaders | — | **TODO** |
| Evaluation harness | — | **TODO** |
| Schema `<evidence>` + `<answer>` sections | `vlm_wrapper/schema.py` | **TODO** |
| Inner MDP hop trace logging in tool loop | `vlm_wrapper/tool_loop.py` | **TODO** |
| Re-observation (Option B) for GROUND hops | `vlm_wrapper/tool_loop.py` | **TODO** |
| Qwen3-VL-8B training pipeline | — | **TODO** |

---

## 15. Reference

- vlm_wrapper repo: https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent/tree/main/vlm_wrapper
- System plan index: [`plans/README.md`](README.md)
