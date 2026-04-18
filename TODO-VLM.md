# TODO-VLM: VLM visual parser trained on native text state

**Goal:** Train a VLM to produce structured scene summaries from visual observations (pixels only), using the native text state that Gym-V and BrowserGym already provide as free supervision. The trained VLM either grounds visually or calls env APIs for information not observable in pixels. Output plugs directly into the existing Game-AI-Agent pipeline (skill retrieval → reasoning → action).

**Key insight:** Both domains already emit rich text state (Gym-V wrapper captions/rules, BrowserGym AXTree/DOM). This text state is the training label — no manual annotation needed. The VLM learns to see what the text state already knows.

---

## 1. Collect paired training data from environments

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

## 2. Normalize text state into a canonical schema

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

---

## 3. Train VLM: pixels → structured summary (Stage A)

Supervised distillation: VLM sees only the visual input, produces the normalized text state.

- **Training signal:** field-level match against normalized text state labels.
- **Metrics:** field accuracy, relation accuracy, target-slot accuracy, format compliance.
- **No action prediction.** The VLM is a parser. Downstream skill bank and decision agent stay unchanged.

Start with a capable open VLM (Qwen-VL or similar) and fine-tune on the paired data.

---

## 4. Learn when to call APIs vs. ground visually (Stage B)

Some text state fields are not visually observable (hidden form validation, off-screen DOM elements, game rules not rendered). The VLM needs to learn:

- **What it can see:** produce directly from pixels.
- **What it can't see:** emit an API call to the environment.

**Training signal:** fields present in text state but not inferable from pixels → API-call targets. Fields that correlate with visual features → visual grounding targets.

**Approach:**

1. First train pure visual grounding (Stage A) — no API calls, filter non-visual fields from labels.
2. Then add tool-use training — reintroduce non-visual fields, teach the model to request them via env API.

Define a minimal API surface per domain:
- **Gym-V:** query object state, query rules, query spatial info not in frame.
- **BrowserGym:** query DOM attribute, query form validation, query off-screen elements.

This is a separate milestone from Stage A. Do not conflate them.

---

## 5. Rollout order

**Phase 1 — Gym-V (controlled lab)**

1. Collect paired data from richest wrapper environments.
2. Normalize wrapper text into schema (this designs the schema).
3. Train VLM on (frame → normalized summary).
4. Plug into existing skill pipeline. Measure: does structured format improve retrieval and decision-making vs. raw wrapper text?

**Phase 2 — BrowserGym (noisy deployment target)**

1. Collect paired data across benchmarks.
2. Normalize AXTree/DOM into the same schema.
3. Fine-tune or adapt the Gym-V-trained VLM.
4. Test whether game-learned visual grounding transfers to web UIs.

**Phase 3 — API calling (non-visual grounding)**

1. Add tool-use training on both domains.
2. Measure: does API fallback recover the information lost by visual-only grounding?

---

## 6. Validation checkpoints

Before investing in each next phase, confirm the previous one actually helps.

- **After Phase 1:** Does the VLM produce structured summaries that the skill pipeline can use? Does structured format beat raw wrapper text for retrieval? If no → fix the schema, not the VLM.
- **After Phase 2:** Does the same schema work for web pages? Does the VLM generalize from games to web, or does it need domain-specific training? If schema breaks → revise shared slots.
- **After Phase 3:** Does API calling recover meaningful information? Is the improvement worth the inference-time cost of tool use?

---

## 7. HallusionBench — canonical schema for visual reasoning QA

The canonical schema (Section 2) applies to HallusionBench without structural changes. HallusionBench is static image-QA (~1,100 samples, yes/no answers), not an interactive environment — but the schema is a scene representation, not an environment interface. Implementation is independent of the VLM training pipeline (Sections 1–4); free supervision is irrelevant here.

**Schema mapping**

Each HallusionBench sample (image + question + answer) maps to one canonical summary:

```text
domain=visual_qa
task=<question text>
goal=determine whether claim is true given image
scene_type=photograph|diagram|illusion|chart

entities=
  e1[type=object, label="triangle", source=vision]
  e2[type=text_overlay, content="circles", source=ocr]
  e3[type=region, desc="left half of image", source=vision]

relations=
  contains(e3, e1)
  contradicts(e1.label, e2.content)
  resembles(e1, "circle", confidence=low)

salient_targets=
  target=e1
  claim_anchor=e2

state_flags=
  illusion_present=true
  text_image_conflict=true

uncertainty=
  e1.label=high
  e1.shape=medium

evidence_slots=
  slot_for=e1
  slot_against=e2
```

**Field additions beyond base schema**

- `state_flags`: `illusion_present`, `text_image_conflict`, `image_edited`, `language_prior_applicable` — flags specific to hallucination/illusion detection.
- `relations`: `contradicts(a, b)` and `resembles(a, b, confidence)` — needed for cases where the visual ground truth conflicts with text or surface appearance.
- `evidence_slots`: `slot_for` / `slot_against` — evidence supporting or contradicting the claim.

Shared slot names (`target`, `constraint`, `candidate_set`) carry over unchanged.

**Hop trace**

The Layer C hop trace maps directly to hallucination reasoning:

```text
trigger=question asks whether the image contains triangles
hop1=locate relevant entities [e1, e3]
hop2=ground entity attributes from pixels [e1.shape → ambiguous]
hop3=check text overlay against visual evidence [e2.content="circles" vs e1.label="triangle"]
hop4=resolve conflict: visual grounding vs language prior
intermediate=visual evidence is ambiguous; text overlay contradicts claim
output=answer=no
grounding=[e1, e2, e3]
```

Each hop forces explicit grounding with source attribution. Hallucination failure modes become auditable — you can trace whether the model trusted language priors over pixel evidence at a specific hop.

**Implementation scope**

1. Define schema dataclass / Pydantic model with HallusionBench-specific fields alongside base fields.
2. Write a loader that reads HallusionBench samples and produces empty schema instances (image + question populated, entity/relation fields to be filled).
3. Populate schema at inference time: the VLM (or a prompted LLM operating over VLM descriptions) fills entities, relations, uncertainty, evidence slots.
4. Evaluate: compare answer accuracy with and without structured schema as intermediate representation. The hypothesis is that forcing explicit grounding and uncertainty reduces hallucination rate.

HallusionBench is small enough (~1,100 samples) that schema population is tractable even with expensive models. No environment integration needed.

---

## 8. Hop trace extraction (deferred)

Multi-hop evidence traces (trigger → hops → subgoal → output) are useful for skill extraction and supervision but are a **separate research problem** from scene parsing.

Do not start hop trace extraction until Stage A parsing is validated and producing clean structured summaries. The hop trace work depends on having reliable structured input to chain over.

---

## 9. Artifacts per phase

| Phase | Artifact |
|-------|----------|
| 1 | Schema (v1), Gym-V adapter, trained VLM (Gym-V), baseline comparison |
| 1→2 | HallusionBench schema instance, loader, populated samples, hallucination-detection ablation |
| 2 | BrowserGym adapter, adapted VLM, cross-domain schema validation |
| 3 | API-calling VLM, tool-use training data, non-visual field coverage |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Gym-V wrapper quality is uneven across 179 envs | Start with richest environments; expand selectively |
| Schema designed from Gym-V doesn't fit BrowserGym | Keep shared slots minimal; allow domain-specific extensions |
| VLM structured output drifts or breaks format | Output validation layer; constrained decoding if needed |
| API-calling adds latency and complexity | Measure visual-only ceiling first; only add API calls where the gap matters |
| Structured summaries don't actually help downstream pipeline | Test early (Phase 1 checkpoint) before investing in Phases 2-3 |

---

## 11. One-sentence framing

We train a VLM as a visual parser — supervised by the text state that game and web environments already provide for free — that converts pixels into structured summaries for the existing skill pipeline, and learns to call environment APIs for information it cannot see.

---

## 12. Concrete observation schemas (for Qwen3-VL-8B training)

### Design constraints for small VLM

Qwen3-VL-8B has 131K context but for efficient GRPO/SFT training we need **tight output budgets**:
- Target structured summary: **300–800 tokens** (output side).
- Input side: 1 image + short system prompt + optional history ≈ 1–2K text tokens + image tokens.
- Flat tagged format (not deeply-nested JSON) — easier for 8B models to produce reliably.
- Every field has a fixed tag name — parse with simple regex, no JSON decoder needed at inference.
- Fields are **ordered** (spatial → semantic → action) so the model can generate left-to-right without backtracking.

### 12a. Shared output schema (both domains emit this)

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

### 12b. BrowserGym adapter: native obs → schema

**Source observation fields** (from `BrowserEnv._get_obs()`):

| BrowserGym field | Type | Maps to schema |
|---|---|---|
| `obs["axtree_object"]` | merged AXTree dict | `<entities>`, `<attributes>`, `<relations>` |
| `obs["dom_object"]` | DOM snapshot dict | backup for AXTree gaps |
| `obs["extra_element_properties"]` | dict[bid → {visibility, bbox, clickable, set_of_marks}] | `e*.pos`, `e*.state` |
| `obs["screenshot"]` | np.array (H,W,3) | VLM visual input (not in text schema) |
| `obs["focused_element_bid"]` | str | `<state_flags> focused_element=` |
| `obs["url"]` | str | `task=` context |
| `obs["goal"]` / `obs["goal_object"]` | str / list[dict] | `goal=` |
| `obs["open_pages_urls"]` | tuple[str] | `<state_flags> num_tabs=` |
| `obs["last_action"]` | str | history context |
| `obs["last_action_error"]` | str | `<state_flags> error=` |
| `obs["elapsed_time"]` | float | `step=` context |

**Conversion procedure (runs at data-collection time, produces training labels):**

```
1. Flatten AXTree via flatten_axtree_to_str() with:
     filter_visible_only=True, filter_with_bid_only=True
   This gives the visible, interactive elements with bids.

2. For each AXTree node with a bid:
   → Create entity: e{n}[type=element, label={node.role + name}, bid={bid}]
   → From extra_element_properties[bid]:
       pos = bbox (scaled)
       state = "visible" + (",clickable" if clickable) + (",focused" if bid == focused_element_bid)
   → If node has value: attribute e{n}.value={value}

3. Build relations from AXTree parent-child structure:
   → contains(parent_eid, child_eid) for meaningful nesting
   → grouped(eid1, eid2, ...) for sibling lists (nav items, form fields)

4. State flags:
   → error = last_action_error (if non-empty)
   → dialog_open = true if modal/dialog role in AXTree
   → input_pending = true if focused element is a text input
   → num_tabs = len(open_pages_urls)

5. Targets (heuristic for training label):
   → target = entity most relevant to goal (BM25 or embedding match)
   → blocker = entity causing last_action_error (if any)
   → candidate_set = clickable entities near target

6. Actions: map to BrowserGym high-level action set:
   → click(bid), fill(bid, value), scroll(direction), etc.
   → Keep top-5 most plausible from candidate_set.

7. Truncation: if > 40 entities, keep:
   → All entities in set_of_marks (SoM)
   → All clickable entities
   → Drop non-interactive, non-visible containers
   Target: ≤ 25 entities for 8B model.
```

**Example output (BrowserGym, WebArena shopping task):**

```text
<state>
domain=browser
task=webarena.shopping.143
goal=Find the cheapest red jacket and add to cart
step=3

<entities>
e1[type=element, label=navigation 'Main Menu', bid=a12, pos=0,0,200,40]
e2[type=element, label=link 'Jackets', bid=a45, pos=30,120,80,20]
e3[type=element, label=combobox 'Sort By', bid=b12, pos=500,90,120,30]
e4[type=element, label=link 'Red Wool Jacket $49.99', bid=c8, pos=100,200,250,300]
e5[type=element, label=link 'Red Down Jacket $39.99', bid=c14, pos=360,200,250,300]
e6[type=element, label=button 'Add to Cart', bid=c15, pos=400,510,100,30]
e7[type=element, label=textbox 'Search', bid=a3, pos=300,5,200,30]

<attributes>
e3.value=Relevance
e4.state=visible,clickable
e5.state=visible,clickable
e6.state=visible,clickable
e7.state=visible,clickable,focused

<relations>
contains(e1,e2)
grouped(e4,e5)
adjacent(e5,e6)

<state_flags>
progress=0.3
phase=mid
error=null
dialog_open=false
input_pending=false

<targets>
target=e5
blocker=null
constraint=cheapest
candidate_set=[e4,e5,e6]

<uncertainty>
e5.label=low
e6.label=low

<actions>
a1=click(c14)
a2=click(c15)
a3=fill(a3, "red jacket")
a4=select_option(b12, "Price: Low to High")
</state>
```

### 12c. Gym-V adapter: native obs → schema

**Source observation fields** (from `gym_v.core.Observation`):

| Gym-V field | Type | Maps to schema |
|---|---|---|
| `obs.image` | PIL Image | VLM visual input (not in text schema) |
| `obs.text` | str | parse into `<entities>`, `<attributes>`, `<state_flags>` |
| `obs.metadata` | dict | additional context for label enrichment |
| `env.description` | str (property) | `goal=`, game rules → training system prompt |
| `info["history"]` (via HistoryRecorder) | list[{obs, action}] | step count, prior context |
| `info["invalid_action"]` | bool | `<state_flags> error=` |

**Conversion procedure (per environment category):**

```
1. Parse obs.text (TextArena game state text):
   → Extract board/grid state if present (2048 grid, Sokoban map, etc.)
   → Extract score/progress if present
   → Extract game messages (invalid move, win/loss, etc.)

2. For multi-turn games, create entities from game state:
   → 2048: each non-zero tile → entity with value, position
   → Sokoban: player, boxes, targets, walls → entities with positions
   → Chess: pieces → entities with position, type
   → Minesweeper: revealed cells, flags → entities

3. For single-turn puzzles, create entities from visual structure:
   → Grid cells, graph nodes, geometric shapes → entities
   → Constraint annotations → relations

4. Build relations from spatial layout:
   → adjacent(e1,e2) for neighboring tiles/cells
   → blocks(wall,path) for movement constraints
   → contains(region,entity) for spatial grouping

5. State flags from obs.text and info:
   → progress = score/target or boxes_on_target/total_boxes
   → phase = early/mid/late from step count vs max_episode_steps
   → error = info["invalid_action"] text if True

6. Targets (heuristic):
   → target = highest-value merge candidate (2048), nearest unplaced box (Sokoban), etc.
   → blocker = wall/obstacle preventing target action
   → constraint = from env.description rules

7. Actions from env.description valid moves:
   → 2048: [Up],[Down],[Left],[Right]
   → Sokoban: [w],[a],[s],[d]
   → Keep only contextually relevant subset

8. Truncation:
   → For large grids (>8x8): only emit entities near player/action zone
   → For repetitive cells: group into regions ("empty_region_1[cells=12]")
   Target: ≤ 20 entities for 8B model.
```

**Example output (Gym-V, 2048):**

```text
<state>
domain=gymv
task=Game2048-v0
goal=Reach a 2048 tile by merging identical numbers
step=14

<entities>
e1[type=object, label=tile_512, pos=0,0,1,1]
e2[type=object, label=tile_256, pos=0,1,1,1]
e3[type=object, label=tile_128, pos=0,2,1,1]
e4[type=object, label=tile_64, pos=1,0,1,1]
e5[type=object, label=tile_32, pos=1,2,1,1]
e6[type=object, label=tile_4, pos=2,1,1,1]
e7[type=object, label=tile_2, pos=3,3,1,1]
e8[type=region, label=empty, cells=9]

<attributes>
e1.value=512
e2.value=256
e3.value=128
e4.value=64
e5.value=32
e6.value=4
e7.value=2

<relations>
adjacent(e1,e2)
adjacent(e2,e3)
adjacent(e1,e4)
merge_candidate(e4,e5)

<state_flags>
progress=0.25
phase=mid
error=null
dialog_open=false
input_pending=false

<targets>
target=e1
blocker=null
constraint=build towards top-left corner
candidate_set=[e1,e2,e3,e4]

<uncertainty>
e8.cells=low

<actions>
a1=[Left]
a2=[Up]
a3=[Right]
a4=[Down]
</state>
```

**Example output (Gym-V, Sokoban):**

```text
<state>
domain=gymv
task=Sokoban-v0
goal=Push all boxes onto target squares
step=8

<entities>
e1[type=object, label=player, pos=3,2,1,1]
e2[type=object, label=box, pos=2,2,1,1]
e3[type=object, label=box, pos=3,4,1,1]
e4[type=object, label=box_on_target, pos=1,3,1,1]
e5[type=object, label=target, pos=2,4,1,1]
e6[type=object, label=target, pos=4,2,1,1]
e7[type=region, label=wall, cells=16]
e8[type=region, label=floor, cells=12]

<attributes>
e4.state=solved
e2.state=free
e3.state=free

<relations>
adjacent(e1,e2)
blocks(e7,e2)
pushable(e1,e2,direction=up)
adjacent(e3,e5)

<state_flags>
progress=0.33
phase=mid
error=null
dialog_open=false
input_pending=false

<targets>
target=e3
blocker=e7
constraint=cannot pull boxes
candidate_set=[e2,e3]

<uncertainty>
e7.cells=low

<actions>
a1=[w]
a2=[a]
a3=[s]
a4=[d]
</state>
```

### 12d. Schema design decisions for 8B model

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

### 12e. Training label generation pipeline (plan only)

```
For each episode step in {Gym-V, BrowserGym}:

  1. Collect native observation:
       Gym-V:  (obs.image, obs.text, env.description, info)
       Browser: (obs["screenshot"], obs["axtree_object"], obs["dom_object"],
                 obs["extra_element_properties"], obs["goal"], obs["url"],
                 obs["focused_element_bid"], obs["last_action_error"])

  2. Run domain adapter → raw structured summary (all fields populated from text state)

  3. Apply truncation/pruning to fit token budget

  4. Validate format: regex check all tags present, entity refs consistent

  5. Store as training pair:
       input:  image + system_prompt(env.description + schema_spec)
       output: structured summary text

  Training label = the output.
  VLM must learn to produce this from image alone (or image + minimal text context).
```

### 12f. Schema versioning

The schema will evolve. Version it explicitly:

- **v0.1** — initial schema from this document, tested on 2048 + Sokoban + MiniWoB++
- **v0.2** — revised after first training run (expect field additions/removals)
- **v1.0** — stable schema validated across ≥ 3 Gym-V categories + ≥ 1 BrowserGym benchmark

Schema version goes in the system prompt so the model can be trained on mixed versions during transition.

---

## 13. Chosen visual reasoning & video reasoning benchmarks

The following benchmarks are selected for multi-step visual reasoning evaluation. The list is mixed image + video — the first three are image-based, the last two are video-based.

### 13a. Image-based benchmarks

**CLEVR** — Synthetic, compositional visual reasoning with functional programs; officially released and downloadable.

- Download: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
- Official page: https://cs.stanford.edu/people/jcjohns/clevr/

**GQA** — Real-image visual reasoning with scene graphs, structured question semantics, and public download files for scene graphs, questions, and images.

- Downloads:
  - https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
  - https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
  - https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
- Official page: https://cs.stanford.edu/people/dorarad/gqa/download.html

**ToolVQA** — A newer benchmark explicitly aimed at multi-step visual reasoning with tool use; the project page and repo point to a public Google Drive dataset release.

- Download: https://drive.google.com/drive/folders/1diRjF2jK0aHoAMximnT7jNg4eN96ppCp?usp=sharing
- Repo: https://github.com/Fugtemypt123/ToolVQA-release
- Project page: https://fugtemypt123.github.io/ToolVQA-website/

### 13b. Video-based benchmarks

**SIV-Bench** — Open and very relevant for social interaction understanding and reasoning. The project page links both Code and Dataset, and the Hugging Face dataset page confirms the public dataset repo. Contains 2,792 real-world video clips and 8,792 high-quality QAs.

- Dataset: https://huggingface.co/datasets/Fancylalala/SIV-Bench
- Project page: https://kfq20.github.io/sivbench/
- Code: https://github.com/kfq20/SIV-Bench

**Video-Holmes** — Very relevant for complex multi-step clue-chaining reasoning. Contains 1,837 questions from 270 manually annotated suspense short films. Videos, questions, and eval code are packaged on GitHub and Hugging Face.

- Repo: https://github.com/TencentARC/Video-Holmes
- Dataset: https://huggingface.co/datasets/TencentARC/Video-Holmes
- Project page: https://video-holmes.github.io/Page.github.io/
- Download method:

```bash
git clone https://github.com/TencentARC/Video-Holmes.git
cd Video-Holmes
pip install huggingface_hub
python download.py --hf_token YOUR_HUGGINGFACE_ACCESS_TOKEN
unzip Benchmark/videos.zip -d Benchmark/
unzip Benchmark/annotations.zip -d Benchmark/
```

### 13c. Summary table

| Benchmark | Modality | Focus | Scale | Public |
|-----------|----------|-------|-------|--------|
| CLEVR | Image | Compositional visual reasoning | ~860K questions | Yes |
| GQA | Image | Scene-graph grounded reasoning | ~22M questions | Yes |
| ToolVQA | Image | Multi-step reasoning with tool use | — | Yes (Google Drive) |
| SIV-Bench | Video | Social interaction understanding | 2,792 clips, 8,792 QAs | Yes (Hugging Face) |
| Video-Holmes | Video | Multi-step clue-chaining | 270 films, 1,837 questions | Yes (Hugging Face) |

---

## 14. Visual grounding for benchmark evaluation via vlm_wrapper

The `vlm_wrapper` pipeline (§12, this repo) already provides the machinery needed to do visual grounding on the chosen benchmarks — the key is matching each benchmark's modality (image vs. video) and reasoning style (compositional, scene-graph, tool-use, temporal, clue-chaining) to the right combination of vlm_wrapper heads and tool registries.

### 14a. What "visual grounding" means here

Visual grounding = for every entity, relation, or evidence hop the model references in its answer, trace it back to a specific pixel region (bounding box) and/or a specific temporal location (frame index / timestamp) in the source image or video. The `vlm_wrapper` produces this in the `<entities>` section (each entity has `pos=x,y,w,h`) and the hop trace (§7, each hop has `grounding=[eid, ...]`).

Three heads are available:

| Head | Input | Grounding method | Latency | Best for |
|------|-------|------------------|---------|----------|
| Head 1 — Heuristic | Text state (AXTree, obs.text) | Regex/tree-walking → schema | ~ms | Interactive envs with text state |
| Head 2 — Vision | Screenshot → GPT-4o / Qwen3 | VLM directly produces schema | ~2-4s | General images, static QA |
| Head 3 — OmniParser-v2 | Screenshot → YOLO + OCR + Florence-2 | Local model ensemble → schema | ~0.6s GPU | UI screenshots, precise bbox grounding |
| Tool loop | Screenshot → VLM calls tools | Multi-hop tool calling → schema | ~5-15s | Complex reasoning with evidence chains |

For benchmarks with no environment text state (all five chosen benchmarks), Head 1 is not applicable. The relevant heads are **Head 2**, **Head 3**, and the **tool-calling loop**.

### 14b. Per-benchmark grounding strategy

**CLEVR (image, compositional)**

- Modality: single synthetic image per question.
- Grounding need: locate each mentioned object (sphere, cube, cylinder) and verify spatial/attribute relations.
- vlm_wrapper approach:
  - `build_visual_registry(image)` → tools: `detect_objects`, `spatial_query`, `count_objects`, `extract_colors`, `visual_search`.
  - The VLM reads the question, calls `detect_objects` to get all shapes with bboxes, calls `spatial_query` to verify "left of" / "behind" relations, calls `extract_colors` to confirm "red" / "blue" attributes.
  - Each hop in the answer maps to a tool call → grounded evidence.
  - Entry point: `visual_generate_label_with_tools(image, goal=question, task_id="clevr.xxx")`.

**GQA (image, scene-graph grounded)**

- Modality: single real-world image per question.
- Grounding need: locate objects mentioned in the question, verify scene-graph relations (on, near, wearing, holding).
- vlm_wrapper approach:
  - Same as CLEVR: `build_visual_registry(image)`.
  - GQA's ground-truth scene graphs provide gold bbox annotations — use these to evaluate whether the vlm_wrapper's `detect_objects` bboxes match (IoU comparison).
  - The hop trace should align with GQA's functional program (select → filter → relate → query) — each program step maps to a tool call.
  - GQA scene graphs can also be loaded as a validation oracle: after the VLM produces its schema, compare `<entities>` bboxes against GQA gold scene-graph bboxes.

**ToolVQA (image, multi-step tool-use)**

- Modality: single image per question, but the benchmark is explicitly about tool-augmented reasoning.
- Grounding need: the benchmark already defines tool use as part of the reasoning — visual grounding is the piece that connects tool outputs back to image regions.
- vlm_wrapper approach:
  - `build_visual_registry(image)` provides the vision-model tool suite.
  - ToolVQA's own tools (table lookup, knowledge graph query, etc.) can be added as custom tools via `ToolRegistry.register()` alongside the visual tools.
  - The combined registry gives the VLM both visual grounding (detect, describe, spatial) and external knowledge tools in the same loop.
  - Hop traces from `run_tool_loop` directly capture the multi-hop reasoning chain with grounded evidence.

**SIV-Bench (video, social interaction)**

- Modality: video clips (2,792 clips, 8,792 QAs).
- Grounding need: temporal grounding (which frames show the relevant social interaction) + spatial grounding (which people/objects are involved).
- vlm_wrapper approach:
  - `build_video_visual_registry(frames=decoded_frames, fps=fps)` → full suite: temporal navigation + vision-model detection + cross-frame tools.
  - Video-level tools: `get_frame`, `sample_frames`, `detect_scene_changes`, `temporal_navigate` — find the relevant temporal window.
  - Cross-frame tools: `track_object` (follow a person across frames), `find_moment` (find when a social interaction starts/ends), `detect_activity` (classify interaction type), `compare_elements` (what changed between frames).
  - Per-frame tools: `detect_objects_at_frame` (ground people/objects at a specific moment), `describe_region` (caption an ambiguous gesture or expression).
  - Entry point: `video_visual_generate_label_with_tools(frames, goal=question, fps=fps)`.
  - Grounding output: both temporal (frame index / timestamp per hop) and spatial (bbox per entity per frame).

**Video-Holmes (video, clue-chaining)**

- Modality: suspense short films (270 films, 1,837 questions).
- Grounding need: multi-step clue chain — each clue is grounded in a specific temporal moment and spatial region. Evidence accumulates across time.
- vlm_wrapper approach:
  - Same registry as SIV-Bench: `build_video_visual_registry(...)`.
  - The tool loop is particularly valuable here because Video-Holmes requires *multi-hop* reasoning: the VLM must find clue A in frame range X, then clue B in frame range Y, then chain them to answer the question.
  - Tool call trace naturally captures the clue chain: `detect_scene_changes()` → `get_frame(idx)` → `detect_objects_at_frame(idx)` → `describe_region(...)` → navigate to next clue → repeat.
  - The hop trace (§7) maps directly to Video-Holmes's reasoning structure: trigger → hop1 (find first clue) → hop2 (find second clue) → ... → output (answer).

### 14c. Implementation plan for benchmark grounding

```
Phase 1 — Image benchmarks (CLEVR, GQA, ToolVQA)

  1. Write benchmark loaders:
       clevr_loader.py    — reads CLEVR questions + images
       gqa_loader.py      — reads GQA questions + images + scene graphs
       toolvqa_loader.py  — reads ToolVQA questions + images + tool defs

  2. For each benchmark sample:
       image, question, gold_answer = loader.load(sample_id)
       result = visual_generate_label_with_tools(
           image, goal=question, task_id=f"{benchmark}.{sample_id}",
       )

  3. Evaluate:
       a) Answer accuracy: compare result vs gold_answer
       b) Grounding quality (GQA only): IoU between predicted entity
          bboxes and GQA gold scene-graph bboxes
       c) Hop trace quality: does the tool call sequence align with
          the gold functional program (GQA) or expected tool chain (ToolVQA)?

Phase 2 — Video benchmarks (SIV-Bench, Video-Holmes)

  1. Write benchmark loaders:
       sivbench_loader.py      — reads video clips + QA pairs
       videoholmes_loader.py   — reads films + questions + annotations

  2. For each benchmark sample:
       frames = decode_video(video_path)
       result = video_visual_generate_label_with_tools(
           frames, goal=question, fps=fps,
           task_id=f"{benchmark}.{sample_id}",
       )

  3. Evaluate:
       a) Answer accuracy: compare result vs gold_answer
       b) Temporal grounding: do the accessed frame indices overlap
          with gold temporal annotations (if available)?
       c) Evidence chain quality: does the tool trace capture the
          expected multi-hop reasoning structure?
```

### 14d. Grounding-aware schema extensions for benchmarks

The base schema (§12a) needs minor benchmark-specific additions:

```text
<evidence>
hop1.tool=detect_objects
hop1.result_ref=e1,e3
hop1.frame=null                  # null for image benchmarks
hop1.confidence=high

hop2.tool=spatial_query
hop2.result_ref=e1,e3
hop2.relation=left_of
hop2.frame=null

<answer>
answer={predicted answer}
grounding=[e1,e3]
evidence_chain=[hop1,hop2]
confidence=high
```

For video benchmarks, add temporal grounding:

```text
hop1.frame=42
hop1.timestamp=14.0
hop2.frame=87
hop2.timestamp=29.0
```

### 14e. What this buys us

1. **Auditable reasoning**: every answer has a tool-call trace mapping to specific image/video regions. Hallucination is detectable by checking whether the cited entities actually exist at the cited locations.
2. **SFT training data**: the (question, tool_trace, answer) triples from benchmark evaluation become supervised fine-tuning data for Qwen3-VL-8B to learn grounded multi-hop reasoning.
3. **Cross-benchmark transfer**: the schema is the same across all five benchmarks — a model trained on CLEVR/GQA tool traces transfers to SIV-Bench/Video-Holmes because the entity/relation/hop format is shared.
4. **Grounding evaluation metric**: beyond answer accuracy, we can measure grounding precision (% of cited entities with correct bboxes) and temporal precision (% of cited frames within gold temporal windows).

### 14f. Reference implementation

The existing `vlm_wrapper` modules already implement all required pieces:

| Component | Module | Status |
|-----------|--------|--------|
| Visual tool registry | `tools_visual.py` | Done — 9 tools (detect_objects, describe_region, spatial_query, visual_search, count_objects, classify_scene, measure_distance, extract_colors, read_text_region) |
| Video tool registry | `tools_video.py` | Done — 8 tools (get_frame, sample_frames, compare_frames, detect_scene_changes, get_video_info, read_text_in_frame, temporal_navigate, list_valid_actions) |
| Cross-frame registry | `tools_video_visual.py` | Done — 6 tools (track_object, summarize_clip, find_moment, detect_activity, compare_elements, detect_objects_at_frame) |
| OmniParser grounding | `grounding.py` | Done — YOLO + OCR + Florence-2 pipeline |
| Tool-calling loop | `tool_loop.py` | Done — multi-turn VLM loop with trace capture |
| Convenience wrappers | `tool_loop.py` | Done — `visual_generate_label_with_tools`, `video_visual_generate_label_with_tools` |
| Benchmark loaders | — | **TODO** — CLEVR, GQA, ToolVQA, SIV-Bench, Video-Holmes |
| Evaluation harness | — | **TODO** — accuracy + grounding IoU + temporal precision + trace quality |
| Schema extensions | `schema.py` | **TODO** — `<evidence>` and `<answer>` sections |

Reference repo: https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent/tree/main/vlm_wrapper
