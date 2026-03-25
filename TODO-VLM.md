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
