# vlm_wrapper — Visual State Parser for Gym-V and BrowserGym

## What this does

Converts screenshots from video games (Gym-V) and web pages (BrowserGym) into a shared structured text schema (`<state>…</state>`), which plugs into the existing Game-AI-Agent pipeline for skill retrieval and decision-making.

Two heads produce the same schema:

| | Head 1 — Heuristic | Head 2 — Vision |
|---|---|---|
| Input | `obs.text` / AXTree (text) | Screenshot (pixels) |
| Method | Regex + tree-walking | GPT-4o vision API call |
| Cost | Free | ~$0.01/call |
| Latency | <1 ms | ~1–3 s |
| Use case | Real-time RL, baselines, label validation | Training-label generation for Qwen3-VL-8B |

## File layout

```
vlm_wrapper/
├── __init__.py              # exports both heads
├── schema.py                # shared schema spec, system prompt, image encoding, parsing
├── gymv_heuristic.py        # Head 1: obs.text → schema (Gym-V)
├── browser_heuristic.py     # Head 1: AXTree/DOM → schema (BrowserGym)
├── gymv_adapter.py          # Head 2: screenshot → GPT-4o → schema (Gym-V)
└── browser_adapter.py       # Head 2: screenshot → GPT-4o → schema (BrowserGym)
```

## End goal

Train Qwen3-VL-8B via SFT distillation from GPT-4o labels so that at inference time the 8B model sees **only a screenshot** and produces the structured schema. The heuristic head provides cheap validation and ground-truth fields that the vision head cannot observe from pixels alone.

---

## Can Qwen3-VL-8B learn this task?

**Yes.** The task is structured template generation (not open-ended reasoning), and Gym-V's own benchmarks show Qwen3-VL-8B already scores 16.5 average *zero-shot* on harder tasks (logic, algorithms, multi-turn strategy). Structured scene parsing is strictly easier, and we fine-tune rather than zero-shot. Distillation from GPT-4o to 8B is well-established for structured output — expect >90% field accuracy after ~3-5K labeled examples per domain.

## Challenges and mitigations

### Challenge 1: Entity position accuracy from pixels

Getting `pos=x,y,w,h` right from screenshots is the hardest perceptual sub-task. GPT-4o itself isn't great at precise pixel coordinates. Grid games are easier (regular grid → integer coords), but browser element positions will be noisy.

**Mitigation — tool use at inference time.**

Instead of requiring the VLM to hallucinate exact pixel coordinates, teach it to **call tools** that return precise positions:

- **Gym-V:** The model emits a tool call like `query_grid_pos(entity_label)` and receives the exact grid coordinates from the environment's internal state. Gym-V wrappers already expose state metadata for this.
- **BrowserGym:** The model emits `query_element_bbox(bid)` and the environment returns the exact bounding box from `extra_element_properties[bid]["bbox"]`. BrowserGym already computes this via CDP.

This means the VLM only needs to **identify and label** entities from pixels (which it's good at), then **delegate coordinate lookup** to a tool (which is exact). The schema output still contains `pos=` fields, but they're filled by tool responses rather than pixel estimation.

The heuristic head provides ground-truth positions for every training example (from AXTree bboxes / grid state), so we can:
1. Train the VLM to produce tool calls for position fields.
2. Validate tool-call accuracy against heuristic-head ground truth.
3. Fall back to heuristic positions if the tool call fails.

Tool-use training is a separate stage after basic SFT — do not conflate them. Sequence: SFT for entity detection first, then add tool-use for precise grounding.

### Challenge 2: Entity coverage on cluttered web pages

A busy web page can have 50+ interactive elements. We cap at 25 entities. The model must learn *which* entities matter for the current task goal — a prioritization problem.

**Mitigation — API inference for hard pages.**

For complex pages where the 8B model misses critical entities or includes irrelevant ones, use a **cascaded approach**:

1. **8B model first (fast, cheap).** Run Qwen3-VL-8B on the screenshot. Check entity count and coverage heuristics.
2. **Escalate to API if needed.** If the schema has <5 entities on a page that should have more, or if key goal-related terms are missing from entity labels, escalate to GPT-4o (or another strong vision model) for that specific observation.
3. **Selective escalation criteria:**
   - Entity count below threshold for the page complexity.
   - Zero entities matching keywords from the `goal=` field.
   - Schema validation failures (missing sections, malformed IDs).
   - Confidence signal: if the 8B model is fine-tuned with uncertainty estimation (e.g., outputting `<uncertainty>` scores), escalate when uncertainty is high.

This keeps average cost low (most frames use the 8B model at ~$0/call) while maintaining quality on the hard tail. In practice, expect escalation on <10% of web pages after good SFT, dropping further with more training data.

**Additional mitigations:**
- The `goal=` field in the prompt helps the model focus on task-relevant elements.
- Start training on simple pages (MiniWoB++) before complex ones (WebArena).
- The heuristic head (AXTree parsing) provides a coverage baseline — compare entity counts to flag under-detection.

### Challenge 3: Format compliance at 8B scale

8B models occasionally drop section tags, mismatch entity IDs, or produce malformed lines.

**Mitigation:** The flat tagged format (no nested JSON) was specifically chosen for 8B reliability. Additional defenses:
- Constrained decoding (vLLM supports it) to force section tag ordering.
- Post-processing regex cleanup for minor formatting issues.
- Expect ~95% format compliance after SFT, improving to ~98%+ with constrained decoding.

### Challenge 4: Relations require game semantics, not just vision

Entity listing is visual grounding (8B models do this). But `blocks(e5,e1)` or `merge_candidate(e4,e5)` requires understanding game rules.

**Mitigation:** Game rules and goal are passed as text context in the prompt — the model doesn't infer rules from pixels alone. Relations that depend on non-visual logic (e.g., "this box is stuck because walls are on two sides") can also be delegated to tools: the model identifies entities visually, then calls `check_relation(e1, e2, relation_type)` against the environment state.

---

## Recommended training sequence

1. **Gym-V games first** (2048, Sokoban, Minesweeper). Grid layouts, limited entities, clear structure. Clean GPT-4o labels, fast iteration. Expect >90% field accuracy within a few thousand examples per game.

2. **Validate with heuristic head.** For every GPT-4o label, run the heuristic head on the same observation. Flag disagreements (entity count mismatch, missing fields) for review. This catches GPT-4o hallucinations before they enter training data.

3. **Add tool-use training.** After basic SFT, add a second training stage where the model learns to emit tool calls for position queries and relation checks. Train on examples where the tool-call version is more accurate than the pixel-only version.

4. **Browser: MiniWoB++ → WebArena.** Start with simple pages, validate that the schema and entity prioritization work, then scale to complex real-web pages. Use API escalation (Challenge 2) for the hard tail during data collection.

5. **Expected data budget:** ~3-5K labeled examples per domain for solid SFT. At ~$0.01/example with GPT-4o, that's $30-50 per domain.

---

## Quick start

```python
# Head 1 — Heuristic (Gym-V)
from vlm_wrapper import gymv_heuristic_schema
schema = gymv_heuristic_schema(
    obs_text="| 2 | 4 | 0 | 0 |\n| 0 | 16 | 8 | 0 |",
    description="You are playing 2048. Valid moves: [Up], [Down], [Left], [Right].",
    task_id="Game2048-v0",
    step=5,
)

# Head 1 — Heuristic (BrowserGym)
from vlm_wrapper import browser_heuristic_schema
schema = browser_heuristic_schema(obs, step=3, task_id="webarena.shopping.143")

# Head 2 — Vision (Gym-V)
from vlm_wrapper import gymv_generate_label
result = gymv_generate_label(frame, goal="Reach 2048", task_id="Game2048-v0", step=5)
print(result["schema"])    # <state>…</state> or None
print(result["warnings"])  # validation issues

# Head 2 — Vision (BrowserGym)
from vlm_wrapper import browser_obs_to_schema
result = browser_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
print(result["schema"])
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLM_LABEL_MODEL` | `gpt-4o` | Vision model for Head 2 |
| `VLM_LABEL_MAX_TOKENS` | `1200` | Max output tokens per call |
| `VLM_LABEL_TEMPERATURE` | `0.2` | Low temp for consistent structured output |

Or pass `model=`, `api_key=`, `base_url=` directly to any `generate_label` call.
