# PLAN: Action Agent (Decision Agent)

**Scope:** The decision-making agent that consumes structured `<state>` schemas from the [Visual Grounding](PLAN-VISUAL-GROUNDING.md) pipeline and selects/executes environment actions guided by skills from the [Skill Bank](PLAN-SKILL-BANK.md).

**Upstream:** Structured schema from visual grounding (vlm_wrapper heads); skill guidance from Skill Bank.
**Downstream:** Environment actions; experience trajectories fed back to Skill Bank and GRPO training.

---

## 1. Architecture overview

The action agent implements the three-stage decision loop from COS-PLAY (COLM 2026):

```
structured_state (from VLM grounding)
    ↓
skill retrieval (from Skill Bank)  ←──── re-select on stall/completion/abort
    ↓
intention update (subgoal inference)
    ↓
action execution (from valid action set)
    ↓
reward computation (r_env + r_follow + r_cost)
    ↓
experience → Skill Bank ingestion + GRPO buffer
```

### Per-step loop

1. **`get_state_summary()`** — deterministic + LLM state compression into `key=value` format (≤400 chars). Prefers `structured_state` from VLM grounding when available.
2. **`infer_intention()`** — LLM produces a `[TAG] subgoal phrase` from summary + context (last actions, task). Tags: SETUP, CLEAR, MERGE, ATTACK, DEFEND, NAVIGATE, POSITION, COLLECT, BUILD, SURVIVE, OPTIMIZE, EXPLORE, EXECUTE.
3. **Skill re-selection check** — triggers re-query when: no active skill, duration exceeded, zero-reward stall (≥4 steps), abort/success criteria matched in current state.
4. **`get_skill_guidance()`** — queries `SkillQueryEngine` (RAG mode) using `game_name + intention + state_text[:1500]` as query, with `structured_state` converted to `{predicate: float}` for applicability scoring. Returns skill_id, protocol, execution_hint, failure_modes.
5. **`action()`** — builds prompt: system prompt + skill guidance + recent actions/rewards + valid action list → LLM → action.
6. **`parse_response()`** — multi-strategy action extraction: exact match → numbered selection → substring → edit distance → token overlap → RAG embedding semantic match as final fallback.
7. **Anti-repetition** — if same action repeated N times with 0 reward, randomly pick alternative.
8. **`env.step(action)`**
9. **`_SkillTracker.update()`** — advance protocol step index, track reward-on-skill, switch count.
10. **Build `Experience`** — state, action, reward, next_state, done, intentions, tasks, sub_tasks (active skill), summary_state, available_actions.

---

## 2. Tiered model architecture

### Design choice: 8B-only vs. tiered 32B/72B + 8B

Two options were considered:

| | Option A: 8B + GRPO LoRA everywhere | Option B: 32B/72B inference + 8B trained executor |
|---|---|---|
| **Decision loop** | 8B (trained) | 8B (trained) — identical |
| **Skill Crafter / Failure Reflector** | 8B (trained) | 32B/72B (inference-only) |
| **Per-step latency** | Fast | Fast (same 8B in the hot path) |
| **Skill/protocol quality** | Limited by 8B reasoning ceiling | Strong — 32B/72B excels at composition, diagnosis, analogy |
| **Trainability** | Fully end-to-end | Real-time loop fully trainable; offline components frozen |
| **Cold-start** | Weak until GRPO converges | 32B/72B generates high-quality trajectories from day one |

**Decision: Option B (tiered).** The pipeline's offline components — Skill Crafter (composition, generalization, hypothesis), Failure Reflector (backward trace analysis, diagnosis, recovery), and cold-start generation — require multi-step counterfactual reasoning, cross-domain analogy, and structured diagnosis that 8B models consistently struggle with. These components run *between* episodes, not per-step, so the larger model adds zero latency to the decision loop.

### Three model tiers

```
┌─────────────────────────────────────────────┐
│  Tier 0: GPT-5.4 / frontier (API, offline)   │
│  • Initial cold-start labeling               │
│  • Reward model / judge for GRPO             │
└──────────────────┬──────────────────────────┘
                   ↓ bootstrap data
┌─────────────────────────────────────────────┐
│  Tier 1: Qwen3-32B/72B (inference, offline)  │
│  • Skill Composer: effect chaining           │
│  • Skill Hypothesizer: failure → new skills  │
│  • Skill Generalizer: cross-domain transfer  │
│  • Failure Reflector: localize + diagnose    │
│  • Trajectory generation for GRPO buffer     │
│  • Skill verification judging                │
└──────────────────┬──────────────────────────┘
                   │ skills, protocols, recovery
                   │ patches, training signal
                   ↓
┌─────────────────────────────────────────────┐
│  Tier 2: Qwen3-8B + GRPO LoRA (trained)     │
│  • schema_gen: screenshot → <state> schema   │
│  • hop_select: schema + trace → next hop     │
│  • skill_select: schema → which skill        │
│  • Action execution + anti-repetition        │
│  • SkillTracker protocol following           │
└─────────────────────────────────────────────┘
```

### Why the 8B real-time loop stays fast

The 32B/72B never enters the per-step loop. Per-step operations (schema_gen, hop_select ×1–8, skill_select on reselect, action execution) are all served by the 8B on vLLM, identical to Option A. The 32B/72B runs as a separate offline process:

| Offline operation | When it runs | Frequency |
|---|---|---|
| Failure Reflection (localize, diagnose) | After a failed episode ends | ~1× per failed episode |
| Skill Composition (effect chaining) | Periodic batch job | Every N episodes |
| Skill Hypothesis (from failure patterns) | When failure memory accumulates | Rare |
| Cross-domain transfer | When adding a new domain | One-time |
| Cold-start trajectory generation | Before GRPO training begins | One-time |

Offline reflection can be pipelined: the 8B rolls out episode batch K+1 while the 32B/72B reflects on batch K. GRPO gradient computation dominates wall-clock time, so the 32B/72B reflection effectively hides inside existing training overhead.

### Multi-run reasoning for 32B/72B

Even at 32B/72B scale, the offline tasks (failure diagnosis, skill composition, cross-domain transfer) are too complex for single-pass inference. The 32B/72B must run **multiple reasoning passes** per task:

| Offline task | Why single-pass is insufficient | Multi-run strategy |
|---|---|---|
| **Failure localization** | Backward trace analysis requires re-evaluating each hop against its context; a single pass often fixates on the symptom step, not the root cause | Pass 1: identify symptom step. Pass 2: re-evaluate each prior hop with targeted prompts. Pass 3: confirm root cause via counterfactual ("would fixing step K have prevented the failure?") |
| **Failure diagnosis** | Diagnosis involves hypothesis generation + verification; one-shot diagnoses frequently hallucinate causal links | Pass 1: generate candidate explanations. Pass 2: verify each against context_snapshot. Pass 3: select most consistent explanation |
| **Skill composition** | Effect chaining across 2+ skills requires verifying precondition/postcondition compatibility, which involves combinatorial checking | Pass 1: propose candidate compositions. Pass 2: verify effect chain validity per pair. Pass 3: generate protocol + test expectations |
| **Skill hypothesis** | Novel skill proposals from failure patterns are speculative; most initial proposals are low-quality | Best-of-N sampling (N=3–5): generate N proposals, score by contract completeness + novelty, keep top-K |
| **Cross-domain transfer** | Schema-slot mapping between domains is ambiguous; naive mappings fail | Pass 1: identify shared structural slots. Pass 2: propose mapping candidates. Pass 3: instantiate and sanity-check with target-domain examples |

This multi-run design means each offline task costs ~3–5× the tokens of a single 32B/72B call, but these costs are amortized across many episodes and remain far cheaper than putting the 32B/72B in the per-step loop. The quality improvement from multi-run (especially for failure localization and diagnosis) is critical — single-pass failure reflection would produce noisy recovery patches that degrade rather than improve skills.

**Budget guideline:** ~500–2000 tokens per offline reasoning pass × 3–5 passes × a few tasks per episode batch = on the order of 10K–50K tokens of 32B/72B inference per training iteration. Negligible compared to GRPO rollout costs.

### Routing

All three tiers share the same code path; `API_func.ask_model` routes to the correct backend:

| Tier | Routing |
|---|---|
| Tier 0: GPT-5.4 | OpenRouter / OpenAI API |
| Tier 1: Qwen3-32B/72B | vLLM serving (offline batch, quantized) |
| Tier 2: Qwen3-8B | vLLM serving (real-time, LoRA hot-swap) |

---

## 3. Skill-guided decision making

### Skill selection (RAG mode)

`select_skill_from_bank()` tries four paths in order:

1. **`SkillQueryEngine.select()`** — richest path (RAG relevance + applicability + structured guidance)
2. **`SkillQueryEngine.query_for_decision_agent()`** — convenience wrapper
3. **`SkillBankAgent.select_skill()`** — agent-level selection
4. **TF-IDF keyword fallback** — only when no query engine available

### Scoring

Each candidate skill is scored on three axes:

| Component | Weight | Source |
|-----------|--------|--------|
| Retrieval relevance | 40% | RAG embedding cosine similarity + keyword Jaccard |
| Execution applicability | 35% | Effect compatibility against current state predicates |
| Historical pass rate | 25% | Success rate from past executions |

### Protocol-aware lifecycle

The `_SkillTracker` manages:
- Protocol step tracking with `>>` marker at current step
- Duration caps per skill
- Stall detection (≥4 steps with reward ≤0)
- Success/abort criteria keyword matching in current state
- Automatic alternate skill selection when same skill re-selected after failure

---

## 4. Reward computation

**r_total = r_env + w_follow × r_follow + r_cost**

| Component | Source | Purpose |
|-----------|--------|---------|
| r_env | Raw environment reward | Task progress |
| r_follow | Skill contract eff_add predicate matching | Skill-following shaping |
| r_cost | Query/call costs + skill switch penalty | Discourage excessive tool use |

### r_follow details

Checks how many `eff_add` predicates from the active skill's contract appear in the observation:
- **+0.05** per newly satisfied predicate
- **+0.20** when all eff_add predicates satisfied
- **-0.01** per step with no predicate progress

### r_cost defaults

| Parameter | Default |
|-----------|---------|
| `query_mem_cost` | -0.05 |
| `query_skill_cost` | -0.05 |
| `call_skill_cost` | -0.02 |
| `skill_switch_cost` | -0.10 |

---

## 5. Two-level MDP (long-horizon reasoning)

The core architectural insight (from `LONG_HORIZON_REASONING.md`): multi-hop visual reasoning is reframed as a **long-horizon interaction** problem. Instead of performing all reasoning hops inside a single LLM inference pass, each hop becomes an explicit step in an MDP — turning the reasoning trace into a proper trajectory that can be optimized end-to-end with RL.

### Current vs. target granularity

|                    | Current (single-call)                                           | Target (two-level MDP)                                     |
|--------------------|------------------------------------------------------------------|-------------------------------------------------------------|
| **One "step"**     | Screenshot → (all hops internally in one LLM call) → env action  | Outer: env action; Inner: each reasoning hop is a step      |
| **"Reasoning"**    | Happens *inside* a single inference pass (hop1→hop2→…→action)    | Happens *across* explicit steps (each hop is RL-trainable)  |
| **Horizon**        | Short (hops are hidden, only env actions count as steps)         | Long (inner hops + outer actions = full trajectory)         |

### Two-level MDP structure

```
┌─────────────────────────────────────────────────────┐
│  OUTER MDP (environment level)                       │
│  State: screenshot + task description                │
│  Action: click / type / game-move                    │
│  Reward: task success (sparse)                       │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │  INNER MDP (reasoning level)                  │   │
│  │  State: visual schema + hop trace so far       │   │
│  │  Action: reasoning operation (one of):         │   │
│  │    - GROUND(entity_query)  → find entities     │   │
│  │    - CHECK(relation/attr)  → verify condition  │   │
│  │    - RETRIEVE(memory_key)  → query skill bank  │   │
│  │    - CONCLUDE(intermediate)→ commit subgoal    │   │
│  │    - EXECUTE(env_action)   → exit inner loop   │   │
│  │  Reward: shaped from hop quality + outer reward│   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

Each outer step contains 1..N inner reasoning steps. The agent **learns when to stop reasoning and act**.

### Inner action vocabulary

The existing VLM Agent hop trace maps directly to the inner action space:

| Hop trace step | Inner action | vlm_wrapper tool |
|---------------|-------------|-----------------|
| Locate relevant entities | `GROUND(query)` | `detect_objects`, `visual_search` |
| Check constraints / relations | `CHECK(predicate)` | `spatial_query`, `check_relation` |
| Query skill bank / memory | `RETRIEVE(key)` | `select_skill_from_bank` |
| Commit intermediate result | `CONCLUDE(subgoal)` | (internal state update) |
| Execute environment action | `EXECUTE(action)` | env.step() — exits inner loop |

### Design decisions

**Re-observation between hops:**
- **Option A (default):** Hops operate on the same `<state>` schema, only updating an internal scratchpad. Cheaper, faster.
- **Option B (selective):** GROUND actions can trigger re-rendering or zooming into a region. More expensive but handles visual detail.
- **Recommendation:** A for games/web, B for visual QA where fine-grained visual grounding matters.

**Reward for inner hops:**
- Schema consistency reward (does the hop produce a valid schema update?)
- Hop trace quality reward (GPT-4o judges full trace, for GRPO)
- Progress shaping (reward for reducing uncertainty or resolving blockers)
- **Recommendation:** GRPO with trajectory-level reward; let the policy learn which hops matter. Aligns with the existing `r_follow`.

**Inner loop length:**
- Hard cap (max 8 hops per outer step) + learned EXECUTE decision.
- The existing `_SkillTracker` abort/success criteria naturally handles this.

### LoRA adapter layout

| Adapter | Purpose |
|---------|---------|
| `schema_gen` | Screenshot → `<state>` schema (Qwen3-VL) |
| `hop_select` | Schema + trace → next reasoning action (replaces single-call action) |
| `skill_select` | Schema → which reasoning skill to invoke |
| `segment` | Trajectory → skill boundary detection |
| `contract` | Segment → effects contract |

### Episode trajectory format

Each episode becomes a proper long-horizon trajectory:

```
(schema_0, GROUND, schema_0') → (schema_0', CHECK, schema_0'') → (schema_0'', EXECUTE(click), schema_1) → ...
```

This is segmentable by the Skill Bank — reasoning hop chains become discoverable skills.

---

## 6. Integration with Visual Grounding

The action agent currently consumes text observations. The integration path:

1. **Phase 1 (current):** `get_state_summary()` compresses raw text observations. Single-call reasoning.
2. **Phase 2 (schema input):** `get_state_summary()` receives the `<state>` schema directly from the VLM grounding pipeline, which replaces text-based compression with structured grounding.
3. **Phase 3 (inner MDP):** Action agent implements the two-level MDP. The `<state>` schema becomes the inner MDP state. GROUND actions call vlm_wrapper tools. Entity references enable grounded actions ("click e5" instead of "click the red jacket").

### Schema as inner MDP state

The `<state>` schema is the state representation for the inner reasoning MDP:
- `<entities>` + `<relations>` — what the agent knows about the scene (updated by GROUND/CHECK hops)
- `<uncertainty>` — drives whether to gather more info (GROUND) or act (EXECUTE)
- `<targets>` (target, blocker, candidate_set) — narrows the inner action space
- `<state_flags>` (progress, error, dialog_open) — lifecycle decisions, including when to CONCLUDE or abort

---

## 7. Two pipeline variants

### Pipeline A — `qwen3_decision_agent.py` (full skill lifecycle)

- Skill bank required (per-game, query engine, tracker)
- `_SkillTracker` with reselect, alternate, protocol steps
- Fuzzy + edit distance + RAG embedding action parsing
- Anti-repetition guard
- Output: `test_rollout/decision_agent/<game>/<timestamp>/`

### Pipeline B — `run_qwen3_8b_eval.py` (lightweight evaluation)

- Skill bank optional (`--bank` flag)
- Single query per step, no tracking
- Exact match + `extract_action()` parsing
- Multi-benchmark (LMGame-Bench + AgentEvolver + Orak)
- Resume support
- Output: `output/<model>/<game>/<timestamp>/`

---

## 8. Supported environments

| # | Stack | Game | Registry Key |
|---|-------|------|-------------|
| 1 | LMGame-Bench | 2048 | `twenty_forty_eight` |
| 2 | LMGame-Bench | Candy Crush | `candy_crush` |
| 3 | LMGame-Bench | Tetris | `tetris` |
| 4 | AgentEvolver | Avalon | `avalon` |
| 5 | AgentEvolver | Diplomacy | `diplomacy` |
| 6 | Orak | Super Mario | `super_mario` |

---

## 9. Uncertainty-driven GROUND triggering

The inner MDP agent actively manages information completeness through the GROUND action.  The `<uncertainty>` section in the schema is the communication channel between grounding and reasoning.

### When to GROUND vs. act

The `hop_select` adapter learns this trade-off end-to-end via GRPO:

| Schema signal | Inner MDP response | Rationale |
|---------------|-------------------|-----------|
| `uncertainty.e5.label=high` + skill needs `$target` | `GROUND(e5)` → re-detect | Skill can't execute without confident target |
| `uncertainty.e3.pos=medium` + action is coarse | `EXECUTE(action)` directly | Position uncertainty tolerable for coarse actions |
| `blocker=null` + skill is `blocker_prerequisite_replan` | `GROUND(scene)` → find blockers | Skill explicitly requires blocker identification |
| `candidate_set=[]` + skill is `locate_filter_select` | `GROUND(candidates)` → populate set | Empty candidate set means grounding was incomplete |
| All slots populated, uncertainty low | Skip GROUND → `CHECK` or `EXECUTE` | Grounding was sufficient, proceed |

### Slot coverage check before skill execution

Before the `_SkillTracker` activates a skill, it checks that the skill's required slots (from `SlotBinding`) are populated in the current state.  Missing slots trigger a GROUND hop rather than skill failure:

```
SkillTracker.activate(skill, current_schema)
  → check slot_bindings against <targets> + <entities>
  → if $target missing: insert GROUND(target_query) as hop 0
  → if $blocker missing but skill needs it: insert GROUND(blocker_query) as hop 0
  → proceed with skill protocol from hop 1
```

This means **grounding doesn't need to be perfect** — it needs to be *good enough* for the reasoning layer to identify what's missing and fill it in.  The inner MDP reward naturally optimises this: unnecessary GROUND hops waste budget, but missing critical information causes task failure.

See [Visual Grounding §12](PLAN-VISUAL-GROUNDING.md#12-schema-completeness-guarantee-grounding--reasoning-contract) for the full 4-layer guarantee.

---

## 10. TODO

| Task | Priority | Status |
|------|----------|--------|
| Integrate VLM schema as primary state input | P0 | Not started |
| Implement inner reasoning MDP (hop_select adapter) | P0 | Not started |
| Entity-referenced actions (click e5 instead of click(400,510)) | P1 | Not started |
| Inner hop reward shaping (schema consistency + progress) | P1 | Not started |
| Slot coverage check in _SkillTracker before skill activation | P1 | Not started |
| Uncertainty-driven GROUND insertion (hop 0 when slots missing) | P1 | Not started |
| Extend to BrowserGym action space | P1 | Not started |
| Extend to OSWorld action space | P2 | Not started |
| Video-based decision making (temporal action selection) | P2 | Not started |
| Learned EXECUTE timing (when to stop reasoning and act) | P2 | Not started |

---

## 11. Implementation

| File | Purpose |
|------|---------|
| `decision_agents/agent.py` | `VLMDecisionAgent`, `run_tool()`, `run_episode_vlm_agent()` |
| `decision_agents/agent_helper.py` | `get_state_summary()`, `infer_intention()`, `select_skill_from_bank()`, `EpisodicMemoryStore` |
| `decision_agents/reward_func.py` | `RewardComputer`, `compute_reward()` |
| `decision_agents/dummy_agent.py` | Baseline agent for comparison |
| `scripts/qwen3_decision_agent.py` | Pipeline A (full skill lifecycle) |
| `inference/run_qwen3_8b_eval.py` | Pipeline B (lightweight evaluation) |
