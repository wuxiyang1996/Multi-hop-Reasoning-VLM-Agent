# Decision Agent

The Decision Agent module from the **COS-PLAY** co-evolution framework (COLM 2026). Implements the three-stage decision loop described in Section 4.1 of the paper: **skill retrieval** → **intention update** → **action execution**, with composite reward shaping (r_total = r_env + λ_f · r_follow + r_cost).

Two agents ship in this package:

| Agent | Input | Use when |
|-------|-------|----------|
| `ActorAgent` (new, schema-native) | Parsed `<state>…</state>` schema from `vlm_wrapper` | You have visual-grounding output. This is the **Agent 1 (Actor)** target from [`plans/02-action-agent/PLAN-ACTION-AGENT.md`](../plans/02-action-agent/PLAN-ACTION-AGENT.md) §2.3 and the future GRPO training target (Phase 1). |
| `VLMDecisionAgent` (legacy, text-native) | Raw observation text | You don't yet have VLM grounding wired in (Pipeline A / B below). Kept for backward compatibility with `scripts/qwen3_decision_agent.py` and `inference/run_qwen3_8b_eval.py`. |

Both run the same decision loop shape; the actor just consumes a richer, pre-parsed state and exposes the inner-MDP / skill-interface seams the plan calls out.

---

## ActorAgent — schema-native decision agent

`ActorAgent` implements §1 of `PLAN-ACTION-AGENT.md` end-to-end:

```
<state> schema (from vlm_wrapper)
    ↓
parse_state_schema → StateSchema          # decision_agents/schema_parser.py
    ↓
compact_summary + infer_intention         # replaces raw-text compression
    ↓
SkillTracker.should_reselect              # decision_agents/skill_tracker.py
    ↓ (if reselect)
SkillProvider.select(query, schema, ...)  # decision_agents/skill_interface.py
    ↓
SkillTracker.activate → slot coverage     # PLAN §10
    ↓
HopPolicy.select_next_hop × N             # decision_agents/inner_mdp.py
    ↓ (EXECUTE)
action prompt → LLM                       # schema + skill + valid actions
    ↓
resolve_entity_action(click(e5))          # PLAN §7 Phase 3
    ↓
anti-repetition guard
    ↓
env.step(action)                          # driven by the runner
    ↓
SkillTracker.record_step + RewardComputer
    ↓
SkillProvider.record_outcome (on skill end)
```

### Dependency injection

Every piece with a `…Provider` / `…Policy` name is injectable so later phases of the plan can swap implementations without changing `ActorAgent`:

| Interface | Default | Swap-in for |
|-----------|---------|-------------|
| `SkillProvider` | `NullSkillProvider` (skill-free) or `SkillBankProvider(bank)` (RAG) | Trained Skill-Use Agent (Agent 2) |
| `HopPolicy` | `HeuristicHopPolicy` (rule-based over schema uncertainty + slot coverage) | `hop_select` LoRA adapter (PLAN §5) |
| `RewardComputer` | Existing `reward_func.RewardComputer` | Extended reward decomposition (PLAN §6) |

### Skill interface contract

`SkillProvider` is the seam between the actor and anything that knows about skills. Three methods:

| Method | Purpose |
|--------|---------|
| `select(query, state_summary, structured_state, current_predicates, top_k) -> list[SkillGuidance]` | Return candidate skills for the current state. |
| `record_outcome(skill_id, outcome, reward, steps_taken, info)` | Called after every skill attempt terminates (`success` / `abort` / `stall` / `switch` / `timeout`). |
| `available_skills() -> list[str]` | Enumerate what the provider can return. |

`SkillGuidance` bundles everything the actor renders into the prompt: name, strategy, protocol steps, preconditions, success/abort criteria, required/optional slots (→ drive the GROUND-insertion rule), `eff_add` / `eff_del` effects (→ feed `r_follow`), and a fallback `micro_plan`.

```python
from decision_agents import ActorAgent, SkillBankProvider, run_actor_episode
from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.query import SkillQueryEngine

bank = SkillBankMVP("path/to/bank.jsonl"); bank.load()
engine = SkillQueryEngine(bank=bank)

agent = ActorAgent(
    model="Qwen/Qwen3-8B",
    skill_provider=SkillBankProvider(engine),   # or NullSkillProvider() for baseline
)

episode = run_actor_episode(env, agent=agent, task="Clear the board", max_steps=200)
```

### Where schemas come from

The runner expects the env (or a wrapper around it) to place the `<state>` text on `info["schema"]` (or `info["schema_text"]`). Override via `schema_from_info=…` when integrating a different wrapper. When the key is missing the actor falls back to the raw-text path (Phase 1), so you can drop `ActorAgent` into existing envs before finishing the VLM wiring.

### Files

| File | What it does |
|------|--------------|
| `actor_agent.py` | `ActorAgent`, `ActorDecision`, `ActorState`, `run_actor_episode` |
| `schema_parser.py` | `StateSchema`, `Entity`, `Targets`, `StateFlags`, `Relation`, `Hop`, `Answer`, `ResolvedAction`, `parse_state_schema`, `resolve_entity_action` |
| `skill_interface.py` | `SkillProvider` protocol, `SkillGuidance`, `NullSkillProvider`, `SkillBankProvider` |
| `skill_tracker.py` | `SkillTracker`, `ActivationCheck`, `TrackerState` — lifecycle + slot-coverage (PLAN §10) |
| `inner_mdp.py` | `HopAction`, `HopStep`, `HopTrace`, `HopPolicy`, `HeuristicHopPolicy` — inner-MDP scaffold (PLAN §5) |

---

## ActorAgent — planned improvements (gaps vs PLAN-ACTION-AGENT.md)

A deep review of `actor_agent.py` against `plans/02-action-agent/PLAN-ACTION-AGENT.md` surfaced a handful of places where the code silently diverges from the plan or leaves a plan-promised feature un-wired. This section is the running TODO for closing those gaps. Everything below is additive — existing callers should not need to change.

**Status legend:** ✅ shipped · 🟡 partial · ⬜ pending.

### P0 — genuine divergences from the plan

| # | Status | Gap | Plan ref | Fix (shipped or proposed) |
|---|--------|-----|----------|---------------------------|
| 1 | 🟡 | **Inner-MDP hops are logged, not executed.** `_run_inner_mdp` emitted `HopStep`s with no side effects. | §5 (Option A/B), §7 Phase 2 | **Shipped (Option A):** added `InnerScratchpad` on `ActorState`; `_run_inner_mdp` now dispatches through `_apply_hop_side_effect` — `GROUND` updates scratchpad + calls `tracker.clear_ground_flag`, `RETRIEVE` calls `self.memory.query` and stores top-3 hits, `CONCLUDE` appends to `scratchpad.notes`. **Deferred:** Option B visual-tool re-observation between hops. |
| 2 | ✅ | **`ActivationCheck` is discarded / `clear_ground_flag` never called.** | §10 | **Shipped.** On activation the tracker's `missing_slots` seed `scratchpad.pending_ground_slots`; after every `GROUND` hop the actor calls `tracker.clear_ground_flag(schema)` so the LoRA doesn't have to re-derive Agent 2's deterministic slot-coverage rule. |
| 3 | ✅ | **`r_cost` never fires for `QUERY_SKILL` / `QUERY_MEM`.** | §4, PLAN-PIPELINE-ORCHESTRATOR §7 | **Shipped.** `ActorDecision` now carries `queried_skill` / `queried_mem`; `observe_result` passes both into `RewardComputer.compute_reward`, which was extended with orthogonal `queried_skill=` / `queried_mem=` kwargs that add the per-event cost on top of the primary `action_type` bucket. |
| 4 | ✅ | **Intention inference runs _after_ action selection.** | §1 step 2 | **Shipped.** `ActorAgent.step` was re-ordered: `_infer_intention` now runs right after `compact_summary` and before both the reselect decision and `_pick_action`, so the skill-bank query and the action prompt see the current step's intention. |
| 5 | 🟡 | **Action parsing lacks the multi-strategy fallbacks.** | §1 step 6 | **Shipped:** `_extract_action_from_reply` now returns `(action, parse_path)` and implements exact → numbered (`"1."`/`"2)"`/`"3:"`) → entity-ref → edit distance (`difflib`, case-sensitive + caseless) → token overlap → loose substring → trailing-digit fallback. `parse_path` is logged on `ActorDecision` + `Experience.extras`. **Deferred:** lift into a shared `ActionParser` consumed by `VLMDecisionAgent` + optional RAG-embedding `ActionEmbeddingMatcher`. |
| 6 | ⬜ | **`r_follow` uses text substring matching even when a schema is present.** | §4 r_follow | **Pending.** Next step: thread an optional `schema` into `RewardComputer.compute_reward` and match `eff_add` against `state_flags` / `entities_by_ontology` / `relations`. |

### P1 — plan-named features the code doesn't expose

| # | Status | Gap | Plan ref | Fix |
|---|--------|-----|----------|-----|
| 7 | ⬜ | **No `ContinueSwitchPolicy` seam for Agent 2 GRPO.** | §6 | Abstract `SkillTracker.should_reselect` into a pluggable policy, symmetric with `HopPolicy`. |
| 8 | ⬜ | **Hop trace is not used for reward shaping.** | §5 "Reward for inner hops" | Cheap online signal: `-cost_per_inner_hop` + bonus when a `GROUND` hop reduces `schema.missing_slots`. Defer full GPT-4o-judged hop-quality reward to offline GRPO. |
| 9 | ⬜ | **Pipeline-orchestrator log shapes are missing.** | PLAN-PIPELINE-ORCHESTRATOR §2.1/§2.2 | Accept a `TraceContext` dataclass in `ActorAgent.step`; thread `run_id` / `episode_id` / `step_id` / `span_id` / `schema_hash` through `decision.to_dict()` and each `HopStep`. |
| 10 | ⬜ | **Budget control is only a hop cap.** | PLAN-PIPELINE-ORCHESTRATOR §7 | Accept an optional `BudgetCounter`; decrement in `_select_skill`, `_pick_action`, `_run_inner_mdp`, `memory.query`. On exhaustion, degrade to the deterministic fallbacks already present. |
| 11 | ✅ | **`progress_notes` are written and never read.** | §1 step 5 | **Shipped.** `_build_action_prompt` now emits `Recent progress: …` from the last 3 notes alongside `Recent actions` / `Recent rewards`. |
| 12 | ⬜ | **Entity-referenced actions aren't prompted for browser/OSWorld.** | §7 Phase 3 | When the domain is `browser` / `osworld`, append an "Entity-referenced actions you may also emit" section to the prompt, enumerated from `schema.interactive_entities()`. |

### P2 — code-hygiene / smaller items

- ✅ **`ActorDecision.to_dict` dropped `reasoning`** — now emitted alongside `queried_skill`, `queried_mem`, and `parse_path`.
- ✅ **`_build_default_memory` silently returned `None`** — now logs `DEBUG` / `WARNING` so the missing-memory mode is traceable.
- ✅ **`_ = json` sentinel at the bottom of `actor_agent.py`** — removed (and the unused `json` import with it).
- ✅ **`skill_interface.py:506` `f"get_slot_bindings"`** — f-string prefix dropped.
- ⬜ **`HopAction.VERIFY` is declared but never emitted or consumed.** The dispatcher in `_apply_hop_side_effect` handles it as a logged-only op, so it is now safe to leave in the action space for the future `hop_select` LoRA. Semantics (`VERIFY = re-check after CONCLUDE`) still need to be pinned.
- ⬜ **Anti-repetition is deterministic** — plan §1 step 7 says "randomly pick". Seed with `Random(hash(episode_id))` or rotate by `len(last_actions)` to avoid 2-action limit cycles.
- ✅ **Schema's own `task` / `goal` were ignored** — `_build_action_prompt` now falls back to `schema.goal or schema.task` when the caller-supplied `task` is empty.

### Plan-side clarifications the code surfaces

The review also exposed four places where `PLAN-ACTION-AGENT.md` is vaguer than it should be; these will be proposed as plan patches in parallel with the code changes:

1. **§10 slot-coverage insertion responsibility** — currently the plan says `SkillTracker` inserts hop 0, but the code puts the rule in `HopPolicy`. The shipped `_apply_hop_side_effect` now routes the deterministic `clear_ground_flag` call through the tracker so the LoRA does **not** have to relearn the rule; the plan should codify this ownership.
2. **§5 "scratchpad"** — the word appears once with no definition. The shipped `InnerScratchpad` dataclass (`pending_ground_slots` / `grounded_slots` / `memory_hits` / `notes`) is a concrete proposal; the plan should adopt it.
3. **§4 cost accounting** — add a table specifying which step emits which cost (reselect → `query_skill_cost`; `RETRIEVE` hop that calls `memory.query` → `query_mem_cost`; `skill_switch_cost` fires iff `active_skill_id` changes). This matches the shipped behaviour.
4. **§1 step ordering** — explicitly pin intention inference to run *before* reselect/action, matching both the plan's written prose and the shipped code.

### Patch-set log

Shipped so far (all additive — no existing-caller API breaks):

1. ✅ Extended `RewardComputer` with orthogonal `queried_skill` / `queried_mem` cost events.
2. ✅ Fixed `f"get_slot_bindings"` typo in `skill_interface.py`.
3. ✅ Added `InnerScratchpad` dataclass and plumbed it onto `ActorState`.
4. ✅ Extended `ActorDecision` with `queried_skill`, `queried_mem`, `parse_path`; re-exposed `reasoning` in `to_dict`.
5. ✅ Re-ordered `ActorAgent.step` — intention inference now runs before reselect.
6. ✅ Refactored `_run_inner_mdp` + new `_apply_hop_side_effect` to actually execute hops (GROUND → `tracker.clear_ground_flag`, RETRIEVE → `self.memory.query`, CONCLUDE → notes).
7. ✅ Multi-strategy action parser (exact → numbered → entity-ref → edit distance → token overlap → loose → trailing-digit), with a `parse_path` log tag.
8. ✅ `_build_action_prompt` now renders `Inner reasoning so far`, `Recent progress`, and falls back to `schema.goal` when the caller didn't pass a `task`.
9. ✅ `observe_result` + `run_actor_episode` thread the new flags into `Experience.extras` (`queried_skill`, `queried_mem`, `parse_path`, `scratchpad`, `reasoning`).
10. ✅ Removed the dead `_ = json` sentinel; logged the `_build_default_memory` fallback path.
11. ✅ Tests: 13 new cases covering reselect cost, scratchpad grounding, RETRIEVE-hop memory wiring, `to_dict` shape, and the parser pipeline.

Still open (see tables above): #1-OptionB, #5-sharedActionParser, #6, #7, #8, #9, #10, #12, VERIFY semantics, anti-repetition randomness.

---

## Legacy VLMDecisionAgent (text-native, Pipeline A / B)

**Two model backends:**

- **GPT-5.4** (training-free) — used for cold-start data generation and labeling via OpenRouter / OpenAI API.
- **Qwen3-8B** (GRPO-trained with LoRA adapters) — served via vLLM for decision agent inference and evaluation.

Both share the same code path; `API_func.ask_model` routes to the correct API based on the model name. Skill bank loading and querying are identical for both backends.

## Supported games

**6 games** across three environment stacks (matching `cold_start/`):

| # | Stack | Game | Registry Key |
|---|-------|------|-------------|
| 1 | LMGame-Bench | **2048** | `twenty_forty_eight` |
| 2 | LMGame-Bench | **Candy Crush** | `candy_crush` |
| 3 | LMGame-Bench | **Tetris** | `tetris` |
| 4 | AgentEvolver | **Avalon** | `avalon` |
| 5 | AgentEvolver | **Diplomacy** | `diplomacy` |
| 6 | Orak | **Super Mario** | `super_mario` |

---

## Decision agent pipelines

Two script-level pipelines drive the decision agent at inference time. Both use `Qwen/Qwen3-8B` served via vLLM and share the same core helpers from `decision_agents/`, but differ in skill-bank integration depth and game coverage.

### Pipeline A — `scripts/qwen3_decision_agent.py` (with skill select)

Skill-bank-guided decision agent with protocol-aware lifecycle management.

**Per-step loop:**

1. **`get_state_summary()`** — deterministic + LLM state compression into `key=value` format (≤400 chars)
2. **`infer_intention()`** — Qwen3-8B produces a `[TAG] subgoal phrase` from summary + context (last actions, task)
3. **Skill re-selection check** (`_SkillTracker.should_reselect()`) — triggers re-query when: no active skill, duration exceeded, zero-reward stall (≥4 steps with reward ≤0), abort/success criteria keyword-matched in current state
4. **`get_skill_guidance()`** — queries `SkillQueryEngine` (RAG mode) using `game_name + intention + state_text[:1500]` as query, with `structured_state` converted to `{predicate: float}` for applicability scoring. Returns skill_id, skill_name, execution_hint, protocol (steps, preconditions, success/abort criteria)
   - If re-selecting and the same skill returns, `_try_alternate_skill()` randomly picks a different skill_id
   - Sets protocol on `_SkillTracker` for step tracking and prompt injection
5. **`qwen3_action()`** — builds prompt: system prompt + `format_skill_guidance_for_prompt()` (active skill name, strategy, plan steps with `>>` marker at current step, preconditions, done-when, abort-if) + recent actions/rewards context + numbered action list → Qwen3-8B via vLLM
6. **`parse_qwen_response()`** — multi-strategy action extraction: exact match → numbered selection → substring → edit distance → token overlap → **RAG embedding semantic match** (`ActionEmbeddingMatcher` using `Qwen3-Embedding-0.6B`) as final fallback
7. **`_apply_anti_repetition()`** — if same action repeated N times with 0 reward, randomly pick an alternative
8. **`env.step(action)`**
9. **`_SkillTracker.update()`** — advance protocol step index, track reward-on-skill, switch count
10. **Build `Experience`** with: state, action, reward, next_state, done, intentions, tasks, sub_tasks (active skill), summary_state, available_actions

**Key features:**

- Protocol-aware skill lifecycle (find-apply loop with duration caps, stall detection, criteria matching)
- RAG `ActionEmbeddingMatcher` for semantic action fallback
- Anti-repetition guard
- Per-game skill bank loading (`bank_dir/<game_name>/`)
- Output: `test_rollout/decision_agent/<game>/<timestamp>/`

**Usage:**

```bash
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m scripts.qwen3_decision_agent --games twenty_forty_eight --episodes 3
python -m scripts.qwen3_decision_agent --one_per_game --gpu 0 -v
python -m scripts.qwen3_decision_agent --no-bank --episodes 3        # baseline without skill bank
python -m scripts.qwen3_decision_agent --bank /path/to/bank --episodes 3
```

### Pipeline B — `inference/run_qwen3_8b_eval.py` (without skill select)

General-purpose evaluation script across multiple benchmarks, with optional skill bank support but no skill lifecycle tracking.

**Per-step loop:**

1. **`get_state_summary()`** — same deterministic + LLM state compression
2. **`infer_intention()`** — same Qwen3-8B intention inference
3. **`get_skill_guidance()`** — optional (via `--bank` flag), simpler query using `state[:500]`, no intention/structured_state scoring, no re-selection logic
4. **`qwen3_agent_action()`** — builds prompt: system prompt + skill guidance text + user template (comma-separated actions) → Qwen3-8B via vLLM
5. **`_parse_qwen_response()`** — simpler parsing: exact match (case-insensitive) → `extract_action()` fallback → first valid action (no fuzzy/edit-distance/RAG)
6. **`env.step(action)`**
7. **Generate experience summary** via LLM: a "short strategic note" from state + action (extra LLM call per step)
8. **Build `Experience`** with same rich fields

**Game-specific episode runners:**

| Runner | Games | Features |
|--------|-------|----------|
| `run_qwen3_episode()` | 2048, Tetris, Candy Crush | Standard LMGame-Bench loop |
| `run_qwen3_avalon_episode()` | Avalon | Multi-agent (all players = Qwen3), `ThreadPoolExecutor` parallel queries |
| `run_qwen3_diplomacy_episode()` | Diplomacy | 7 powers, order parsing, SC delta tracking, 20-phase cap |
| `run_qwen3_orak_episode()` | Super Mario | Orak env wrappers |

**Key features:**

- Multi-benchmark (LMGame-Bench + AgentEvolver + Orak)
- Resume interrupted runs (`--resume`)
- Per-experience LLM summary generation
- Output: `output/<model_slug>/<game>/<timestamp>/`

**Usage:**

```bash
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m inference.run_qwen3_8b_eval --games twenty_forty_eight --episodes 3
python -m inference.run_qwen3_8b_eval --episodes 10                   # all 6 games
python -m inference.run_qwen3_8b_eval --resume                       # resume interrupted run
python -m inference.run_qwen3_8b_eval --bank path/to/bank.jsonl      # with optional skill bank
python -m inference.run_qwen3_8b_eval --list-games                   # show available games
```

### Pipeline comparison

| Aspect | `qwen3_decision_agent.py` | `run_qwen3_8b_eval.py` |
|--------|---------------------------|------------------------|
| Skill bank | Required (per-game, query engine, tracker) | Optional (`--bank` flag) |
| Skill lifecycle | `_SkillTracker` with reselect, alternate, protocol steps | Single query per step, no tracking |
| Skill query key | `game + intention + state[:1500]` | `state[:500]` |
| Applicability scoring | Yes (structured_state → predicate floats) | No (pass_rate proxy only) |
| Action parsing | Fuzzy + edit distance + RAG embedding | Exact match + `extract_action()` |
| Anti-repetition | Yes | No |
| Action format in prompt | Numbered list | Comma-separated |
| Game coverage | LMGame-Bench | LMGame-Bench + Avalon + Diplomacy + Orak |
| Experience summary | State summary only | Extra LLM call for strategic note |
| Output dir | `test_rollout/decision_agent/` | `output/<model>/` |
| Resume support | No | Yes |

---

## Skill selection (RAG mode)

Skill selection is RAG-based by default. When `SkillQueryEngine` initializes, it auto-loads the `Qwen3-Embedding-0.6B` embedder and pre-embeds all skill descriptions. The TF-IDF keyword fallback in `agent_helper._rank_skills_by_relevance()` only fires if the query engine fails to initialize.

### How `select_skill_from_bank()` routes

The function tries four paths in order, stopping at the first success:

1. **`SkillQueryEngine.select()`** — richest path (RAG relevance + applicability + structured guidance)
2. **`SkillQueryEngine.query_for_decision_agent()`** — convenience wrapper that delegates to `select()` when state is available
3. **`SkillBankAgent.select_skill()`** — alternative agent-level selection
4. **TF-IDF keyword fallback** via `_rank_skills_by_relevance()` — only when no query engine is available

### `SkillQueryEngine.select()` scoring

Each candidate skill is scored on three axes and combined into a final confidence:

| Component | Weight | Source |
|-----------|--------|--------|
| Retrieval relevance | 40% | RAG embedding cosine similarity + keyword Jaccard |
| Execution applicability | 35% | Effect compatibility against current state predicates |
| Historical pass rate | 25% | Success rate from past executions |

Skills are sorted by confidence and top-k returned as `SkillSelectionResult` objects containing: `skill_id`, `skill_name`, `why_selected`, `relevance`, `applicability_score`, `confidence`, `expected_effects`, `preconditions`, `termination_hint`, `failure_modes`, `execution_hint`, `micro_plan`, `contract`, `pass_rate`.

---

## Files

| File | What it does |
|------|-------------|
| `agent.py` | `VLMDecisionAgent` (LLM decision agent), `run_tool()`, `run_episode_vlm_agent()`, tool handlers (e.g. `TOOL_SELECT_SKILL` → `active_skill_plan` from protocol steps) |
| `agent_helper.py` | `get_state_summary()`, `build_rag_summary()`, `extract_game_facts()`, `infer_intention()`, `EpisodicMemoryStore`, `skill_bank_to_text()`, `query_skill_bank()` / `select_skill_from_bank()`, `_get_protocol_for_skill()` |
| `reward_func.py` | `RewardConfig`, `RewardResult`, `RewardComputer`, `compute_reward()` (r_follow uses skill contract `eff_add`) |
| `dummy_agent.py` | Baseline `language_agent_action()` + game detection + action extraction for all 6 supported games (LMGame-Bench, AgentEvolver, Orak) |
| `__init__.py` | Re-exports the above |

---

## Quick start — run a full episode

`run_episode_vlm_agent()` returns an **`Episode`** object (from `data_structure.experience`) with fully-populated `Experience` objects per step.

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig

episode = run_episode_vlm_agent(
    env,
    model="Qwen/Qwen3-8B",   # or "gpt-5.4" for training-free cold-start
    task="Complete level 1",
    max_steps=200,
    verbose=True,
)

print(episode.get_length())
print([e.reward for e in episode.experiences])
print([e.reward_details for e in episode.experiences])
print(episode.metadata["cumulative_reward"])
print(episode.experiences[-1].done)

exp = episode.experiences[0]
print(exp.summary_state)   # key=value format
print(exp.intentions)      # [TAG] phrase
print(exp.sub_tasks)       # active skill ID
print(exp.reward_details)  # full reward breakdown dict
```

### With a skill bank and custom reward config

```python
from decision_agents import (
    VLMDecisionAgent,
    run_episode_vlm_agent,
    EpisodicMemoryStore,
    RewardConfig,
)
from skill_agents.skill_bank.bank import SkillBankMVP

bank = SkillBankMVP("path/to/bank.jsonl")
bank.load()

from rag import get_text_embedder
memory = EpisodicMemoryStore(max_entries=500, embedder=get_text_embedder())

reward_cfg = RewardConfig(
    w_follow=0.1,
    query_mem_cost=-0.05,
    query_skill_cost=-0.05,
    call_skill_cost=-0.02,
    skill_switch_cost=-0.10,
)

agent = VLMDecisionAgent(
    model="Qwen/Qwen3-8B",
    skill_bank=bank,
    memory=memory,
    reward_config=reward_cfg,
    retrieval_budget_n=10,
    skill_abort_k=5,
)

episode = run_episode_vlm_agent(env, agent=agent, task="Clear all boxes", max_steps=500, verbose=True)
```

---

## Step-by-step control (manual loop)

```python
from decision_agents import VLMDecisionAgent

agent = VLMDecisionAgent(model="Qwen/Qwen3-8B")
obs, info = env.reset()

last_tool_name = None
last_tool_result = None

for t in range(200):
    decision = agent.step(str(obs), info, last_tool_name, last_tool_result)
    tool = decision["tool"]
    args = decision["args"]

    if tool == "take_action":
        obs, reward, term, trunc, info = env.step(args["action"])
        agent.update_from_tool_result("take_action", args["action"], str(obs))
        if term or trunc:
            break
    elif tool == "reward":
        rr = agent.reward_computer.compute_reward(r_env=reward, action_type="primitive", observation=str(obs))
        agent.update_from_tool_result("reward", rr, str(obs))
    else:
        from decision_agents import run_tool
        result = run_tool(tool, args, agent, str(obs), info)
        agent.update_from_tool_result(tool, result, str(obs))

    last_tool_name = tool
    last_tool_result = decision.get("result")
```

---

## Skill bank: protocol store vs contract

The skill bank stores each skill as a **Skill** object with two logical parts (see `skill_agents.stage3_mvp.schemas`):

- **Protocol store** — What the decision agent sees: `name`, `strategic_description`, `tags`, `protocol` (steps, preconditions, success_criteria, abort_criteria, expected_duration), `confidence`. Used by `skill_bank_to_text()`, `query_skill_bank()`, and to set `active_skill_plan` from `protocol.steps`.
- **Contract** — Effects (`eff_add`, `eff_del`, `eff_event`) used for segmentation, verification, and **reward shaping**. The agent still gets the contract via `bank.get_contract(skill_id)` when computing r_follow.

So: the agent **plans** from protocols (when present) and is **rewarded** for making progress on the contract's eff_add predicates.

---

## Helper functions

### `get_state_summary(observation, structured_state=None, *, max_chars=400, use_llm_fallback=False, llm_callable=None)`

Produces a compact `key=value` state summary optimised for LLM context windows, retrieval, skill-bank indexing, and trajectory segmentation. Summaries are **never** raw observation text and always ≤ 400 characters.

**Priority order:**
1. `structured_state` → `compact_structured_state()` (preferred; wrapper-produced dict)
2. `observation` → `compact_text_observation()` (deterministic boilerplate removal + clause compression)
3. LLM fallback (optional, disabled by default)

```python
from decision_agents import get_state_summary

summary = get_state_summary(
    obs_text,
    structured_state=info.get("structured_state"),
)
# → "game=tetris | phase=midgame | stack_h=14 | holes=32 | next=T,Z,I,J | level=1"
```

**Supported wrappers with `build_structured_state_summary()`:**

| Wrapper | Key fields | Example |
|---------|-----------|---------|
| GamingAgent (LMGame-Bench) | game, step, self, objective, critical, affordance | `game=2048 \| self=highest:256 \| objective=merge tiles` |
| Avalon | game, phase, self, progress, critical, objective | `game=avalon \| phase=team_vote \| self=role:Percival(G)` |
| Diplomacy | game, phase, self, resources, critical, objective | `game=diplomacy \| phase=S1902M \| self=power:FRANCE centers:5` |
| Orak (Mario) | game, step, self, objective, critical, affordance | `game=super_mario \| self=pos:(120,80) \| objective=reach flag` |

### `build_rag_summary(state, game_name, *, step_idx, total_steps, reward, max_chars)`

Fully deterministic (no LLM) `key=value` summary optimised for RAG embedding retrieval. Combines game-aware fact extraction with phase estimation and reward.

```python
from decision_agents.agent_helper import build_rag_summary

summary = build_rag_summary(
    state_text,
    game_name="tetris",
    step_idx=50,
    total_steps=86,
    reward=1.0,
)
# → "game=tetris | phase=midgame | step=50/86 | stack_h=14 | holes=32 | next=T,Z,I,J | level=1 | reward=+1"
```

Uses `extract_game_facts()` internally — game-specific parsers for Tetris (stack_h, holes, piece, next), 2048 (highest, empty, tiles, merges), Candy Crush (score, moves, pairs), Super Mario (mario position, enemies, items), Avalon (phase, role, quest), and Diplomacy (phase, power, centers, units).

### `infer_intention(summary_or_observation, game=None, model=None, context=None)`

Returns a `[TAG] subgoal phrase` (≤15 words) describing the agent's current subgoal. Tags:

```
SETUP | CLEAR | MERGE | ATTACK | DEFEND | NAVIGATE | POSITION |
COLLECT | BUILD | SURVIVE | OPTIMIZE | EXPLORE | EXECUTE
```

```python
from decision_agents import infer_intention

intention = infer_intention(
    summary,
    context={
        "last_actions": ["up", "left"],
        "progress_notes": ["pushed box onto goal"],
        "task": "push all boxes to goals",
    },
)
# e.g. "[NAVIGATE] Push remaining box right toward goal tile"
```

### `EpisodicMemoryStore`

RAG-embedding retrieval memory for the `query_memory` tool. When an embedder is supplied (or auto-loaded from `rag/`), memories are embedded on `add` and queries use cosine similarity blended with keyword overlap.

```python
from decision_agents import EpisodicMemoryStore
from rag import get_text_embedder

mem = EpisodicMemoryStore(
    max_entries=500,
    embedder=get_text_embedder(),
    embedding_weight=0.7,
)

mem.add_experience(
    state_summary="game=tetris | stack_h=14 | holes=32 | next=T,Z,I,J | level=1",
    action="rotate_cw",
    next_state_summary="game=tetris | stack_h=14 | holes=30 | next=Z,I,J,S | level=1",
    done=False,
)

results = mem.query("game=tetris | stack_h=high | holes=many", k=3)
```

### `skill_bank_to_text(skill_bank)` and `query_skill_bank(skill_bank, state, task, ...)`

**`skill_bank_to_text(skill_bank)`** — Formats the skill bank for agent prompts. When a skill has a protocol (name, strategic_description, steps), the string shows those; otherwise it falls back to effect counts.

**`query_skill_bank(skill_bank, state, task, ...)`** — Alias for `select_skill_from_bank`. Picks the best-matching skill for the current state/task and returns it with a protocol dict (steps, preconditions, success_criteria, expected_duration).

---

## Reward function

### Standalone usage

```python
from decision_agents import RewardComputer, RewardConfig

cfg = RewardConfig(w_follow=0.1, skill_switch_cost=-0.10)
rc = RewardComputer(cfg)

rr = rc.compute_reward(
    r_env=1.0,
    action_type="primitive",
    observation="checkpoint area",
    active_skill_id="nav_to_cp",
    skill_contract=contract,
)
print(rr)
# RewardResult(r_env=1.0000, r_follow=0.0500, r_cost=0.0000, r_total=1.0050)
```

### RewardConfig defaults

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `w_follow` | 0.1 | Weight on r_follow in r_total |
| `query_mem_cost` | -0.05 | Cost per QUERY_MEM action |
| `query_skill_cost` | -0.05 | Cost per QUERY_SKILL action |
| `call_skill_cost` | -0.02 | Cost per CALL_SKILL action |
| `skill_switch_cost` | -0.10 | Penalty when active skill changes |
| `follow_predicate_bonus` | 0.05 | Bonus per newly satisfied eff_add predicate |
| `follow_completion_bonus` | 0.20 | Bonus when all eff_add predicates satisfied |
| `follow_no_progress_penalty` | -0.01 | Penalty per step with no predicate progress |

### Reward components

- **r_env**: Raw environment reward passed through.
- **r_follow**: Skill-following shaping (termination-free). Checks how many `eff_add` predicates from the active skill's contract appear in the observation. Awards bonuses for newly satisfied predicates and a completion bonus when all are met.
- **r_cost**: Negative costs for queries, skill calls, and skill switching.
- **r_total**: `r_env + w_follow * r_follow + r_cost`.

---

## Dummy agent (baseline)

The original single-call LLM agent, for comparison or simple use:

```python
from decision_agents import language_agent_action

action = language_agent_action(
    state_nl=observation_text,
    game="gamingagent",
    model="Qwen/Qwen3-8B",    # or "gpt-5.4"
)
```

Supports all 6 games: 2048, Candy Crush, Tetris (LMGame-Bench), Avalon, Diplomacy (AgentEvolver), Super Mario (Orak).

---

## Per-step loop (LLMDecisionAgent protocol)

Every timestep the runner executes:

1. **`get_state_summary`** — required; runner computes it before action (returns `key=value` facts).
2. **(Optional)** **`select_skill`** — choose a skill when no active skill, skill exhausted, or agent is stuck. Returns full structured guidance (protocol steps, preconditions, termination hints, failure modes). Budget-limited to once every N steps unless stuck.
3. **`take_action`** — required; exactly one environment action. Agent has intention (from previous step), fresh state summary, and any active skill guidance in context.
4. **`get_intention`** — required; runner updates intention after observing action result (returns `[TAG] subgoal phrase`).
5. **`reward`** — required; compute `(r_env, r_follow, r_cost, r_total)` for logging/training.

### Format consistency

The agent prompt uses consistent formats across cold-start labeling and runtime inference:
- **Intention**: `"[TAG] subgoal phrase"` (e.g., `"[CLEAR] Reduce holes before stack overflows"`)
- **State summary**: `"key=value"` pairs (e.g., `"game=tetris | phase=endgame | stack_h=15 | holes=42"`)
- **Memory results**: `key=value` summaries from `EpisodicMemoryStore`
