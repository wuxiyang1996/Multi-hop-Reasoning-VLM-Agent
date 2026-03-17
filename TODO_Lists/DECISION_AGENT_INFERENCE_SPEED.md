# Decision Agent Inference Speed & Parallelism

**Created:** 2026-03-15  
**Status:** Open  
**Related:** `SKILLBANK_INFERENCE_SPEED.md`, `SKILLBANK_GRPO_PLAN_REWRITE.md`

---

## Problem

The decision agent runs 2–4 **serial** LLM calls per environment step via two LoRA adapters (`skill_selection` and `action_taking`). With HuggingFace `.generate()` at batch size 1, each step takes ~1.3–1.8s.

In the co-evolution framework, **every co-evolution step** collects rollouts across all 8 games × 5–10 episodes each = **40–80 episodes**. The games have wildly different step budgets and env speeds, so episode durations are highly uneven. Serial rollout collection is the dominant bottleneck — far exceeding the skill bank update time.

---

## Co-Evolution Rollout Budget

The 8 skill bank games (from `labeling/output/gpt54_skillbank/`) have different step budgets and estimated inference profiles:

| Game | max_steps | Env speed | Steps/ep (typical) | LLM calls/step | Est. serial ep time |
|------|-----------|-----------|---------------------|-----------------|---------------------|
| candy_crush | 50 | Fast (~5ms/step) | ~45 | 2–3 | ~65s |
| sokoban | 100 | Fast (~5ms/step) | ~80 | 2–3 | ~120s |
| twenty_forty_eight | 200 | Fast (~2ms/step) | ~180 | 2–3 | ~270s |
| tetris | 200 | Fast (~5ms/step) | ~150 | 2–3 | ~225s |
| pokemon_red | 200 | Medium (~20ms/step) | ~200 | 3–4 | ~360s |
| super_mario | 500 | Medium (~15ms/step) | ~300 | 3–4 | ~500s |
| avalon | varies | Medium | ~100 | 3–4 | ~180s |
| diplomacy | varies | Slow (~50ms/step) | ~150 | 3–4 | ~300s |

**Key observations:**
- Episode durations vary **8x** across games (65s for candy_crush vs 500s for super_mario)
- With 5–10 episodes per game, the total serial rollout time is **~2,800–5,600s (47–93 min)**
- Short games (candy_crush, sokoban) finish much earlier than long games, creating natural staggering opportunities
- Env step latency varies from ~2ms (2048) to ~50ms (diplomacy) — for fast envs, LLM inference dominates; for slow envs, env time is non-negligible

### Total LLM call budget per co-evolution step

| Scenario | Episodes | Total steps | LLM calls (2.5 avg/step) | Serial time |
|----------|----------|-------------|--------------------------|-------------|
| 8 games × 5 eps | 40 | ~5,400 | ~13,500 | ~47 min |
| 8 games × 8 eps | 64 | ~8,640 | ~21,600 | ~75 min |
| 8 games × 10 eps | 80 | ~10,800 | ~27,000 | ~93 min |

This is the dominant cost in co-evolution. Skill bank updates (~12 min per 3 EM iterations from `SKILLBANK_INFERENCE_SPEED.md`) are dwarfed by rollout collection.

---

## Current Per-Step Pipeline (Serial)

Each step in `run_episode()` (`scripts/qwen3_decision_agent.py`) runs:

| Step | Name | LLM calls | Latency | Depends on |
|------|------|-----------|---------|------------|
| 1 | `summary_state` | 0 (deterministic) | ~0ms | obs, game, step, reward |
| 2 | `summary_prose` | 1 (cheap) | ~500ms | summary_state, prev_summary_state |
| 3 | `skill_selection` | 1 (conditional) | ~500ms | summary_state, **prev** intention, skill_bank |
| 4 | `intention` | 1 | ~300ms | summary_state, **skill_guidance from step 3** |
| 5 | `action_taking` | 1 | ~500ms | summary_state, **intention from step 4**, **skill_guidance from step 3** |

**Total per step: ~1.3–1.8s** (3–4 serial LLM calls when skill reselection triggers, 2–3 when not)

---

## Dependency Analysis — Three Critical Findings

### Finding 1: `summary_prose` is fire-and-forget

Its output (`current_summary`) is only stored as `exp.summary` for logging. No downstream LLM call (skill selection, intention, or action) reads it. It can be removed entirely during inference or run asynchronously.

### Finding 2: `summary_prose` and `skill_selection` are independent

`skill_selection` uses `current_intention` from the **previous** step (line 1499 in `qwen3_decision_agent.py`), not the intention generated in this step. `summary_prose` uses `summary_state` + `prev_summary_state`. Neither depends on the other — they can run concurrently.

### Finding 3: The true critical path is 3 serial LLM calls

```
summary_state (0ms) ──┬──► skill_selection (~0.5s) ──► intention (~0.3s) ──► action (~0.5s)
                       │
                       └──► summary_prose (~0.5s, fire-and-forget, not on critical path)
```

When no skill reselection is needed (the common case — ~70% of steps), the critical path shortens to:

```
summary_state (0ms) ──► intention (~0.3s) ──► action (~0.5s)
```

---

## Acceleration Strategies

### Strategy A — Drop `summary_prose` during inference

`generate_summary_prose()` adds a ~10-word strategic note to `summary_state`. This note is never consumed by any downstream LLM call — it's purely for Experience logging. During inference (not GRPO data collection), skip it entirely.

```python
# In run_episode():
if not inference_mode:
    summary = generate_summary_prose(obs_nl, ...)
else:
    summary = summary_state  # skip LLM call
```

- **Savings:** ~500ms per step (eliminates 1 LLM call)
- **Risk:** None — the note is metadata only
- **Implementation:** Add `inference_mode=True` flag to `run_episode()`

### Strategy B — Merge intention into `action_taking` prompt

Currently, `generate_skill_aware_intention()` is a separate LLM call producing `[TAG] subgoal`. This is then injected into the `action_taking` prompt as `Current intention: {intention}`. Instead, incorporate intention generation directly into the `action_taking` LoRA prompt:

```
You are an expert game-playing agent.
Given the state, skill guidance, and available actions:
1. First, state your subgoal as [TAG] phrase (max 15 words)
2. Then choose the best action

Output format (strict):
INTENTION: [TAG] subgoal phrase
REASONING: 1-2 sentences
ACTION: <number>
```

The `action_taking` LoRA already receives all the context that the intention prompt uses (state, skill guidance, delta, urgency). Training the LoRA to produce both outputs adds minimal overhead (~15 extra output tokens) but eliminates an entire serial LLM call.

- **Savings:** ~300ms per step
- **Risk:** Requires either retraining the `action_taking` LoRA (Option 2 below) or using the base model for the merged prompt (Option 1 below)
- **Tradeoff:** Intention is now jointly optimized with the action — may actually be *better* since intention and action are coherent by construction

**Option 1 (inference-only merge):** Keep separate intention call during GRPO training for cleaner reward attribution, but merge at inference time using the base model. No retraining needed.

**Option 2 (full merge):** Retrain `action_taking` LoRA to jointly produce intention + action. Better coherence, but intention quality can't be independently evaluated.

**Recommendation:** Start with Option 1, upgrade to Option 2 once the LoRA is retrained.

### Strategy C — vLLM serving with both decision agent LoRAs

Serve `skill_selection` and `action_taking` adapters through vLLM alongside the 3 GRPO-trained skill bank adapters (`segment`, `contract`, `curator`). Total: 5 LoRAs on one base model. Stage 1 (boundary) uses the base model without a LoRA adapter; `retrieval` is legacy/not actively trained.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --enable-lora \
    --lora-modules \
        segment=runs/lora_adapters/segment \
        contract=runs/lora_adapters/contract \
        curator=runs/lora_adapters/curator \
        skill_selection=runs/lora_adapters/skill_selection \
        action_taking=runs/lora_adapters/action_taking \
    --max-loras 5 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --trust-remote-code
```

Client specifies adapter per request via `model="skill_selection"` or `model="action_taking"` in the OpenAI API call.

- **Savings:** Enables all cross-episode and cross-system batching (Strategies D and E)
- **Prerequisite for Strategies D and E**

### Strategy D — Async cross-episode batching (main throughput win)

Rewrite `run_episode()` as `run_episode_async()`. LLM calls use `await` (async HTTP to vLLM via `AsyncOpenAI`). `env.step()` — which is synchronous — runs in an executor (ThreadPoolExecutor for pure-Python games, ProcessPoolExecutor for emulator games).

**Why async beats threads for training speed:**

With threads, each thread blocks waiting for vLLM's response. A blocked thread occupies a worker slot but does no work. With 20 threads, you get at most ~20-30 concurrent vLLM requests.

With async, when one coroutine `await`s the vLLM response, the event loop immediately runs another coroutine's `env.step()` or prompt formatting. All 80 episodes are "live" simultaneously — each with 1 in-flight request whenever it's between its `env.step()` and its next LLM call. This means **40-60 concurrent vLLM requests** from 80 episodes, **2-3x more GPU utilization** than the threading model.

```
Threading model (20 workers):
  Thread 1: [===LLM wait===][env][===LLM wait===][env]...
  Thread 2: [===LLM wait===][env][===LLM wait===][env]...
  ...
  Thread 20: [===LLM wait===][env][===LLM wait===][env]...
  ← 20 slots occupied, ~20 concurrent vLLM requests →

Async model (80 coroutines, no worker limit):
  Coro 1:   [await LLM][env][await LLM][env]...
  Coro 2:   [env][await LLM][env][await LLM]...
  Coro 3:   [await LLM][env][await LLM]...
  ...
  Coro 80:  [await LLM][env][await LLM]...
  ← 80 coroutines, event loop multiplexes, ~40-60 concurrent vLLM requests →
```

The async model keeps vLLM's request queue fuller, which means better GPU batch fill rate, which means higher tokens/sec, which means faster training.

**Architecture: async LLM + executor env.step()**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional

EMULATOR_GAMES = {"pokemon_red", "super_mario", "diplomacy"}

# Executors for synchronous env.step() calls
_thread_executor = ThreadPoolExecutor(max_workers=20)
_process_executor = ProcessPoolExecutor(max_workers=8)

vllm_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


async def vllm_generate(prompt: str, adapter: str, max_tokens: int = 512,
                         temperature: float = 0.3) -> str:
    """Single async LLM call to vLLM with LoRA adapter selection."""
    r = await vllm_client.completions.create(
        model=adapter,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].text


async def run_episode_async(
    game: str,
    max_steps: int,
    skill_bank: Any = None,
    on_step_done: Optional[callable] = None,
) -> Dict[str, Any]:
    """Async episode runner. LLM calls are awaited; env.step() runs in executor.

    The event loop multiplexes across all concurrent episodes: while this
    coroutine awaits a vLLM response, other coroutines run their env.step()
    or prompt formatting. This maximizes vLLM request concurrency.
    """
    loop = asyncio.get_event_loop()
    executor = _process_executor if game in EMULATOR_GAMES else _thread_executor

    env = make_gaming_env(game=game, max_steps=max_steps)
    obs_nl, info = env.reset()
    # ... (init same as run_episode) ...

    for step in range(max_steps):
        summary_state = generate_summary_state(obs_nl, ...)  # deterministic, instant

        if need_reselect:
            candidates = get_top_k_skill_candidates(skill_bank, ...)  # CPU, fast
            skill_prompt = build_skill_selection_prompt(candidates, summary_state, ...)
            skill_reply = await vllm_generate(skill_prompt, adapter="skill_selection",
                                               max_tokens=256)
            guidance = parse_skill_selection(skill_reply, ...)

        # Strategy B: intention merged into action prompt
        action_prompt = build_merged_action_prompt(summary_state, guidance, actions, ...)
        action_reply = await vllm_generate(action_prompt, adapter="action_taking",
                                            max_tokens=512)
        action, reasoning, intention = parse_merged_response(action_reply, ...)

        # env.step() is synchronous — run in executor to avoid blocking event loop
        next_obs, reward, done, _, next_info = await loop.run_in_executor(
            executor, env.step, action
        )

        # ... (record GRPO data, update trackers, same as run_episode) ...

        if done:
            break

    env.close()
    return {"episode": episode, "stats": stats, "grpo_records": grpo_records}


async def collect_rollouts(
    game_episode_configs: List[Dict[str, Any]],
    max_concurrent: int = 40,
    on_episode_done: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """Launch all episodes as coroutines with semaphore-based concurrency control.

    Unlike a thread pool where each worker is occupied for the full episode
    duration, the semaphore only limits how many episodes are *active* — not
    how many are waiting on I/O. With 80 episodes and max_concurrent=40,
    all 40 active episodes' LLM calls land in vLLM's batch simultaneously.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def _run_one(config):
        async with semaphore:
            result = await run_episode_async(
                game=config["game"],
                max_steps=config["max_steps"],
                skill_bank=config["skill_bank"],
            )
            if on_episode_done:
                on_episode_done(result)
            return result

    tasks = [_run_one(cfg) for cfg in game_episode_configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        print(f"  {len(failures)} episodes failed: {failures[0]}")
    return successes
```

**Episode launch order matters.** All 80 coroutines are created immediately, but the semaphore gates which ones run. The order in the config list determines which coroutines acquire the semaphore first. A naive ordering (all candy_crush, then all sokoban, ...) creates problems:

```
BAD ordering (grouped by game):
t=0s    [candy1][candy2]...[candy10][sokoban1]...[sokoban10][2048_1]...[mario1]...[mario5]
        └── 10 candy_crush finish in 13s, GPU underloaded briefly ──┘
t=13s   candy done → mario6-10 start, but now GPU only has long-game episodes
t=100s  mario10 finishes last → long tail, GPU underloaded at the end
```

The optimal strategy: **interleave games sorted by descending duration, round-robin across episodes.**

This is the Longest Processing Time (LPT) scheduling heuristic — proven optimal for minimizing makespan on parallel machines:

```python
GAME_DURATION_ORDER = [
    "super_mario",          # ~100s per episode (longest)
    "pokemon_red",          # ~65s
    "diplomacy",            # ~55s
    "twenty_forty_eight",   # ~50s
    "tetris",               # ~45s
    "avalon",               # ~35s
    "sokoban",              # ~24s
    "candy_crush",          # ~13s (shortest)
]

def build_interleaved_configs(games, episodes_per_game, game_banks):
    """Build episode configs in interleaved order: longest games first, round-robin.

    Round 1: mario_ep0, pokemon_ep0, diplomacy_ep0, 2048_ep0, tetris_ep0, ...
    Round 2: mario_ep1, pokemon_ep1, diplomacy_ep1, 2048_ep1, tetris_ep1, ...
    ...

    This ensures:
    1. Long episodes acquire the semaphore first → they start generating
       tokens immediately while short episodes queue behind them
    2. Each semaphore "wave" contains a mix of game types → uniform GPU load
    3. Emulator games (mario, pokemon, diplomacy) are spread across different
       positions → CPU load is distributed
    4. Short games (candy, sokoban) finish early in each wave → their slots
       are refilled AND their trajectories feed the skill bank pipeline (Strategy E)
    """
    sorted_games = sorted(
        games,
        key=lambda g: GAME_DURATION_ORDER.index(g) if g in GAME_DURATION_ORDER else 4,
    )

    configs = []
    for ep_idx in range(episodes_per_game):
        for game in sorted_games:
            game_cfg = GAME_CONFIGS[game]
            configs.append({
                "game": game,
                "max_steps": game_cfg.max_steps,
                "skill_bank": game_banks.get(game),
                "episode_idx": ep_idx,
            })
    return configs
```

```
GOOD ordering (interleaved, longest first):
t=0s    [mario0][pokemon0][diplo0][2048_0][tetris0][avalon0][sokoban0][candy0]  ← wave 1 (8 eps)
        [mario1][pokemon1][diplo1][2048_1][tetris1][avalon1][sokoban1][candy1]  ← wave 2 (8 eps)
        ... (remaining 24 eps queue behind the semaphore)

t=13s   candy0,candy1 done → sokoban2,candy2 acquire semaphore
        → candy trajectories feed skill bank pipeline (Strategy E)
t=24s   sokoban0,sokoban1 done → tetris2,avalon2 acquire semaphore
t=35s   avalon0,avalon1 done → more episodes start
        GPU stays fully loaded — mix of long and short games always in flight

t=100s  Last mario episode completes
        By now, candy/sokoban/avalon trajectories are already through
        skill bank Stages 1-3 → significant head start on bank update
```

**Why LPT ordering specifically helps:**

| Problem | How LPT fixes it |
|---------|------------------|
| Long tail (GPU idles waiting for last mario ep) | Long episodes start first → they finish closer to when short ones finish |
| Burst-then-idle (all short games finish at once) | Interleaving ensures each wave has a mix → uniform vLLM request rate |
| CPU contention (all emulator games running simultaneously) | Emulator games (mario, pokemon, diplomacy) are positions 0,1,2 in each round — spread across waves |
| Strategy E starvation (no early completions for skill bank) | Short games still finish earliest within each wave → skill bank pipeline starts early |
| Uneven vLLM batch sizes (40 same-game prompts have similar length) | Mixed games produce varied prompt lengths → more efficient GPU memory usage |

At any moment the GPU batch contains requests from many different games at different step counts:
- candy_crush ep3 step 40's `action_taking` (token 45)
- pokemon_red ep1 step 12's `skill_selection` (token 12)
- tetris ep5 step 90's `action_taking` (prefilling)
- sokoban ep2 step 55's `skill_selection` (token 30)
- 2048 ep7 step 150's `action_taking` (token 88)
- ... up to ~40-60 concurrent requests

**Concurrency tuning:**

| max_concurrent | Peak vLLM requests | KV cache pressure | GPU utilization |
|----------------|---------------------|-------------------|-----------------|
| 20 | ~25-35 | Medium | ~60% |
| 40 | ~40-60 | High | ~85% |
| 60 | ~50-70 | Near limit | ~90% |

With async, we can safely push to `max_concurrent=40` because the overhead per coroutine is negligible (unlike threads which each consume ~8MB stack). The bottleneck shifts to vLLM's KV cache capacity (~50-70 concurrent requests on A100-80GB).

**Recommendation:** `max_concurrent=40` for maximum GPU utilization. If vLLM starts preempting (evicting KV cache entries), reduce to 30.

**Full launch example (8 games × 10 episodes = 80 episodes):**

```python
configs = build_interleaved_configs(games, episodes_per_game=10, game_banks=per_game_banks)
# configs order: mario0, pokemon0, diplo0, 2048_0, tetris0, avalon0, sokoban0, candy0,
#                mario1, pokemon1, diplo1, 2048_1, tetris1, avalon1, sokoban1, candy1, ...

results = await collect_rollouts(configs, max_concurrent=40, on_episode_done=on_episode_done)
```

- **Savings:** With 40 concurrent episodes across 8 games in LPT order, throughput scales ~20-30x vs serial — significantly better than the 15-20x from the threading model due to higher vLLM request concurrency, and ~10-15% better than naive ordering due to reduced long-tail effect.

### Strategy E — Cross-system batching (decision agent + skill bank share vLLM)

During co-evolution training, the decision agent rollout phase and skill bank update phase currently alternate. With all 5 adapters served through the same vLLM instance and everything running on the same event loop, we can **overlap** them naturally:

```
t=0s    Launch all 80 episode coroutines with max_concurrent=40

t=15s   candy_crush ep1–10 complete (shortest game)
        → on_episode_done fires: launch skill bank coroutines for
          candy_crush trajectories (boundary → segment → contract)
        → semaphore slots freed → queued episodes acquire them

t=30s   sokoban ep1–10 complete
        → skill bank Stage 1+2 on sokoban trajectories
        → candy_crush trajectories now in Stage 2 (segment adapter)

t=60s   2048, tetris episodes completing
        → skill bank processing continues concurrently

t=100s  super_mario (longest) completes → all rollouts collected
        → skill bank pipeline finishes remaining stages
```

The key insight: **short-game rollouts feed the skill bank pipeline while long-game rollouts are still collecting.** All requests — decision agent and skill bank — are async coroutines sharing the same event loop and the same `AsyncOpenAI` client. vLLM's batch contains a mix:

```
vLLM batch at t=30s:
  - 30 ongoing episodes: ~35 skill_selection/action_taking requests
  - candy_crush completed: ~20 segment/contract adapter requests  
  - Total: ~55 mixed-adapter requests in one GPU forward pass
```

Since the skill bank pipeline from `SKILLBANK_INFERENCE_SPEED.md` already uses `generate_batch()` with async, both systems share the same async client and event loop. No thread synchronization, no queues — just coroutines.

```python
async def co_evolution_step(games, episodes_per_game, skill_bank_pipeline, ...):
    """Run rollouts + skill bank updates concurrently on one event loop.

    Both decision agent episodes and skill bank stages are coroutines.
    All their vLLM requests land in the same continuous batch. Short-game
    trajectories feed the skill bank while long-game episodes still run.
    """
    completed_trajectories = asyncio.Queue()

    def on_episode_done(result):
        completed_trajectories.put_nowait(result)

    async def skill_bank_consumer():
        """Process completed trajectories through skill bank as they arrive."""
        batch = []
        while True:
            try:
                traj = await asyncio.wait_for(completed_trajectories.get(), timeout=5.0)
                batch.append(traj)
                if len(batch) >= 4:
                    await skill_bank_pipeline.process_batch_async(batch, vllm_client)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await skill_bank_pipeline.process_batch_async(batch, vllm_client)
                    batch = []
                break
        if batch:
            await skill_bank_pipeline.process_batch_async(batch, vllm_client)

    rollout_task = collect_rollouts(configs, max_concurrent=40,
                                    on_episode_done=on_episode_done)
    consumer_task = skill_bank_consumer()

    rollout_results, _ = await asyncio.gather(rollout_task, consumer_task)
    return rollout_results
```

- **Savings:** Skill bank processing (~4 min with vLLM batching) is almost entirely hidden behind rollout collection. Net co-evolution step time ≈ max(rollouts, skill_bank), not the sum.
- **Synergy with async model:** Since both systems are async coroutines on the same event loop, their vLLM requests naturally interleave — no thread synchronization overhead, no queue latency.

### Strategy F — Conditional intention caching

When the same skill remains active (no reselection), the intention often changes minimally. Cache and reuse the previous intention, only regenerating when:

- Skill changes (reselection triggered)
- State delta is significant (urgency detected by `_detect_urgency()`)
- Every K steps (e.g., K=3) as a freshness guarantee

```python
should_regen_intention = (
    need_reselect
    or urgency
    or step_count % intention_refresh_k == 0
)
if should_regen_intention:
    intention = generate_skill_aware_intention(...)
# else: reuse current_intention from previous step
```

- **Savings:** Eliminates ~60-70% of intention LLM calls (~0.3s each)
- **Risk:** Stale intentions may cause slightly worse action quality; mitigated by K-step refresh

---

## Projected Timeline

### Per Step

| Component | Current (serial) | Optimized (A+B+C) | Savings |
|-----------|------------------|--------------------|---------|
| summary_state | 0ms | 0ms | — |
| summary_prose | ~500ms | 0ms (skipped) | 500ms |
| skill_selection | ~500ms (conditional) | ~200ms (vLLM batched) | 300ms |
| intention | ~300ms | 0ms (merged into action) | 300ms |
| action_taking | ~500ms | ~200ms (vLLM batched) | 300ms |
| **Total** | **~1.3–1.8s** | **~0.2–0.4s** | **~4-6x** |

### Per Episode (by game)

| Game | Steps (typical) | Current (serial) | Optimized (A+B+C, single ep) |
|------|-----------------|------------------|-------------------------------|
| candy_crush | 45 | ~65s | ~13s |
| sokoban | 80 | ~120s | ~24s |
| avalon | 100 | ~180s | ~35s |
| tetris | 150 | ~225s | ~45s |
| diplomacy | 150 | ~300s | ~55s |
| twenty_forty_eight | 180 | ~270s | ~50s |
| pokemon_red | 200 | ~360s | ~65s |
| super_mario | 300 | ~500s | ~100s |

### Per Co-Evolution Rollout Collection (8 games × N episodes)

| Approach | 5 eps/game (40 total) | 8 eps/game (64 total) | 10 eps/game (80 total) |
|----------|----------------------|----------------------|------------------------|
| Current (fully serial) | ~47 min | ~75 min | ~93 min |
| vLLM + call reduction only (A+B+C, still serial) | ~10 min | ~16 min | ~20 min |
| + Async cross-episode batching (D, 40 concurrent) | **~1.5 min** | **~2.5 min** | **~3 min** |

**How the async estimate works (10 eps/game example):**
- 80 episodes total, 40 coroutines active simultaneously
- All 40 coroutines issue LLM calls concurrently → ~40-60 concurrent vLLM requests
- vLLM processes ~800-1000 tok/s at this concurrency → each LLM call returns in ~100-150ms (batched)
- Per optimized step: ~200-300ms (1-2 LLM calls + env.step)
- Wall time dominated by the longest game: super_mario (300 steps × 0.25s/step ≈ 75s) × 10 eps
- But with 40 concurrent coroutines, we run ~5 mario eps simultaneously → ~75s × 2 waves = **~2.5 min**
- Short-game episodes are essentially "free" — they fit in the gaps while long-game episodes run
- Async gives ~1.5x better throughput than 20-thread model because of ~2x higher vLLM request concurrency

### Per Co-Evolution Step (rollouts + skill bank + GRPO)

| Phase | Current | Optimized | Notes |
|-------|---------|-----------|-------|
| Rollout collection (8 games × 8 eps) | ~75 min | ~2.5 min | Strategy A+B+C+D (async, 40 concurrent) |
| Skill bank update (3 EM iterations) | ~12 min | ~4 min | From SKILLBANK_INFERENCE_SPEED.md |
| Cross-system overlap (E) | — | −3.5 min | Skill bank processes early-finishing episodes while late ones run |
| GRPO updates (decision agent) | ~5 min | ~5 min | Sequential, GPU-bound |
| **Total co-evolution step** | **~92 min** | **~8 min** | **~11.5x speedup** |

### Full Co-Evolution Training

| Training run | Current | Optimized |
|-------------|---------|-----------|
| 10 co-evolution steps | ~15 hours | ~1.3 hours |
| 30 co-evolution steps | ~46 hours | ~4 hours |
| 50 co-evolution steps | ~77 hours | ~6.7 hours |

---

## GPU Memory Budget (5 LoRAs)

| Component | Memory |
|-----------|--------|
| Qwen3-8B base weights (bf16) | ~16 GB |
| 5 LoRA adapters (rank 16 each) | ~1.0 GB total |
| vLLM KV cache (at 0.85 utilization) | ~38 GB |
| **Total** | ~67.4 GB / 80 GB |

With ~38 GB KV cache and mixed adapter requests, vLLM supports ~50-70 concurrent requests — enough for 8 concurrent episodes each with 1-2 in-flight requests plus skill bank pipeline requests during cross-system batching.

---

## What Cannot Be Parallelized

| Component | Why sequential |
|-----------|---------------|
| Steps within an episode | Each step's action must be executed in env before next step starts |
| skill_selection → action_taking within a step | Action prompt needs the chosen skill's guidance |
| GRPO parameter updates | Gradient computation needs all group rewards |
| Cross-iteration EM (iter 1 → 2 → 3) | Next iteration uses the bank/adapters from the previous |
| LoRA hot-reload after GRPO update | vLLM must reload updated adapter weights (~1-2s per adapter) |

---

## Implementation Checklist

### P0 — Critical (eliminates wasted LLM calls, ~4-6x per step)

- [ ] **Drop `summary_prose` in inference mode** — add `inference_mode` flag to `run_episode()` that skips `generate_summary_prose()`. Store `summary_state` as `exp.summary` instead.
- [ ] **Merge intention into `action_taking`** — modify `qwen3_action()` prompt template to include intention generation; update `parse_qwen_response()` to extract `INTENTION:` line; add flag to toggle merged vs separate mode.
- [ ] **Async vLLM client for decision agent** — reuse the async vLLM client from the skill bank plan (`skill_agents/lora/vllm_client.py`). Update `DualLoRAManager.call_skill_selection()` and `call_action_taking()` to use async vLLM API instead of HuggingFace `.generate()`.
- [ ] **Update vLLM launch script** — add `skill_selection` and `action_taking` to `--lora-modules`, set `--max-loras 5`.

### P1 — Important (cross-episode throughput, ~20-30x total)

- [ ] **`run_episode_async()`** — rewrite `run_episode()` as async coroutine. LLM calls use `await vllm_client.completions.create(...)`. `env.step()` runs in executor via `await loop.run_in_executor(executor, env.step, action)`.
- [ ] **Game-to-executor routing** — classify each game as `EMULATOR_GAMES` (ProcessPoolExecutor) or pure-Python (ThreadPoolExecutor); `run_episode_async()` selects the correct executor based on game name.
- [ ] **`build_interleaved_configs()` with LPT ordering** — sort games by descending episode duration, interleave round-robin across episodes. Ensures long games start first, each wave has a mix of game types, and emulator games are spread across positions.
- [ ] **`collect_rollouts()` with asyncio** — launch all 40–80 episodes as coroutines in LPT-interleaved order, gate concurrency with `asyncio.Semaphore(max_concurrent=40)`. `asyncio.gather` handles all scheduling; short-game episodes release the semaphore early for queued long-game episodes.
- [ ] **Conditional intention caching** — skip intention regeneration when skill unchanged and no urgency/significant delta detected. Add `intention_refresh_k` parameter (default 3).

### P2 — Co-evolution optimization (~10x total)

- [ ] **Cross-system batching with `on_episode_done` callback** — when an episode coroutine completes, push the trajectory to an `asyncio.Queue`. A consumer coroutine pulls trajectories and runs them through the skill bank pipeline (boundary → segment → contract) on the same event loop and same `AsyncOpenAI` client. All vLLM requests (decision agent + skill bank) naturally interleave in the same GPU batch.
- [ ] **`co_evolution_step()` orchestrator** — async function that runs `collect_rollouts()` and `skill_bank_consumer()` concurrently via `asyncio.gather`. Coordinates the producer-consumer pattern between rollout collection and skill bank updates.
- [ ] **Speculative candidate prefetch** — launch skill candidate retrieval (CPU, in executor) concurrently with the current step's `env.step()` via `asyncio.gather`, so candidates are ready for the next step's potential reselection.
- [ ] **Adaptive concurrency** — poll vLLM queue depth via `/metrics` endpoint periodically; dynamically adjust `semaphore._value` if queue depth exceeds safe threshold (~50 concurrent requests). Also useful during cross-system batching when skill bank requests compete for GPU bandwidth.

---

## Environment Concurrency Model

All LLM calls are async (coroutines using `AsyncOpenAI`). The synchronous `env.step()` calls run in executors via `loop.run_in_executor()`. Each game is routed to the appropriate executor based on its CPU profile:

| Game | Env type | Executor for env.step() | GIL during env.step() | env.step() latency |
|------|----------|-------------------------|-----------------------|--------------------|
| twenty_forty_eight | Pure Python / NumPy | **ThreadPoolExecutor** | Released (NumPy) | ~2ms |
| candy_crush | Pure Python / NumPy | **ThreadPoolExecutor** | Released (NumPy) | ~5ms |
| sokoban | Pure Python | **ThreadPoolExecutor** | Brief hold | ~5ms |
| tetris | Pure Python / NumPy | **ThreadPoolExecutor** | Released (NumPy) | ~5ms |
| avalon | Custom Python | **ThreadPoolExecutor** | Brief hold | ~10ms |
| pokemon_red | PyBoy emulator (C) | **ProcessPoolExecutor** | Held during frame render | ~20ms |
| super_mario | gym-super-mario-bros (C) | **ProcessPoolExecutor** | Held during frame render | ~15ms |
| diplomacy | Custom (multi-agent) | **ProcessPoolExecutor** | Varies | ~50ms |

**Why the executors only matter for env.step():**
- LLM calls are `await vllm_client.completions.create(...)` — pure async I/O, no GIL involvement
- `env.step()` is the only synchronous blocking call in the loop
- For pure-Python games, `env.step()` is 2-10ms — even blocking the event loop briefly is acceptable, but running in a thread executor keeps the event loop fully responsive for other coroutines
- For emulator games, `env.step()` is 15-50ms with heavy C code that holds the GIL — process executor avoids blocking other threads' env.step() calls

**Process executor for emulators:**
- `ProcessPoolExecutor` requires picklable arguments
- `env.step(action)` only passes a string action — trivially picklable
- The env object itself lives in the coroutine (main process); only the step call is shipped to the worker process
- **Caveat:** `loop.run_in_executor()` with `ProcessPoolExecutor` requires the callable to be module-level (not a lambda or closure). Wrap in a helper function.
- Alternative: if process serialization is too complex for some envs, fall back to `ThreadPoolExecutor` — the GIL impact of 5-8 emulator episodes is manageable (~100ms/cycle total GIL contention)

---

## Verification Metrics

| Metric | Target | How to measure |
|--------|--------|----------------|
| LLM calls per step | ≤ 2 (down from 3–4) | Counter in `run_episode()` |
| Per-step latency (single ep) | ≤ 400ms | Timer in `run_episode()` |
| Candy crush episode (shortest) | ≤ 15s | Timer per episode |
| Super mario episode (longest) | ≤ 100s | Timer per episode |
| Rollout collection (8 games × 8 eps) | ≤ 3 min | Timer in `collect_rollouts()` |
| Co-evolution step (rollout + bank + GRPO) | ≤ 8 min | Timer in `co_evolution_step()` |
| GPU utilization during rollouts | ≥ 70% | `nvidia-smi` |
| vLLM throughput (5 LoRAs, mixed) | ≥ 700 tok/s | `vllm.metrics` endpoint |
| vLLM queue depth (peak) | ≤ 50 | `vllm.metrics` endpoint |
| Action quality (reward) | No regression vs serial | Compare per-game mean episode rewards |
| Full 30-step co-evolution | ≤ 4 hours | End-to-end timer |
