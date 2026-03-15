# Co-Evolution Loop — Unified Plan

**Created:** 2026-03-15  
**Status:** Open  
**Depends on:** `DECISION_AGENT_INFERENCE_SPEED.md`, `SKILLBANK_INFERENCE_SPEED.md`  
**Existing code:** `trainer/launch_coevolution.py`, `trainer/decision/coevolution_callback.py`

---

## Overview

The co-evolution framework alternates between two agents:

1. **Decision Agent** — plays games using two LoRA adapters (`skill_selection`, `action_taking`), producing rollouts
2. **Skill Bank Agent** — processes rollouts to discover, segment, and refine skills via three GRPO-trained LoRA adapters (`segment`, `contract`, `curator`). Stage 1 (boundary) uses the base model (reward is too indirect for GRPO). `retrieval` is legacy/planned.

Both share one Qwen3-14B base model served through a single vLLM instance with 5 LoRA adapters loaded simultaneously.

---

## The Loop

```
                    ┌──────────────────────────────────────────────────────┐
                    │                  CO-EVOLUTION LOOP                   │
                    └──────────────────────────────────────────────────────┘

Step 0 (cold start):
    Bank = empty
    Decision agent collects rollouts WITHOUT skill selection
    (action_taking LoRA only, no skill_selection calls)

Step 1:
    Skill bank processes Step 0 rollouts → Bank_v1
    (boundary → segment → contract → maintenance)

Step 2:
    Decision agent collects rollouts WITH skill selection using Bank_v1
    (both skill_selection + action_taking LoRAs active)

Step 3:
    Skill bank processes Step 2 rollouts → Bank_v2

    ... repeat Step 2-3 ...

Step 2k:
    Decision agent collects rollouts with Bank_v{k}

Step 2k+1:
    Skill bank processes rollouts → Bank_v{k+1}
    GRPO updates all 5 LoRAs
```

---

## Detailed Phase Breakdown

### Phase A — Rollout Collection (Decision Agent)

**What:** Run 8 games × N episodes (5–10 each) = 40–80 episodes concurrently.

**Two modes:**
- **Cold-start (bank empty):** Skip skill selection entirely. The `run_episode_async()` coroutine sets `skill_bank=None` → `get_top_k_skill_candidates()` returns `[]` → no `skill_selection` LoRA call → only `action_taking` LoRA runs per step. This produces baseline rollouts that the skill bank can segment.
- **Warm (bank populated):** Full pipeline — `skill_selection` LoRA picks from bank candidates, then `action_taking` LoRA uses the chosen skill's protocol.

**How (from DECISION_AGENT_INFERENCE_SPEED.md):**
- Async coroutines with `asyncio.Semaphore(max_concurrent=40)`
- LPT-ordered episode configs (longest games first, interleaved round-robin)
- `env.step()` in ThreadPoolExecutor (pure-Python games) or ProcessPoolExecutor (emulators)
- All LLM calls via `AsyncOpenAI` to shared vLLM server

**Output:** List of `(Episode, stats, grpo_records)` per episode. The `grpo_records` contain prompt/completion/reward for both `action_taking.jsonl` and `skill_selection.jsonl`.

**Estimated time (optimized):** ~2.5 min for 8 games × 8 episodes

### Phase B — Skill Bank Update (Skill Bank Agent)

**What:** Process the collected rollout trajectories through the 4-stage skill bank pipeline.

**Stages:**
1. **Boundary Proposal** (base model, no LoRA) — find segment cut points in trajectories
2. **Segmentation Decode** (segment LoRA) — assign skill labels to segments via DP + LLM re-ranking
3. **Contract Learning** (contract LoRA) — learn effect contracts (preconditions/postconditions) per skill
4. **Bank Maintenance** (curator LoRA) — refine, merge, split, materialize skills

**How (from SKILLBANK_INFERENCE_SPEED.md):**
- Pipelined micro-batch execution through Stages 1→2→3 with staggered async
- Stage 4 is sequential (tool-calling agent with shared sandbox state)
- 3 EM iterations per update (convergence check between iterations)

**Output:** Updated `SkillBankMVP` with new/refined skills, contracts, and protocols.

**Estimated time (optimized):** ~4 min for 3 EM iterations

### Phase C — GRPO Training

**What:** Update LoRA adapter weights using the collected rollouts.

**Two GRPO systems:**
1. **Decision Agent GRPO** — updates `skill_selection` and `action_taking` LoRAs using episode rewards
2. **Skill Bank GRPO** — updates `segment`, `contract`, `curator` LoRAs using stage-specific rewards (decode quality, pass rate, etc.)

**How:**
- Pause vLLM serving (time-slice approach from SKILLBANK_INFERENCE_SPEED.md)
- Run `GRPOLoRATrainer.train_step()` for each adapter with data
- Hot-reload updated adapter weights into vLLM (~1-2s per adapter)

**Output:** Updated LoRA adapter weights in vLLM.

**Estimated time:** ~5 min (sequential, GPU-bound)

---

## Full Co-Evolution Step

```
                    Phase A                    Phase B              Phase C
              ┌─────────────────┐      ┌──────────────────┐   ┌───────────────┐
              │ Rollout collect  │      │ Skill bank update │   │ GRPO training  │
              │ 8 games × 8 eps │ ───► │ 3 EM iterations   │──►│ 5 LoRA updates │
              │ (~2.5 min)       │      │ (~4 min)          │   │ (~5 min)       │
              └─────────────────┘      └──────────────────┘   └───────────────┘
                                              ▲
                                              │ cross-system overlap:
                                              │ short-game rollouts feed
                                              │ Stages 1-3 while long-game
                                              │ episodes still running
```

### With Cross-System Overlap (Strategy E)

Phases A and B overlap — as short-game episodes complete, their trajectories immediately enter the skill bank pipeline:

```
t=0s    Phase A starts: launch 64 episode coroutines (8 games × 8 eps)
        All use skill_selection + action_taking adapters

t=15s   candy_crush episodes complete → Phase B starts for candy_crush
        skill bank pipeline uses segment + contract + curator adapters
        vLLM batch: ~30 decision agent + ~20 skill bank = ~50 mixed requests

t=30s   sokoban episodes complete → Phase B processes sokoban
        candy_crush already in Stage 2 (segment)

t=60s   2048, tetris, avalon episodes complete → Phase B continues

t=90s   pokemon_red, diplomacy complete

t=120s  super_mario (longest) completes → Phase A done
        Phase B has already processed 6/8 games → only ~1 min remaining

t=150s  Phase B completes → all trajectories processed, bank updated
        Effective Phase B time: ~0.5 min (most was hidden behind Phase A)

t=155s  Phase C starts: GRPO updates, vLLM paused
        Update skill_selection, action_taking, segment, contract, curator LoRAs

t=450s  Phase C done, vLLM resumes with new weights
        → Next co-evolution step begins
```

**Total optimized co-evolution step: ~7.5 min**

---

## Cold-Start Handling

The first co-evolution step is special because the skill bank is empty:

```
Step 0 (one-time):
    ┌────────────────────────────────────────────────┐
    │ Phase A (cold-start mode):                     │
    │   - skill_bank=None passed to run_episode()    │
    │   - skill_selection LoRA never called           │
    │   - Only action_taking LoRA + base model run   │
    │   - Rollouts have no skill labels              │
    │   - GRPO records: action_taking only           │
    │                                                │
    │   → Produces raw game trajectories             │
    └────────────────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────┐
    │ Phase B (initial extraction):                  │
    │   - Boundary proposal discovers segments       │
    │   - Decode assigns __NEW__ labels              │
    │   - Contract learning creates initial contracts│
    │   - Bank maintenance materializes proto-skills │
    │                                                │
    │   → Bank_v1 created from scratch               │
    │   → Typically 30-60 skills across 8 games      │
    └────────────────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────┐
    │ Phase C (GRPO, limited):                       │
    │   - action_taking LoRA updated (has data)      │
    │   - skill_selection LoRA skipped (no data)     │
    │   - Skill bank LoRAs updated (have data)       │
    └────────────────────────────────────────────────┘
```

After Step 0, all subsequent steps run in warm mode with full skill selection.

### Smooth transition to warm mode

The decision agent needs graceful handling when the bank transitions from empty to populated:

```python
async def run_episode_async(game, max_steps, skill_bank=None, ...):
    bank_available = skill_bank is not None and len(skill_bank) > 0

    for step in range(max_steps):
        summary_state = generate_summary_state(...)

        guidance = None
        if bank_available:
            if need_reselect:
                candidates = get_top_k_skill_candidates(skill_bank, ...)
                if candidates:
                    skill_reply = await vllm_generate(skill_prompt, adapter="skill_selection")
                    guidance = parse_skill_selection(skill_reply, ...)

        # action_taking always runs, with or without skill guidance
        action_reply = await vllm_generate(
            build_action_prompt(summary_state, guidance, actions, ...),
            adapter="action_taking",
        )
        ...
```

---

## GRPO Integration — What Gets Updated When

| Adapter | Agent | GRPO reward signal | Updated when | Source data |
|---------|-------|-------------------|--------------|-------------|
| `action_taking` | Decision | Step reward from env | Every step | `action_taking.jsonl` |
| `skill_selection` | Decision | Step reward from env | Every step with candidates ≥ 2 | `skill_selection.jsonl` |
| `segment` | Skill Bank | Stage 3 contract pass rate + decision agent follow score | Phase C | Stage 2 GRPO rollouts |
| `contract` | Skill Bank | Holdout verification pass rate | Phase C | Stage 3 GRPO rollouts |
| `curator` | Skill Bank | Bank quality delta (filtered vs unfiltered) | Phase C | Stage 4 GRPO rollouts |

**Not GRPO-trained:**
- `boundary` (Stage 1) — uses base model. Boundary reward is too indirect (depends on downstream decode quality). Predicate extraction and significance scoring run without a LoRA adapter.
- `retrieval` — legacy/planned. Not actively trained in either GRPO system.

### GRPO training order within Phase C

```
1. Update action_taking (reward = step reward, independent)
2. Update skill_selection (reward = step reward, independent)
   ↕ can run in parallel with #1

3. Update segment (reward = Stage 3 pass rate + follow score)
4. Update contract (reward = holdout verification)
5. Update curator (reward = bank quality delta)
   ↕ must be sequential: #3 → #4 → #5 (reward cascades)
```

**Optimization:** Steps 1-2 (decision agent GRPO) and Steps 3-5 (skill bank GRPO) are independent — they use different data and different reward signals. They could run concurrently on separate GPUs. On a single GPU, run decision agent GRPO first (faster, ~2 min), then skill bank GRPO (~3 min).

---

## Orchestrator Implementation

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class CoEvolutionConfig:
    games: List[str]
    episodes_per_game: int = 8
    max_concurrent_episodes: int = 40
    bank_dir: str = "runs/skillbank"
    checkpoint_dir: str = "runs/coevolution/checkpoints"
    checkpoint_interval: int = 5
    grpo_enabled: bool = True
    vllm_base_url: str = "http://localhost:8000/v1"
    total_steps: int = 30
    wandb_project: str = "game-ai-coevolution"
    wandb_enabled: bool = True
    resume_from_step: Optional[int] = None


async def co_evolution_loop(config: CoEvolutionConfig):
    """Main co-evolution training loop.

    Step 0:  Rollouts (no bank) → Skill bank extraction → Bank_v1
    Step 1+: Rollouts (with bank) → Skill bank update → Bank_v{k+1}
             + GRPO updates for all 5 LoRAs

    Checkpoints saved every `checkpoint_interval` steps.
    All metrics logged to W&B in real time.
    """
    import wandb

    vllm_client = AsyncOpenAI(base_url=config.vllm_base_url, api_key="EMPTY")

    bank_store = VersionedBankStore(bank_dir=config.bank_dir)
    skill_bank_pipeline = SkillBankPipeline(bank_store, vllm_client)

    # ── W&B init ──────────────────────────────────────────────────
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            config={
                "games": config.games,
                "episodes_per_game": config.episodes_per_game,
                "max_concurrent": config.max_concurrent_episodes,
                "total_steps": config.total_steps,
                "checkpoint_interval": config.checkpoint_interval,
            },
            resume="allow",
        )

    # ── Resume from checkpoint if requested ───────────────────────
    start_step = 0
    if config.resume_from_step is not None:
        start_step = load_checkpoint(config.checkpoint_dir, config.resume_from_step,
                                     bank_store)
        print(f"  Resumed from checkpoint at step {start_step}")

    for step in range(start_step, config.total_steps):
        t0 = time.time()
        bank = bank_store.current_bank
        bank_available = bank is not None and len(bank) > 0

        # ── Phase A: Collect rollouts ─────────────────────────────
        configs = build_interleaved_configs(
            config.games, config.episodes_per_game,
            game_banks={g: bank for g in config.games} if bank_available else {},
        )

        completed_queue = asyncio.Queue()

        def on_episode_done(result):
            completed_queue.put_nowait(result)

        async def skill_bank_consumer():
            batch = []
            while True:
                try:
                    traj = await asyncio.wait_for(completed_queue.get(), timeout=5.0)
                    batch.append(traj)
                    if len(batch) >= 4:
                        await skill_bank_pipeline.process_batch_async(batch)
                        batch = []
                except asyncio.TimeoutError:
                    if batch:
                        await skill_bank_pipeline.process_batch_async(batch)
                        batch = []
                    break
            if batch:
                await skill_bank_pipeline.process_batch_async(batch)

        rollout_task = collect_rollouts(
            configs,
            max_concurrent=config.max_concurrent_episodes,
            on_episode_done=on_episode_done,
        )
        consumer_task = skill_bank_consumer()

        rollout_results, _ = await asyncio.gather(rollout_task, consumer_task)

        # ── Phase B: Finalize skill bank update ───────────────────
        await skill_bank_pipeline.finalize_em_iteration(bank_store)

        if not skill_bank_pipeline.passes_quality_gate():
            bank_store.rollback()
            print(f"  Step {step}: bank update REJECTED, rolled back")

        # ── Phase C: GRPO training ────────────────────────────────
        grpo_stats = {}
        if config.grpo_enabled:
            await pause_vllm_serving()

            grpo_records = collect_all_grpo_records(rollout_results)
            decision_stats = await train_decision_grpo(grpo_records)
            skillbank_stats = await train_skillbank_grpo(skill_bank_pipeline.grpo_data)
            grpo_stats = {**decision_stats, **skillbank_stats}

            await resume_vllm_serving()

        elapsed = time.time() - t0
        bank = bank_store.current_bank
        n_skills = len(bank) if bank else 0
        mode = "cold-start" if step == 0 else "warm"

        # ── W&B logging ───────────────────────────────────────────
        if config.wandb_enabled:
            episode_metrics = compute_episode_metrics(rollout_results)
            wandb.log({
                "step": step,
                "wall_time_s": elapsed,
                "bank_version": bank_store.version,
                "n_skills": n_skills,
                "mode": 0 if step == 0 else 1,

                # Per-game reward curves
                **{f"reward/{game}/mean": m["mean_reward"]
                   for game, m in episode_metrics["per_game"].items()},
                **{f"reward/{game}/max": m["max_reward"]
                   for game, m in episode_metrics["per_game"].items()},
                **{f"reward/{game}/episode_length": m["mean_length"]
                   for game, m in episode_metrics["per_game"].items()},

                # Aggregate reward
                "reward/all_games_mean": episode_metrics["overall_mean_reward"],
                "reward/all_games_max": episode_metrics["overall_max_reward"],

                # GRPO losses
                **{f"grpo/{k}": v for k, v in grpo_stats.items()},

                # Skill bank quality
                "bank/n_skills": n_skills,
                "bank/accepted": int(skill_bank_pipeline.passes_quality_gate()),

                # Timing breakdown
                "timing/phase_a_s": episode_metrics.get("phase_a_time", 0),
                "timing/phase_b_s": episode_metrics.get("phase_b_time", 0),
                "timing/phase_c_s": grpo_stats.get("train_time", 0),
                "timing/total_s": elapsed,
            }, step=step)

        # ── Checkpoint every N steps ──────────────────────────────
        if (step + 1) % config.checkpoint_interval == 0 or step == 0:
            save_checkpoint(
                checkpoint_dir=config.checkpoint_dir,
                step=step,
                bank_store=bank_store,
                adapter_paths={
                    "action_taking": "runs/lora_adapters/action_taking",
                    "skill_selection": "runs/lora_adapters/skill_selection",
                    "segment": "runs/lora_adapters/segment",
                    "contract": "runs/lora_adapters/contract",
                    "curator": "runs/lora_adapters/curator",
                },
            )
            print(f"  Checkpoint saved at step {step}")

        print(
            f"  Co-evolution step {step} ({mode}): "
            f"{elapsed:.0f}s, bank_v{bank_store.version}, "
            f"{n_skills} skills"
        )

    if config.wandb_enabled:
        wandb.finish()


def save_checkpoint(checkpoint_dir, step, bank_store, adapter_paths):
    """Save full co-evolution state for resumption."""
    import json, shutil
    ckpt_path = Path(checkpoint_dir) / f"step_{step:04d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Save bank snapshot
    bank_store.current_bank.save(str(ckpt_path / "skill_bank.jsonl"))

    # Copy all LoRA adapter weights
    for name, src in adapter_paths.items():
        dst = ckpt_path / "adapters" / name
        if Path(src).exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Save metadata
    meta = {
        "step": step,
        "bank_version": bank_store.version,
        "n_skills": len(getattr(bank_store.current_bank, "skill_ids", [])),
        "timestamp": time.time(),
    }
    (ckpt_path / "meta.json").write_text(json.dumps(meta, indent=2))


def load_checkpoint(checkpoint_dir, step, bank_store):
    """Restore co-evolution state from a checkpoint."""
    import json, shutil
    ckpt_path = Path(checkpoint_dir) / f"step_{step:04d}"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Restore bank
    bank_store.load_from(str(ckpt_path / "skill_bank.jsonl"))

    # Restore adapter weights
    adapters_dir = ckpt_path / "adapters"
    if adapters_dir.exists():
        for name in adapters_dir.iterdir():
            dst = Path(f"runs/lora_adapters/{name.name}")
            shutil.copytree(name, dst, dirs_exist_ok=True)

    meta = json.loads((ckpt_path / "meta.json").read_text())
    return meta["step"] + 1  # resume from next step
```

---

## Timing Budget

### Per Co-Evolution Step (8 games × 8 episodes, optimized)

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase A: Rollout collection | ~2.5 min | 64 async episodes, 40 concurrent, LPT order |
| Phase B: Skill bank update | ~0.5 min visible | Most hidden behind Phase A via cross-system batching |
| Phase C: GRPO training | ~5 min | Sequential, GPU-bound (vLLM paused) |
| Overhead (hot-reload, logging) | ~0.5 min | 5 LoRA reloads + metrics |
| **Total** | **~8.5 min** | |

### Cold-Start Step (Step 0)

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase A: Rollout collection (no bank) | ~2 min | Fewer LLM calls per step (no skill_selection) |
| Phase B: Initial bank extraction | ~5 min | More work — all skills are __NEW__, more segments |
| Phase C: GRPO training (partial) | ~3 min | Only action_taking + skill bank LoRAs |
| **Total** | **~10 min** | |

### Full Training Run

| Scenario | Steps | Wall time |
|----------|-------|-----------|
| 10 steps (quick experiment) | 1 cold + 9 warm | ~1.4 hours |
| 30 steps (standard) | 1 cold + 29 warm | ~4.2 hours |
| 50 steps (thorough) | 1 cold + 49 warm | ~7 hours |

### Compared to Serial Baseline

| | Serial (current) | Optimized | Speedup |
|-|------------------|-----------|---------|
| Per rollout collection | ~75 min | ~2.5 min | 30x |
| Per skill bank update | ~47 min | ~4 min (0.5 min visible) | 12x |
| Per GRPO training | ~5 min | ~5 min | 1x |
| **Per step total** | **~127 min** | **~8.5 min** | **~15x** |
| **30-step training** | **~63 hours** | **~4.2 hours** | **~15x** |

---

## Dependencies Between Plans

```
SKILLBANK_INFERENCE_SPEED.md
  └── vLLM serving with multi-LoRA (P0)
  └── Async vLLM client + generate_batch() (P0)
  └── Pipelined micro-batch EM executor (P1)
       │
       ▼
DECISION_AGENT_INFERENCE_SPEED.md
  └── Drop summary_prose + merge intention (P0)
  └── run_episode_async() with executor env.step() (P1)
  └── collect_rollouts() with LPT ordering (P1)
  └── Cross-system batching with on_episode_done (P2)
       │
       ▼
CO_EVOLUTION_PLAN.md (this file)
  └── co_evolution_loop() orchestrator
  └── Cold-start handling
  └── GRPO integration for all 5 LoRAs
  └── Phase A-B overlap via async producer-consumer
```

---

## Implementation Checklist

### P0 — Infrastructure (required before anything runs)

- [ ] **vLLM server with 5 LoRAs** — single launch script serving all GRPO-trained adapters:
  `segment`, `contract`, `curator`, `skill_selection`, `action_taking`
- [ ] **Shared `AsyncOpenAI` client** — one client instance used by both decision agent (`run_episode_async`) and skill bank pipeline (`generate_batch`)
- [ ] **`run_episode_async()` with cold-start mode** — `skill_bank=None` → skip skill_selection; `skill_bank` populated → full pipeline

### P1 — Core loop

- [ ] **`co_evolution_loop()` orchestrator** — async function implementing the Step 0 / Step 1+ pattern. Integrates `collect_rollouts()`, `SkillBankPipeline`, and GRPO training.
- [ ] **`build_interleaved_configs()` with bank-awareness** — when `bank=None`, set `skill_bank=None` in configs so episodes run without skill selection.
- [ ] **GRPO data collection in `run_episode_async()`** — record prompt/completion/reward for both `action_taking` and `skill_selection` (when active) in the same format as `GRPOStepRecord`.
- [ ] **`train_decision_grpo()` + `train_skillbank_grpo()`** — wrapper functions that pause vLLM, run `GRPOLoRATrainer.train_step()` for each adapter, hot-reload updated weights.

### P2 — Cross-system optimization

- [ ] **Phase A-B overlap** — `on_episode_done` callback feeds completed trajectories to `skill_bank_consumer()` coroutine. Both share the same event loop and vLLM client.
- [ ] **VersionedBankStore integration** — bank snapshots, rollback on quality gate failure, diff logging between versions.
- [ ] **W&B logging** — log per-game reward curves (`reward/{game}/mean`, `reward/{game}/max`, `reward/{game}/episode_length`), aggregate reward (`reward/all_games_mean`), GRPO losses per adapter (`grpo/action_taking_loss`, etc.), skill bank quality (`bank/n_skills`, `bank/accepted`), and timing breakdown (`timing/phase_a_s`, `timing/phase_b_s`, `timing/phase_c_s`). Logged every step in real time.
- [ ] **Checkpoint every 5 steps** — `save_checkpoint()` stores bank snapshot (`skill_bank.jsonl`), all 5 LoRA adapter weights, and metadata (`step`, `bank_version`, `n_skills`, `timestamp`) under `runs/coevolution/checkpoints/step_NNNN/`. Also checkpoint at step 0 (cold start). `load_checkpoint()` restores bank + adapters for resumption via `--resume-from-step N`.
- [ ] **`compute_episode_metrics()`** — aggregate rollout results into per-game and overall reward stats for W&B logging.

---

## What Cannot Be Parallelized

| Constraint | Why |
|-----------|-----|
| Steps within an episode | env.step() depends on previous action |
| Phase C (GRPO) needs Phase A complete | GRPO needs all group rewards from rollouts |
| Skill bank EM iteration N+1 needs N | Next iteration uses the bank from the previous |
| Phase C pauses vLLM | GRPO training needs exclusive GPU access |
| Cross-step dependency | Step k+1 uses Bank_v{k} from step k |

---

## GPU Resource Timeline

### Option 1: Single A100-80GB

```
t=0          t=2.5min      t=3min        t=8min        t=8.5min
├────────────┼─────────────┼─────────────┼─────────────┤
│ vLLM serve │ vLLM serve  │             │ vLLM serve  │
│ (Phase A)  │ (Phase A+B) │ GRPO train  │ (next step) │
│            │             │ (Phase C)   │             │
│ 5 LoRAs    │ 5 LoRAs     │ vLLM paused │ 5 LoRAs     │
│ ~85% util  │ ~90% util   │ ~100% util  │ ~85% util   │
├────────────┼─────────────┼─────────────┼─────────────┤
                                          ↑ hot-reload
                                            new weights
```

- During Phase A+B: vLLM serves 40-60 concurrent requests across 5 adapters
- During Phase C: GRPO training uses full GPU for gradient computation (vLLM paused)
- Between steps: ~5s for 5 LoRA hot-reloads into vLLM
- **Per step: ~8.5 min | 30 steps: ~4.2 hours**

### Option 2: 8 GPUs (4 inference + 4 training) — Recommended

Split GPUs by role, not by agent. This eliminates the vLLM pause entirely.

**GPU assignment:**

| GPUs | Role | What runs |
|------|------|-----------|
| 0-3 | Inference (always running) | vLLM with `tensor_parallel=4`, ALL 5 LoRAs |
| 4-5 | Training (decision agent) | GRPO for `action_taking` + `skill_selection` |
| 6-7 | Training (skill bank) | GRPO for `segment` + `contract` + `curator` |

**Why 4 inference + 4 training, NOT 4 decision + 4 skill bank:**

Splitting by agent (4 for decision, 4 for skill bank) wastes ~50% of GPU time. During Phase A (rollouts), the skill bank GPUs sit idle. During Phase B, the decision GPUs sit idle. Splitting by role keeps all GPUs busy: inference GPUs serve both agents' LLM calls continuously, training GPUs update both agents' adapters concurrently.

**Timeline per co-evolution step:**

```
GPUs 0-3 (vLLM TP=4, never paused):
Step k:  [Phase A: rollouts ~1.5min][B: ~0.3min]──[Phase A: step k+1...]
          action_taking, skill_selection   segment,               ↑ hot-reload
          requests across 8 games          contract,              new adapter
                                           curator                weights
                                           requests

GPUs 4-5 (decision GRPO):
Step k:                              [action_taking GRPO][skill_sel GRPO]
                                     ├────── ~1.5 min ──┤├─── ~0.5 min ─┤
                                                                         ↓ save to disk

GPUs 6-7 (skill bank GRPO):
Step k:                              [segment GRPO][contract GRPO][curator GRPO]
                                     ├─── ~1 min ──┤├── ~1 min ──┤├── ~1 min ──┤
                                                                                ↓ save to disk
```

**Three key wins:**

1. **Phase C is hidden** — GRPO training on GPUs 4-7 runs concurrently with inference on GPUs 0-3. The 5-minute GRPO pause is eliminated. After GRPO finishes, adapter weights are saved to disk and hot-reloaded into vLLM (~1-2s per adapter, no restart needed).

2. **TP=4 speeds up per-request latency** — with tensor parallelism across 4 GPUs, the 14B model's forward pass is split 4 ways. Per-token latency drops ~3x. Combined with ~200 GB of KV cache (vs ~38 GB on 1 GPU), vLLM can handle 150+ concurrent requests. Phase A drops from ~2.5 min to ~1.5 min.

3. **Decision GRPO and skill bank GRPO overlap** — GPUs 4-5 update `action_taking` + `skill_selection` while GPUs 6-7 simultaneously update `segment` + `contract` + `curator`. These are independent (different data, different rewards, different adapters). Phase C drops from ~5 min sequential to ~3 min parallel.

**Step-over-step dependency:**

Phase A of step k+1 needs both the updated bank (from Phase B of step k) and updated adapter weights (from Phase C of step k). The per-step time is therefore:

```
per_step = max(Phase_A + Phase_B, Phase_C)
         = max(1.5 + 0.3, 3.0)
         = 3.0 min
```

Phase A+B finishes in ~1.8 min, then waits ~1.2 min for GRPO to complete and hot-reload. Still ~3x faster than single-GPU.

**vLLM launch command (GPUs 0-3):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --tensor-parallel-size 4 \
    --enable-lora \
    --lora-modules \
        segment=runs/lora_adapters/segment \
        contract=runs/lora_adapters/contract \
        curator=runs/lora_adapters/curator \
        skill_selection=runs/lora_adapters/skill_selection \
        action_taking=runs/lora_adapters/action_taking \
    --max-loras 5 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --enable-prefix-caching \
    --trust-remote-code
```

**GRPO training (GPUs 4-7):**

```python
import torch

def train_all_grpo(rollout_results, skillbank_grpo_data):
    """Train decision + skill bank adapters concurrently on GPUs 4-7."""

    # Decision GRPO on GPUs 4-5 (independent of skill bank GRPO)
    decision_process = mp.Process(
        target=train_decision_grpo,
        args=(rollout_results,),
        kwargs={"devices": [4, 5]},
    )

    # Skill bank GRPO on GPUs 6-7 (independent of decision GRPO)
    skillbank_process = mp.Process(
        target=train_skillbank_grpo,
        args=(skillbank_grpo_data,),
        kwargs={"devices": [6, 7]},
    )

    decision_process.start()
    skillbank_process.start()
    decision_process.join()
    skillbank_process.join()

    # Hot-reload all updated adapters into vLLM (GPUs 0-3)
    for adapter in ["action_taking", "skill_selection",
                     "segment", "contract", "curator"]:
        hot_reload_adapter(adapter)
```

**Memory budget (per GPU):**

| Component | Per GPU (TP=4, GPUs 0-3) | Per GPU (GPUs 4-5 or 6-7) |
|-----------|--------------------------|---------------------------|
| Qwen3-14B base (bf16, sharded) | ~7 GB | ~14 GB (replicated) |
| 5 LoRA adapters (rank 16) | ~0.25 GB | ~0.6 GB (subset) |
| KV cache | ~50 GB | — |
| Optimizer states + gradients | — | ~4 GB |
| **Total** | ~57 GB / 80 GB | ~19 GB / 80 GB |

Training GPUs have ample headroom for large batch sizes during GRPO.

**Comparison:**

| | Single GPU | 8 GPUs (4+4) |
|---|---|---|
| Phase A (rollouts) | ~2.5 min | ~1.5 min (TP=4) |
| Phase B (skill bank, visible) | ~0.5 min | ~0.3 min |
| Phase C (GRPO) | ~5 min (blocks vLLM) | ~0 min visible (overlapped) |
| Wait for GRPO | — | ~1.2 min (dependency) |
| **Per step** | **~8.5 min** | **~3 min** |
| **30-step training** | **~4.2 hours** | **~1.5 hours** |
| **50-step training** | **~7 hours** | **~2.5 hours** |

---

## Additional Speed Optimizations

**Constraint:** Do NOT change precision. All inference and training stays at bf16. No quantization (AWQ/GPTQ/bitsandbytes). This preserves model quality.

### Inference — vLLM flags (no code changes)

- [ ] **Prefix caching** — add `--enable-prefix-caching` to vLLM launch. The decision agent reuses the same system prompt (`SYSTEM_PROMPT`, `SKILL_SELECTION_SYSTEM_PROMPT`) across every step of every episode — that's 500-2000 tokens of shared prefix whose KV cache entries are recomputed on every request today. With prefix caching, vLLM stores and reuses them. Skill bank prompts also share per-stage instruction prefixes. Estimated savings: 20-30% reduction in time-to-first-token, zero accuracy impact.

- [ ] **Chunked prefill** — add `--enable-chunked-prefill` to vLLM launch. Allows vLLM to interleave prefilling new requests with decoding existing ones. Without this, a batch of 40 new requests forces all in-flight requests to wait during the prefill phase. With it, prefill is broken into chunks that alternate with decode iterations. Reduces latency variance under high concurrency.

- [ ] **Speculative decoding** — add `--speculative-model Qwen/Qwen3-1.8B --num-speculative-tokens 5`. The draft model proposes tokens that the full 14B model verifies in a single forward pass. Mathematically identical output distribution (rejection sampling guarantees this). For structured outputs like `REASONING: ...\nACTION: 3`, acceptance rates are typically 70-90%. Estimated savings: 1.5-2x faster autoregressive generation. Requires a compatible draft model to be available.

### Inference — output efficiency (minor code changes)

- [ ] **Guided decoding** — add `--guided-decoding-backend outlines` to vLLM and pass `extra_body={"guided_regex": r"REASONING: .{10,200}\nACTION: \d+"}` in the API call. This constrains generation to the expected output format, eliminating malformed outputs that require retries. Particularly useful for skill bank JSON outputs where a missing bracket wastes an entire LLM call.

### Inference — episode-level (medium code changes)

- [ ] **Early episode termination** — in `run_episode_async()`, detect stuck episodes and terminate early. This avoids wasting 200+ LLM calls on episodes where the agent is clearly failing:

```python
STUCK_WINDOW = 15
MIN_STEPS_BEFORE_CHECK = 20

if step > MIN_STEPS_BEFORE_CHECK:
    recent = rewards[-(STUCK_WINDOW):]
    if all(r <= 0 for r in recent):
        break  # agent stuck, stop wasting LLM calls
```

For super_mario (300 max steps), a failing episode drops from 300 steps to ~35 steps — saving ~265 LLM calls per bad episode. With 10 episodes, even 3 bad ones save ~800 calls.

- [ ] **Adaptive episode count** — instead of fixed N episodes per game, allocate more to games with improving reward trends and fewer to plateaued games. Over 30 co-evolution steps, reward trends stabilize for some games early:

```python
def adaptive_episodes(game, base=8, history=reward_history[game]):
    trend = np.polyfit(range(len(history[-5:])), history[-5:], 1)[0]
    if trend > 0.01:
        return base + 2   # improving, collect more
    elif trend < -0.01:
        return max(3, base - 2)  # declining, collect fewer
    return base
```

### Training — GRPO efficiency (no precision changes)

- [ ] **Batched forward/backward** — the `GRPOLoRATrainer` (`skill_agents_grpo/grpo/trainer.py`) currently processes one `(prompt, completion, advantage)` at a time. Batch 8-16 samples into a single forward pass with padding. Same precision, same computation, but ~4-8x fewer GPU kernel launches. This is the single biggest GRPO training speedup.

- [ ] **Gradient accumulation** — accumulate gradients over N micro-batches before calling `optimizer.step()`. This decouples effective batch size from GPU memory without changing precision:

```python
for i, sample in enumerate(batch):
    loss = compute_loss(sample) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

- [ ] **Learning rate schedule** — add warmup + cosine decay. Over 30+ co-evolution steps with 4 epochs each, the adapter sees ~120+ gradient updates at a fixed LR. Warmup prevents early instability; cosine decay prevents overshooting as the adapter converges:

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

- [ ] **Early stopping per GRPO epoch** — instead of fixed `epochs_per_batch` (currently 4 for decision, 2-3 for skill bank), monitor loss and stop when it plateaus. Saves 1-2 unnecessary epochs per adapter per co-evolution step.

### Impact summary (precision-safe optimizations only)

| Optimization | Speed impact | Effort |
|---|---|---|
| Prefix caching | 20-30% faster TTFT | 1 flag |
| Chunked prefill | 10-15% latency reduction | 1 flag |
| Speculative decoding | 1.5-2x generation speed | 1 flag + draft model |
| Guided decoding | Fewer retries | Medium |
| Early episode termination | 30-50% fewer calls on bad eps | Easy |
| Batched GRPO forward/backward | 4-8x GRPO training speed | Medium |
| Gradient accumulation | Better gradient quality | Easy |
| LR schedule | Better convergence | Easy |
