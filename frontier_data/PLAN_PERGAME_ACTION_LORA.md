# Per-Game action_taking LoRA — Policy Collapse Fix

## Problem

gymv games share a **fixed 12-button action space** every step:
`['B','A','MODE','START','UP','DOWN','LEFT','RIGHT','C','Y','X','Z']`

But each game's optimal action distribution is radically different:

| Game | Top-1 | Top-1% | Top-3% | KL from avg |
|------|-------|--------|--------|-------------|
| Strider | RIGHT | 70.1% | 94.6% | 0.89 |
| DynamiteHeaddy | RIGHT | 44.8% | 74.3% | 0.57 |
| Columns | DOWN | 42.0% | 77.0% | 1.16 |
| SpaceHarrierII | B | 41.8% | 85.4% | 0.68 |
| StreetsOfRage2 | RIGHT | 37.8% | 74.9% | 0.53 |
| AlteredBeast | RIGHT | 35.3% | 77.9% | 0.61 |
| Airstriker | B | 28.4% | 60.8% | 0.71 |
| ThunderForceIII | B | 25.1% | 71.1% | 0.72 |

A shared SFT model averages these into `RIGHT=34% / B=23%` — wrong for
every game. The model learns a **marginal frequency** (always output the
most common action) instead of a **conditional policy** (choose action
based on game state). This causes policy collapse.

### Why more/better data doesn't fix it

- Switching from single-teacher (GPT-5.4) to multi-teacher (GPT + Claude
  + Gemini + Qwen3-VL) high-reward data does not change the fundamental
  distribution skew. Strider is still 70% RIGHT regardless of teacher.
- The action list is identical every step, so the model has no structural
  signal to differentiate "when to go RIGHT" vs "when to attack" — only
  the schema text (game state) varies, buried deep in the prompt.

---

## Solution: Per-Game action_taking LoRA

Train one `action_taking` LoRA adapter per game. Only `action_taking` is
per-game; the other 4 adapters remain shared:

| Adapter | Mode | Rationale |
|---------|------|-----------|
| **action_taking** | **per-game** | Action distributions differ radically across games |
| skill_selection | shared | Candidate skill lists change every step (no fixed-set problem) |
| segment | shared | Skill ranking is game-agnostic |
| contract | shared | Effect summarization is game-agnostic |
| curator | shared | Bank maintenance is game-agnostic |

### Disk layout

```
adapter_dir/
├── decision/
│   ├── skill_selection/                    ← shared
│   └── action_taking/
│       ├── Temporal_Airstriker-v0/         ← per-game LoRA
│       ├── Temporal_AlteredBeast-v0/
│       ├── Temporal_Columns-v0/
│       ├── Temporal_DynamiteHeaddy-v0/
│       ├── Temporal_SpaceHarrierII-v0/
│       ├── Temporal_StreetsOfRage2-v0/
│       ├── Temporal_Strider-v0/
│       └── Temporal_ThunderForceIII-v0/
└── skillbank/                              ← all shared
    ├── segment/
    ├── contract/
    └── curator/
```

### Naming convention

vLLM adapter name = `action_taking__{game}` (double underscore separator).

Example: `action_taking__Temporal_Strider-v0`

Fallback: games without a per-game LoRA (e.g. new games, env_wrapper)
fall back to base model (existing adapter-not-found logic in
`vllm_client.py` handles this).

---

## Files to modify

### SFT training side (2 files)

**1. `trainer/SFT/config.py`**
- Add `per_game_adapters: List[str] = ["action_taking"]`
- `adapter_output_path()`: for per-game adapters, return
  `<output>/decision/action_taking/<game>/`

**2. `trainer/SFT/train.py`**
- When `action_taking in config.per_game_adapters`:
  - Loop over `config.games`, load per-game data, train independent LoRA
  - Each game gets ~3200 rows, ~2 epochs, a few minutes on H200
- Other adapters: unchanged shared training path

### Inference side (4 files)

**3. `trainer/coevolution/config.py`**
- `adapter_path("action_taking", game=game)` returns per-game path
- `init_or_load_adapters()` iterates games for action_taking init
- Add `per_game_adapter_names()` helper

**4. `trainer/coevolution/vllm_client.py`**
- `ADAPTER_MAP`: accept any `action_taking__*` pattern dynamically
  (currently static dict with 5 entries)

**5. `trainer/coevolution/episode_runner.py`**
- Line ~1612: `adapter="action_taking"` → `adapter=f"action_taking__{game}"`
- Line ~1743: `GRPORecord(adapter=...)` uses same per-game name
- `skill_selection` calls unchanged

**6. `trainer/coevolution/vllm_server.py`**
- `reload_adapters()` / `_build_lora_args()`: scan
  `decision/action_taking/*/` subdirectories
- Register each as `action_taking__<game>`

### GRPO + checkpoint side (2 files)

**7. `trainer/coevolution/grpo_training.py`**
- `GRPORecord.adapter` is already a string → per-game names route
  naturally
- LoRA update logic groups records by adapter name, updates only the
  matching LoRA weights

**8. `trainer/coevolution/checkpoint.py`**
- Expand `ADAPTER_NAMES` to include per-game names
- Save/restore iterates `action_taking/*/` subdirectories

---

## Resource impact

| Metric | Current (shared) | Per-game (8 games) |
|--------|------------------|--------------------|
| action_taking LoRA count | 1 | 8 |
| Total LoRA adapters | 5 | 12 |
| GPU memory (vLLM) | ~250 MB | ~600 MB |
| Memory delta | — | +350 MB (<0.3% of H200) |
| SFT training time | 1 run, ~31K rows | 8 runs, ~3.2K rows each |
| GRPO per-step cost | unchanged | unchanged (same episode count) |

vLLM batch-internal LoRA switching is zero-cost (adapter mask indexing).
Different games don't batch together anyway (synchronizer groups by game).

---

## Multi-teacher SFT data

The per-game LoRA training uses multi-teacher high-reward SFT data
generated by `frontier_data/scripts/build_multiteacher_sft.py`:

| Source | Teachers | Filter | Output |
|--------|----------|--------|--------|
| skip8 runs (80 steps, frame_skip=8) | GPT-5.4, Claude-4.6, Gemini-3.1, Qwen3-VL | reward > 0 | 8 games, 320 eps, 25,316 rows |
| Legacy (100 steps, frame_skip=1) | GPT-5.4 | reward > 0 | env_wrapper only (gymv all zero) |

Per-game episode selection uses round-robin across teachers (max 40
episodes/game) to maximize teacher diversity.

Data location: `frontier_data/output/decision_sft_multiteacher/<game>/action_taking.jsonl`

### Per-game data sizes

| Game | Episodes | Rows | Teachers |
|------|----------|------|----------|
| Airstriker | 40 | 3,200 | gemini:13 / claude:13 / gpt54:13 / qwen:1 |
| AlteredBeast | 40 | 3,200 | gemini:12 / claude:12 / gpt54:12 / qwen:4 |
| Columns | 40 | 2,916 | qwen:10 / gemini:10 / gpt54:10 / claude:10 |
| DynamiteHeaddy | 40 | 3,200 | claude:14 / gemini:13 / gpt54:13 |
| SpaceHarrierII | 40 | 3,200 | claude:10 / gpt54:10 / gemini:10 / qwen:10 |
| StreetsOfRage2 | 40 | 3,200 | gemini:10 / gpt54:10 / claude:10 / qwen:10 |
| Strider | 40 | 3,200 | qwen:16 / gemini:14 / claude:10 |
| ThunderForceIII | 40 | 3,200 | gemini:13 / gpt54:13 / claude:14 |

### Zero-data games (no high-reward episodes from any teacher)

CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe, KidChameleon,
MortalKombatII — excluded from per-game LoRA. These need skip8 rollouts
to be collected first.

---

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Per-game LoRA overfits on ~3.2K rows | LoRA r=16 has ~3M params; 3.2K rows is sufficient for low-rank fine-tuning |
| GRPO updates too noisy (8-16 eps/game/step) | Cross-step ReplayBuffer already implemented; can lower per-game LR |
| New game has no per-game LoRA | Fallback to base model (adapter-not-found logic in vllm_client.py) |
| env_wrapper games have different action spaces | Keep shared fallback for non-gymv; per-game only for gymv initially |

## Key insight

SFT alone cannot teach optimal game policies — gymv action selection is
fundamentally an RL problem. Per-game LoRA fixes the **cold-start
distribution mismatch** (shared model averages conflicting signals), but
the real policy learning happens during GRPO. The goal of per-game SFT
is a better warm-start, not a solved policy.
