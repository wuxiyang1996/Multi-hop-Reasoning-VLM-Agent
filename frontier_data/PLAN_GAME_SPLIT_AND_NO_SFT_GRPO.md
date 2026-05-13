# Game Split & No-SFT GRPO Training Plan

## Overview

Two questions addressed:
1. How to split 12 games into source (skill mining) vs target (skill transfer)?
2. Can we skip SFT entirely and train with only skill transfer + GRPO from raw Qwen?

---

## 1. Game Split: Source vs Target

### Current Phase 1/2 Split (recommended — keep as-is)

Defined in `trainer/coevolution/config.py` (`PHASE1_DEFAULT_GAMES` / `PHASE2_HOLDOUT_GAMES`).

| Phase | Role | Games | Genres covered |
|-------|------|-------|----------------|
| **Phase 1** (source) | Mine concrete skills, populate per-task banks | ThunderForceIII, AlteredBeast, Columns, DynamiteHeaddy, candy_crush, tetris | shooter, brawler, platformer, puzzle |
| **Phase 2** (target) | Transfer mega-skill skeletons to held-out games | SpaceHarrierII, StreetsOfRage2, Airstriker, Strider, twenty_forty_eight, super_mario | shooter, brawler, platformer, puzzle |

### Why this split works

**Genre coverage is complete.** Every Phase 2 target genre has a matching
Phase 1 source:

| Target game (Phase 2) | Genre | Source game (Phase 1) | Shared mega-skills |
|------------------------|-------|-----------------------|-------------------|
| SpaceHarrierII | shooter | ThunderForceIII | 7 |
| Airstriker | shooter | ThunderForceIII | 4 |
| StreetsOfRage2 | brawler | AlteredBeast | 5 |
| Strider | platformer | DynamiteHeaddy | 7 |
| super_mario | platformer | DynamiteHeaddy | - |
| twenty_forty_eight | puzzle | Columns / tetris / candy_crush | 4 |

**Mega-skill transfer paths confirmed.** The shared skill bank contains
354 mega-skills, of which 14 span 2+ games. The strongest transfer
bridges are DynamiteHeaddy-Strider (7 shared) and AlteredBeast-Strider
(7 shared), both crossing the Phase 1/2 boundary.

### Potential improvements (not blocking)

- Phase 1 is puzzle-heavy (3/6 are puzzle games). Consider swapping
  candy_crush to Phase 2 and moving Airstriker to Phase 1 for better
  balance (shooter:2, brawler:1, platformer:1, puzzle:2 in each phase).
- 5 zero-data games (CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe,
  KidChameleon, MortalKombatII) are excluded entirely. The fighter genre
  (MortalKombatII) has no representation in either phase. These games
  need skip8 rollouts to be collected before they can participate.

---

## 2. No-SFT Training: Raw Qwen + Skill Transfer + GRPO

### What SFT currently teaches

SFT cold-start trains the model on 3 things:

1. **Output format** — `SUBGOAL: [TAG] ...\nREASONING: ...\nACTION: N`
2. **Action semantics** — which button number corresponds to which action
3. **Base strategy** — rough move/attack/dodge patterns

### Why SFT might not be necessary for Phase 2

For Phase 2 (transfer) games, the skill bank already provides richer
guidance than SFT ever did. At runtime, the episode runner injects
structured skill protocols into the prompt:

```
--- Active Skill: dodge_and_engage ---
  Strategy: Move to avoid enemy attacks, then counter
  Plan (4 steps):
  >> 1. Observe enemy position and attack pattern
     2. Move to safe position
     3. Time counter-attack when enemy is vulnerable
     4. Execute attack and verify hit
  Done when: enemy defeated
--- end skill ---
```

This structured guidance tells the model WHAT to do (reasoning framework).
GRPO then teaches HOW to execute (concrete action selection through
trial-and-error reward). Meanwhile, SFT's action_taking data has a
critical flaw: the REASONING field is always "Expert play." (empty) and
the action distribution teaches marginal frequency rather than
conditional policy (see `PLAN_PERGAME_ACTION_LORA.md`).

### Architecture already supports no-SFT training

`config.py` has `start_mode="from_scratch"` which random-initializes all
LoRA adapters. GRPO from scratch is a designed code path:

```python
# trainer/coevolution/config.py
start_mode: str = "auto"  # "from_scratch" | "resume" | "auto"
pretrained_adapter_paths: Dict[str, str] = field(default_factory=dict)
```

The action parser in `episode_runner.py` already falls back to random
action on parse failure, providing natural exploration when the model
hasn't yet learned the output format.

### How raw Qwen handles the cold-start

| Concern | How it's handled |
|---------|-----------------|
| Output format unknown | GRPO reward includes format bonus; Qwen's instruction-following ability picks up the format within ~3-5 steps |
| Action semantics | Schema prompt lists `Available actions: 1. B  2. A ...` — Qwen can read this |
| No base strategy | Skill bank protocol provides the strategy; GRPO rewards correct execution |
| Early episodes are random | Expected — this IS exploration. Replay buffer smooths early noise |

### Proposed experiment design

| Experiment | Training | Eval on | Tests |
|------------|----------|---------|-------|
| **Baseline** | Per-game SFT + GRPO + skill bank | Phase 2 games | Standard pipeline |
| **No-SFT-A** | Raw Qwen + GRPO + skill bank (no SFT) | Phase 2 games | Can skill bank replace SFT warm-start? |
| **No-SFT-B** | Raw Qwen + GRPO + no skill bank (no SFT) | Phase 2 games | GRPO-only baseline |
| **SFT-only** | Per-game SFT, no GRPO | Phase 2 games | SFT upper bound without RL |

**Expected results:**
- No-SFT-A: slower start (format learning overhead ~3-5 steps), but may
  match or exceed Baseline by step 10+ (no marginal frequency bias from
  SFT)
- No-SFT-B: significantly worse (random exploration without skill
  guidance)
- If No-SFT-A approaches Baseline, this proves **skill bank is the real
  contribution**, and SFT is merely a format warm-start

This corresponds to ablations A1 and A5 in `PIPELINE_GUIDE.md`.

### Launch command (no code changes needed)

```bash
# No-SFT-A: raw Qwen + GRPO + transferred skill bank
python -m trainer.coevolution.orchestrator \
  --start_mode from_scratch \
  --games gymv_space_harrier_ii gymv_streets_of_rage_2 \
         gymv_airstriker gymv_strider \
  --seed_bank_dir frontier_data/output/per_task_banks \
  --total_steps 60

# No-SFT-B: raw Qwen + GRPO, no skill bank
python -m trainer.coevolution.orchestrator \
  --start_mode from_scratch \
  --games gymv_space_harrier_ii gymv_streets_of_rage_2 \
         gymv_airstriker gymv_strider \
  --total_steps 60
```

---

## 3. Recommended Path

1. **Phase 1** — normal training: per-game SFT + GRPO on 6 source games.
   Mine concrete skills, build shared skill bank.

2. **Phase 2** — run two parallel experiments on 6 target games:
   - **With SFT**: per-game SFT warm-start then GRPO + transferred skills
   - **No SFT**: raw Qwen `from_scratch` then GRPO + transferred skills
     (same skill bank)

3. **Compare** Phase 2 reward curves. If no-SFT matches SFT within 10
   GRPO steps, the skill bank fully substitutes for SFT warm-start. This
   is the strongest evidence that structured skill transfer (not teacher
   distillation) is the core contribution.

---

## 4. Relationship to Other Plans

- **`PLAN_PERGAME_ACTION_LORA.md`** — per-game LoRA fixes SFT's marginal
  frequency problem. If we skip SFT entirely for Phase 2, per-game LoRA
  is only needed for Phase 1 source games.
- **`PIPELINE_GUIDE.md` §7** — the Phase 1/2 split defined there is
  unchanged. The no-SFT experiment is a new ablation row (A5 variant).
- **`PLAN_FEW_SHOT_SKILL_BANK.md`** — Phase 3 (non-game tasks) already
  uses self-rollout ICL exemplars instead of SFT. The no-SFT approach
  for Phase 2 games aligns with Phase 3's philosophy.
