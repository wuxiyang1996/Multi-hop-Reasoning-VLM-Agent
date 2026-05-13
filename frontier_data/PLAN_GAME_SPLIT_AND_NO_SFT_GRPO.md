# Game Split & No-SFT GRPO Training Plan

## Overview

Two questions addressed:
1. How to split 12 games into source (skill mining) vs target (skill transfer)?
2. Can we skip SFT entirely and train with only skill transfer + GRPO from raw Qwen?

---

## 1. Game Split: Source vs Target

### Optimized Split (1 game per genre in Phase 1)

Defined in `trainer/coevolution/config.py` (`PHASE1_DEFAULT_GAMES` /
`PHASE2_HOLDOUT_GAMES`).

The split was computed by exhaustive search over all valid 1-per-genre
assignments, maximizing the number of cross-phase mega-skill transfer
links. The optimal split scores 50 cross-phase transfer links (vs 43
for the previous split, +16%).

| Phase | Game | Genre | Skills | Role |
|-------|------|-------|--------|------|
| **Phase 1** | ThunderForceIII | shooter | 32 | Source: mine shooter skills |
| | StreetsOfRage2 | brawler | 34 | Source: mine brawler skills |
| | Strider | platformer | 45 | Source: mine platformer skills (richest bank) |
| | Columns | puzzle (spatial) | 27 | Source: mine spatial puzzle skills |
| | tetris | puzzle (spatial) | - | Env_wrapper: block placement like Columns |
| | candy_crush | puzzle (match) | - | Env_wrapper: pattern-match optimization |
| **Phase 2** | SpaceHarrierII | shooter | 33 | Target: receive shooter skills from TF3 |
| | AlteredBeast | brawler | 21 | Target: receive brawler skills from SoR2 |
| | DynamiteHeaddy | platformer | 32 | Target: receive platformer skills from Strider |
| | Airstriker | shooter | 31 | Target: receive shooter skills from TF3 |
| | twenty_forty_eight | puzzle (strategy) | - | Env_wrapper: receives puzzle skills from Columns |
| | super_mario | platformer | - | Env_wrapper: receives platformer skills from Strider |

### Genre taxonomy

The 12 games span 4 primary genres. The 3 puzzle env_wrapper games are
sub-classified by gameplay mechanics:

| Genre | Sub-genre | Games | Defining mechanic |
|-------|-----------|-------|-------------------|
| shooter | - | ThunderForceIII, SpaceHarrierII, Airstriker | Aim + fire + dodge projectiles |
| brawler | - | StreetsOfRage2, AlteredBeast | Melee combat + movement |
| platformer | - | Strider, DynamiteHeaddy, super_mario | Jump + navigate + avoid obstacles |
| puzzle | spatial | Columns, tetris | Block placement + row/line clearing |
| puzzle | match | candy_crush | Pattern recognition + swap optimization |
| puzzle | strategy | twenty_forty_eight | Lookahead + merge planning |

### Cross-phase transfer bridges (9 links, 50 mega-skills total)

| Source (Phase 1) | Target (Phase 2) | Shared mega-skills |
|------------------|-------------------|--------------------|
| Strider | DynamiteHeaddy | 7 |
| Strider | AlteredBeast | 7 |
| Strider | SpaceHarrierII | 7 |
| ThunderForceIII | SpaceHarrierII | 7 |
| StreetsOfRage2 | AlteredBeast | 5 |
| ThunderForceIII | DynamiteHeaddy | 5 |
| StreetsOfRage2 | DynamiteHeaddy | 4 |
| Strider | Airstriker | 4 |
| Columns | DynamiteHeaddy | 4 |

Strider is the strongest transfer hub — it shares 7 mega-skills with
3 different Phase 2 games across 3 genres (platformer, brawler, shooter).

### Why this split is optimal

1. **Maximized cross-phase transfer:** 50 mega-skill links vs 43 in the
   previous split (+16%). Every Phase 2 game has at least one strong
   source in Phase 1.
2. **Phase 1 has the richest skill banks:** Strider (45 skills),
   StreetsOfRage2 (34), ThunderForceIII (32), Columns (27) = 138 total.
   More skills mined in Phase 1 means more to transfer in Phase 2.
3. **Genre-balanced:** exactly 1 game per primary genre in Phase 1.
   Phase 2 has 2 shooters, 1 brawler, 1 platformer — the extra shooter
   provides a same-genre difficulty gradient (Airstriker is easier than
   SpaceHarrierII).
4. **Env_wrapper placement is principled:** tetris and candy_crush go
   to Phase 1 (source puzzle varieties), twenty_forty_eight and
   super_mario go to Phase 2 (transfer targets matching Columns and
   Strider respectively).

### Excluded games (no data)

5 gymv games have zero reward across all 4 teachers and no skill banks:
CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe, KidChameleon,
MortalKombatII. The fighter genre (MortalKombatII) has no representation.
These need skip8 rollouts before they can participate.

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
