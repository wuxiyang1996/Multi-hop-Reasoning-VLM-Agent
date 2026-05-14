# Balanced Game SFT Data Analysis

## Goal

Select a **balanced** set of high-reward teacher demonstrations across
12 games so that per-game SFT or GRPO training does not collapse toward
any single game or model.

Balance criterion: **equal episode count per game**, selecting the
highest-reward episodes regardless of which teacher model produced them.

---

## Source Data

`SFT_Data/game_sft/` — 712 high-reward episodes from 4 teacher models
across 12 games (8 gymv + 4 env\_wrapper).

### Raw episode counts (before balancing)

| Game | Total | GPT-5.4 | Claude | Gemini | Qwen | avg reward | min | max |
|------|------:|--------:|-------:|-------:|-----:|-----------:|----:|----:|
| Airstriker | 62 | 16 | 16 | 14 | 16 | 80.3 | 20 | 160 |
| AlteredBeast | 59 | 13 | 16 | 14 | 16 | 298.3 | 100 | 1,300 |
| Columns | 60 | 16 | 16 | 13 | 15 | 119.3 | 23 | 223 |
| DynamiteHeaddy | 55 | 15 | 15 | 13 | 12 | 100.0 | 100 | 100 |
| SpaceHarrierII | 64 | 16 | 16 | 16 | 16 | 22,670 | 10,900 | 74,600 |
| StreetsOfRage2 | 61 | 15 | 16 | 15 | 15 | 302.0 | 50 | 650 |
| **Strider** | **24** | **0** | **10** | **14** | **0** | 95.8 | 50 | 150 |
| ThunderForceIII | 61 | 13 | 16 | 16 | 16 | 537.7 | 100 | 1,500 |
| tetris | 67 | 20 | 16 | 15 | 16 | 293.6 | 21 | 960 |
| candy\_crush | 68 | 20 | 16 | 16 | 16 | 431.4 | 262 | 672 |
| super\_mario | 66 | 20 | 14 | 16 | 16 | 587.0 | 283 | 1,946 |
| 2048 | 65 | 20 | 15 | 14 | 16 | 917.8 | 4 | 2,312 |
| **Total** | **712** | 184 | 182 | 176 | 170 | | | |

**Imbalance:** per-game counts range 24–68 (2.8× ratio). Training on raw
data would over-represent env\_wrapper games and under-represent Strider.

---

## Balancing Strategy

1. **K = 55** — the minimum episode count among the 11 non-Strider games
   (DynamiteHeaddy has exactly 55).
2. For each of the 11 games, keep the **top-55 episodes by reward**
   (best demonstrations from any teacher model).
3. **Strider** is the exception: only Claude (10) and Gemini (14) have
   non-zero reward data (GPT and Qwen scored 0 on all episodes). Keep
   all 24 episodes.
4. Total: **55 × 11 + 24 = 629 episodes**.

This drops 83 low-reward episodes (712 → 629, −12%) while equalizing
the per-game contribution to prevent distribution shift.

---

## Balanced Dataset — `SFT_Data/game_sft_balanced/`

### Per-game breakdown

| Game | Kept | GPT-5.4 | Claude | Gemini | Qwen | avg reward |
|------|-----:|--------:|-------:|-------:|-----:|-----------:|
| Airstriker | 55 | 14 | 16 | 13 | 12 | 86.9 |
| AlteredBeast | 55 | 11 | 16 | 13 | 15 | 312.7 |
| Columns | 55 | 16 | 12 | 13 | 14 | 126.2 |
| DynamiteHeaddy | 55 | 15 | 15 | 13 | 12 | 100.0 |
| SpaceHarrierII | 55 | 16 | 13 | 11 | 15 | 24,134.5 |
| StreetsOfRage2 | 55 | 14 | 15 | 14 | 12 | 323.8 |
| **Strider** | **24** | **—** | **10** | **14** | **—** | **95.8** |
| ThunderForceIII | 55 | 13 | 10 | 16 | 16 | 585.5 |
| tetris | 55 | 19 | 15 | 14 | 7 | 348.3 |
| candy\_crush | 55 | 20 | 14 | 5 | 16 | 463.2 |
| super\_mario | 55 | 20 | 14 | 15 | 6 | 645.8 |
| 2048 | 55 | 20 | 9 | 10 | 16 | 1,081.7 |
| **Total** | **629** | **178** | **159** | **151** | **141** | — |

### Model source distribution (natural, not forced)

| Model | Episodes | Share |
|-------|--------:|------:|
| GPT-5.4 | 178 | 28.3% |
| Claude | 159 | 25.3% |
| Gemini | 151 | 24.0% |
| Qwen | 141 | 22.4% |

Model contribution is naturally balanced (max/min ratio = 1.26×) because
top-reward filtering draws roughly evenly from all four teachers.

### Supporting data (unchanged from `game_sft/`)

- **Decision SFT:** 22,086 `action_taking` + 21,086 `skill_selection` rows
- **Skill banks:** 328 skills across 12 games

---

## Known Issues

1. **Strider under-represented:** only 24 episodes vs 55 for other
   games. GPT-5.4 and Qwen scored 0 reward on all Strider runs. This
   game will receive less SFT weight. Mitigation: up-weight Strider in
   the SFT loss, or rely on skill transfer from Phase 1.

2. **SpaceHarrierII reward scale:** avg reward ~24k vs ~100–1000 for
   other games. Reward normalization is required before mixing into a
   single SFT objective. Per-game z-score normalization is recommended.

3. **DynamiteHeaddy constant reward:** all episodes have reward = 100
   (no variance). Top-K selection is effectively random for this game.

---

## Directory Structure

```
SFT_Data/game_sft_balanced/
├── MANIFEST.json
├── gymv/
│   ├── gpt54/     ─┐
│   ├── claude/     │  each contains Temporal_{game}-v0/episode_*.json
│   ├── gemini/     │
│   └── qwen/      ─┘
├── env_wrapper/
│   ├── gpt54/     ─┐
│   ├── claude/     │  each contains {tetris,candy_crush,...}/episode_*.json
│   ├── gemini/     │
│   └── qwen/      ─┘
├── decision_sft/
│   └── {game}/action_taking.jsonl + skill_selection.jsonl
└── skill_banks/
    └── {game}/skill_bank.jsonl
```

---

## Relationship to Training Plans

- **`PLAN_GAME_SPLIT_AND_NO_SFT_GRPO.md`** — the balanced SFT data is
  used for Phase 1 source-game training and optionally for Phase 2
  warm-start (the no-SFT ablation skips it for Phase 2).
- **`PIPELINE_GUIDE.md` §5** — SFT data generation. This analysis
  post-processes the pipeline output to equalize game representation.
- **`PLAN_PERGAME_ACTION_LORA.md`** — per-game LoRA adapters trained on
  this balanced data avoid marginal-frequency collapse.
