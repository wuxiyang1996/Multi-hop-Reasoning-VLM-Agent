# Stage 1 Results

Best co-evolution runs per game, with peak reward and corresponding step.

| Game | Run Path | Best Step | Reward |
|------|----------|-----------|--------|
| Candy Crush | `runs/candy_crush_coevo_v4_20260519_093912` | 6 | 620.0 |
| Columns | `runs/gymv_columns_coevo_v4_20260519_001840` | 3 | 155.98 |
| Strider | `runs/gymv_strider_coevo_v5_20260519_184613` | 9 | 79.17 |
| ThunderForce III | `runs/gymv_thunder_force_iii_coevo_v9_grpo_unclip` | 3 | 800.0 |
| StreetsOfRage2 | `runs/gymv_streets_of_rage_2_coevo_v5_20260520_010806` | 4 | 360.83 |

---

## Stage 2 — LLM-Judge Mega-Skill + 1-shot ICL Transfer (v3, game-to-game)

Scope: **game-to-game transfer only**.  Source: 5 best Stage-1 GRPO banks
(82 skills).  Targets: 6 Phase-2 holdout games.

### Pipeline

```
5 best GRPO banks (82 skills)
  → scripts/judge_plan_level_similarity.py    [content-based shortlist + gpt-5-mini]
  → frontier_data/output/plan_level_similarity_judgments.json   (1230 directed edges, 173 score=5)

  → scripts/build_megaskills_from_judge.py    [2-stage: mutual ≥5 core + ≥4 satellite attach]
  → frontier_data/output/megaskills_2stage/mega_skills.jsonl   (10 mega-skills, all multi-task)

  → scripts/stage2_seeds_from_megaskills.py   [genre-affinity ranking + 1-shot ICL]
  → frontier_data/output/stage2_seeds_v3/<target>/skill_bank.jsonl   (10 seeds × 6 targets)
```

### Why the LLM judge

The judge scores similarity from the **actual NL reasoning steps**, not
from the canonical-intent label sequence.  Shortlist is also
content-based (bag-of-stem cosine over rep.protocol_steps + name +
description) so the LLM gets candidates that *look similar in procedure*,
not just in label.  Cleanup strips `"Valid actions: …"` prompt-template
leaks from source protocol steps so the seed templates aren't
contaminated with the actor's action menu.

### Mega-skill library (10, all cross-game)

| mega_skill_id                     | n_members | n_tasks | template_signature |
|-----------------------------------|----------:|--------:|--------------------|
| `mega.000.explore`                |        36 |       5 | ACT → VERIFY |
| `mega.001.late_recover_survive`   |        13 |       5 | EVALUATE → ACT → PERCEIVE → ACT |
| `mega.002.recover_survive`        |         7 |       3 | ACT → VERIFY → EVALUATE |
| `mega.003.recover_reshuffle`      |         4 |       2 | EVALUATE → ACT → PERCEIVE |
| `mega.004.inspect_setup`          |         4 |       4 | EVALUATE → ACT → PERCEIVE → EVALUATE → ACT |
| `mega.005.early_recover_survive`  |         3 |       2 | EVALUATE → ACT → PERCEIVE → ACT |
| `mega.006.commit_explore`         |         3 |       3 | EVALUATE → ACT → PERCEIVE → ACT |
| `mega.007.mid_execute`            |         3 |       3 | EVALUATE → ACT → PERCEIVE → EVALUATE |
| `mega.008.explore.f63b`           |         3 |       2 | EVALUATE → VERIFY |
| `mega.009.commit_evade`           |         3 |       2 | ACT → VERIFY → ACT |

### Phase-2 seed bank summary

| target | genre | seeds | w/ICL | within-genre |
|---|---|---:|---:|---:|
| `gymv_space_harrier_ii` | shmup    | 10 | 10 | 8 |
| `gymv_airstriker`       | shmup    | 10 | 10 | 8 |
| `gymv_altered_beast`    | beatemup | 10 | 10 | 7 |
| `gymv_dynamite_headdy`  | platform | 10 | 10 | 6 |
| `twenty_forty_eight`    | puzzle   | 10 | 10 | 7 |
| `super_mario`           | platform | 10 | 10 | 6 |

Each seed entry has `{ name, template_signature, protocol{preconditions,
steps, step_checks}, contract.description, exemplars[0]{source, kind,
reasoning_steps}, tags, provenance.{source_mega_skill, ranking_score} }`.
Ranking uses genre affinity weighted by per-binding-task similarity to
the target.

(For cross-domain transfer to WEB / VR see `frontier_data/PLAN_FEW_SHOT_SKILL_BANK.md`
§0a — separate teacher-bootstrap step needed; not covered here.)

---

## Stage 2 — Earlier attempt (deprecated)

The earlier attempt used `build_reasoning_aligned_bank.py` (canonical-intent
signature clustering) and `stage2_mega_skill_plus_icl.py`.  That pipeline:

- clustered by intent-label sequence (PERCEIVE/EVALUATE/ACT/etc.) which
  produced 59 plans but the resulting templates were uninformative
  (every seed got `PERCEIVE → DECIDE → COMMIT`), and
- included WEB/VR targets that lacked within-cohort ICL exemplars (the
  "phase 0a pending" gap below).
- was based on the old `frontier_data/output/per_task_banks/` (GPT-extracted)
  not the GRPO-validated banks.

Output preserved at `frontier_data/output/stage2_seed_v2/` for comparison.

### Method

1. **Mega-skill clustering** (`build_reasoning_aligned_bank.py`):
   normalize every protocol step to a canonical reasoning intent
   (PERCEIVE, RECALL, EVALUATE, DECIDE, NAVIGATE, ACT, VERIFY), extract a
   compressed plan signature, cluster by shared plan across domains.
   324 skills → 59 reasoning plans → 10 real cross-domain bridges.

2. **Seed generation** (`scripts/stage2_mega_skill_plus_icl.py`):
   For each target task T, pick top-K mega-skill templates relevant to T's
   domain, then build a seed bank entry containing
   `{template_signature, protocol.steps, step_checks, exemplar}` where
   the exemplar is a 1-shot ICL trace pulled from a within-cohort source
   skill's `protocol_raw.steps` (extracted by GPT-5.4 during Stage 1).

### Mega-Skill Library (10 cross-domain bridges)

10 reasoning plans appear in ≥ 2 of {GAME, WEB, VR}:

```
ACT → DECIDE → ACT → VERIFY    27 skills (GAME=11, WEB=5)
ACT → VERIFY                   13 skills (GAME=5, WEB=2, VR=1)
ACT → DECIDE → ACT             10 skills (GAME=4, WEB=5, VR=1)
ACT → DECIDE → VERIFY           6 skills (GAME=1, WEB=1, VR=3)
PERCEIVE → ACT → VERIFY         6 skills (GAME=2, WEB=2)
ACT → PERCEIVE → DECIDE         5 skills (GAME=1, WEB=3, VR=1)
ACT → PERCEIVE → ACT → VERIFY   3 skills (GAME=1, WEB=1)
EVALUATE → ACT → DECIDE         2 skills (GAME=1, WEB=1)
ACT → DECIDE → EVALUATE         2 skills (GAME=1, VR=1)
ACT → EVALUATE → ACT            2 skills (GAME=1, VR=1)
```

### Seed Bank Bundles (template + 1-shot ICL)

Each seed entry contains:
- `template_signature`     — abstract reasoning plan (e.g. PERCEIVE → DECIDE → ACT → VERIFY)
- `protocol.steps`         — NL rendering of each canonical intent
- `protocol.step_checks`   — per-step predicates (e.g. `entity_grounded=true`)
- `exemplars[0]`           — 1-shot ICL trace (reasoning steps from a within-cohort source skill's `protocol_raw`)
- `tags`                   — provenance (`exemplar_from:<task>`, `mega_domains:GAME+VR+WEB`, …)

ICL exemplar source policy (per PLAN_FEW_SHOT_SKILL_BANK.md §"What non-game tasks contribute to each other"):

| Target domain | ICL source priority | Status |
|---------------|--------------------|--------|
| GAME          | same-genre Phase 1 game | ✅ within-cohort |
| VR            | siv_bench/tir_bench/video_holmes `protocol_raw` | ✅ within-cohort |
| WEB           | miniwob/webshop `protocol_raw` is empty → fallback to VR or Phase 0a teacher demos | ⚠ Phase 0a pending |

### Phase 2 Seed Banks

| Target | Domain | Seeds | w/ICL | Within-cohort ICL | Top exemplar sources |
|--------|--------|-------|-------|------------------|----------------------|
| gymv_space_harrier_ii | GAME | 12 | 8/12 | 8/8 ✅ | Airstriker, ThunderForceIII |
| gymv_airstriker       | GAME | 12 | 8/12 | 8/8 ✅ | Airstriker, ThunderForceIII |
| gymv_altered_beast    | GAME | 12 | 8/12 | 8/8 ✅ | AlteredBeast, StreetsOfRage2 |
| gymv_dynamite_headdy  | GAME | 12 | 8/12 | 8/8 ✅ | DynamiteHeaddy, Strider |
| twenty_forty_eight    | GAME | 12 | 8/12 | 8/8 ✅ | Columns, tetris |
| super_mario           | GAME | 12 | 8/12 | 8/8 ✅ | DynamiteHeaddy, Strider |
| vr_new_bench          | VR   | 12 | 6/12 | 6/6 ✅ | tir_bench, siv_bench, video_holmes |
| webshop_new           | WEB  | 12 | 6/12 | 0/6 ⚠ | VR fallback — Phase 0a teacher demos required |
| miniwob_unseen        | WEB  | 12 | 6/12 | 0/6 ⚠ | VR fallback — Phase 0a teacher demos required |

Scripts:
- `frontier_data/scripts/build_reasoning_aligned_bank.py` — mega-skill clustering
- `scripts/stage2_mega_skill_plus_icl.py` — seed = template + 1-shot ICL bundle

Output: `frontier_data/output/stage2_seed_v2/<task>/skill_bank.jsonl`

### Next Step: Phase 0a (Teacher Demos for WEB)

Per PLAN_FEW_SHOT_SKILL_BANK.md §0a, WEB targets need teacher-bootstrapped
ICL exemplars (one-time GPT-5.4 / Gemini call on 200 train samples per task).
The current within-cohort ICL gap (0/6 for WEB) is the documented gap that
Phase 0a fills.
