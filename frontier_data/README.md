# Frontier Data — SFT Inventory & Shared Skill Bank Pipeline

Full skill extraction, shared bank construction, and cross-task binding
pipeline built on top of the `emnlp2026_download/workspace/` archive.

**4 frontier teachers:** GPT-5.4, Claude Sonnet 4.5, Gemini 3.1 Pro,
Qwen3-VL-235B-A22B.

> **Excluded from SFT:** assistantbench, osworld (per-episode traces),
> and 5 zero-reward gymv games (CastleOfIllusion, CastlevaniaBloodlines,
> GoldenAxe, KidChameleon, MortalKombatII — 0 reward across all 4 teachers).

---

## Contribution Framing & Data Leakage Analysis

### What this pipeline IS vs IS NOT

| | Description |
|---|---|
| **Infrastructure** (this pipeline) | Frontier teachers extract skills, GPT-5.4 lifts to Layer-C, GPT-5.4 re-grounds to new domains. All "transfer" here is performed by frontier models that already know how to solve both source and target tasks. |
| **Contribution** (training experiments) | A **9B student model**, through structured skill banks, reuses reasoning patterns discovered in one domain to adapt faster in a new domain — something it **cannot do** from raw SFT alone. |

### Data leakage concern

The frontier teacher (GPT-5.4) extracts skills, lifts them to
modality-agnostic templates, and re-grounds them to new domains. The
"9 cross-domain reasoning plans covering 221 skills" is a property of
**GPT-5.4's knowledge**, not of the skill bank architecture. A reviewer
would correctly ask: "Is this skill transfer, or just frontier model
distillation with extra steps?"

### What constitutes valid evidence

The contribution is **not** "frontier models can transfer reasoning"
(trivial). The contribution is: **structured skill banks enable a small
model to transfer reasoning across domains better than unstructured
alternatives.** This requires three ablation comparisons:

| Comparison | What it tests | Expected evidence |
|---|---|---|
| **A1: 9B + seed bank vs 9B + no bank** | Does the bank help at all? | Seed bank reaches same reward in 5 steps that no-bank needs 15 steps |
| **A2: 9B + seed bank vs 9B + raw SFT** (same data) | Does *structure* matter, or is raw distillation enough? | Structured bank outperforms flat SFT with identical source data |
| **A3: 9B + cross-domain seeds vs 9B + same-domain seeds** | Does cross-domain transfer add value? | Game→Web/VR seeds improve reward beyond same-domain-only seeds |

**A1** proves the bank is useful. **A2** proves the bank *architecture*
(not just data) is the contribution. **A3** proves cross-domain transfer
is real, not just within-domain skill reuse.

### Where each ablation maps to the training plan

| Ablation | Phase | Configuration |
|---|---|---|
| A1 | Phase 2 (held-out games) | `BANK_MODE=shared` vs `BANK_MODE=none` |
| A2 | Phase 2 | `BANK_MODE=shared` vs SFT-only (same frontier episodes, no bank structure) |
| A3 | Phase 3 (OOD: web + VR) | `seed_source=all_domains` vs `seed_source=target_domain_only` |

**Headline metric** (from coevo plan §7.3):
`reward(seed-bank @ 5 steps) / reward(no-seed @ 15 steps GRPO)` — if
≥ 1.0 on ≥ 4/6 held-out targets, cross-task transfer holds.

### What the frontier data pipeline provides

This pipeline is the **data preparation layer** that feeds the training
experiments. It is not the contribution itself, but without it the
ablation experiments cannot run:

1. **Per-task skill banks** (406 skills) → seed candidates for A1, A3
2. **Layer-C templates** (406 templates) → modality-agnostic reasoning
   plans that enable cross-domain seeding for A3
3. **Decision SFT** (22k rows) → the raw-SFT baseline for A2
4. **Shared bank + bindings** → the structured bank for A1, A2
5. **Cross-domain reasoning plans** (7 collapsed plans, 314 skills,
   77.3%) → evidence that the same reasoning structure exists across
   domains (motivation for A3, not the result itself); includes 1
   three-way (GAME+WEB+VR) plan covering 55 skills

---

## Pipeline Results (2026-05-12)

| Metric | Count |
|---|---:|
| Per-task skill banks collected | **18** |
| Per-task skills (native) | **406** |
| Abstract mega-skills (shared bank) | **354** |
| Multi-task mega-skills (span ≥ 2 tasks) | **14** |
| Forward cross-task bindings (offline) | **186** |
| Total bindings (native + forward) | **592** |
| **Layer-C templates (GPT-5.4)** | **406 / 406** |
| **Cross-domain plans (exact 8-op)** | **9** (221 skills, 54.4%) |
| **Cross-domain plans (collapsed 5-op)** | **7** (314 skills, **77.3%**) |
| **Three-way plans (GAME+WEB+VR)** | **1** (55 skills, 13.5%) |
| **Sig-level judge validated (score ≥ 4)** | **52/72** pairs (72%), all 7 sigs MODERATE+ |
| **Plan-level judge (full plan text)** | **377/415** pairs (≥4), **310 NEW** cross-sig |
| **Non-game skills with cross-domain match** | **78/78 (100%)** via plan-level judge |
| **Cross-domain mega-skill families (plan-judge)** | **12** (10 three-way GAME+WEB+VR, 2 two-way) |
| **Bottom-up mega-skill taxonomy (LLM-extract)** | **18** families (5 three-way, 5 two-way, 8 single-domain) |
| Decision SFT coverage | 12 / 18 skill-banked tasks |
| action_taking rows | 22,086 |
| skill_selection rows | 21,086 |
| Cohort coverage | gymv_game 211 · env_wr_game 72 · web 48 · vr_image 13 · vr_video 17 |

---

## 1. Source Layout in `emnlp2026_download/workspace/main_project/`

### 1a. Game rollouts

| Data type | Path pattern |
|---|---|
| GPT gym_v (13 games, e20) | `Cold-start-out-gymv/2026-04-28_20-20-10/Temporal_*/` |
| GPT gym_v skip8 testbed (8 games, e16) | `Cold-start-out-gymv/gpt54_skip8_e16_s80_20260503_093654/` |
| Claude/Gemini/Qwen gym_v (13 games, e20) | `openrouter-baselines-out/2026-05-01_08-06-49/{teacher}/gymv/` |
| Claude/Gemini/Qwen gym_v skip8 (8 games, e16) | `openrouter-baselines-out/openrouter_skip8_e16_s80_20260503_093707/{teacher}/gymv/` |
| GPT env_wrappers (tetris, candy_crush, 2048) | `Cold-start-out/2026-04-28_18-25-38/game-ai-agent/` |
| GPT SFT env_wrappers (13 gymv games) | `Cold-start-out-gymv/sft_gpt5p4_e20_s100_20260429_*/` |
| Qwen-local env_wrappers (4 games) | `qwen-baselines-out/2026-04-29_10-34-29/35B-A3B/env_wrappers/` |
| Qwen-local gym_v (13 games) | `qwen-baselines-out/2026-04-29_10-34-29/35B-A3B/gymv/` |
| Qwen-API gym_v | `qwen-api-baselines-out/qwen_api_v2_20260504_035222/qwen3.5-35b-a3b/gymv/` |
| Skill Bank SFT (per-game) | `skill_bank_sft/Temporal_*/` |

### 1b. Non-game rollouts

| Data type | Path pattern |
|---|---|
| GPT miniwob (125 tasks) | `Cold-start-out-browsergym/miniwob.*/` |
| GPT webshop (50 tasks) | `Cold-start-out-browsergym/webshop_50task_low/` |
| Claude/Gemini/Qwen webshop | `Cold-start-out-browsergym/webshop_50task_{claude,gemini,qwen}/` |
| GPT assistantbench (181 tests) | `Cold-start-out-browsergym/assistantbench.test.*/` (**excluded**) |
| Claude/Gemini/Qwen assistantbench | `openrouter-transfer-baselines-out/.../browsergym/assistantbench.*` (**excluded**) |
| GPT osworld | `Cold-start-out-osworld/` (**excluded**) |
| Claude/Gemini/Qwen osworld | `openrouter-transfer-baselines-out/.../{teacher}/osworld/` (**excluded**) |
| GPT visual-reasoning (image) | `Cold-start-out-visual-reasoning/{tir_bench,visual_toolbench}/` |
| GPT visual-reasoning (video) | `Cold-start-out-visual-reasoning-video/{siv_bench,video_holmes}/` |
| Claude/Gemini/Qwen VR (image) | `openrouter-transfer-baselines-out/.../{teacher}/vr_image/` |
| Claude/Gemini/Qwen VR (video) | `openrouter-transfer-baselines-out/.../{teacher}/vr_video/` |
| GPT VisualWebArena (smoke) | `Cold-start-out-browsergym/vwa_real_agent_smoke*/` |

### 1c. Other data

| Data type | Path pattern |
|---|---|
| Decision SFT JSONL | `labeling/decision_sft_jsonl/run_20260430_082516/{task}/` |
| Cross-domain skill lift | `skill_transfer_test/skill_bank_local/full_v5/` |
| Cross-domain transfer eval | `cross_domain_results/{_phase0,_phase5,_final}/` |

---

## 2. 8-Game × 4-Model Testbed (Game SFT Core)

The **skip8** configuration (`frame_skip=8`, 16 episodes per model per game)
is the primary multi-teacher testbed for game SFT. All 8 games have
**complete 4-model coverage**.

| Game | Genre | GPT-5.4 | Claude | Gemini | Qwen | Teacher range |
|---|---|---:|---:|---:|---:|---|
| ThunderForceIII | shmup | 306 | 269 | **725** | **750** | 269–750 |
| AlteredBeast | beat-em-up | 119 | 294 | **425** | 263 | 119–425 |
| Columns | puzzle | **154** | 63 | 99 | 132 | 63–154 |
| DynamiteHeaddy | action-platformer | **94** | **94** | 81 | 75 | 75–94 |
| SpaceHarrierII | shmup | 23,850 | **29,431** | 22,931 | 14,469 | 14k–29k |
| StreetsOfRage2 | beat-em-up | 259 | 281 | **409** | 202 | 202–409 |
| Airstriker | shmup | 53 | 94 | **98** | 68 | 53–98 |
| Strider | action-platformer | 0 | 31 | **113** | 0 | 0–113 |

**Total: 8 games × 4 models × 16 episodes = 512 rollout episodes.**

> **Phase assignment** (from coevo plan):
> - **Phase 1 source** (6 games): ThunderForceIII, AlteredBeast, Columns,
>   DynamiteHeaddy, candy_crush, tetris
> - **Phase 2 transfer targets**: SpaceHarrierII (30× reward scale test),
>   StreetsOfRage2, Airstriker, Strider (in-genre transfer targets)
> - **SpaceHarrierII** — moved to Phase 2 because 14k–29k reward is ~30×
>   larger than any other gymv game, would dominate aggregates
> - **Strider** — GPT and Qwen score 0; 50%-zero teacher distribution
>   would poison Phase 1 SFT

---

## 3. Full Per-Task Skill Inventory

### 3a. gym_v games (8 games — all 4 teachers)

Mean reward from `rollout_summary.json` (skip8 e16). **Bold** = best
across 4 teachers.

| Task | Genre | GPT | Claude | Gemini | Qwen | Skills | SFT rows | Fwd binds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Airstriker | shmup | 53 | 94 | **98** | 68 | 31 | 2,000 | 10 |
| AlteredBeast | beat-em-up | 119 | 294 | **425** | 263 | 21 | 2,000 | 8 |
| Columns | puzzle | **154** | 63 | 99 | 132 | 27 | 2,000 | 9 |
| DynamiteHeaddy | action-plat | **94** | **94** | 81 | 75 | 32 | 2,000 | 7 |
| SpaceHarrierII | shmup | 23,850 | **29,431** | 22,931 | 14,469 | 33 | 2,000 | 8 |
| StreetsOfRage2 | beat-em-up | 259 | 281 | **409** | 202 | 34 | 2,000 | 9 |
| Strider | action-plat | 0 | 31 | **113** | 0 | 45 | 2,000 | 2 |
| ThunderForceIII | shmup | 306 | 269 | 725 | **750** | 32 | 2,000 | 8 |
| **gym_v total** | | | | | | **255** | **16,000** | **61** |

Rollout data: e20 cold-start (GPT) + e20 full (Claude/Gemini/Qwen) +
e16 skip8 testbed (all 4 models).

> **Excluded (zero reward):** CastleOfIllusion, CastlevaniaBloodlines,
> GoldenAxe, KidChameleon, MortalKombatII — all 4 teachers score 0;
> env returns no reward signal. 161 skills + 10,000 SFT rows removed.

### 3b. env_wrapper games (4 games — GPT + Qwen-local)

| Task | GPT reward | Qwen reward | Skills | SFT rows | Fwd binds |
|---|---:|---:|---:|---:|---:|
| tetris | 297 | 65 | 28 | 1,573 | 10 |
| super_mario | 1,138 | 500 | 16 | 326 | 13 |
| candy_crush | 552 | 509 | 10 | 1,000 | 14 |
| twenty_forty_eight | 1,542 | 959 | 19 | 3,187 | 15 |
| **env_wrapper total** | | | **73** | **6,086** | **52** |

> GPT rewards from SFT e20 run; Qwen-local from
> `qwen-baselines-out/.../35B-A3B/env_wrappers/` (e16).
> Claude / Gemini never collected for env_wrapper games.

### 3c. Non-game environments

#### Web tasks

| Task | GPT | Claude | Gemini | Qwen | Episodes | Archetypes | Cluster field | Fwd binds |
|---|:---:|:---:|:---:|:---:|---:|---:|---|---:|
| miniwob (125 sub-tasks) | ✅ | — | — | — | 125 | **45** | task family (click, email, ...) | 14 |
| webshop (50 tasks) | ✅ | ✅ | ✅ | ✅ | 50 | **3** | step-count bucket | 14 |
| assistantbench (181 tests) | ✅ | ✅ | ✅ | ✅ | 181 | **excluded** | — | — |
| osworld | ✅ | ✅ | ✅ | ✅ | ~30 | **excluded** | — | — |
| VisualWebArena (smoke) | ✅ | — | — | — | 1 | — | — | — |

miniwob and webshop now use **archetype aggregation** matching the VR
benchmark format. The `build_web_skill_banks.py` script lifts each
episode into a per-episode skill, then clusters by `cluster_key`:
- **miniwob**: task family = first word of task name (click → 31 members,
  email → 10, drag → 9, use → 7, enter → 6, choose → 5, ...)
- **webshop**: step-count bucket (short ≤5 steps → 13 members,
  medium 6-12 → 12, long 13+ → 25)

#### Visual reasoning benchmarks (in skill bank, 4 tasks — all 4 teachers)

| Task | Corpus | GPT | Claude | Gemini | Qwen | Bank skills | n_instances | Forward binds |
|---|---|:---:|:---:|:---:|:---:|---:|---:|---:|
| tir_bench | vr_image | ✅ | ✅ | ✅ | ✅ | 11 | 105 | 15 |
| visual_toolbench | vr_image | ✅ | ✅ | ✅ | ✅ | 2 | 31 | 15 |
| siv_bench | vr_video | ✅ | ✅ | ✅ | ✅ | 10 | 220 | 15 |
| video_holmes | vr_video | ✅ | ✅ | ✅ | ✅ | 7 | 396 | 15 |
| **VR total** | | **4×4** | | | | **30** | **752** | **60** |

### 3d. Totals (skill-banked tasks only)

| | gym_v | env_wrapper | Web | VR | All |
|---|---:|---:|---:|---:|---:|
| Tasks | 8 | 4 | 2 | 4 | **18** |
| Native skills | 255 | 73 | 48 | 30 | **406** |
| Forward bindings | 61 | 52 | 28 | 45 | **186** |
| Total bindings | 316 | 125 | 76 | 75 | **592** |

---

## 4. Shared Skill Bank — TwoLayerSkillStore

The shared bank lifts 406 per-task skills into abstract mega-skills
(SharedAbstractSkill). The **default clustering method** is plan-level
LLM-as-judge (`build_plan_clustered_bank.py`), which groups skills by
shared reasoning procedure rather than name or structural signature.

### Clustering method (default: plan-level LLM judge)

| Method | Script | How it clusters | Coverage |
|---|---|---|---|
| **Plan-level LLM judge** ✅ | `build_plan_clustered_bank.py` | Union-find on judge edges (score ≥ 4) | 100% |
| Name-based (fallback) | `build_shared_bank.py` | `normalise_skill_id` stem | 0% cross-domain |
| Structural signature | `build_reasoning_aligned_bank.py` | Collapsed 5-op signature | 77.3% |

The plan-level judge clustering works by:
1. Loading judge results (`plan_level_similarity_judgments.json`)
2. Creating an edge between skills A and B when the judge scores them ≥ 4
   ("same transferable cognitive procedure")
3. Running union-find to extract connected components as mega-skill clusters
4. Falling back to signature/name grouping for skills without judge edges

Override: `CLUSTER_METHOD=name bash frontier_data/scripts/run_full_pipeline.sh`

### Storage layout

```
frontier_data/output/shared_skill_bank/
├── abstract.jsonl                  ← 354 SharedAbstractSkill (protocol skeletons)
├── by_task/<task>/bindings.jsonl   ← BoundConcreteSkill per task (406 native + 186 forward)
└── SUMMARY.json
```

### Template signature distribution

| Signature | Count | % |
|---|---:|---:|
| PERCEIVE → DECIDE → COMMIT | 259 | 73.2 |
| COMMIT → COMMIT | 14 | 4.0 |
| VERIFY | 22 | 6.2 |
| COMMIT | 11 | 3.1 |
| COMPARE → VERIFY | 11 | 3.1 |
| PERCEIVE → COMMIT → COMMIT | 10 | 2.8 |
| PERCEIVE → COMMIT | 8 | 2.3 |
| PERCEIVE → RECALL → DECIDE → COMMIT | 3 | 0.8 |
| PERCEIVE → RECALL → DECIDE → COMMIT → VERIFY | 3 | 0.8 |
| Other | 13 | 3.7 |

### Multi-task mega-skills (top 15, span ≥ 2 tasks)

| Mega-skill | Signature | Tasks | Cohorts |
|---|---|---:|---|
| **INSPECT/SETUP** | PERCEIVE → COMPARE → DECIDE | 14 | gymv_game, env_wr_game |
| **COMMIT/EXPLORE** | PERCEIVE → RECALL → FILTER → DECIDE → COMMIT | 12 | gymv_game |
| **COMMIT/POSITION** | VERIFY | 11 | gymv_game, env_wr_game |
| **RECOVER/EVADE** | PERCEIVE → RECALL → DECIDE → COMMIT → VERIFY | 11 | gymv_game |
| **COMMIT/ATTACK** | PERCEIVE → DECIDE → COMMIT | 9 | gymv_game |
| **COMMIT/NAVIGATE** | PERCEIVE → RECALL → DECIDE → COMMIT | 9 | gymv_game, env_wr_game |
| **COMMIT/EVADE** | COMPARE | 8 | gymv_game, env_wr_game |
| **COMMIT/COLLECT** | PERCEIVE → FILTER → DECIDE → COMMIT | 5 | gymv_game |
| **RECOVER/SURVIVE** | PERCEIVE → RECALL → DECIDE → COMMIT → VERIFY | 5 | gymv_game |
| **COMMIT/CLEAR** | PERCEIVE → COMPARE → FILTER → DECIDE → COMMIT | 4 | gymv_game, env_wr_game |
| **COMMIT/SETUP** | PERCEIVE → RECALL → DECIDE → COMMIT | 4 | gymv_game, env_wr_game |
| **COMMIT/SURVIVE** | VERIFY | 4 | gymv_game, env_wr_game |
| **COMMIT/EXECUTE** | PERCEIVE → DECIDE → COMMIT → VERIFY | 3 | gymv_game |
| **RECOVER/DEFEND** | PERCEIVE → RECALL → DECIDE → COMMIT → VERIFY | 3 | gymv_game |
| **COMPARE/ATTACK** | PERCEIVE → DECIDE → COMMIT | 2 | gymv_game |

### How mega-skills bridge games and domains

The same abstract reasoning skeleton materialises as completely different
concrete skills depending on the task:

**`RECOVER/EVADE` (11 tasks, PERCEIVE → RECALL → DECIDE → COMMIT → VERIFY)**

| Task | Concrete skill | What it does |
|---|---|---|
| StreetsOfRage2 | "Slip Past Threats" | Sidestep or back away to dodge punches/kicks |
| ThunderForceIII | "Dodge and Probe" | Veer up/down when enemy fire crowds the ship's lane |
| Strider | "Evade Air Hazards" | Jump/reposition when aerial enemies appear overhead |
| Airstriker | "Dodge Left" | Sharp left to slip between incoming bullets |
| CastlevaniaBloodlines | "Defensive Retreat" | Back off and time whip to avoid damage |

**`COMMIT/CLEAR` (4 tasks, PERCEIVE → COMPARE → FILTER → DECIDE → COMMIT)**

| Task | Concrete skill | What it does |
|---|---|---|
| candy_crush | "Clear Match Chain" | Scan board → compare match candidates → pick longest chain |
| CastlevaniaBloodlines | "Clear Corridor" | Perceive obstacles → compare paths → push through |
| DynamiteHeaddy | "Clear Hazard Zone" | Spot hazards → compare approach angles → headbutt clear |
| Strider | "Clear Enemy Wave" | Scan wave → compare threat priority → slash through |

### Bidirectional data flow

```
FORWARD (transfer to new task):
  SharedAbstractSkill ──LLM re-ground to target vocab──▶
    candidate BoundConcreteSkill (status=PENDING)
      ──harness FewShotAdapter validate──▶
        status=VALIDATED, sub_episodes appended

BACKWARD (discovery from task):
  New skill mined in task X ──LLM lift to modality-agnostic skeleton──▶
    upsert into SharedAbstractBank
      (new record or new lineage entry on existing abstract)
```

### Runtime configuration

| Component | Per-Task Bank | Shared Bank |
|---|---|---|
| Storage | `<bank_dir>/<game>/skill_bank.jsonl` | `abstract.jsonl` + `by_task/*/bindings.jsonl` |
| Manager | `PerGameSkillBankManager` | `SharedSkillBankManager` + `TwoLayerSkillStore` |
| Identity | task-local `skill_id` | `stable_key = (skill_id_stem, template_signature)` |
| Transfer | LoRA carry-over only | LLM skeleton-to-binding at phase boundaries |
| Config | `BANK_MODE=per_game` (default) | `BANK_MODE=shared` + `TRANSLATE_ON_BOUNDARY=1` |

---

## 5. Pipeline Scripts

### Master orchestrator

```bash
bash frontier_data/scripts/run_full_pipeline.sh            # full 9-stage run
STAGE=5 bash frontier_data/scripts/run_full_pipeline.sh    # resume from stage 5
DRY_RUN=1 bash frontier_data/scripts/run_full_pipeline.sh  # print commands only
```

| Stage | Script | What it does | LLM needed |
|---:|---|---|:---:|
| 1 | `extract_skillbank_{gpt54,gymv_gpt54}.py` | Extract per-task skill banks from cold-start rollouts | ✅ GPT-5.4 |
| 2 | `label_skill_actions_*.py` → `build_decision_sft_jsonl.py` | Label episodes with skills → decision SFT | ✅ GPT-5.4 |
| 3 | (copy + reorganise) | Build `frontier_distill_jsonl` from decision SFT | — |
| 4 | `lift_skill_templates_gpt54.py` | Lift Layer-C procedural templates | ✅ GPT-5.4 |
| 5 | `build_plan_clustered_bank.py` | **Default:** plan-level judge clustering → SharedAbstractBank | — (uses judge results) |
| 5-alt | `build_shared_skill_bank.py` | Fallback: name-based clustering → SharedAbstractBank | — |
| 6 | `discover_skill_to_shared_bank.py` | Backward: per-task skills → shared bank | ✅ GPT-5.4 |
| 7 | `bind_abstract_to_task.py` | Forward: mega-skills → per-task bindings | ✅ GPT-5.4 |
| 8 | `crafter_v2_batch_pipeline.py` | Crafter v2 refine/compose skills | ✅ 35B judge |
| 9 | `build_inventory.py` | Rebuild SFT data inventory | — |

### Individual scripts under `frontier_data/scripts/`

| Script | What it does | Output |
|---|---|---|
| `collect_all_per_task_banks.py` | Collect per-task `skill_bank.jsonl` from all download sources | `output/per_task_banks/` |
| `collect_decision_sft.py` | Collect decision SFT + produce gap report | `output/decision_sft/` |
| `build_plan_clustered_bank.py` | **Default:** plan-level judge clustering → mega-skills (union-find on judge edges) | `output/shared_skill_bank/` |
| `build_shared_bank.py` | Fallback: name-based clustering → 354 SharedAbstractSkill mega-skills | `output/shared_skill_bank/` |
| `build_web_skill_banks.py` | Lift miniwob (45 archetypes) + webshop (3 archetypes) | `output/per_task_banks/{miniwob,webshop}/` |
| `bind_and_validate.py` | Forward-bind mega-skills → per-task via harness + crafter | `output/shared_skill_bank/by_task/` |
| `test_game_to_nongame_transfer.py` | Harness structural validation: game → non-game transfer matrix | `output/transfer_matrix.json` |
| `build_reasoning_aligned_bank.py` | Reasoning-intent normalizer + cross-domain plan matcher | `output/reasoning_aligned_mega_skills.json` |
| `judge_plan_level_similarity.py` | Plan-level LLM-as-judge: batch 1-vs-N comparison (feeds default clustering) | `output/plan_level_similarity_judgments.json` |
| `judge_plan_similarity.py` | Signature-level LLM-as-judge: pairwise comparison within same signature | `output/plan_similarity_judgments.json` |
| `extract_mega_skills.py` | Bottom-up per-skill LLM classification into 18 cognitive families | `output/mega_skill_labels.json` |
| `cluster_mega_skills.py` | Merge raw mega-skill labels into canonical families | `output/mega_skill_clusters.json` |
| `inject_layerc_protocols.py` | Convert Layer-C templates → runtime protocol in skill banks | per-task `skill_bank.jsonl` |

### Output directory layout

```
frontier_data/
├── scripts/
│   ├── run_full_pipeline.sh              ← 9-stage master orchestrator
│   ├── collect_all_per_task_banks.py     ← step 1: collect all native skills
│   ├── collect_decision_sft.py           ← step 2: collect decision SFT + gap report
│   ├── build_plan_clustered_bank.py      ← DEFAULT step 3: plan-judge clustering
│   ├── build_shared_bank.py              ← fallback step 3: name-based clustering
│   ├── bind_and_validate.py              ← step 4: forward-bind + crafter
│   ├── judge_plan_level_similarity.py    ← plan-level LLM judge (feeds clustering)
│   ├── judge_plan_similarity.py          ← signature-level LLM judge
│   ├── extract_mega_skills.py            ← bottom-up LLM classification
│   ├── cluster_mega_skills.py            ← label merging
│   └── inject_layerc_protocols.py        ← Layer-C → runtime protocol
├── output/
│   ├── per_task_banks/                   ← 406 skills across 18 tasks
│   │   ├── <task>/skill_bank.jsonl
│   │   └── MANIFEST.json
│   ├── decision_sft/                     ← 12/18 skill-banked tasks with SFT data
│   │   ├── <task>/{action_taking,skill_selection}.jsonl
│   │   ├── MANIFEST.json
│   │   └── GAP_REPORT.json              ← 6 non-game gaps documented
│   ├── shared_skill_bank/                ← TwoLayerSkillStore
│   │   ├── abstract.jsonl                ← mega-skills (plan-judge clustered)
│   │   ├── by_task/<task>/bindings.jsonl
│   │   └── SUMMARY.json
│   ├── layer_c_templates/                ← GPT-5.4 Layer-C procedural templates
│   │   ├── <cohort>/<task>/template_bank.jsonl
│   │   └── _lift_summary.json
│   ├── plan_level_similarity_judgments.json  ← plan-level judge results (feeds clustering)
│   ├── plan_similarity_judgments.json        ← signature-level judge results
│   ├── mega_skill_labels.json               ← per-skill LLM classification
│   ├── mega_skill_clusters.json             ← merged canonical families
│   ├── bind_reports/                     ← binding audit trail
│   ├── transfer_matrix.json              ← game→non-game harness validation
│   └── reasoning_aligned_mega_skills.json ← cross-domain reasoning plans
├── README.md                             ← this file
└── PIPELINE_GUIDE.md                     ← pipeline walkthrough
```

---

## 6. Upstream Script Reference

Scripts in the main repo that the pipeline wraps:

| Script | Role | Dependencies |
|---|---|---|
| `labeling/extract_skillbank_gpt54.py` | SkillBankAgent pipeline for env_wrappers | labeled episodes |
| `labeling/extract_skillbank_gymv_gpt54.py` | SkillBankAgent for gym-v ROMs | cold-start rollouts |
| `labeling/build_skillbank_qa_gpt54.py` | CLUSTER→CONTRACT→CURATOR for QA/MiniWob | `qa_multihop_out` |
| `labeling/build_decision_sft_jsonl.py` | Convert skill-labeled episodes → SFT JSONL | `skill_actions_out` |
| `scripts/lift_skill_templates_gpt54.py` | GPT-5.4 Layer-C template lifting | per-task skill banks |
| `scripts/build_shared_skill_bank.py` | Merge templates + mining + production → SharedAbstractBank | templates + skill banks |
| `scripts/bind_abstract_to_task.py` | Forward-bind abstract → target task | SharedAbstractBank |
| `scripts/discover_skill_to_shared_bank.py` | Backward-lift new skill → shared bank | per-task skill bank |
| `scripts/seed_per_task_bank_cold_start.py` | Cold-start a new task from shared bank | SharedAbstractBank |
| `scripts/crafter_v2_batch_pipeline.py` | Crafter v2: failure detection → skill proposal → dedup | training run artifacts |

---

## 7. Training Plan Integration

| Phase | Tasks | Bank mode | What happens |
|---|---|---|---|
| Phase 1 | ThunderForceIII, AlteredBeast, Columns, DynamiteHeaddy, candy_crush, tetris | `per_game` | Mine concrete skills; populate per-task banks |
| Phase 2 | SpaceHarrierII, StreetsOfRage2, Airstriker, Strider, twenty_forty_eight, super_mario | `shared` | Transfer: mega-skill skeletons to held-out games |
| Phase 3 | miniwob, webshop, tir_bench, visual_toolbench, siv_bench, video_holmes | `shared` | OOD: reasoning patterns cross domain boundaries |

Key config:
- `BANK_MODE=shared` — one SharedAbstractBank across all phases
- `TRANSLATE_ON_BOUNDARY=1` — re-ground skills at phase transition
- `feasible_tasks` on each skill — runtime eligibility veto via `EligibilityFilter`

### 7a. Ablation experiment design

Three controlled comparisons to isolate the contribution of structured
skill banks from frontier-model distillation:

**A1 — Bank vs No-Bank** (Phase 2, per held-out game)

| Arm | Config | Budget |
|---|---|---|
| `seed-bank` | `BANK_MODE=shared`, seed from Phase 1 bank | 5 steps infer → 15 steps GRPO |
| `no-bank` | `BANK_MODE=none`, empty skill bank | 15 steps GRPO |
| `no-bank-long` | `BANK_MODE=none`, empty skill bank | 35 steps GRPO (upper bound) |

Metric: step at which `seed-bank` matches `no-bank@15` reward.
Claim: structured bank saves ≥ 10 GRPO steps per held-out game.

**A2 — Structured Bank vs Raw SFT** (Phase 2, per held-out game)

| Arm | Config | Data |
|---|---|---|
| `seed-bank` | `BANK_MODE=shared`, cold-start seeds | Structured skill bank (protocol + contract + effects) |
| `raw-sft` | SFT warm-start, no bank structure | Same frontier episodes, flattened to (obs, action) pairs |

Both arms see the **same source data** (identical frontier teacher
rollouts). The only difference is whether the data arrives as structured
skill records (with reasoning plans, contracts, protocol steps) or as
flat SFT rows. This isolates the architecture contribution.

**A3 — Cross-Domain vs Same-Domain Seeds** (Phase 3, per OOD target)

| Arm | Config | Seed source |
|---|---|---|
| `cross-domain` | Seeds from all 12 game tasks (Layer-C matched) | 7 collapsed cross-domain plans (314 skills) → target |
| `same-domain-only` | Seeds from same-domain tasks only | VR→VR or Web→Web, no game skills |
| `no-seed` | Empty bank | Cold start |

Metric: reward delta between `cross-domain` and `same-domain-only`.
Claim: game-domain reasoning plans (PERCEIVE→EVALUATE→DECIDE→ACT etc.)
transfer usefully to web/VR tasks.

### 7b. What a positive result looks like

```
Phase 2 (in-domain transfer):
  seed-bank@5  ≥  no-bank@15   on ≥ 4/6 held-out games     → bank saves steps (A1)
  seed-bank@15 >  raw-sft@15   on ≥ 4/6 held-out games     → structure matters (A2)

Phase 3 (cross-domain transfer):
  cross-domain > same-domain-only  on ≥ 3/6 OOD targets    → cross-domain works (A3)
  cross-domain > no-seed           on ≥ 5/6 OOD targets    → seeds help OOD (A1+A3)
```

### 7c. What a negative result means

- A1 fails → bank architecture doesn't help; the 9B model learns
  equally fast with or without seeds. Possible cause: seeds are too
  abstract for the student to operationalize.
- A2 fails → structure doesn't matter; raw SFT is as good as the bank.
  This would mean the contribution is in the *data* (frontier teacher
  quality), not the *architecture* (skill bank).
- A3 fails → cross-domain reasoning plans don't transfer. Game skills
  and web/VR skills share the same Layer-C signatures but the concrete
  grounding is too different for the 9B model to bridge. Possible cause:
  re-grounding quality depends on frontier model, not student capability.

---

## 8. Cross-Domain Transfer Analysis (Game → Non-Game)

### 8a. Harness structural validation

`test_game_to_nongame_transfer.py` ran all 14 game multi-task abstracts
through the existing harness pipeline (predicate translator + FewShotAdapter
+ skill_bank_bridge) against every non-game target.

| Abstract | Signature | Game tasks | miniwob | webshop | tir | vtool | siv | vholmes |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| COMMIT/ATTACK | P→D→C | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/CLEAR | P→Co→F→D→C | 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/COLLECT | P→F→D→C | 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/EXECUTE | P→D→C→V | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/EXPLORE | P→R→F→D→C | 7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/NAVIGATE | P→R→D→C | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/SETUP | P→R→D→C | 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMPARE/ATTACK | P→D→C | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INSPECT/SETUP | P→Co→D | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RECOVER/EVADE | P→R→D→C→V | 7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RECOVER/SURVIVE | P→R→D→C→V | 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COMMIT/EVADE | Co | 6 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| COMMIT/POSITION | V | 8 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| COMMIT/SURVIVE | V | 3 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**Result: 11/14 (78.6%) pass structural validation.** The 3 rejected are
single-step reactive skills (< 2 hops, not multi-step reasoning).

Predicate translation tables (`gymv → browser`, `gymv → visual_reasoning`,
`gymv → video`) in `harness/predicate_translator.py` provide full coverage.

### 8b. Reasoning-plan alignment analysis (protocol structure, not names)

**Critical finding:** the current shared bank clusters skills by
`skill_id_stem` (name), **not by reasoning procedure**. Deep inspection
of actual protocol steps reveals three mismatched vocabularies:

| Domain | Avg steps | Op vocabulary | Protocol level |
|---|---:|---|---|
| Game (328 skills) | 0.9 | EXEC, MOVE, VERIFY, APPROACH, KEEP | **action-level** |
| VR (30 skills) | 5.9 | INSPECT, EVALUATE, VERIFY, EXECUTE | **reasoning-level** |
| Web (48 skills) | 3.9 | COMMIT, PERCEIVE | **raw-trace-level** |

When skills are normalized to a domain-agnostic reasoning vocabulary
(PERCEIVE / RECALL / EVALUATE / DECIDE / NAVIGATE / ACT / VERIFY) and
matched by **compressed reasoning plan**, only **3 cross-domain plans**
emerge from 406 skills:

| Reasoning plan | Domains | Skills | Example |
|---|---|---:|---|
| PERCEIVE → VERIFY → ACT | GAME + VR | 4 | game setup + VR color QA |
| PERCEIVE → ACT | GAME + WEB | 3 | compare/setup + order items |
| PERCEIVE → ACT → VERIFY → ACT | GAME + VR | 2 | mario setup + video temporal |

**Root cause:** all three domains actually share the
`PERCEIVE → EVALUATE → ACT → VERIFY` reasoning cycle, but it is **hidden**
because:
1. Game skills encode reasoning as action verbs (EXEC, MOVE, APPROACH)
2. Web skills are raw click/fill traces with no reasoning structure
3. Only VR skills have explicit reasoning-level protocols

**Top reasoning plans per domain (compressed):**

| Game | Web | VR |
|---|---|---|
| ACT → NAVIGATE → ACT (7) | ACT → PERCEIVE → ACT (6) | PERCEIVE → EVALUATE → VERIFY → ACT (17) |
| NAVIGATE → ACT (6) | PERCEIVE → ACT (2) | PERCEIVE → EVALUATE → PERCEIVE → VERIFY → ACT (4) |
| NAVIGATE → ACT → VERIFY (6) | ACT → PERCEIVE → ACT → ... (2) | PERCEIVE → ACT → PERCEIVE → EVALUATE → VERIFY → ACT (2) |

### 8c. Layer-C reasoning-intent re-lift (GPT-5.4, completed)

`scripts/lift_skill_templates_gpt54.py` re-lifted all 406 per-task skills
into 2-5 step modality-agnostic procedural templates using 8 controlled
reasoning operators: `{PERCEIVE, RECALL, COMPARE, FILTER, DECIDE, COMMIT,
VERIFY, RECOVER}`. These are **reasoning intents**, not action verbs.

**Result: 406/406 skills successfully lifted** (98.4 s, 16 workers).

Output: `frontier_data/output/layer_c_templates/<cohort>/<task>/template_bank.jsonl`

**Top Layer-C signatures per domain:**

| Game (328) | Web (48) | VR (30) |
|---|---|---|
| PERCEIVE→DECIDE→COMMIT→VERIFY (161) | PERCEIVE→DECIDE→COMMIT→COMMIT (6) | PERCEIVE→COMPARE→DECIDE→VERIFY (5) |
| PERCEIVE→FILTER→DECIDE→COMMIT→VERIFY (13) | PERCEIVE→DECIDE→COMMIT→VERIFY (5) | PERCEIVE→RECALL→COMPARE→DECIDE→VERIFY (4) |
| PERCEIVE→COMPARE→FILTER→DECIDE→COMMIT (9) | PERCEIVE→COMPARE→DECIDE→COMMIT (4) | PERCEIVE→COMPARE→DECIDE→VERIFY→COMMIT (3) |
| PERCEIVE→COMPARE→DECIDE→COMMIT→VERIFY (8) | RECALL→PERCEIVE→FILTER→DECIDE→COMMIT (4) | PERCEIVE→COMPARE→FILTER→DECIDE→VERIFY (3) |

### 8d. Cross-domain reasoning plans (Layer-C)

#### Raw 8-op signatures (exact match)

After re-lift, **9 reasoning plans are shared across ≥ 2 domains**,
covering **221 of 406 skills (54.4%)**. However, the top signature
(`PERCEIVE→DECIDE→COMMIT→VERIFY`, 166 skills) is too generic to
represent a meaningful "transfer" — it's the default 4-step loop.

#### Collapsed 5-op signatures (semantic equivalence)

The 8 Layer-C ops are collapsed to **5 semantic equivalence classes**:

| Collapse | Rationale |
|---|---|
| COMPARE + FILTER → **EVALUATE** | Both assess/evaluate perceived state |
| COMMIT + VERIFY + RECOVER → **ACT** | All are execution/action steps |
| PERCEIVE, DECIDE, RECALL | Kept distinct |

This lifts coverage from 54.4% → **77.3%** (314/406 skills) and —
crucially — unlocks the **first true THREE-WAY (GAME+WEB+VR) plan**:

| Collapsed plan | Domains | Skills | Example |
|---|---|---:|---|
| **PERCEIVE → DECIDE → ACT** | GAME+WEB | **201** | navigation / click tasks |
| **PERCEIVE → EVALUATE → DECIDE → ACT** | **GAME+WEB+VR** ★ | **55** | tetris optimize + siv emotion + miniwob compare |
| **PERCEIVE → ACT** | GAME+WEB | **21** | charge attack + drag/draw |
| **RECALL → PERCEIVE → EVALUATE → DECIDE → ACT** | VR+WEB | **11** | video_holmes + webshop search |
| **PERCEIVE → RECALL → EVALUATE → DECIDE → ACT** | GAME+VR | **10** | sequence recall + VR temporal |
| **PERCEIVE → EVALUATE → ACT** | GAME+WEB | **8** | scene scan + ascending order |
| **RECALL → PERCEIVE → DECIDE → ACT** | VR+WEB | **8** | VR+miniwob recall tasks |

**Still domain-locked (22.7%, 92 skills):**
- GAME: 80 skills — mostly contain RECOVER-heavy loops or trivially short (ACT only)
- WEB: 11 skills — start with RECALL (which games rarely do)
- VR: 1 skill — unique PERCEIVE→RECALL→EVALUATE→DECIDE (no ACT)

**Key improvement summary:**

| Metric | Name-based | Exact 8-op | Collapsed 5-op |
|---|---|---|---|
| Cross-domain plans | 0 | 9 | **7** |
| Skills covered | 0 | 221 (54.4%) | **314 (77.3%)** |
| Three-way plans | 0 | 0 | **1 (55 skills)** |
| Meaningful (non-generic) | 0 | 55 | **314** |

**Notable three-way example — `PERCEIVE → EVALUATE → DECIDE → ACT` (55 skills):**

| Domain | Task | Skill | What the reasoning plan does |
|---|---|---|---|
| GAME | tetris | COMMIT/OPTIMIZE | Assess board → evaluate sequences → pick best → place |
| GAME | Columns | puzzle_analysis | Perceive columns → compare patterns → decide placement → commit |
| WEB | miniwob | compare_selection | Observe options → evaluate against criteria → select → click |
| VR | siv_bench | Emotion inference | Observe cues → match against emotions → decide → commit |
| VR | video_holmes | IMC inference | Observe behavior → compare to states → select → commit |

### 8d′. LLM-as-judge plan similarity validation

Pure structural matching (collapsed signatures) cannot distinguish "same
reasoning procedure" from "same structure, different cognitive challenge".
We ran **GPT-4.1-mini as a judge** on all 72 cross-domain skill pairs
(3 samples per domain, all 7 collapsed sig groups), scoring 1–5 on
whether the full plan context (predicate text) represents the SAME
transferable cognitive procedure.

| Collapsed plan | Domains | Pairs | Avg | ≥4 | Verdict |
|---|---|---:|---:|---:|---|
| **RECALL → PERCEIVE → DECIDE → ACT** | VR↔WEB | 9 | **4.0** | 100% | STRONG_TRANSFER |
| **PERCEIVE → EVALUATE → DECIDE → ACT** | GAME+WEB+VR | 27 | **3.8** | 78% | MODERATE_TRANSFER |
| **RECALL → PERCEIVE → EVALUATE → DECIDE → ACT** | VR↔WEB | 9 | **3.8** | 78% | MODERATE_TRANSFER |
| PERCEIVE → ACT | GAME↔WEB | 6 | 3.5 | 67% | MODERATE_TRANSFER |
| PERCEIVE → EVALUATE → ACT | GAME↔WEB | 9 | 3.4 | 44% | MODERATE_TRANSFER |
| PERCEIVE → DECIDE → ACT | GAME↔WEB | 9 | 3.3 | 67% | MODERATE_TRANSFER |
| PERCEIVE → RECALL → EVALUATE → DECIDE → ACT | GAME↔VR | 3 | 3.3 | 33% | MODERATE_TRANSFER |

**Overall: 72 pairs judged, 52 (72%) rated as same procedure (score ≥ 4).**

Key findings:
- **VR↔WEB transfer is strongest** (avg 4.0) — both involve target recall
  + evidence matching, just with different modalities (image vs DOM)
- **Three-way PERCEIVE→EVALUATE→DECIDE→ACT is validated** (78% same
  procedure) — the shared cognitive strategy is "perceive options →
  evaluate against criteria → select best → commit"
- **GAME↔WEB (PERCEIVE→DECIDE→ACT) is weakest** (avg 3.3, bimodal:
  67% score=4, 33% score=2) — "navigate environment" and "click UI
  element" have same structure but different cognitive challenge

Implication for seed pipeline: the judge scores can be used as
**confidence weights** when `pick_seed_candidates` selects seeds.
Plans with avg ≥ 4.0 get full weight; 3.0–4.0 get 0.5; < 3.0
are excluded from cross-domain seeding.

### 8d″. Plan-level LLM judge (beyond signatures)

The signature-based judge (§8d′) only compares skills within the same
collapsed-signature group.  **Plan-level judging** compares the FULL
reasoning plan text across ALL cross-domain pairs, regardless of
structural signature — finding matches that signatures miss entirely.

Method: for each of the 78 non-game skills, present its full plan
alongside 20 diverse game skill plans and ask GPT-4.1-mini which
(if any) share the same cognitive procedure.  Similarly for 30 VR
skills vs 15 WEB skills.  Total: 108 batch LLM calls.

| Direction | Targets | Total matches (≥3) | High (≥4) | NEW (different sig) |
|---|---:|---:|---:|---:|
| WEB → GAME | 48 | 225 | 202 | **168** |
| VR → GAME | 30 | 87 | 79 | **66** |
| VR → WEB | 30 | 103 | 96 | **76** |
| **Total** | **108** | **415** | **377** | **310** |

**Key finding: 310 NEW high-confidence cross-domain pairs discovered
that collapsed-signature matching completely missed.**

75 unique non-game skills gained cross-domain matches they didn't have
from structural matching alone.  ALL 78 non-game skills now have ≥ 1
cross-domain match — **effective coverage: 100% of non-game skills**.

Top discoveries (score = 5, DIFFERENT collapsed signatures):

| Target | Candidate | Shared procedure |
|---|---|---|
| VR: tir "how many colors" (P→E→D→A) | WEB: miniwob "how many aqua" (R→P→E→D→A) | Count items with attribute → submit total |
| VR: siv "action likely to" (P→R→E→D→A) | WEB: webshop "i need some" (R→P→E→D→A) | Perceive options → recall criteria → filter → decide → verify |
| WEB: miniwob "create 30min event" (R→P→D→A) | GAME: Columns "Execute" (P→A) | Perceive state → commit placement → verify result |
| VR: video "arrange following" (P→E→D→A) | WEB: miniwob "click numbers" (P→E→A) | Perceive items → compare order → commit sequence → verify |

**Why signatures miss these:** structural differences like
`PERCEIVE→RECALL→EVALUATE` vs `RECALL→PERCEIVE→EVALUATE` (order of
first two ops) or `PERCEIVE→DECIDE→ACT` vs `RECALL→PERCEIVE→DECIDE→ACT`
(presence/absence of RECALL prefix) are structurally distinct but
cognitively equivalent — the LLM judge recognizes the shared procedure.

**Impact on seed bank sizing:**

| Matching strategy | Non-game coverage | Seedable skills |
|---|---:|---:|
| Exact 8-op sig | 55 (70%) | 221/406 (54%) |
| Collapsed 5-op sig | 66 (85%) | 314/406 (77%) |
| **Plan-level LLM judge** | **78 (100%)** | **406/406 (100%)** |

### 8d‴. Cross-domain mega-skill families (plan-level)

Classifying the 377 high-confidence cross-domain pairs by their shared
reasoning procedure description yields **12 distinct mega-skill families**:

| # | Family | Domains | Skills | Representative procedure |
|---|---|---|---:|---|
| 1 | **COMPARE_AND_RANK** | GAME+WEB+VR ★ | 51 | Perceive targets → rank by priority → select best → execute |
| 2 | **DECIDE_ACT_VERIFY** | GAME+WEB+VR ★ | 33 | Perceive state → decide optimal action → execute → verify |
| 3 | **MATCH_AND_CLASSIFY** | GAME+WEB+VR ★ | 23 | Perceive → match to known categories → commit classification |
| 4 | **PERCEIVE_DECIDE_ACT** | GAME+WEB+VR ★ | 22 | Observe → decide on criteria-based target → commit action |
| 5 | **FILTER_AND_SELECT** | GAME+WEB+VR ★ | 19 | Identify cues → filter options → select best match → verify |
| 6 | **RECALL_FILTER_SELECT** | GAME+WEB+VR ★ | 16 | Recall goal → perceive options → filter → select |
| 7 | **SEQUENTIAL_EXECUTION** | GAME+WEB+VR ★ | 11 | Perceive initial state → multi-step transform → verify |
| 8 | **CREATE_AND_VERIFY** | GAME+WEB+VR ★ | 8 | Perceive target → create/place → verify completion |
| 9 | **COUNT_AND_REPORT** | GAME+WEB+VR ★ | 6 | Count attribute instances → submit total |
| 10 | **INPUT_AND_SUBMIT** | GAME+WEB+VR ★ | 4 | Perceive field → input value → submit |
| 11 | GENERAL_PROCEDURE | GAME+WEB | 4 | Evaluate multiple options → select best config |
| 12 | OBSERVE_AND_ACT | GAME+WEB | 1 | Perceive item → finalize action |

**10 of 12 families are three-way (GAME+WEB+VR)** — structural signature
matching found only 1 three-way plan. Plan-level LLM judging discovers
10× more cross-domain reasoning structure.

**Evolution of cross-domain mega-skill discovery:**

| Method | Cross-domain mega-skills | 3-way | Coverage |
|---|---:|---:|---:|
| Name-based clustering | 0 | 0 | 0% |
| Collapsed 5-op signatures | 7 | 1 | 77% |
| Plan-level LLM judge | **12** | **10** | **100%** |

Scripts:
- `judge_plan_level_similarity.py` — plan-level LLM-as-judge (batch, 108 queries)
- `judge_plan_similarity.py` — signature-level LLM-as-judge (pairwise, 72 pairs)
- `build_reasoning_aligned_bank.py` — offline reasoning-intent normalizer
- `test_game_to_nongame_transfer.py` — harness validation of game→non-game
- `inject_layerc_protocols.py` — convert Layer-C templates → runtime protocol

### 8e. Bottom-up mega-skill extraction (per-skill LLM classification)

An alternative to pairwise plan-level judging (§8d″):
classify **each skill independently** into a fixed taxonomy of
18 cognitive mega-skill families via LLM, then cluster by label.

**Method**: For each of 406 skills, send its Layer-C template steps +
description to `gpt-4.1-mini` with a fixed 18-category taxonomy.
Cost: O(n) = 406 LLM calls (~40s with 10-thread parallelism).

| # | Mega-skill family | Count | Domains | Procedure |
|---|---|---:|---|---|
| 1 | **DODGE_AND_SURVIVE** | 65 | GAME | Monitor threats → evade → survive |
| 2 | **INFER_AND_DECIDE** | 43 | GAME+WEB+VR ★ | Perceive evidence → reason → select action |
| 3 | **ENGAGE_AND_DEFEAT** | 41 | GAME | Identify target → approach → attack → verify |
| 4 | **NAVIGATE_AND_REACH** | 37 | GAME+WEB+VR ★ | Plan path → move → arrive at target |
| 5 | **EXPLORE_AND_DISCOVER** | 35 | GAME | Probe unknown → observe → update knowledge |
| 6 | **COMPARE_AND_RANK** | 23 | GAME+WEB+VR ★ | Perceive options → rank by criterion → select |
| 7 | **SEQUENCE_AND_COMPLETE** | 22 | GAME+WEB+VR ★ | Ordered sub-steps → execute each → verify |
| 8 | **TIME_AND_REACT** | 22 | GAME | Wait for trigger → timed action |
| 9 | **EVALUATE_AND_OPTIMIZE** | 21 | GAME | Assess quality → try improvement → compare |
| 10 | **RECALL_MATCH_AND_SELECT** | 18 | GAME+WEB+VR ★ | Retrieve criteria → perceive → match → select |
| 11 | **POSITION_AND_PLACE** | 15 | GAME+WEB ● | Perceive target → place item → verify |
| 12 | **TRANSFORM_AND_VERIFY** | 14 | GAME+WEB ● | Apply transformation → verify outcome |
| 13 | **MONITOR_AND_SUSTAIN** | 14 | GAME | Track process → maintain → ensure completion |
| 14 | **INPUT_AND_SUBMIT** | 12 | GAME+WEB ● | Fill form → submit → verify |
| 15 | **FILTER_AND_NARROW** | 11 | VR+WEB ● | Apply criteria → eliminate → act on match |
| 16 | **COLLECT_AND_ACCUMULATE** | 8 | GAME | Find collectible → acquire → track |
| 17 | **COUNT_AND_REPORT** | 3 | VR+WEB ● | Count attribute → report total |
| 18 | **RETRIEVE_AND_EXECUTE** | 2 | GAME | Recall procedure → execute → confirm |

**Cross-domain summary:**

| Coverage | Families | Skills |
|---|---:|---:|
| Three-way (GAME+WEB+VR) ★ | 5 | 143 (35%) |
| Two-way ● | 5 | 55 (14%) |
| Single-domain | 8 | 208 (51%) |

**Key insight**: The 5 three-way families (**INFER_AND_DECIDE**,
**NAVIGATE_AND_REACH**, **COMPARE_AND_RANK**, **SEQUENCE_AND_COMPLETE**,
**RECALL_MATCH_AND_SELECT**) represent the core transferable cognitive
procedures. They cover 143 skills across all three domains and are
the best candidates for cross-domain seed transfer.

Scripts:
- `extract_mega_skills.py` — Phase 1: per-skill LLM classification (O(n))
- `cluster_mega_skills.py` — Phase 2: optional label merging (for open-ended extraction)

### 8f. Layer-C protocol injection (runtime-ready)

`inject_layerc_protocols.py` converted all 406 Layer-C templates into
runtime protocol dicts and patched the per-task skill banks:

| Category | Count |
|---|---:|
| Skills patched (empty/thin → 3-5 step reasoning plan) | **304** |
| Already-rich protocols (preserved, enriched with step_checks) | **102** |

The agent now sees structured reasoning plans during execution:

```
--- Active Skill: archetype.siv_bench.Action_Recognition ---
  Plan (5 steps):
  >> 1. Observe current scene cues and immediate preceding events
     2. Recall task goal of predicting the next likely action
     3. Compare candidate continuations against causal and contextual evidence
     4. Select the continuation most strongly implied by context
     5. Submit the chosen next-action prediction
  Done when: prediction_submitted=true
--- end skill ---
```

The `>>` marker advances via `SkillProgressTracker.compute_step_advancement`
with +0.1 intrinsic bonus per step — GRPO learns to follow reasoning plans.

### 8g. V1 transfer approach: skills as suggestions

**Design principle:** Layer-C reasoning plans are **suggestions**, not
commands. The 9B agent sees them in its prompt, attempts to follow, and
GRPO + crafter refine what works. No complex cross-domain compose needed.

**Why previous compose attempts failed:**
- R4 compose in `decide_skill_crafting_gpt54.py` uses co-occurrence
  statistics (A→B frequent ≥ 5% transitions) — behavioral correlation,
  not reasoning complementarity
- Crafter v2 35B proposer gets action-level inputs → outputs caught
  by ban lists or too abstract (0.5% abstract-share per audit)
- `GeneralizeProposal` does 1:1 translation only, cannot compose

**V1 approach (skills as suggestions):**

```
Phase 1: train on 6 source games → build per-task banks
         ↓
Phase 2: seed_per_task_bank_cold_start.py
         pick_seed_candidates now prioritizes by:
           1st: THREE-WAY collapsed signatures (GAME+WEB+VR)
           2nd: TWO-WAY cross-domain (collapsed 5-op, ≥ 2 domains)
           3rd: cohort diversity
           4th: task breadth
           5th: production successes
         Collapsed 5-op matching: 314/406 skills eligible (77.3%)
         ↓
         GPT-5.4 re-grounds to target vocab (bind_abstract_to_task.py)
         ↓
         Seeds land as confidence_tag="candidate" (down-weighted)
         ↓
Phase 2+3: Agent tries seed reasoning plans
           GRPO rewards good execution → plans get reinforced
           Crafter patches bad plans → plans get refined
           Eligibility filter demotes useless plans
           New skills discovered in target feed back to shared bank
```

**Key insight:** the agent doesn't need a perfect plan. It needs a
**reasonable reasoning structure** to bootstrap from. "Scan → Compare →
Filter → Decide → Commit" is useful in webshop even if the original
predicates are from tetris — the STRUCTURE guides the agent's reasoning,
and GRPO fine-tunes the execution.

---

## 9. Known Gaps & Next Steps

### Gap 0: Ablation experiments not yet run

The three ablation comparisons (A1/A2/A3 in §7a) are the **primary
deliverable** that separates the contribution from frontier distillation.
Without these results, the pipeline is infrastructure only.

**Blocking on:** Phase 1 curriculum training (coevo plan §4), which
produces the rolling skill bank that seeds Phase 2 and Phase 3.

### Gap 1: Non-game decision SFT (6 tasks missing)


siv_bench, tir_bench, video_holmes, visual_toolbench, miniwob, webshop —
no `action_taking.jsonl` or `skill_selection.jsonl`.

**To fill:**
```bash
# Step 1: Run QA labeling
python labeling/build_skillbank_qa_gpt54.py \
    --multihop-run labeling/qa_multihop_out/run_<ts> \
    --miniwob-run  labeling/qa_miniwob_labeled/run_<ts>

# Step 2: Build multimodal decision SFT
python scripts/build_multimodal_decision_sft.py \
    --sources video_holmes,siv_bench,tir_bench,visual_toolbench,miniwob,webshop
```

### Gap 2: `frontier_distill_jsonl` partially filled

Covers 17 game tasks from available decision SFT. Non-game VR tasks blocked
on Gap 1.

### Gap 3: 186 forward bindings are PENDING

Offline heuristic binding done; LLM-driven re-grounding + harness
validation needed to promote PENDING → VALIDATED:

```bash
python frontier_data/scripts/bind_and_validate.py --model gpt-5.4
```

### Gap 4: env_wrapper games — partial teacher coverage

tetris, super_mario, candy_crush, twenty_forty_eight have GPT-5.4 +
Qwen-local (35B-A3B) rollouts. Claude / Gemini never collected.

### Gap 5: Template signature coverage — ✅ RESOLVED

~~84.6 % of abstracts used the fallback signature.~~
Layer-C lift completed (§8c): all 406 skills now have GPT-5.4 generated
procedural templates with rich 8-op signatures. Output in
`frontier_data/output/layer_c_templates/`.

### Gap 6: Decision SFT is GPT-only

Current 22,086 action_taking rows all come from GPT-5.4 cold-start
labeling. The skip8 testbed has Claude / Gemini / Qwen rollouts for
8 games (512 episodes total) that haven't been labeled yet. Multi-teacher
SFT would need: skill labeling → decision SFT generation on those rollouts.

### Gap 7: miniwob & webshop — no decision SFT yet

The 48 archetype skills (45 miniwob + 3 webshop) have been added to the
skill bank and shared bank, but decision SFT labeling has not been run.
Need: skill labeling on 125 miniwob + 50 webshop episodes → decision SFT.
Multi-model webshop coverage (Claude/Gemini/Qwen) can increase diversity.

### Gap 8: Cross-domain reasoning alignment — ✅ RESOLVED (collapsed 5-op)

~~Only 3 reasoning plans spanned ≥ 2 domains (9 skills).~~
After Layer-C re-lift (§8c) + collapsed 5-op equivalence (§8d):
**7 cross-domain reasoning plans** covering **314 of 406 skills (77.3%)**,
including 1 true three-way (GAME+WEB+VR) plan covering 55 skills.

- GAME↔WEB: 4 shared plans (230 skills)
- GAME↔VR: 1 plan (10 skills)
- VR↔WEB: 2 plans (19 skills)
- **GAME+WEB+VR: 1 plan (55 skills)** — `PERCEIVE→EVALUATE→DECIDE→ACT`

The `collapse_signature()` function in `seed_per_task_bank_cold_start.py`
and the `collapsed_signature` field on every template/skill record make
this matching automatic at seed selection time.

> **Note:** this gap being "resolved" means the *data pipeline* can now
> produce cross-domain seed candidates. Whether those seeds actually help
> the 9B student model is tested by ablation A3 (§7a) — see Gap 0.

### Excluded tasks

- **assistantbench** — 181 GPT + multi-model test episodes; per-episode
  traces only (`n_instances=1`, no archetype aggregation).
- **osworld** — 30 per-episode traces, same issue.
- **CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe, KidChameleon,
  MortalKombatII** — 0 reward across all 4 frontier teachers (env returns
  no reward signal). 161 skills + 10,000 SFT rows removed.

Rollout data for all excluded tasks remains available in
`emnlp2026_download/` if re-processed in the future.
