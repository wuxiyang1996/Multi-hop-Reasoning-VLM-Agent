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
5. **Cross-domain reasoning plans** (9 plans, 221 skills) → evidence
   that the same reasoning structure exists across domains (motivation
   for A3, not the result itself)

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
| **Cross-domain reasoning plans** | **9** (covering 221 skills, 54.4%) |
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

The shared bank lifts 406 per-task skills into **354 abstract mega-skills**
(SharedAbstractSkill). Of these, **14 span ≥ 2 tasks** — the transferable
multi-step reasoning patterns.

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
| 5 | `build_shared_skill_bank.py` | Merge all sources → SharedAbstractBank | — |
| 6 | `discover_skill_to_shared_bank.py` | Backward: per-task skills → shared bank | ✅ GPT-5.4 |
| 7 | `bind_abstract_to_task.py` | Forward: mega-skills → per-task bindings | ✅ GPT-5.4 |
| 8 | `crafter_v2_batch_pipeline.py` | Crafter v2 refine/compose skills | ✅ 35B judge |
| 9 | `build_inventory.py` | Rebuild SFT data inventory | — |

### Individual scripts under `frontier_data/scripts/`

| Script | What it does | Output |
|---|---|---|
| `collect_all_per_task_banks.py` | Collect per-task `skill_bank.jsonl` from all download sources | `output/per_task_banks/` |
| `collect_decision_sft.py` | Collect decision SFT + produce gap report | `output/decision_sft/` |
| `build_shared_bank.py` | Lift 406 skills → 354 SharedAbstractSkill mega-skills | `output/shared_skill_bank/` |
| `build_web_skill_banks.py` | Lift miniwob (45 archetypes) + webshop (3 archetypes) | `output/per_task_banks/{miniwob,webshop}/` |
| `bind_and_validate.py` | Forward-bind mega-skills → per-task via harness + crafter | `output/shared_skill_bank/by_task/` |
| `test_game_to_nongame_transfer.py` | Harness structural validation: game → non-game transfer matrix | `output/transfer_matrix.json` |
| `build_reasoning_aligned_bank.py` | Reasoning-intent normalizer + cross-domain plan matcher | `output/reasoning_aligned_mega_skills.json` |

### Output directory layout

```
frontier_data/
├── scripts/
│   ├── run_full_pipeline.sh           ← 9-stage master orchestrator
│   ├── collect_all_per_task_banks.py  ← step 1: collect all native skills
│   ├── collect_decision_sft.py        ← step 2: collect decision SFT + gap report
│   ├── build_shared_bank.py           ← step 3: build SharedAbstractBank
│   └── bind_and_validate.py           ← step 4: forward-bind + crafter
├── output/
│   ├── per_task_banks/                ← 406 skills across 18 tasks
│   │   ├── <task>/skill_bank.jsonl
│   │   └── MANIFEST.json
│   ├── decision_sft/                  ← 12/18 skill-banked tasks with SFT data
│   │   ├── <task>/{action_taking,skill_selection}.jsonl
│   │   ├── MANIFEST.json
│   │   └── GAP_REPORT.json            ← 6 non-game gaps documented
│   ├── shared_skill_bank/             ← TwoLayerSkillStore
│   │   ├── abstract.jsonl             ← 354 mega-skills
│   │   ├── by_task/<task>/bindings.jsonl
│   │   └── SUMMARY.json
│   ├── layer_c_templates/             ← GPT-5.4 Layer-C procedural templates
│   │   ├── <cohort>/<task>/template_bank.jsonl  ← 406 templates (8-op vocabulary)
│   │   └── _lift_summary.json         ← lift run metadata
│   ├── bind_reports/                  ← binding audit trail
│   │   └── bind_report_*.json
│   ├── transfer_matrix.json           ← game→non-game harness validation (84 pairs)
│   └── reasoning_aligned_mega_skills.json  ← 9 cross-domain reasoning plans (Layer-C)
└── README.md                          ← this file
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
| `cross-domain` | Seeds from all 12 game tasks (Layer-C matched) | 9 cross-domain reasoning plans → target |
| `same-domain-only` | Seeds from same-domain tasks only | VR→VR or Web→Web, no game skills |
| `no-seed` | Empty bank | Cold start |

Metric: reward delta between `cross-domain` and `same-domain-only`.
Claim: game-domain reasoning plans (PERCEIVE→DECIDE→COMMIT→VERIFY etc.)
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

After re-lift, **9 reasoning plans are shared across ≥ 2 domains**,
covering **221 of 406 skills (54.4%)**:

| Reasoning plan | Domains | Skills | Example |
|---|---|---:|---|
| **PERCEIVE→DECIDE→COMMIT→VERIFY** | GAME+WEB | **166** | mario navigation + miniwob focus/resize |
| **PERCEIVE→COMPARE→FILTER→DECIDE→COMMIT** | GAME+VR | **11** | tetris optimize + siv_bench emotion inference |
| **PERCEIVE→DECIDE→COMMIT→COMMIT** | GAME+WEB | **10** | positioning + drag/draw/bisect |
| **PERCEIVE→COMPARE→DECIDE→COMMIT→VERIFY** | GAME+WEB | **9** | tetris evade + tic-tac-toe |
| **PERCEIVE→COMMIT→COMMIT→VERIFY** | GAME+WEB | **8** | charge attack + copy paste |
| **PERCEIVE→COMPARE→DECIDE→VERIFY** | GAME+VR | **7** | columns labeling + VR symbol/math/emotion |
| **PERCEIVE→DECIDE→COMMIT→COMMIT→VERIFY** | GAME+WEB | **4** | dodge-and-strike + text styling |
| **PERCEIVE→COMPARE→COMMIT→VERIFY** | GAME+WEB | **4** | scene scan + ascending order |
| **PERCEIVE→COMMIT→COMMIT** | GAME+WEB | **2** | hazard dodge + circle center |

**Key improvement over name-based clustering:**
- Before (name-based): 14 multi-task mega-skills, **0 cross-domain**, 0 skills bridging game↔non-game
- After (Layer-C reasoning plans): **9 cross-domain plans**, **221 skills** bridging game↔non-game
  - GAME↔WEB: 7 shared plans
  - GAME↔VR: 2 shared plans
  - GPT-5.4 also annotated `transferable_to_cohorts` per skill

**Notable cross-domain example — "Perceive → Compare → Filter → Decide → Commit":**

| Domain | Task | Skill | What the reasoning plan does |
|---|---|---|---|
| GAME | tetris | COMMIT/OPTIMIZE | Assess board → evaluate sequences → discard blocked → pick best → place |
| GAME | tetris | skill-903e63c5e3 | Assess state → evaluate actions → discard risky → pick stable → execute |
| VR | siv_bench | Emotion inference | Observe cues → match against emotions → eliminate inconsistent → decide → commit |
| VR | video_holmes | IMC inference | Observe behavior → relate to states → discard mismatches → select → commit |

Scripts:
- `build_reasoning_aligned_bank.py` — offline reasoning-intent normalizer
- `test_game_to_nongame_transfer.py` — harness validation of game→non-game

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

### Gap 8: Cross-domain reasoning alignment — ✅ RESOLVED (Layer-C re-lift)

~~Only 3 reasoning plans spanned ≥ 2 domains (9 skills).~~
After Layer-C re-lift (§8c–8d): **9 cross-domain reasoning plans** covering
**221 of 406 skills (54.4%)**. GAME↔WEB: 7 plans, GAME↔VR: 2 plans.
Remaining gap: no WEB↔VR plans yet (web skills tend toward COMMIT-heavy
signatures while VR uses COMPARE-heavy).

**Next step:** re-cluster shared bank by Layer-C signatures to create
cross-domain mega-skills (replace name-based clustering).

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
