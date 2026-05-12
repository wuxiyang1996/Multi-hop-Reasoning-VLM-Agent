# Frontier Data — SFT Inventory & Shared Skill Bank Pipeline

Full skill extraction, shared bank construction, and cross-task binding
pipeline built on top of the `emnlp2026_download/workspace/` archive.

**4 frontier teachers:** GPT-5.4, Claude Sonnet 4.5, Gemini 3.1 Pro,
Qwen3-VL-235B-A22B.

> **Excluded from SFT:** assistantbench, osworld (per-episode traces),
> and 5 zero-reward gymv games (CastleOfIllusion, CastlevaniaBloodlines,
> GoldenAxe, KidChameleon, MortalKombatII — 0 reward across all 4 teachers).

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
│   └── bind_reports/                  ← binding audit trail
│       └── bind_report_*.json
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

| Phase | Games | Bank mode | What happens |
|---|---|---|---|
| Phase 1 | ThunderForceIII, AlteredBeast, Columns, DynamiteHeaddy, candy_crush, tetris | `per_game` | Mine concrete skills; populate per-task banks |
| Phase 2 | SpaceHarrierII, StreetsOfRage2, Airstriker, Strider | `shared` | Transfer: mega-skill skeletons to held-out games |
| Phase 3 | miniwob, webshop, tir_bench, visual_toolbench, siv_bench, video_holmes | `shared` | OOD: reasoning patterns cross domain boundaries |

Key config:
- `BANK_MODE=shared` — one SharedAbstractBank across all phases
- `TRANSLATE_ON_BOUNDARY=1` — re-ground skills at phase transition
- `feasible_tasks` on each skill — runtime eligibility veto via `EligibilityFilter`

---

## 8. Known Gaps & Next Steps

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

### Gap 5: Template signature coverage

84.6 % of abstracts use the fallback `PERCEIVE → DECIDE → COMMIT` signature
because raw per-task skill records lack explicit `template_signature` or
`protocol_steps`. Running `scripts/lift_skill_templates_gpt54.py` with
GPT-5.4 would produce richer Layer-C templates and improve shared bank
signature diversity.

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

### Excluded tasks

- **assistantbench** — 181 GPT + multi-model test episodes; per-episode
  traces only (`n_instances=1`, no archetype aggregation).
- **osworld** — 30 per-episode traces, same issue.
- **CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe, KidChameleon,
  MortalKombatII** — 0 reward across all 4 frontier teachers (env returns
  no reward signal). 161 skills + 10,000 SFT rows removed.

Rollout data for all excluded tasks remains available in
`emnlp2026_download/` if re-processed in the future.
