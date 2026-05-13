# Frontier Data Pipeline Guide

Complete walkthrough of skill extraction, mega-skill construction, clustering,
and cross-domain skill transfer.

**4 frontier teachers:** GPT-5.4, Claude Sonnet 4.5, Gemini 3.1 Pro, Qwen3-VL-235B-A22B

**Coverage:** 18 tasks (8 gymv games + 4 env_wrapper games + 2 web + 4 VR), 406 native skills

---

## 0. Data Provenance & Training Strategy

### Games (12 tasks) — frontier teacher distillation

Games use frontier model outputs for SFT. This is safe because game
environments are **interactive** — no model can "memorize" the correct action
at frame N. Every rollout is generated fresh against a live emulator.

| Data | Source | Safe? |
|------|--------|:-----:|
| SFT (action_taking, skill_selection) | GPT-5.4 + Claude-4.6 + Gemini-3.1 + Qwen3-VL-235B | ✅ |
| Skill banks | GPT-5.4 extraction from teacher rollouts | ✅ |
| GRPO training rollouts | Qwen (self-play) | ✅ |

Teacher coverage: gym_v has all 4 teachers; env_wrapper has GPT-5.4 only.

```
Game pipeline: Multi-teacher SFT → GRPO → Phase 1 (6 source) → Phase 2 (6 hold-out)
```

### Non-game tasks (6 tasks) — self-rollout exemplars, no teacher answers

Non-game benchmarks (Video Holmes, TIR-bench, etc.) are **static Q&A
datasets**. Frontier models may have been trained on these benchmarks, so
their reasoning traces could be post-hoc rationalization of memorized answers.

| Data | Source | Safe? | Notes |
|------|--------|:-----:|-------|
| Seed skills (Layer-C templates) | GPT-5.4 | ✅ | Abstract skeletons, no answers |
| SFT warmup | Game-SFT checkpoint | ✅ | Teaches format, not benchmark answers |
| ICL exemplars (success/fail) | **Qwen (self-rollout)** | ✅ | Must NOT come from GPT/Gemini |
| GRPO rollouts | Qwen (self) | ✅ | |

```
Non-game pipeline:
  Game-SFT'd Qwen → self-rollout on 200 train split → collect success/fail
  → ICL exemplars in prompt → GRPO iterations → eval on 800 held-out
```

**Train/eval split:** 200 train / ~800 eval per task, fixed seed. All
exemplars and training signals come exclusively from the train split.

See `PLAN_FEW_SHOT_SKILL_BANK.md` for the full non-game training plan.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  9-STAGE MASTER PIPELINE                        │
│               (run_full_pipeline.sh)                            │
└─────────────────────────────────────────────────────────────────┘

Stage 1: SKILL EXTRACTION (per-task)
 ┌──────────────────────────────────────────────────────────┐
 │ extract_skillbank_gymv_gpt54.py  → 13 gymv games        │ ← GPT-5.4
 │ extract_skillbank_gpt54.py      → 4 env_wrapper games   │
 │ build_skillbank_qa_gpt54.py     → web + VR tasks        │
 │ build_web_skill_banks.py        → miniwob / webshop     │ ← no LLM
 └───────────────────────┬──────────────────────────────────┘
                         │
    collect_all_per_task_banks.py
    → output/per_task_banks/ (406 skills, 18 tasks)
                         │
                         ▼
Stage 2-3: DECISION SFT
 ┌──────────────────────────────────────────────────────────┐
 │ label_skill_actions → build_decision_sft_jsonl.py       │ ← GPT-5.4
 │ collect_decision_sft.py → output/decision_sft/          │
 └───────────────────────┬──────────────────────────────────┘
                         │
                         ▼
Stage 4: LAYER-C TEMPLATE LIFT
 ┌──────────────────────────────────────────────────────────┐
 │ lift_skill_templates_gpt54.py                           │ ← GPT-5.4
 │ → output/layer_c_templates/ (406 templates, 8-op)       │
 └───────────────────────┬──────────────────────────────────┘
                         │
                         ▼
Stage 5: BUILD SHARED BANK (mega-skills)
 ┌──────────────────────────────────────────────────────────┐
 │ build_plan_clustered_bank.py  [DEFAULT]                 │ ← uses judge results
 │ 406 skills → plan-level judge clusters → mega-skills    │
 │ → output/shared_skill_bank/abstract.jsonl               │
 └───────────────────────┬──────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
Stage 6: BACKWARD             Stage 7: FORWARD
(discover → shared bank)      (bind → per-task)
 discover_skill_to_            bind_abstract_to_task.py
 shared_bank.py                bind_and_validate.py
                         │
                         ▼
Stage 8: CRAFTER V2 REFINEMENT
 crafter_v2_batch_pipeline.py → failure detection → skill proposal → dedup

Stage 9: INVENTORY BUILD
 build_inventory.py → rebuild SFT data inventory
```

---

## 1. Skill Extraction (Stage 1-2)

### Entry Scripts

| Script | Target | LLM needed |
|---|---|:---:|
| `labeling/extract_skillbank_gymv_gpt54.py` | 13 gym_v ROM games | ✅ GPT-5.4 |
| `labeling/extract_skillbank_gpt54.py` | 4 env_wrapper games | ✅ GPT-5.4 |
| `labeling/build_skillbank_qa_gpt54.py` | Non-game tasks (miniwob, webshop, VR) | ✅ GPT-5.4 |
| `scripts/build_web_skill_banks.py` | miniwob (45 archetypes) + webshop (3 archetypes) | ❌ |

### How It Works

1. GPT-5.4 acts as a SkillBankAgent over frontier-model cold-start rollouts,
   extracting per-task concrete skills.
2. Each skill record contains:
   - `skill_id` — unique identifier
   - `strategic_description` — what the skill does at a strategic level
   - `contract` — preconditions and postconditions
   - `protocol` — ordered execution steps
3. Output: one `skill_bank.jsonl` per task in `output/per_task_banks/<task>/`

### Web Task Special Handling

`build_web_skill_banks.py` uses **archetype aggregation** instead of per-episode skills:
- **miniwob**: clusters by task-name prefix (click → 31 members, email → 10, drag → 9 ...)
- **webshop**: clusters by step-count bucket (short ≤5 → 13, medium 6-12 → 12, long 13+ → 25)

### Current Scale

| Category | Tasks | Skills | SFT rows |
|---|---:|---:|---:|
| gym_v | 8 | 255 | 16,000 |
| env_wrapper | 4 | 73 | 6,086 |
| Web | 2 | 48 | — |
| VR | 4 | 30 | — |
| **Total** | **18** | **406** | **22,086** |

---

## 2. Mega-Skill Extraction

Two complementary methods build mega-skills.

### Method A: Name-Based Clustering → TwoLayerSkillStore

**Script:** `scripts/build_shared_bank.py` (Stage 5)

**Core logic:**

```
Phase 1: Cluster by normalised skill_id stem
  stem = normalise_skill_id(sid)   # strip version/decoration, extract core name
  clusters[stem].append((task, raw))

Phase 2: Lift each cluster → one SharedAbstractSkill (mega-skill)
  sig = infer_template_signature(rep)   # infer from contract/protocol/name

Phase 3: Write TwoLayerSkillStore
  abstract.jsonl            → 354 mega-skills
  by_task/*/bindings.jsonl  → per-task concrete bindings
```

**Signature inference priority:**
1. Existing `template_signature` field (contains `→`)
2. `protocol_steps` / `template_steps` op fields
3. Skill-name verbs → predefined signature map (e.g. ATTACK → `PERCEIVE → DECIDE → COMMIT`)
4. Final fallback: `PERCEIVE → DECIDE → COMMIT`

**Result:** 406 skills → **354 mega-skills**, of which **14 span ≥ 2 tasks**

### Method B: Bottom-Up LLM Classification

**Phase 1 — Per-skill classification** (`scripts/extract_mega_skills.py`)

Sends each skill's Layer-C template steps + description to `gpt-4.1-mini`,
which classifies it into one of **18 fixed cognitive families**:

| # | Family | Core cognitive procedure |
|---|---|---|
| 1 | COMPARE_AND_RANK | Perceive options → rank by criterion → select best |
| 2 | NAVIGATE_AND_REACH | Plan path → move to target → arrive |
| 3 | DODGE_AND_SURVIVE | Monitor threats → decide evasive action → execute |
| 4 | ENGAGE_AND_DEFEAT | Identify target → approach → attack → verify |
| 5 | COLLECT_AND_ACCUMULATE | Find collectible → acquire → track count |
| 6 | SEQUENCE_AND_COMPLETE | Break into ordered sub-steps → execute each → verify |
| 7 | RECALL_MATCH_AND_SELECT | Retrieve criteria → perceive options → match → select |
| 8 | FILTER_AND_NARROW | Apply criteria → eliminate non-matching → act on rest |
| 9 | INPUT_AND_SUBMIT | Perceive form → enter value → submit → verify |
| 10 | COUNT_AND_REPORT | Identify target attribute → count/aggregate → output |
| 11 | POSITION_AND_PLACE | Perceive target location → arrange item → verify |
| 12 | TIME_AND_REACT | Wait for trigger → execute precisely timed action |
| 13 | TRANSFORM_AND_VERIFY | Apply transformation → verify new state matches goal |
| 14 | EXPLORE_AND_DISCOVER | Probe unknown → observe result → update knowledge |
| 15 | MONITOR_AND_SUSTAIN | Track process → maintain/adjust → ensure completion |
| 16 | INFER_AND_DECIDE | Perceive evidence → reason about causes → select action |
| 17 | RETRIEVE_AND_EXECUTE | Recall known procedure → execute → confirm |
| 18 | EVALUATE_AND_OPTIMIZE | Assess quality → try improvement → compare |

- Complexity: O(n) = 406 LLM calls, 10-thread parallel, ~40 seconds
- Output: `output/mega_skill_labels.json`

**Phase 2 — Label merging** (`scripts/cluster_mega_skills.py`)

If Phase 1 produces more than 18 raw labels, an LLM merges semantically
equivalent labels down to 15-20 canonical families.

- Output: `output/mega_skill_clusters.json`

**Result:** 18 mega-skill families — 5 three-way (GAME+WEB+VR), 5 two-way

---

## 3. Clustering

The pipeline's **default** clustering method is **plan-level LLM-as-judge**
(`build_plan_clustered_bank.py`), which groups skills by whether an LLM
judge determines they share the same reasoning procedure. Two alternative
methods are available as fallbacks or for analysis:

| Method | Script | Clusters by | Cross-domain coverage | Default? |
|---|---|---|---|:---:|
| **Plan-level LLM judge** | `build_plan_clustered_bank.py` | Shared reasoning procedure (judge score ≥ 4) | **100%** | **✅** |
| Name clustering | `build_shared_bank.py` | normalised `skill_id_stem` | 0% | fallback |
| Structural signature | `build_reasoning_aligned_bank.py` | Layer-C collapsed 5-op signature | 77.3% | analysis |

### 3a. Default: Plan-Level LLM Judge Clustering (`build_plan_clustered_bank.py`)

The default clustering method works as follows:

1. **Load judge results** — reads `plan_level_similarity_judgments.json`
   (108 batch LLM queries, 377 high-confidence matches) and optionally
   `plan_similarity_judgments.json` (72 pairwise judgments)

2. **Build similarity graph** — creates an edge between two skills if the
   plan-level judge scored them ≥ threshold (default 4, meaning "same
   transferable cognitive procedure")

3. **Union-Find connected components** — skills linked by judge edges form
   a single mega-skill cluster. Transitive: if A↔B and B↔C, then {A,B,C}
   are one cluster

4. **Fallback for orphans** — skills with no judge edges (e.g. game-only
   skills never compared to non-game) are grouped by collapsed signature
   or name stem, so nothing is orphaned

5. **Emit TwoLayerSkillStore** — same output format as name-based clustering:
   `abstract.jsonl` + `by_task/<task>/bindings.jsonl`

```bash
# Default (used by run_full_pipeline.sh Stage 5)
python frontier_data/scripts/build_plan_clustered_bank.py

# Lower threshold to include moderate matches
python frontier_data/scripts/build_plan_clustered_bank.py --threshold 3

# Dry run — show clusters without writing
python frontier_data/scripts/build_plan_clustered_bank.py --dry-run
```

To switch back to name-based clustering:

```bash
CLUSTER_METHOD=name bash frontier_data/scripts/run_full_pipeline.sh
```

### 3b. Reasoning-Intent Normalisation (`build_reasoning_aligned_bank.py`)

Unifies three domains' disparate verb vocabularies into 7 domain-agnostic
reasoning intents:

| Intent | Meaning | Game verbs | Web verbs | VR verbs |
|---|---|---|---|---|
| PERCEIVE | Observe / scan current state | INSPECT, SCAN, LOOK | PERCEIVE | OBSERVE, INSPECT |
| RECALL | Retrieve prior knowledge | RECALL, REMEMBER | RECALL | RECALL |
| EVALUATE | Compare / assess / reason | COMPARE, FILTER, ASSESS | EVALUATE, COMPARE | EVALUATE, COMPARE |
| DECIDE | Select among alternatives | DECIDE, SELECT, CHOOSE | DECIDE, SELECT | DECIDE, CHOOSE |
| NAVIGATE | Move to a target | MOVE, APPROACH | NAVIGATE, SCROLL | NAVIGATE |
| ACT | Execute a concrete action | EXEC, ATTACK, COMMIT | COMMIT, PLACE | EXECUTE, COMMIT |
| VERIFY | Confirm outcome | VERIFY, CHECK, KEEP | VERIFY, CONFIRM | VERIFY, CHECK |

**Classification priority:** direct op mapping → evidence_role mapping → notes keyword match → default ACT

**Plan compression:** removes consecutive repeated intents:

```
[PERCEIVE, PERCEIVE, EVALUATE, ACT, ACT, VERIFY]
→ [PERCEIVE, EVALUATE, ACT, VERIFY]
```

### 3c. Layer-C 8-op → Collapsed 5-op

The 8 Layer-C operators are collapsed into 5 semantic equivalence classes:

| Collapse rule | Rationale |
|---|---|
| COMPARE + FILTER → **EVALUATE** | Both assess / evaluate perceived state |
| COMMIT + VERIFY + RECOVER → **ACT** | All are execution / action steps |
| PERCEIVE, DECIDE, RECALL | Kept distinct |

This lifts coverage from 54.4% → **77.3%** (314/406 skills) and unlocks the
first **three-way (GAME+WEB+VR) plan**:

| Collapsed plan | Domains | Skills |
|---|---|---:|
| PERCEIVE → DECIDE → ACT | GAME+WEB | 201 |
| **PERCEIVE → EVALUATE → DECIDE → ACT** | **GAME+WEB+VR** ★ | **55** |
| PERCEIVE → ACT | GAME+WEB | 21 |
| RECALL → PERCEIVE → EVALUATE → DECIDE → ACT | VR+WEB | 11 |
| PERCEIVE → RECALL → EVALUATE → DECIDE → ACT | GAME+VR | 10 |
| PERCEIVE → EVALUATE → ACT | GAME+WEB | 8 |
| RECALL → PERCEIVE → DECIDE → ACT | VR+WEB | 8 |

### 3d. LLM-as-Judge Validation

**Signature-level** (`judge_plan_similarity.py`):
- 72 cross-domain pairs, GPT-4.1-mini scores 1-5
- Result: 52/72 (72%) judged as same cognitive procedure

**Plan-level** (`judge_plan_level_similarity.py`):
- Compares full reasoning-plan text (not limited to same signature)
- 108 batch LLM calls
- Discovered **310 new** high-confidence cross-domain pairs that signature
  matching completely missed
- All 78 non-game skills reached 100% cross-domain coverage

---

## 4. Skill Transfer

### 4a. Layer-C Template Lift (Stage 4)

`scripts/lift_skill_templates_gpt54.py`:
- GPT-5.4 lifts all 406 skills into **modality-agnostic procedural templates**
- Uses 8 controlled reasoning operators:
  `{PERCEIVE, RECALL, COMPARE, FILTER, DECIDE, COMMIT, VERIFY, RECOVER}`
- Output: `output/layer_c_templates/<cohort>/<task>/template_bank.jsonl`

### 4b. Forward Binding — Mega-Skills to New Tasks (Stage 7)

**Script:** `scripts/bind_and_validate.py`

```
SharedAbstractSkill ──LLM re-ground to target vocab──▶
  BoundConcreteSkill (status=PENDING)
    ──harness FewShotAdapter validate──▶
      status=VALIDATED, sub_episodes appended
```

Two modes:

| Mode | Command | Behaviour |
|---|---|---|
| **Offline** | `--offline` | Heuristic mapping, creates PENDING bindings |
| **LLM-driven** | `--model gpt-5.4` | Semantic re-grounding + harness validation |

### 4c. Backward Discovery — New Tasks to Shared Bank (Stage 6)

```
New skill mined in task X ──LLM lift to modality-agnostic skeleton──▶
  Upsert into SharedAbstractBank
    (new record, or new lineage entry on existing abstract)
```

### 4d. V2 Transfer Approach — "Teacher-First Bootstrap + GRPO"

This is the **currently adopted strategy** (replaces V1's GPT-binding + Crafter):

```
Phase 0: Games (12 tasks)
         Multi-teacher SFT (GPT/Claude/Gemini/Qwen3-VL) → GRPO
         → Game-SFT'd Qwen checkpoint (warm-start for non-game tasks)
         → Layer-C templates distilled from game skill banks
         ↓
Phase 0: Non-game tasks (6 tasks)
         Train/Eval split: 200 / ~800 (fixed seed)
         ↓
         Teacher (GPT/Gemini) rollout on 200 train → teacher demonstrations
         ↓
         Assign teacher demos to archetypes → initial skill bank
         (each skill has teacher exemplar + Layer-C protocol + step_checks)
         ↓
Phase 1: First GRPO iteration
         Game-SFT'd Qwen + skill bank with teacher exemplars → rollout
         Collect Qwen self-traces → replace teacher exemplars where valid
         ↓
Phase 2+: Iterative GRPO refinement
         Each iteration: rollout → diagnose → update bank → GRPO
         Teacher exemplars gradually replaced by Qwen self-traces
         Statistics-driven: enrich/demote/retire (zero online LLM calls)
         ↓
Eval:    800 held-out cases
         Ablation: teacher-bootstrapped vs self-only exemplars
```

**Key insight:** instead of GPT-5.4 binding + Crafter (unreliable LLM
generation), use **teacher demonstrations as initial ICL exemplars**
that get naturally replaced by Qwen's own traces through GRPO iterations.
No new skill text is generated online — only real rollout traces enter the bank.

See `PLAN_FEW_SHOT_SKILL_BANK.md` for full details.

### 4e. Protocol Injection (Runtime-Ready)

**Script:** `scripts/inject_layerc_protocols.py`

Converts Layer-C templates into runtime protocol dicts and patches per-task
skill banks in place:
- 304 empty/thin protocols → 3-5 step reasoning plans
- 102 already-rich protocols → preserved, enriched with step_checks

What the agent sees at runtime:

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

---

## 5. How to Run

### Full pipeline (all 9 stages)

```bash
bash frontier_data/scripts/run_full_pipeline.sh
```

### Resume from a specific stage

```bash
STAGE=5 bash frontier_data/scripts/run_full_pipeline.sh
```

### Dry run (print commands only)

```bash
DRY_RUN=1 bash frontier_data/scripts/run_full_pipeline.sh
```

### Run individual steps

```bash
# 1. Collect all per-task skill banks
python frontier_data/scripts/collect_all_per_task_banks.py

# 2. Run LLM-as-judge (prerequisite for default clustering)
python frontier_data/scripts/judge_plan_level_similarity.py      # plan-level (default)
python frontier_data/scripts/judge_plan_similarity.py            # signature-level (supplemental)

# 3. Build shared mega-skill bank (DEFAULT: plan-level judge clustering)
python frontier_data/scripts/build_plan_clustered_bank.py

# 3-alt. Alternative clustering methods
python frontier_data/scripts/build_shared_bank.py                # name-based (fallback)
python frontier_data/scripts/build_reasoning_aligned_bank.py     # structural signature (analysis)
python frontier_data/scripts/extract_mega_skills.py              # bottom-up LLM classification
python frontier_data/scripts/cluster_mega_skills.py              # label merging

# 4. Forward binding (offline / LLM)
python frontier_data/scripts/bind_and_validate.py --offline
python frontier_data/scripts/bind_and_validate.py --model gpt-5.4

# 7. Inject Layer-C protocols into runtime skill banks
python frontier_data/scripts/inject_layerc_protocols.py

# 8. Game → non-game transfer matrix validation
python frontier_data/scripts/test_game_to_nongame_transfer.py
```

---

## 6. Output Directory Layout

```
frontier_data/
├── scripts/                               ← 14 scripts
│   ├── run_full_pipeline.sh               ← 9-stage master orchestrator
│   ├── collect_all_per_task_banks.py
│   ├── collect_decision_sft.py
│   ├── build_plan_clustered_bank.py       ← DEFAULT: plan-judge clustering → mega-skills
│   ├── build_shared_bank.py               ← fallback: name clustering → mega-skills
│   ├── build_web_skill_banks.py           ← miniwob/webshop archetypes
│   ├── bind_and_validate.py               ← forward binding + crafter
│   ├── build_reasoning_aligned_bank.py    ← reasoning-intent normalisation
│   ├── test_game_to_nongame_transfer.py   ← transfer matrix validation
│   ├── extract_mega_skills.py             ← per-skill LLM classification
│   ├── cluster_mega_skills.py             ← label merging
│   ├── judge_plan_similarity.py           ← signature-level judge
│   ├── judge_plan_level_similarity.py     ← plan-level judge (feeds default clustering)
│   └── inject_layerc_protocols.py         ← Layer-C → runtime protocol
│
├── output/
│   ├── per_task_banks/                    ← 406 skills across 18 tasks
│   │   ├── <task>/skill_bank.jsonl
│   │   └── MANIFEST.json
│   ├── shared_skill_bank/                 ← TwoLayerSkillStore
│   │   ├── abstract.jsonl                 ← 354 mega-skills
│   │   ├── by_task/<task>/bindings.jsonl
│   │   └── SUMMARY.json
│   ├── layer_c_templates/                 ← GPT-5.4 Layer-C templates
│   │   ├── <cohort>/<task>/template_bank.jsonl
│   │   └── _lift_summary.json
│   ├── decision_sft/                      ← 12/18 tasks with SFT data
│   │   ├── <task>/{action_taking,skill_selection}.jsonl
│   │   ├── MANIFEST.json
│   │   └── GAP_REPORT.json
│   ├── bind_reports/                      ← binding audit trail
│   ├── transfer_matrix.json               ← game → non-game validation
│   ├── reasoning_aligned_mega_skills.json ← cross-domain reasoning plans
│   ├── mega_skill_labels.json             ← per-skill classification
│   ├── mega_skill_clusters.json           ← merged families
│   ├── plan_similarity_judgments.json     ← signature-level judge results
│   └── plan_level_similarity_judgments.json ← plan-level judge results
│
├── README.md                              ← full technical documentation
├── PIPELINE_GUIDE.md                      ← this file
└── PLAN_FEW_SHOT_SKILL_BANK.md           ← non-game training plan (data provenance + exemplars)
```

---

## 7. Training Plan Integration

### Game Training (Phase 1 + Phase 2)

| Phase | Tasks | Bank mode | Data source | Purpose |
|---|---|---|---|---|
| Phase 1 | ThunderForceIII, AlteredBeast, Columns, DynamiteHeaddy, candy_crush, tetris | `per_game` | Frontier teacher SFT + GRPO | Mine concrete skills; populate per-task banks |
| Phase 2 | SpaceHarrierII, StreetsOfRage2, Airstriker, Strider, twenty_forty_eight, super_mario | `shared` | Frontier teacher SFT + GRPO | Transfer: mega-skill skeletons to held-out games |

Key configuration:
- `BANK_MODE=shared` — single SharedAbstractBank across all phases
- `TRANSLATE_ON_BOUNDARY=1` — re-ground skills at phase transitions
- `feasible_tasks` on each skill — runtime eligibility filter
- SFT data from frontier teachers is standard distillation (safe for games)

### Non-Game Training (Phase 3)

| Phase | Tasks | Bank mode | Data source | Purpose |
|---|---|---|---|---|
| Phase 3 | miniwob, webshop, tir_bench, visual_toolbench, siv_bench, video_holmes | `shared` + `exemplar` | Seed skills from GPT + **self-rollout ICL exemplars** + GRPO | OOD: reasoning patterns cross domain boundaries |

Phase 3 uses a different pipeline from games (see `PLAN_FEW_SHOT_SKILL_BANK.md`):

```
Game-SFT checkpoint (from Phase 1+2)
  → Self-rollout on 200 train samples per task (Qwen only, no frontier model)
  → Collect success/fail reasoning traces as ICL exemplars
  → Seed skill templates (Layer-C) from shared bank (abstract, no answers)
  → GRPO iterations with exemplar-enriched skill prompts
  → Bank update between iterations (enrich/demote/retire based on statistics)
  → Eval on 800 held-out samples
```

**Critical boundary:** ICL exemplars come from Qwen's own rollouts, not from
frontier models. This avoids benchmark contamination on static Q&A datasets.

### Ablation Experiment Design

| Experiment | Tests | Expected positive result |
|---|---|---|
| **A1: seed-bank vs no-bank** | Does the bank help at all? | seed-bank@5 steps ≥ no-bank@15 steps on ≥ 4/6 games |
| **A2: seed-bank vs raw-SFT** | Does structure matter (same data)? | Structured bank outperforms flat SFT |
| **A3: cross-domain vs same-domain seeds** | Does cross-domain transfer work? | Game → web/VR seeds improve reward |
| **A4: self-exemplar vs no-exemplar** | Do ICL exemplars help non-game tasks? | Exemplar-enriched GRPO > bare GRPO on non-game eval |
| **A5: game-SFT warmup vs raw model** | Does game SFT help non-game? | Game-SFT'd Qwen > raw Qwen on first non-game rollout |

---

## 8. Current Status

| Status | Detail |
|---|---|
| ✅ Done | 406 skills extracted, 354 mega-skills, Layer-C template lift (406/406) |
| ✅ Done | Protocol injection, cross-domain plan analysis (77.3% collapsed / 100% plan-level) |
| ✅ Done | 12 mega-skill families (plan-judge), 18 families (bottom-up) |
| ✅ Done | Game SFT data: 12 games, 22,086 SFT rows from frontier teachers |
| ✅ Done | Data provenance policy: games = teacher distillation, non-game = self-rollout ICL |
| ⏳ Pending | Train/eval split for 6 non-game tasks (200/800, fixed seed) |
| ⏳ Pending | Self-rollout collection: run Qwen on train split to gather success/fail exemplars |
| ⏳ Pending | Exemplar-enriched skill bank construction for non-game tasks |
| ⏳ Pending | A1-A5 ablation experiments (A1-A3 games, A4-A5 non-game) |
| ⏳ Pending | env_wrapper games missing Claude / Gemini teacher data |

**Bottom line:** the pipeline has two distinct training paths:
- **Games**: standard frontier-teacher SFT → GRPO (data-ready, no leakage concerns)
- **Non-game**: seed skills + game-SFT warmup → self-rollout ICL exemplars → GRPO (clean boundary, no benchmark contamination)

The contribution is that structured skill banks + self-rollout exemplars
enable a 9B model to transfer reasoning across domains using only ~200 cases
per target task — without relying on frontier model answers for the target benchmarks.
