# Frontier SFT Data Inventory

Status snapshot from the `emnlp2026_download/workspace/` archive.

**18 tasks** (12 games + 6 non-game) × **4 frontier teachers**
(GPT-5.4, Claude Sonnet 4.5, Gemini 3.1 Pro, Qwen3-VL-235B-A22B).

---

## Source Layout in `emnlp2026_download/workspace/main_project/`

| Data type | Path pattern |
|---|---|
| GPT game rollouts | `Cold-start-out-gymv/{batch}/` |
| Claude / Gemini / Qwen game rollouts | `openrouter-baselines-out/{batch}/{teacher}/gymv/` |
| GPT miniwob rollouts | `Cold-start-out-browsergym/miniwob.*` |
| Claude / Gemini / Qwen miniwob rollouts | `openrouter-transfer-baselines-out/2026-05-01_08-06-44/{teacher}/browsergym/miniwob.*` |
| GPT webshop rollouts | `Cold-start-out-browsergym/webshop_50task_low/` |
| Claude / Gemini / Qwen webshop rollouts | `Cold-start-out-browsergym/webshop_50task_{teacher}/` |
| GPT visual-reasoning (image) | `Cold-start-out-visual-reasoning/{tir_bench,visual_toolbench}/` |
| GPT visual-reasoning (video) | `Cold-start-out-visual-reasoning-video/{siv_bench,video_holmes}/` |
| Claude / Gemini / Qwen VR (image) | `openrouter-transfer-baselines-out/2026-05-01_08-06-44/{teacher}/vr_image/{bench}/` |
| Claude / Gemini / Qwen VR (video) | `openrouter-transfer-baselines-out/2026-05-01_08-06-44/{teacher}/vr_video/{bench}/` |
| Skill Bank SFT (per-game) | `skill_bank_sft/Temporal_*/` |
| Decision SFT JSONL | `labeling/decision_sft_jsonl/run_20260430_082516/{task}/` |
| Unified skill index | `skill_bank_sft/_unified/` |
| SFT training logs | `runs/sft_coldstart*` |

---

## Games (12) — Teacher Coverage

### gym_v (8 core + 5 extended = 13 in rollouts, 12 in SFT inventory)

| Task | GPT-5.4 | Claude | Gemini | Qwen | action_taking | skill_selection | Bank skills |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|
| Temporal_Airstriker-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 31 |
| Temporal_AlteredBeast-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 21 |
| Temporal_CastleOfIllusion-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | — |
| Temporal_CastlevaniaBloodlines-v0 | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Temporal_Columns-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 27 |
| Temporal_DynamiteHeaddy-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 32 |
| Temporal_GoldenAxe-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | — |
| Temporal_KidChameleon-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | — |
| Temporal_MortalKombatII-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | — |
| Temporal_SpaceHarrierII-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 33 |
| Temporal_StreetsOfRage2-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 34 |
| Temporal_Strider-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 45 |
| Temporal_ThunderForceIII-v0 | ✅ | ✅ | ✅ | ✅ | 2,000 | 2,000 | 32 |

### env_wrappers (4 games — GPT-5.4 only for rollouts)

| Task | GPT-5.4 | Claude | Gemini | Qwen | action_taking | skill_selection | Bank skills |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|
| tetris | ✅ | ❌ | ❌ | ❌ | 1,573 | 1,573 | 12 |
| super_mario | ✅ | ❌ | ❌ | ❌ | 326 | 326 | 7 |
| candy_crush | ✅ | ❌ | ❌ | ❌ | 1,000 | 1,000 | 3 |
| twenty_forty_eight | ✅ | ❌ | ❌ | ❌ | 3,187 | 3,187 | 8 |

---

## Non-game (6) — All 4 Teachers ✅

### miniwob (BrowserGym)

| Teacher | Source | Episodes |
|---|---|---:|
| GPT-5.4 | `Cold-start-out-browsergym/miniwob.*` | 125 |
| Claude Sonnet 4.5 | `openrouter-transfer-baselines-out/.../claude/browsergym/miniwob.*` | 125 |
| Gemini 3.1 Pro | `openrouter-transfer-baselines-out/.../gemini/browsergym/miniwob.*` | 125 |
| Qwen3-VL-235B | `openrouter-transfer-baselines-out/.../qwen/browsergym/miniwob.*` | 125 |

SFT inventory: 29 bank skills · 905 action_taking · 2,529 skill_selection

### webshop (BrowserGym)

| Teacher | Source | Episodes |
|---|---|---:|
| GPT-5.4 | `Cold-start-out-browsergym/webshop_50task_low/` | 50 |
| Claude Sonnet 4.5 | `Cold-start-out-browsergym/webshop_50task_claude/` | 50 |
| Gemini 3.1 Pro | `Cold-start-out-browsergym/webshop_50task_gemini/` | 50 |
| Qwen3-VL-235B | `Cold-start-out-browsergym/webshop_50task_qwen/` | 50 |

SFT inventory: 16 bank skills · decision SFT **pending** (skill_query at 100 % coverage)

### Visual Reasoning (4 benchmarks)

| Benchmark | Modality | GPT-5.4 | Claude | Gemini | Qwen |
|---|---|---:|---:|---:|---:|
| tir_bench | image | 300 | 300 | 300 | 300 |
| visual_toolbench | image | 300 | 300 | 300 | 300 |
| siv_bench | video | 382 | 382 | 382 | 382 |
| video_holmes | video | 1,000 | 1,000 | 1,000 | 1,000 |

SFT inventory totals for VR:

| Bench | Bank skills | action_taking | skill_selection |
|---|---:|---:|---:|
| tir_bench | 32 | 302 | 5,247 |
| visual_toolbench | 32 | 74 | 6,356 |
| siv_bench | 23 | 803 | 6,818 |
| video_holmes | 31 | 1,271 | 18,451 |

---

## Aggregate Totals

| Metric | Count |
|---|---:|
| Skill banks | 18 |
| Contract-rich skills | 448 |
| Modality-agnostic templates | 448 |
| action_taking rows | ~38,416 |
| skill_selection rows | ~74,462 |

---

## Skill Bank Architecture — Shared Bank vs Per-Task Bank

The system has a **two-layer** skill bank architecture. The key insight:
**skills capture *how to think*, not just *what to do***. The shared bank
stores transferable multi-step reasoning patterns (mega-skills); the
per-task bank stores concrete skills bound to specific games/domains.

### Per-Task Bank — concrete, executable skills

Each task has its own `skill_bank.jsonl` with skills grounded in that
task's specific action vocabulary, entity types, and reward structure.

The same abstract intent (`RECOVER/EVADE`) produces completely different
concrete skills depending on the task:

| Task | Concrete Skill | What it actually does |
|---|---|---|
| StreetsOfRage2 | "Slip Past Threats" | Sidestep or back away to dodge punches/kicks on the street |
| ThunderForceIII | "Dodge and Probe" | Veer up/down when enemy fire crowds your ship's lane |
| Strider | "Evade Air Hazards" | Jump/reposition when aerial enemies or projectiles appear overhead |
| Airstriker | "Dodge Left" | Sharp left to slip between incoming bullets |
| Tetris | "Commit to Clean Placements" | Lock pieces flat to keep stack low, avoid holes |

Each concrete skill carries:
- `protocol` — multi-step executable plan (EXEC, MOVE, KEEP, PLACE, ...)
  with `${target}`, `${direction}` slot bindings to actual game entities
- `contract` — task-specific preconditions/postconditions/effects
- `sub_episodes` — pointers to actual rollout evidence segments

### Shared Bank — transferable multi-step reasoning mega-skills

The shared bank (`skill_bank/shared_abstract_bank.py`) stores
**modality-agnostic procedural skeletons** — the multi-step reasoning
patterns that transfer across games and domains. These are the
"mega-skills" that capture **how to reason**, not what buttons to press.

```
shared_skill_bank/
├── abstract.jsonl                  ← SharedAbstractSkill (reasoning skeletons)
└── by_task/<task>/bindings.jsonl   ← BoundConcreteSkill (task-specific executables)
```

A `SharedAbstractSkill` has:
- `template_signature` — the abstract reasoning chain, e.g.
  `PERCEIVE → COMPARE → FILTER → DECIDE → COMMIT → VERIFY`
- `template_steps` — each step is one of 8 reasoning operations:
  `PERCEIVE`, `RECALL`, `COMPARE`, `FILTER`, `DECIDE`, `COMMIT`,
  `VERIFY`, `RECOVER`
- `protocol_steps` — the protocol with task-specific tokens stripped:
  `op` preserved, slot *names* kept but values dropped, semantic types
  (`tracked_entity`, `navigable_region`, ...) kept because they're
  already modality-agnostic
- `lineage` — which concrete skills across which tasks bind to this
  skeleton (task, cohort, discovery channel, usage counts)
- `stable_key = (skill_id_stem, template_signature)` — identity

### How mega-skills bridge games and domains

The same reasoning skeleton works across completely different modalities:

**Example: `PERCEIVE → COMPARE → FILTER → DECIDE` skeleton**

| Cohort | Task | Concrete skill | What the reasoning chain does |
|---|---|---|---|
| env_wr_game | candy_crush | `COMMIT/CLEAR` | Scan board → compare match candidates → filter by chain length → pick best clear |
| env_wr_game | 2048 | `TRACK/MERGE` | Scan grid → compare merge options → filter by score potential → pick direction |
| vr_image | tir_bench | `COMPARE/RULE_OUT` | Perceive image → compare answer options → filter contradicted ones → decide answer |
| vr_image | visual_toolbench | `REASON/OPTIMIZE` | Perceive chart → compare data points → filter outliers → decide conclusion |
| web | miniwob | `COMMIT/SETUP` | Perceive page → compare form fields → filter filled ones → decide next input |

All 5 share the **same abstract reasoning chain** despite zero overlap in
their predicate vocabularies (game entities vs. HTML elements vs. image
regions). The shared bank captures this procedural commonality.

**Example: `PERCEIVE → DECIDE → COMMIT → VERIFY` skeleton** (most common,
142 skills across 10 tasks)

This is the "pure action" template — perceive state, decide, act, verify.
Dominates gym_v games (shoot, dodge, navigate) but also appears in web
tasks (click target, verify navigation).

**Example from `LONG_HORIZON_REASONING.md` — reasoning skill families:**

```
skill: constraint_satisfaction               ← mega-skill in shared bank
trigger: state_flags.error != null OR target.blocker != null
protocol:
  hop1: GROUND(blocker entity)              ← PERCEIVE
  hop2: CHECK(what constraint is violated)  ← COMPARE
  hop3: RETRIEVE(similar past resolution)   ← RECALL
  hop4: CONCLUDE(subgoal = resolve first)   ← DECIDE
  hop5: EXECUTE(action addressing blocker)  ← COMMIT
```

This pattern transfers across:
- **Games**: piece blocked → clear obstacle → retry move
- **Web**: form invalid → fill missing field → resubmit
- **Visual QA**: weak evidence → gather another anchor → conclude

### Bidirectional data flow

```
FORWARD (transfer to new task):
  SharedAbstractSkill ──LLM bind to target task vocab──▶
    candidate BoundConcreteSkill (status=PENDING) ──harness validate──▶
      status=VALIDATED, sub_episodes appended to per-task bank

BACKWARD (discovery from task):
  New skill mined in task X ──LLM lift protocol to modality-agnostic skeleton──▶
    upsert to SharedAbstractBank (new record or new lineage on existing)
```

### Cross-cohort transfer statistics (Layer C templates)

From 448 skills across 18 tasks:

| Top cross-cohort signatures | # skills | Cohorts spanned |
|---|---:|---|
| `PERCEIVE → DECIDE → COMMIT → VERIFY` | 142 | env_wr_game, gymv_game, web, vr_image |
| `PERCEIVE → COMPARE → DECIDE → COMMIT → VERIFY` | 10 | env_wr_game, gymv_game, vr_image, web |
| `PERCEIVE → COMPARE → DECIDE → VERIFY` | 9 | env_wr_game, vr_image, vr_video, web |
| `PERCEIVE → COMPARE → FILTER → DECIDE` | 69 pairings | env_wr_game + vr_image (strongest cross-domain) |

Key finding: **1,719 cross-cohort skill pairs share an exact template
signature**, even though Layer A (skill ID) gives 0 universal skills and
Layer B (predicate vocabulary) gives only 4 honest cross-cohort pairs.

### Runtime implementation

| Component | Per-Task Bank | Shared Bank |
|---|---|---|
| Storage | `<bank_dir>/<game>/skill_bank.jsonl` | `shared_skill_bank/abstract.jsonl` + `by_task/*/bindings.jsonl` |
| Manager | `PerGameSkillBankManager` | `SharedSkillBankManager` + `TwoLayerSkillStore` |
| Skill identity | task-local `skill_id` | `stable_key = (skill_id_stem, template_signature)` |
| Transfer mechanism | LoRA carry-over only | LLM skeleton-to-binding at phase boundaries |
| Config | `BANK_MODE=per_game` (default) | `BANK_MODE=shared` + `TRANSLATE_ON_BOUNDARY=1` |

Related scripts:
- `scripts/lift_skill_templates_gpt54.py` — extract Layer C templates from concrete skills
- `scripts/build_shared_skill_bank.py` — merge all sources into TwoLayerSkillStore
- `scripts/bind_abstract_to_task.py` — forward-bind abstract skeleton to target task
- `scripts/discover_skill_to_shared_bank.py` — reverse-lift new skill to shared bank
- `scripts/seed_per_task_bank_cold_start.py` — cold-start a new task from shared bank

### Training plan usage

| Phase | Bank mode | Rationale |
|---|---|---|
| Phase 1 (6 source games) | `per_game` | Mine concrete skills, populate per-task banks |
| Phase 2 (6 held-out games) | `shared` | Test: do mega-skill skeletons transfer to new games? |
| Phase 3 (OOD: video, VR) | `shared` | Test: do reasoning patterns cross domain boundaries? |

---

## Known Gaps

### 1. `frontier_distill_jsonl` — missing

The `COLD_START_VALIDATION_ROOT` in `scripts/run_phase1_curriculum.sh` points to:

```
${PROJECT_ROOT}/labeling/frontier_distill_jsonl/run_20260506_055632_with_labeled
```

This directory does **not** exist in either `emnlp2026_download` or the main
repo. It serves two purposes:

- **SFT training corpus** — high-quality frontier teacher distillation data
- **Phase B Crafter validation** — offline gate that verifies each
  Crafter PATCH/HYPOTHESIS contract against teacher-derived
  (state, next_state) pairs before the actor sees the skill

Without it, co-evolution falls back to Phase A inheritance only.

### 2. Webshop decision SFT — not yet generated

Skill bank (16 skills) and per-step `skill_query` are complete, but
`action_taking.jsonl` + `skill_selection.jsonl` have not been emitted.
Run `scripts/build_multimodal_decision_sft.py` over the labeled rollouts to
materialise them.

### 3. env_wrapper games — single teacher

tetris, super_mario, candy_crush, and twenty_forty_eight only have GPT-5.4
rollouts. Claude / Gemini / Qwen rollouts were never collected for these
4 games.
