# Skill Paradigm Comparison: How Skills Guide the Agent

## Background

Two paradigms exist in the codebase for how skills are surfaced to the
agent at action-taking time. They were never explicitly compared.

Evidence source:
- **Paradigm A**: Successful game training runs in `Game-AI-Agent/runs/`
- **Paradigm B**: Designed in `frontier_data/` Layer-C pipeline, rendered
  by `_format_skill_guidance_for_prompt()` — never trained

---

## Paradigm A: Subgoal Tag (Games, verified)

**What the agent sees:**

```
game=candy crush | phase=opening | step=0/50 | score=0 | moves=50

Assigned subgoal: [OPTIMIZE] clear blockers to open matches

Available actions (pick ONE by number):
  1. ((0,5),(1,5))
  2. ((0,6),(1,6))
  ...

Output:
SUBGOAL: [TAG] <objective in ≤15 words>
REASONING: <1-2 sentences>
ACTION: <number>
```

**What the skill provides:**
- A subgoal **tag** (SETUP / CLEAR / MERGE / ATTACK / NAVIGATE / OPTIMIZE ...)
- A one-line **objective** ("clear blockers to open matches")
- Effects predicates (`Achieve: world.moves=2` / `Remove: world.moves=48`)
  stored in bank but NOT shown in prompt

**What the skill does NOT provide:**
- Multi-step reasoning plan
- Per-step progress tracking
- Exemplars or failure cases

**Prompt overhead:** ~20 tokens for the skill block

**Verified results:**
- Candy Crush: 14 GRPO iterations, reward=17.0
- Super Mario: best checkpoint saved at step 16
- Tetris, 2048: multiple successful runs

**Why it works for games:**
- Each step is one env action (click, move, swap) — agent doesn't need
  multi-step reasoning guidance
- Reward is dense (score changes every few steps) — GRPO has strong signal
- Subgoal tag reduces action selection space — "you're in ATTACK mode"
  is enough directional guidance
- Agent does its own reasoning in 1-2 sentences

---

## Paradigm B: Multi-step Reasoning Protocol (Non-game, unverified)

**What the agent would see:**

```
--- Active Skill: Core Theme Inference ---
  Strategy: Identify thematic patterns through evidence chains
  Progress: step 2/5
  Plan (5 steps):
     1. Identify the strongest evidence linking events to a cause
  >> 2. Assess which thematic interpretation best fits all observed cues
     3. Eliminate themes unsupported or contradicted by the evidence
     4. Select the most likely core theme from remaining candidates
     5. Confirm the chosen theme matches the central evidence pattern
  Preconditions: video_loaded; options_visible
  Done when: answer_emitted
  Abort if: No progress after several moves
--- end skill ---
```

**What the skill provides:**
- 5-step reasoning **protocol** with domain-specific step descriptions
- Current step **marker** (`>>`) tracking progress
- `step_checks` for per-step verification (currently all empty)
- `+0.1` intrinsic bonus per verified step → reward shaping for GRPO
- Success/abort criteria

**What the skill does NOT provide (yet):**
- Exemplars (planned but not implemented)
- Failure cases

**Prompt overhead:** ~250 tokens per skill block

**Verified results:** None. This paradigm has never been trained.

**Design rationale for non-game QA:**
- Single-turn reasoning: all 5 steps happen in one model output
- Reward is binary (correct/incorrect) — intrinsic per-step bonus could
  provide gradient signal when env reward is sparse
- VR QA requires structured reasoning: perceive evidence → compare options
  → eliminate → decide. Without guidance, the model may skip steps.

**Risks:**
- If step descriptions are wrong, they mislead the model
- If step_checks are empty (current state), intrinsic bonus is free → noise
- 250 tokens of protocol may overwhelm a 9B model's context budget
- Agent loses freedom to reason its own way

---

## Paradigm C: Hybrid — Archetype + Direction + Exemplar (implemented)

**Status:** Implemented (2026-05-13). Data bridge complete: `protocol_raw.steps`
flows from skill bank through `SkillGuidance.exemplar_steps` to the agent's
prompt as "Example reasoning:" block. Optional 35B LLM enricher rewrites
step text for clarity and diagnoses failure patterns.

**What the agent sees:**

```
--- Active Skill: Core Theme Inference ---
  Strategy: Identify thematic patterns through evidence chains
  Progress: step 2/5
  Plan (5 steps):
     1. Identify the strongest evidence linking events to a cause
  >> 2. Assess which thematic interpretation best fits all observed cues
     3. Eliminate themes unsupported or contradicted by the evidence
     4. Select the most likely core theme from remaining candidates
     5. Confirm the chosen theme matches the central evidence pattern
  Example reasoning:
    - Message links horror to game participation
    - Multiple props confirm dangerous novelty games theme
    - Eliminated technology dependence (only one prop supports it)
  Common mistake: Focused on smartphone prop instead of tracing causal chains
--- end skill ---
```

**What the skill provides:**
- Archetype **name** and one-line reasoning **direction**
- One success **exemplar** (from `protocol_raw.steps` — 9B self-rollout or teacher demo)
- One failure **counter-example** (optional, from `failure_lesson` — 35B diagnosed)

**What the skill does NOT provide:**
- Per-step progress tracking or intrinsic bonus (kept from Paradigm B when step_checks exist)
- Step checks (Paradigm C doesn't depend on them)

**Prompt overhead:** ~120-150 tokens per skill block

**Data sources for exemplars:**

| Source | How | Quality |
|---|---|---|
| 9B self-rollout (default) | `enrich_protocol_raw()` extracts best-reward intention sequence | Okay — raw agent reasoning, may be messy |
| 35B step grounding (optional) | `_enrich_steps_llm_async()` rewrites for clarity | Better — grounded in archetype context |
| 35B exemplar curation (optional) | `_enrich_exemplar_selection_llm_async()` picks clearest trace | Best — LLM selects for pedagogical value |
| 35B failure diagnosis (optional) | `_enrich_failure_diagnosis_llm_async()` → `failure_lesson` | Complementary — teaches what NOT to do |

**Rationale:**
- Exemplar shows concrete reasoning, not abstract instructions
- Closer to few-shot ICL — a paradigm with strong empirical support
- Agent retains freedom to reason its own way, but has a concrete reference
- No dependency on step_checks correctness (currently broken)
- `protocol_raw.steps` already exists in the skill bank as exemplar source
- 35B enrichment is offline and fail-soft — never blocks training

---

## Side-by-side Comparison

| Dimension | A: Subgoal Tag | B: Multi-step Protocol | C: Hybrid + Exemplar |
|---|---|---|---|
| **Implemented** | Yes (games, verified) | Yes (rendering only) | **Yes (2026-05-13)** |
| **Prompt tokens** | ~20 | ~250 | ~120-150 |
| **Agent freedom** | High | Low | Medium |
| **Reasoning guidance** | Direction only | Step-by-step | By example |
| **Failure attribution** | Episode-level | Per-step (if checks work) | Episode-level + failure_lesson |
| **Intrinsic reward** | No | Yes (per-step bonus) | No |
| **Exemplar** | No | No (planned) | **Yes (from protocol_raw.steps)** |
| **Failure counter-example** | No | No | **Yes (from failure_lesson, 35B)** |
| **Dependency on step_checks** | None | Critical (currently broken) | None |
| **35B enrichment** | N/A | N/A | **Optional (3 passes, fail-soft)** |
| **Best for** | Games (dense reward, multi-turn) | QA if step_checks work | QA (sparse reward, single-turn) |

## Resolution: Different Tasks Need Different Paradigms

The two task types have fundamentally different needs:

**Games** — the agent knows HOW to act (click, move, swap). It only needs
to know WHAT to do (ATTACK vs NAVIGATE vs COLLECT). A subgoal tag is
sufficient directional guidance. Dense per-step reward from the environment
provides strong GRPO signal. Multi-step protocol would be overhead.

**Non-game QA** — the agent needs to know HOW TO THINK. The reasoning
chain (perceive evidence → compare options → eliminate → decide) IS the
skill. Multi-step protocol is not overhead — it defines the reasoning
methodology that leads from question to answer.

The critical difference is reward density:

```
Games:    env_reward per step (dense)  → GRPO has strong signal naturally
Non-game: env_reward = 0/1 (binary)   → GRPO needs artificial partial reward

Without per-step reward (non-game QA):
  correct   → +1.0  (model doesn't know which step it got right)
  incorrect → +0.0  (model doesn't know where it went wrong)

With per-step artificial reward (non-game QA):
  PERCEIVE ✓ → +0.1  (found evidence entities)
  COMPARE  ✓ → +0.1  (compared answer options)
  FILTER   ✗ → +0.0  (failed to eliminate — bottleneck identified)
  DECIDE     → skip
  VERIFY     → skip
  correct    → +1.0
  total = 1.2 (success) or 0.2 (failure with partial credit)
```

GRPO goes from binary 0/1 signal to continuous partial reward. The model
learns "I got perceive and compare right but failed at filter" — actionable
gradient signal for improvement.

**Prerequisite:** step_checks MUST be populated (Phase 0e in the plan).
Without real checks, the intrinsic bonus fires unconditionally — free +0.5
per episode that dilutes the reward signal instead of sharpening it.

### Final paradigm assignment

```
Games (12 tasks):
  → Paradigm A: Subgoal tag
  → Verified in training runs. Don't change.
  → Dense env_reward makes per-step intrinsic unnecessary.

Non-game QA (6 tasks):
  → Paradigm B+C: Multi-step protocol + exemplar + per-step artificial reward
  → protocol.steps: existing reasoning steps from archetype extraction
  → step_checks: Phase 0e deterministic fill (OPERATOR_TO_EFFECT mapping)
  → exemplars: from protocol_raw (already in bank) or teacher demonstrations
  → intrinsic bonus: +0.1 per VERIFIED step (earned, not free)
  → Binary env_reward alone is too sparse for GRPO. Per-step partial reward
    turns "correct/incorrect" into a curriculum signal.
```

---

## Per-task-type Recommendation (pending ablation)

| Task type | Recommended | Rationale |
|---|---|---|
| **Games** (12 tasks) | **A** (subgoal tag) | Verified. Dense reward. Multi-turn. Don't fix what works. |
| **VR QA** (video_holmes, siv_bench, tir_bench, visual_toolbench) | **B or C** (needs ablation) | Single-turn reasoning. Binary reward. Need to compare. |
| **Web** (miniwob, webshop) | **A or B** (needs ablation) | Multi-turn like games, but more complex action sequences. |

## Ablation Design

**Task:** video_holmes (7 archetypes, 200 train / 800 eval)

**Setup:** For each paradigm, run N GRPO iterations on the 200 train split
with the same base model (game-SFT'd Qwen 3.5-9B).

| Ablation | Prompt format | Intrinsic reward | Exemplar |
|---|---|---|---|
| A1 | Subgoal tag only | None | None |
| B1 | 5-step protocol (empty checks) | Free +0.5 (useless) | None |
| B2 | 5-step protocol (real checks) | Earned per-step | None |
| C1 | Archetype + exemplar | None | From protocol_raw |
| C2 | Archetype + exemplar | None | From teacher demo |
| BC1 | 5-step protocol + exemplar | Earned per-step | From protocol_raw |

**Metric:** Accuracy on 800 eval split after 3 GRPO iterations.

**Expected outcome:**
- A1 < B1 ≈ A1 (free bonus is noise, not signal)
- B2 > B1 (if step_checks work, earned bonus helps)
- C1 > A1 (exemplar provides concrete guidance)
- C2 ≈ C1 (teacher vs protocol_raw exemplar quality)
- BC1 = best case if both protocol guidance AND exemplar help

**Minimum viable test:** Compare A1 vs C1 — subgoal-only vs exemplar-only.
This tells us whether exemplars add value at all, without needing to fix
step_checks first.

---

## Data Availability for Smoke Test

All data for paradigm rendering already exists in the skill bank:

```python
skill_bank.jsonl per skill record:
  skill.protocol.steps          → Paradigm B step descriptions
  skill.protocol.action_vocab   → Paradigm B operator names
  skill.strategic_description   → Paradigm A objective text
  skill.skill_id                → Paradigm A tag (archetype.video_holmes.CTI → CTI)
  skill.protocol_raw.steps      → Paradigm C exemplar source (per-sample reasoning)
  skill.protocol_raw.failure_lesson → Paradigm C counter-example (35B diagnosed)
  skill.protocol_raw.llm_grounded  → True if steps were rewritten by 35B
  report.expected_answer        → Paradigm C exemplar gold answer
  report.model_answer           → Paradigm C exemplar model answer
```

Run `frontier_data/scripts/smoke_test_paradigms.py` to see all three
prompt variants rendered side-by-side for every video_holmes skill.

## Enabling 35B LLM Enrichment

The 35B enricher is off by default. To enable:

```bash
# Environment variable (simplest)
export LLM_ENRICHMENT_ENABLED=1
export LLM_ENRICHMENT_MODEL=""   # empty = use BACKBONE_JUDGE_MODEL (35B-A3B)

# Or in CoEvolutionConfig
config.llm_enrichment_enabled = True
config.llm_enrichment_model = ""
config.llm_enrichment_step_grounding = True      # pass 1
config.llm_enrichment_exemplar_curation = True   # pass 2
config.llm_enrichment_failure_diagnosis = True   # pass 3
```

Non-LLM enrichment (`enrich_protocol_raw` from 9B self-rollout) always
runs regardless of the LLM enrichment toggle. The 35B passes layer on
top: grounding rewrites raw 9B steps for clarity, curation selects the
clearest exemplar, and diagnosis adds failure counter-examples.
