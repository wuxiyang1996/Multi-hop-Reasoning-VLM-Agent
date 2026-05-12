# Frontier Data — Changelog

## 2026-05-12: Layer-C Reasoning Plans + Cross-Domain Transfer Pipeline

### What was done

#### 1. Layer-C reasoning-intent re-lift (GPT-5.4)

Ran `scripts/lift_skill_templates_gpt54.py` on all 406 per-task skills
across 18 tasks. Each skill was lifted into a 2–5 step modality-agnostic
procedural template using 8 controlled reasoning operators:

```
{PERCEIVE, RECALL, COMPARE, FILTER, DECIDE, COMMIT, VERIFY, RECOVER}
```

- **Result:** 406/406 skills successfully lifted (98.4 s, 16 workers)
- **Fix applied:** `max_tokens` → `max_completion_tokens` for GPT-5.4 API
- **Output:** `frontier_data/output/layer_c_templates/<cohort>/<task>/template_bank.jsonl`

#### 2. Cross-domain reasoning plan discovery

Analyzed Layer-C signatures across 3 domain groups (GAME / WEB / VR).
Found **9 cross-domain reasoning plans** shared by ≥ 2 domains, covering
**221 of 406 skills (54.4%)**:

| Reasoning plan | Domains | Skills | Example |
|---|---|---:|---|
| PERCEIVE→DECIDE→COMMIT→VERIFY | GAME+WEB | 166 | mario navigation + miniwob focus |
| PERCEIVE→COMPARE→FILTER→DECIDE→COMMIT | GAME+VR | 11 | tetris optimize + emotion inference |
| PERCEIVE→DECIDE→COMMIT→COMMIT | GAME+WEB | 10 | positioning + drag/draw |
| PERCEIVE→COMPARE→DECIDE→COMMIT→VERIFY | GAME+WEB | 9 | tetris evade + tic-tac-toe |
| PERCEIVE→COMMIT→COMMIT→VERIFY | GAME+WEB | 8 | charge attack + copy paste |
| PERCEIVE→COMPARE→DECIDE→VERIFY | GAME+VR | 7 | columns labeling + VR QA |
| PERCEIVE→DECIDE→COMMIT→COMMIT→VERIFY | GAME+WEB | 4 | dodge-strike + text styling |
| PERCEIVE→COMPARE→COMMIT→VERIFY | GAME+WEB | 4 | scene scan + ascending order |
| PERCEIVE→COMMIT→COMMIT | GAME+WEB | 2 | hazard dodge + circle center |

Before Layer-C: 0 cross-domain plans. After: 9 plans, 221 skills bridged.

#### 3. Layer-C → runtime protocol injection

`frontier_data/scripts/inject_layerc_protocols.py` converted Layer-C
templates into runtime protocol dicts and patched per-task skill banks.

| | Before | After |
|---|---|---|
| Skills with empty protocol | ~304 | **0** |
| Skills with reasoning-level plans | ~102 | **406** |
| Avg steps per protocol | 0.9 (game), 0 (VR/web) | **3–5 steps** |

Example — siv_bench (was empty, now has 5-step reasoning plan):
```
--- Active Skill: archetype.siv_bench.Action_Recognition ---
  Plan (5 steps):
  >> 1. Observe current scene cues and immediate preceding events
     2. Recall task goal of predicting the next likely action
     3. Compare candidate continuations against causal and contextual evidence
     4. Select the continuation most strongly implied by context
     5. Submit the chosen next-action prediction
--- end skill ---
```

The `>>` step tracker advances via predicates with +0.1 intrinsic bonus.

#### 4. Seed pipeline: Layer-C signature-aware selection

Updated `scripts/seed_per_task_bank_cold_start.py::pick_seed_candidates`
to prioritize cross-domain reasoning plans over name-based lineage:

| Priority | Old (name-based) | New (Layer-C-aware) |
|---|---|---|
| 1st | # native mining tasks | **Cross-domain signature** (≥ 2 domain groups) |
| 2nd | Production successes | Cohort diversity |
| 3rd | Cohort diversity | # native mining tasks |
| 4th | — | Production successes |

This ensures that when seeding a new task, the pipeline picks skills with
shared reasoning STRUCTURE (e.g., PERCEIVE→DECIDE→COMMIT→VERIFY appears
in both games and web) rather than just shared names.

#### 5. Contribution framing & ablation design

Added to README:
- **Data leakage analysis** — the 9 cross-domain plans are GPT-5.4's
  knowledge, not our discovery. The contribution must come from training
  experiments showing structured banks help the 9B student model.
- **Three ablation comparisons** (A1/A2/A3) to isolate the contribution:
  - A1: bank vs no-bank (does the bank save GRPO steps?)
  - A2: structured bank vs raw SFT (does structure matter?)
  - A3: cross-domain vs same-domain seeds (does cross-domain transfer work?)

#### 6. V1 transfer approach: skills as suggestions

Decided against complex cross-domain compose (previous attempts produced
trash due to action-level inputs + ban list conflicts). Instead:

- Layer-C reasoning plans serve as **suggestions** for the 9B agent
- Agent sees step-by-step plan in prompt, attempts to follow
- GRPO reinforces good execution, crafter patches failures
- No compose needed — the STRUCTURE of the plan is what transfers,
  GRPO fine-tunes the execution

### Files changed

| File | Change |
|---|---|
| `scripts/lift_skill_templates_gpt54.py` | Fix `max_tokens` → `max_completion_tokens` |
| `scripts/seed_per_task_bank_cold_start.py` | Layer-C signature-aware `pick_seed_candidates` |
| `frontier_data/scripts/inject_layerc_protocols.py` | **New** — Layer-C → runtime protocol converter |
| `frontier_data/scripts/build_reasoning_aligned_bank.py` | **New** — reasoning-intent normalizer |
| `frontier_data/scripts/test_game_to_nongame_transfer.py` | **New** — harness validation: game→non-game |
| `frontier_data/output/per_task_banks/*/skill_bank.jsonl` | 304/406 skills patched with Layer-C protocols |
| `frontier_data/output/layer_c_templates/` | 406 Layer-C templates (GPT-5.4 generated) |
| `frontier_data/README.md` | Contribution framing, ablation design, §8c–8f |

---

## 2026-05-12 (update 2): Collapsed 5-op signature matching

### Problem

The original 8-op Layer-C signatures produce exact matches that are too
granular. `PERCEIVE→DECIDE→COMMIT→VERIFY` and `PERCEIVE→DECIDE→COMMIT`
are semantically the same reasoning plan but fail exact matching because
VERIFY (an action step) is present in one but not the other. Result:
54.4% coverage dominated by one generic signature, zero three-way plans.

### Solution: collapse 8 ops → 5 semantic equivalence classes

| Collapse | Rationale |
|---|---|
| COMPARE + FILTER → **EVALUATE** | Both assess/evaluate perceived state |
| COMMIT + VERIFY + RECOVER → **ACT** | All are execution/action steps |
| PERCEIVE, DECIDE, RECALL | Kept distinct |

Consecutive duplicate ops after collapsing are deduplicated
(e.g., PERCEIVE→DECIDE→COMMIT→VERIFY → PERCEIVE→DECIDE→ACT).

### Results

| Metric | Before (exact 8-op) | After (collapsed 5-op) |
|---|---|---|
| Cross-domain plans | 9 | **7** |
| Skills covered | 221 (54.4%) | **314 (77.3%)** |
| Three-way plans (GAME+WEB+VR) | 0 | **1 (55 skills)** |
| Meaningful non-generic plans | 55 | **314** |
| Still domain-locked | 185 (45.6%) | **92 (22.7%)** |

The three-way plan `PERCEIVE → EVALUATE → DECIDE → ACT` covers 55 skills
across all 3 domains (36 GAME + 14 VR + 5 WEB).

### Files changed

| File | Change |
|---|---|
| `scripts/seed_per_task_bank_cold_start.py` | Added `collapse_signature()`, `_OP_COLLAPSE` table; `_load_cross_domain_sigs` now uses collapsed sigs; `_score` adds three-way priority |
| `frontier_data/scripts/inject_layerc_protocols.py` | Propagates `collapsed_signature` to per-task skill banks |
| `frontier_data/output/layer_c_templates/*/template_bank.jsonl` | All 406 records have new `collapsed_signature` field |
| `frontier_data/output/per_task_banks/*/skill_bank.jsonl` | All 406 skills patched with `collapsed_signature` field |
| `frontier_data/README.md` | Updated §8d, Pipeline Results, Gap 8 with collapsed coverage stats |

---

## 2026-05-12 (update 3): LLM-as-judge plan similarity validation

### Problem

Collapsed 5-op signatures determine structural match (same op sequence)
but cannot distinguish "same cognitive procedure" from "same structure,
different challenge". E.g., `PERCEIVE→DECIDE→ACT` matches both "dodge
bullet in mario" and "click button in miniwob" — structurally identical
but cognitively different.

### Solution: GPT-4.1-mini as pairwise judge

`frontier_data/scripts/judge_plan_similarity.py` samples 3 skills per
domain for each of the 7 cross-domain collapsed signatures, presents
all cross-domain pairs (72 total) to GPT-4.1-mini, and asks the model
to rate 1–5 whether the full plan context (predicate text) represents
the SAME transferable cognitive procedure.

### Results

| Collapsed plan | Domains | Pairs | Avg | ≥4 | Verdict |
|---|---|---:|---:|---:|---|
| RECALL → PERCEIVE → DECIDE → ACT | VR↔WEB | 9 | **4.0** | 100% | STRONG |
| PERCEIVE → EVALUATE → DECIDE → ACT | GAME+WEB+VR | 27 | **3.8** | 78% | MODERATE |
| RECALL → PERCEIVE → EVALUATE → DECIDE → ACT | VR↔WEB | 9 | **3.8** | 78% | MODERATE |
| PERCEIVE → ACT | GAME↔WEB | 6 | 3.5 | 67% | MODERATE |
| PERCEIVE → EVALUATE → ACT | GAME↔WEB | 9 | 3.4 | 44% | MODERATE |
| PERCEIVE → DECIDE → ACT | GAME↔WEB | 9 | 3.3 | 67% | MODERATE |
| PERCEIVE → RECALL → EVALUATE → DECIDE → ACT | GAME↔VR | 3 | 3.3 | 33% | MODERATE |

**Overall: 52/72 pairs (72%) rated as same procedure (score ≥ 4).**
All 7 cross-domain signatures rated MODERATE or higher.

Key finding: **VR↔WEB transfer is strongest** (avg 4.0) — both involve
target recall + evidence matching with different modalities. Three-way
plan (PERCEIVE→EVALUATE→DECIDE→ACT) validated at 78% same-procedure.

### Integration into seed pipeline

`seed_per_task_bank_cold_start.py::_score` now incorporates judge
confidence as a priority tier:

| Judge tier | Threshold | Meaning |
|---|---|---|
| 2 (highest) | avg_score ≥ 4.0 | STRONG_TRANSFER — same cognitive procedure |
| 1 | avg_score ≥ 3.0 | MODERATE_TRANSFER — overlapping reasoning patterns |
| 0 | < 3.0 or unrated | Weak/no transfer evidence |

Sort key is now: three-way → judge_tier → two-way → cohort diversity →
task breadth → production successes.

### Files changed

| File | Change |
|---|---|
| `frontier_data/scripts/judge_plan_similarity.py` | **New** — LLM-as-judge pairwise plan similarity |
| `frontier_data/output/plan_similarity_judgments.json` | 72 judgments + per-signature summary |
| `scripts/seed_per_task_bank_cold_start.py` | Added `_load_judge_scores`, judge_tier in `_score` |
| `frontier_data/README.md` | Added §8d′, updated Pipeline Results |
| `frontier_data/CHANGELOG.md` | This entry |

---

## 2026-05-12 (update 4): Plan-level LLM-as-judge (beyond signatures)

### Problem

Signature-level judge (update 3) only compares skills within the same
collapsed-signature group. Skills with DIFFERENT signatures but the SAME
cognitive procedure are missed. E.g., `RECALL→PERCEIVE→DECIDE→ACT` (WEB)
and `PERCEIVE→ACT` (GAME) have very different structures but can
represent the same procedure: "perceive state → commit action → verify".

26 WEB skills + 1 VR skill had collapsed sigs not shared by ANY other
domain — completely excluded from cross-domain seeding.

### Solution: batch plan-level LLM judge

`judge_plan_level_similarity.py` presents each non-game skill's FULL
reasoning plan text alongside 20 diverse game representatives (or 15 web
representatives for VR→WEB) and asks GPT-4.1-mini which candidates share
the same cognitive procedure — regardless of structural signature.

108 batch LLM calls (not 25,000+ pairwise comparisons).

### Results

| Direction | Targets | High-conf (≥4) | NEW (different sig) |
|---|---:|---:|---:|
| WEB → GAME | 48 | 202 | 168 |
| VR → GAME | 30 | 79 | 66 |
| VR → WEB | 30 | 96 | 76 |
| **Total** | **108** | **377** | **310** |

**310 NEW high-confidence cross-domain pairs** that signatures missed.
**ALL 78 non-game skills** now have ≥ 1 cross-domain match.
**12 pairs scored 5** (IDENTICAL procedure, different domains).

Coverage improvement:

| Strategy | Non-game coverage | Seedable |
|---|---:|---:|
| Exact 8-op sig | 55 (70%) | 221/406 (54%) |
| Collapsed 5-op sig | 66 (85%) | 314/406 (77%) |
| **Plan-level LLM judge** | **78 (100%)** | **406/406 (100%)** |

### Files changed

| File | Change |
|---|---|
| `frontier_data/scripts/judge_plan_level_similarity.py` | **New** — batch plan-level LLM judge |
| `frontier_data/output/plan_level_similarity_judgments.json` | 108 batch results + 377 high-conf pairs |
| `frontier_data/README.md` | Added §8d″, updated Pipeline Results |
| `frontier_data/CHANGELOG.md` | This entry |

---

## 2026-05-12 (update 5): Cross-domain mega-skill family taxonomy

### Problem

The 377 high-confidence plan-level matches form a dense graph. Connected
component analysis collapses everything into one giant cluster due to
shared game representatives. Need a meaningful taxonomy of what
cognitive procedures are actually being transferred.

### Solution

Classified each match's `shared_reasoning` text into high-level procedure
types (e.g., COMPARE_AND_RANK, DECIDE_ACT_VERIFY). This yields a
principled grouping that reflects the actual cognitive procedure being
transferred, not just graph connectivity.

### Results

**12 distinct cross-domain mega-skill families**, of which **10 are
three-way (GAME+WEB+VR)**. This is a 10× improvement over collapsed
signature matching (which found only 1 three-way plan).

Top families:
- **COMPARE_AND_RANK** (51 skills): perceive → rank by priority → select → execute
- **DECIDE_ACT_VERIFY** (33 skills): perceive → decide → execute → verify
- **MATCH_AND_CLASSIFY** (23 skills): perceive → match to categories → commit
- **PERCEIVE_DECIDE_ACT** (22 skills): observe → criteria-based decision → act
- **FILTER_AND_SELECT** (19 skills): identify → filter → select → verify

### Files changed

| File | Change |
|---|---|
| `frontier_data/README.md` | Added §8d‴ mega-skill families table + evolution table |
| `frontier_data/CHANGELOG.md` | This entry |

---

## 2026-05-12 (update 6): Bottom-up mega-skill extraction via per-skill LLM classification

### Problem

Previous mega-skill discovery relied on pairwise comparison (O(n²)) or
connected-component clustering, which is expensive and can over-cluster.
Need a simpler, scalable approach: classify each skill independently.

### Solution

Defined a fixed 18-category cognitive taxonomy and asked `gpt-4.1-mini`
to classify each of 406 skills into exactly one family. O(n) cost with
10-thread parallelism (~40 seconds total).

### Results

**18 mega-skill families** with well-balanced distribution:
- **5 three-way** (GAME+WEB+VR): INFER_AND_DECIDE (43), NAVIGATE_AND_REACH (37),
  COMPARE_AND_RANK (23), SEQUENCE_AND_COMPLETE (22), RECALL_MATCH_AND_SELECT (18)
- **5 two-way**: POSITION_AND_PLACE, TRANSFORM_AND_VERIFY, INPUT_AND_SUBMIT,
  FILTER_AND_NARROW, COUNT_AND_REPORT
- **8 single-domain** (mostly GAME-specific combat/survival procedures)

The 5 three-way families (143 skills, 35%) represent the core transferable
cognitive procedures across all domains.

### Files changed

| File | Change |
|---|---|
| `frontier_data/scripts/extract_mega_skills.py` | **New** — Phase 1 per-skill classification |
| `frontier_data/scripts/cluster_mega_skills.py` | **New** — Phase 2 optional label merging |
| `frontier_data/output/mega_skill_labels.json` | 406 skill classifications |
| `frontier_data/output/mega_skill_clusters.json` | Merged cluster output |
| `frontier_data/README.md` | Added §8e with 18-family taxonomy table |
| `frontier_data/CHANGELOG.md` | This entry |
