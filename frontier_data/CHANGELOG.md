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

### Commits

| Hash | Message |
|---|---|
| `b85d21d` | frontier_data: Layer-C reasoning-intent re-lift (GPT-5.4) + cross-domain analysis |
| `f216b94` | frontier_data: inject Layer-C reasoning plans into 304/406 skill protocols |
| `18c9d2c` | seed pipeline: Layer-C signature-aware candidate selection + V1 transfer design |
