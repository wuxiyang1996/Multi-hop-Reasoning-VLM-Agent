# Co-evolution 3-phase training plan — source GRPO → in-domain few-shot → OOD transfer

> **Status (2026-05-03 AM):** 🟡 **PLAN — pending 3 disambiguation
> decisions before launch.** Defines a curriculum that extends the
> COS-PLAY paper's per-game training (Appendix C / Table 3) into a
> cross-game skill-transfer + cross-domain few-shot adaptation
> experiment. The per-game **step budget is paper-grounded**
> (7–25 steps per game, average ~15); the **cross-game transfer**
> protocol and **OOD few-shot adaptation** are explicitly identified
> as open problems in the paper's Limitation / Future Work section
> and are this plan's novel contribution.

> **Cross-refs:**
> - [COS-PLAY paper](https://arxiv.org/pdf/2604.20987) — Appendix C,
>   Table 3 (per-game hyperparams); §5.3 + Table 2 (within-game skill
>   reusability); §6 Conclusion (cross-domain transfer = future work).
> - [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md)
>   — backbone stack, the 5 LoRA adapters, current Stage-1 readiness.
> - [`implementation_notes/coevolution-cross-domain-integration.md`](../implementation_notes/coevolution-cross-domain-integration.md)
>   — the C/A/D-layer cross-domain measurement infrastructure that
>   Phase-3 of this plan reuses.
> - [`scripts/run_2048.sh`](../scripts/run_2048.sh) /
>   [`run_tetris.sh`](../scripts/run_tetris.sh) /
>   [`run_super_mario.sh`](../scripts/run_super_mario.sh) — existing
>   per-game wrappers; this plan adds two more (`run_candy_crush.sh`
>   and a generic-gymv wrapper) and a 3-phase orchestrator script.
> - [`trainer/coevolution/config.py`](../trainer/coevolution/config.py)
>   — `CoEvolutionConfig.curriculum_schedule`, `--seed-bank-dir`,
>   `total_steps`, GRPO hyperparam fields.
> - [`canvases/coevolution-architecture-overview.canvas.tsx`](../../../.cursor/projects/workspace/canvases/coevolution-architecture-overview.canvas.tsx)
>   — visual outline of the per-step training cycle.

---

## 1. Why this plan exists (one paragraph)

The project's headline contribution is **skill discovery, adaptation,
and transfer**. The COS-PLAY paper (Appendix C / Table 3) shows each
game converges in 7–25 GRPO steps with the bank crystallising into 6–64
reusable skills, but **all six paper experiments train a single game in
isolation**. The paper's own Conclusion explicitly defers
cross-game and cross-domain transfer: "We also aim to improve
cross-domain transfer, so that learned skills generalize across
multi-modal games, agentic environments, and broader visual reasoning
tasks." This plan turns that future work into a concrete 3-phase
curriculum: (1) reproduce the paper's per-game training on six gymv
games, (2) measure how fast the merged skill bank adapts to held-out
games, (3) measure how fast it adapts to entirely new domains
(`video`, `visual_reasoning`, optionally `browser` / `osworld`).

## 2. Scope and non-goals

In scope:

* **Phase 1** — independent GRPO runs on six gymv source games using
  the paper's per-game hyperparams (Table 3 below).
* **Phase 2** — few-shot adaptation on six held-out gymv games seeded
  with the merged Phase-1 bank, reported at two adaptation budgets
  (5-step inference-only baseline + 15-step GRPO main result).
* **Phase 3** — same two-budget few-shot protocol on the cross-domain
  targets `video` and `visual_reasoning` (cheap VLM-only envs);
  `browser` / `osworld` deferred behind a go/no-go gate after
  Phase-1 + Phase-2 results land.
* A **bank merge + de-dup utility** that produces a single
  `<source_merged>/skill_bank.jsonl` per game from the six independent
  Phase-1 runs.
* A **LoRA-merge policy** for warm-starting Phase-2 / Phase-3 actors.

Out of scope:

* Multi-player social games (Avalon, Diplomacy). Paper's per-game
  training already covers them; their bank dynamics differ enough
  (self-play, role-conditioned skills) that they are a separate
  experimental track.
* Beating the OSWorld / VWA leaderboards. Phase 3's role is
  "few-shot adapts faster than no-bank baseline", not "SOTA".
* Replacing the per-step Layer-A transfer gate or Layer-D dashboard
  hook. They run as before; this plan only adds an outer-level
  curriculum on top.
* Re-running cold-start SFT. Phase 1 starts from the existing
  cold-started `runs/sft_coldstart_20260502_025737` adapters.

## 3. Paper grounding — what's settled vs novel

| Question | Answer | Evidence |
|---|---|---|
| Per-game step budget | **7–25 steps**, mean ~15 | Appendix C, Table 3 |
| Per-game hyperparams | LR 1e-5–5e-5, KL 0.04–0.08, clip 0.10–0.20, max-epochs 2–4, advantage clip on harder games | Appendix C, Table 3 |
| Bank size after training | 6 (Tetris) – 64 (Diplomacy) skills, average ~30/skill instances | §5.3, Table 2 |
| Cross-game GRPO transfer | **Not measured** in the paper | §6 Conclusion (future work) |
| Cross-domain few-shot adaptation | **Not measured** in the paper | §6 Conclusion (future work) |
| "Few-shot" definition in paper | "≤25 iterations to reach strong performance from scratch" — about the **whole** training, not adaptation | Abstract + §5 |

This plan uses (1)–(3) verbatim, treats (4)–(5) as the novel
contribution, and is careful **not** to conflate (6)'s "few-shot"
with our Phase-2 / Phase-3 definition (see §7 below).

## 4. Phase 1 — independent source GRPO runs

### 4.1 Game roster

Four games come straight from the paper (so Phase-1 numbers can be
sanity-checked against published rewards) and two are gymv held-in
games we extend the setting with:

| # | Game | Source | Phase-1 step budget | Phase-1 hyperparams |
|---|---|---|---|---|
| 1 | `tetris` | paper Table 3 | **7** | LR 2e-5 · KL 0.08 · clip 0.10 · max_ep 2 · adv-clip 3.0 · ep/step 8 |
| 2 | `2048` | paper Table 3 | **10** | LR 5e-5 · KL 0.05 · clip 0.20 · max_ep 4 · no adv-clip · ep/step 8 |
| 3 | `candy_crush` | paper Table 3 | **10** | LR 5e-5 · KL 0.05 · clip 0.20 · max_ep 4 · no adv-clip · ep/step 8 |
| 4 | `super_mario_bros` | paper Table 3 | **20** | LR 3e-5 · KL 0.04 · clip 0.15 · max_ep 3 · adv-clip 5.0 · ep/step 8 |
| 5 | `<gymv-A>` _(TBD — see §11.3)_ | new | 15 | defaults |
| 6 | `<gymv-B>` _(TBD — see §11.3)_ | new | 15 | defaults |

Total Phase-1 step budget: **77 steps** spread across six independent
runs. Wall-clock ~36 min/step ⇒ **~46 h sequentially / ~12 h with
6× parallelism** (one run per GPU group, e.g. 1 game per single-GPU vLLM
+ FSDP-2 trainer, or 3 games concurrent on an 8×H200 box with TP=2).

### 4.2 Per-run output

Each run writes to `runs/phase1_<game>_<timestamp>/` with the standard
contents (`adapters/`, `checkpoints/`, `skillbank/<game>/skill_bank.jsonl`,
`rewards/step_NNNN.jsonl`, wandb run tagged
`phase1-source-<game>`).

### 4.3 Sanity bar

Reward at the final step within ±20% of the paper's Figure 4 endpoint
for `tetris` / `2048` / `candy_crush` / `super_mario_bros`. Anything
outside that band before merging means a hyperparam drift vs Table 3 —
debug before proceeding to Phase 2.

## 5. Bank-merge utility

Phase 2 / Phase 3 need a single seed bank, not six. We need
`scripts/merge_banks.py` with this contract:

```
merge_banks.py \
  --inputs runs/phase1_tetris_*/skillbank/tetris/skill_bank.jsonl \
           runs/phase1_2048_*/skillbank/2048/skill_bank.jsonl \
           ... (6 total)
  --output runs/source_merged_<timestamp>/skill_bank.jsonl \
  --dedupe-policy curator-score-keep-best \
  --dedupe-threshold 0.85   # cosine sim on contract effects
```

Behaviour:

* Read every input as JSONL of `{"skill": …, "report": …}` records.
* Group by **canonical contract signature** (`schema_hash` of
  `(eff_add ∪ eff_del ∪ eff_event)`); within a group, keep the record
  with the highest curator score (or the most recent contract version
  if tied).
* Cross-group dedupe: cosine similarity ≥ `--dedupe-threshold` on a
  bag-of-effects embedding ⇒ collapse, again keeping the highest-scored.
* Write atomically (tempfile + fsync + os.replace, mirroring the
  `SkillBankMVP.save()` contract).
* Emit `runs/source_merged_<timestamp>/_merge_meta.json` with
  pre/post counts, per-input contribution, and the dedupe distribution
  so the paper appendix can report bank composition.

Expected post-merge size: ~40–80 unique skills (paper has 6+10+10+20=46
across the four reproduced games; new gymv games add ~10–20 each).

## 6. LoRA warm-start policy for Phase 2 / 3

Three options, ranked by complexity:

| Option | Mechanism | Cost | Risk |
|---|---|---|---|
| **L1** — pick best | Use the single-game adapters from the **best-reward** Phase-1 run as cold start | $0 | source-distribution-specific; may not generalise |
| **L2** — sequential warm-start | Load adapters from Phase-1 run #1, fine-tune through #2, …, #6 (curriculum-style) | extra ~46 h | catastrophic forgetting on early games (the §11.4 risk in this plan) |
| **L3** — weight-merge | Average / SLERP-merge the six per-game adapters, optionally re-balanced by per-game reward | <1 h | empirically robust; paper §E shows merged-LoRA underperforms split-LoRA on a single game, but here we keep the split (5 adapters) and merge **across games** within each adapter slot |

**Recommendation: L3** with uniform weighting as the default and
reward-weighted as the ablation. Implement as
`scripts/merge_adapters.py` with the same per-adapter slot semantics
(merge `skill_selection` across the six runs into one
`skill_selection`, etc.).

## 7. Phase 2 — in-domain few-shot adaptation

### 7.1 Held-out gymv roster

Six gymv games **not** in the Phase-1 source set. Pre-screen for
"at least 3 episodes complete in 2 min on the live actor" so we don't
accidentally grade adaptation on an env where the actor can't even
emit valid actions. Roster TBD with the user.

### 7.2 Two-budget reporting protocol

The paper's "few-shot" terminology refers to the *entire* training
being short (≤25 steps from scratch). Our Phase-2 / Phase-3
"few-shot" means **adaptation given a pre-trained bank + LoRAs**.
To prevent reviewer confusion, we report **both** budgets per
held-out game:

| Budget | Purpose | Bank | LoRA | Reports |
|---|---|---|---|---|
| **5 steps · inference-only** | Zero-shot bank reuse — does the merged Phase-1 bank fire on the new game *at all*? | merged | frozen Phase-1 (no GRPO) | reward, % steps with skill bound, harness rejection rate |
| **15 steps · GRPO** | Online adaptation — does adding GRPO compound the bank advantage over no-bank baseline? | merged, hot-reloaded | unfrozen, full 5-LoRA GRPO | reward curve, new-skill discovery rate, bank growth |

Add a **no-seed control** at 15 steps GRPO (same hyperparams, empty
bank) per held-out game so the lift attributable to the bank is
unambiguous.

Phase-2 wall-clock: 6 games × (5 + 15 + 15) steps × 36 min ≈
**126 h** (5.25 days) sequentially / **~21 h** with 6× parallel
(matches Phase-1 GPU budget).

### 7.3 Headline number

`reward(seed-bank @ 5 steps inference) / reward(no-seed @ 15 steps GRPO)`
— "**how many GRPO steps does the bank save us per held-out game?**"

If this ratio is ≥ 1.0 on ≥ 4 of 6 held-out games, the cross-game
transfer claim holds. Below that we either need more Phase-1 source
games or richer skill abstractions before publishing.

## 8. Phase 3 — out-of-domain few-shot adaptation

### 8.1 Target roster (cost-stratified)

| Target | Per-step cost | Stage-1 inclusion | Why |
|---|---|---|---|
| `video` | ~30 min · VLM-only | **YES** (mandatory) | cheap, mature wrapper, clean signal |
| `visual_reasoning` | ~30 min · VLM-only | **YES** (mandatory) | cheap, mature wrapper, clean signal |
| `browser` | ~50–70 min · Playwright + WebArena/VWA mirrors | **gated** | mirrors landed 2026-05-03 (`docs(vwa)` commit `e30c473`); stability not yet stress-tested at training-loop scale |
| `osworld` | ~80–120 min · Docker + VM | **gated** | per-step cost 5×; current Pass@1 ~4.8% (see [`canvases/osworld-results-diagnosis.canvas.tsx`](../../../.cursor/projects/workspace/canvases/osworld-results-diagnosis.canvas.tsx)) — even a 2× lift might fall inside CI |

Go/no-go for `browser` / `osworld` decided after Phase-2 numbers land.

### 8.2 Adaptation protocol (same as Phase 2)

Two budgets per target: 5-step inference-only + 15-step GRPO with the
merged-bank seed. **No** no-seed control here — Phase 3's claim is
"the gymv-derived bank transfers across domains", not "GRPO works on
this domain in isolation".

Phase-3 wall-clock (mandatory targets only): 2 targets × 20 steps ×
36 min ≈ **24 h**. Browser adds ~14 h, OSWorld ~32 h if gates open.

### 8.3 Headline number

% of skills in the merged Phase-1 bank that **bind successfully**
(harness `validate_choice` returns true) ≥ 1× on each target, plus
the reward delta vs random-skill-selection baseline. Mirrors Layer-A
transfer-gate semantics so the metric is comparable to the in-loop
gate from `_transfer_hook.py`.

## 9. Concrete launch sequence

Three new orchestration scripts, plus reuse of existing
`run_<game>.sh`:

```
scripts/
  run_phase1_source.sh        ← NEW · launches 6 Phase-1 runs (parallel or sequential)
  merge_banks.py              ← NEW · §5
  merge_adapters.py           ← NEW · §6 (L3 strategy)
  run_phase2_holdout.sh       ← NEW · 6 held-out games × 2 budgets + no-seed control
  run_phase3_ood.sh           ← NEW · video + visual_reasoning × 2 budgets
  run_candy_crush.sh          ← NEW · template'd from run_2048.sh, paper hyperparams
  run_2048.sh / tetris.sh /
   super_mario.sh             ← VERIFY paper hyperparams wired (§11.1)
```

Recommended dual-stack runtime layout (8×H200):

```
GPUs 0–3 → 9B trainer + actor vLLM :8000 (per-game Phase-1 run)
GPUs 4–7 → 35B-A3B teacher + judge vLLM :8001
                                  (one instance shared across all phases)
[optional] GPU 8 → scripts/dashboard_sidecar.py (Layer-D, opt-in)
```

Phase-1 parallelisation requires either multi-machine deployment or
sequential runs on a single 8×H200 node. Six concurrent Phase-1 runs
on one node is **not** feasible (would need 6 × 4 GPU = 24 GPUs).

## 10. Wall-clock & cost

| Phase | Step budget | Sequential wall-clock | Parallel wall-clock |
|---|---|---|---|
| 1 — source GRPO | 77 (across 6 games) | ~46 h | ~12 h (6× single-GPU vLLM) |
| Bank + LoRA merge | offline | <2 h | <2 h |
| 2 — held-out adaptation | 6 × (5+15+15) = 210 | ~126 h | ~21 h (6× parallel) |
| 3 — OOD (video + VR) | 2 × 20 = 40 | ~24 h | ~12 h (2× parallel) |
| **Total (mandatory)** | **327 steps** | **~196 h ≈ 8.2 days** | **~47 h ≈ 2 days** |

OOD `browser` / `osworld` (gated): +14 h / +32 h respectively.

LLM-as-judge spend: $0 (35B-A3B local). SFT teacher (gpt-5.5) is not
re-invoked in this plan — the cold-start data is reused from the
existing SFT adapters.

## 11. Open decisions (block launch)

### 11.1 Existing per-game wrappers vs paper Table 3 hyperparams

The wrappers `run_2048.sh` / `run_tetris.sh` / `run_super_mario.sh`
already exist but have not been audited against Table 3 since the
35B-judge / atomic-save commits landed (2026-05-03). **Action**:
diff each wrapper against Table 3 row before Phase-1 launch; record
deltas (and rationale, if any) in this note's §13 changelog.

### 11.2 Cross-game transfer mechanism (A vs B vs C)

| Option | Mechanism | Verdict |
|---|---|---|
| A | Independent runs · merge banks · merge adapters | **RECOMMENDED** — clean per-game baselines, novel-claim isolation |
| B | Single run with `config.games=[6]` (default behaviour) | *games train concurrently every step — loses per-game step-budget control; rejected* |
| C | Single run with `curriculum_schedule={0:[g1], 7:[g2], 17:[g3], …}` | *appealing but causes early-game LoRA drift as later games dominate gradient updates; rejected* |

Selection: **A**.

### 11.3 Two new gymv games (`<gymv-A>` / `<gymv-B>`)

Pending user choice from the gymv catalog. Heuristic: pick one
**deterministic / sparse-reward** game (clean contract learning) and
one **dense / continuous-reward** game (harder binding, more
transferable). Candidates: `sokoban`, `ace_attorney`, `doom`,
`monopoly_deal`, `clue`. **Action**: user to confirm.

### 11.4 Phase-2 budget — accept 5-step + 15-step or compress?

If wall-clock pressure forces compression, drop the 5-step
inference-only baseline first (it's the cheapest to add later as a
post-hoc eval since it doesn't need GRPO). Keep the no-seed 15-step
control unconditionally — without it the lift claim is unfalsifiable.

## 12. Risks

* **R1 — Phase 1 reward drift vs paper.** If our reproduction of
  `tetris` / `2048` / `candy_crush` / `super_mario_bros` lands
  outside ±20% of Figure 4 endpoints, the comparison story falls
  apart at the start. Mitigation: §11.1 hyperparam audit; small (2-step)
  smoke per game before committing the full 7–20 step run.
* **R2 — Bank merge collapse.** If the cross-game dedupe collapses too
  aggressively (threshold too low), the merged bank drops below ~30
  skills and Phase-2 zero-shot has nothing to bind to. Mitigation:
  emit `_merge_meta.json` per §5; sweep threshold ∈ {0.80, 0.85, 0.90}
  if first attempt under-fires.
* **R3 — Phase-2 no-seed control beats seeded 15-step.** Plausible if
  the gymv held-out games are too distant from Phase-1 source
  distribution (e.g. switching from grid puzzles to fast-action games).
  Mitigation: §11.3 game choice heuristic; report all six held-out
  results, even negative.
* **R4 — Cross-domain Phase-3 zero binding.** If no merged-bank skill
  binds on `video` / `visual_reasoning`, the contract effects must be
  too gymv-grid-specific. Mitigation: paper §6 already calls this out
  as future work; partial result is publishable; skill-bridging
  abstraction (paper's PolySkill cross-ref) is the recovery path.
* **R5 — 35B judge family bias inflates Phase-2 / Phase-3 admit rates.**
  Mitigation: schedule a 5% gpt-5.5 spot-check over the final reports
  (per `common/models.py` "Judge family-bias spot-check"); flip
  `VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5` for the spot-check pass.

## 13. Changelog

* **2026-05-03**: initial draft. Per-game step budget grounded in
  COS-PLAY paper Appendix C / Table 3. Plan structure, bank-merge
  utility, two-budget reporting protocol, OOD cost stratification,
  and 5 risks captured. **3 open decisions** still block launch
  (§11.1 hyperparam audit, §11.3 two new gymv games, §11.4 budget
  trim if needed).
