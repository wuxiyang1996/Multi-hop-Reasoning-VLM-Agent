# Phase-5/6 cross-domain transfer - measurement plan

> **Status:** plan written 2026-05-02. **All six stages shipped**
> (Stage 0: `skill_transfer_test/extract/audits/`; Stages 1-4: per-target
> harness wiring under `harness/`; Stage 5: `archetype_aggregator.py`
> + `labeling_supplement/_phase5_matrix.py`; Stage 6:
> `labeling_supplement/_phase4_transfer_matrix.py` +
> `labeling_supplement/_phase4_transfer_report.py`).
> Builds on
> [`implementation_notes/cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md)
> Section 11.5 (transferability assessment, revised 2026-05-02 after the
> `visual_reasoning_wrapper` audit) and the Phase-1.5 cross-corpus
> skill bank shipped at
> [`skill_transfer_test/skill_bank_local/full_v5/`](../skill_transfer_test/skill_bank_local/)
> (1,083 records / 885 verified across 6 corpora).
> **Cross-refs:**
> [`skill_transfer_test/TODO.md`](../skill_transfer_test/TODO.md)
> (Phase-1.5b deferred items - TODO-1 archetype-aggregator gates
> Stage 5; TODO-6 vocab-jaccard closes in Stage 0),
> [`labeling_supplement/_phase4_transfer_cycle.py`](../labeling_supplement/_phase4_transfer_cycle.py)
> (the within-gymv transfer cycle every Stage extends),
> [`harness/few_shot_adapter.py`](../harness/few_shot_adapter.py)
> (the rebinding mechanism that all Stages call into).

---

## 1. Goal

Fill the missing cells of the Phase-6 transfer matrix (game-source skills x {osworld, browser, image-VR, video-VR} targets, plus within-VR/video 4x4) using the `FewShotAdapter` mechanism that already passed within-gymv (4/6 admits on tetris->2048; 2/3 on 2048->tetris per `labeling_supplement/harness_io_out/_phase4_report.md`). What's missing is **target-side wiring** (executor + schema producer + demos + success_fn) per target.

## 2. Six stages

```
Day  1  2  3  4  5  6  7  8  9 10
S0  [==]                                 pre-flight static audits      [SHIPPED]
S1     [====]                            image-VR live measurement     [SHIPPED]
S2     [======]                          video-VR live measurement     [SHIPPED]
S3              [========]               osworld live measurement      [SHIPPED]
S4              [========]               browsergym live measurement   [SHIPPED]
S5                       [==]            within-VR/video 4x4 + TODO-1  [SHIPPED]
S6                          [====]       full NxN matrix + report      [SHIPPED]
```

Critical path: S0 -> S2 -> S5 -> S6 = 8 days. Stages 1+2 and 3+4 each form parallel pairs.

---

## 3. Stage 0 - Pre-flight static audits *(Day 1, ships in this commit)*

Three scripts in `skill_transfer_test/extract/audits/` that produce upper-bound estimates without any new harness wiring.

| File | Purpose |
|---|---|
| `audits/vocab_jaccard.py` | Reproducible Jaccard audit (closes [`TODO-6`](../skill_transfer_test/TODO.md)). Per-layer overlap (protocol-op / slot-type / predicate-type) for every (source-bank x target-bank) pair. |
| `audits/predicate_firing_static.py` | For each source skill, count how many of its `effects_add` / `effects_del` predicate types match the target domain's *aspirational* `success_fn` vocabulary. Emits `cell_max_admit_rate[source x target]`. |
| `audits/slot_binding_feasibility.py` | For each skill's `${slot}` payloads + `slot_types`, check whether the target's predicate evaluators + schema accept that slot-type. Emits `cell_max_bind_feasible[source x target]`. |
| `audits/_runner.py` | Runs all three; emits combined `cross_domain_results/_phase0/<run_id>/upper_bounds.csv`. |

**Acceptance**:
- emits `cross_domain_results/_phase0/<run_id>/upper_bounds.csv` with one row per (source_corpus, target_domain) cell
- emits per-audit JSON + per-skill JSONL for debugging
- gracefully handles missing game banks (`labeling/skill_bank_out/run_*/` is gitignored - emits within-cross-domain numbers + a "missing inputs" warning if absent)
- closes `TODO-6` (vocab_jaccard half)

**Downstream contract**: every Stage 1-6 measured admit rate must satisfy `measured <= upper_bound + slack(0.10)`. Violation indicates either a Stage 0 vocabulary table is wrong or the Stage's success_fn is over-permissive.

**Dependencies**: none. Runs against shipped banks today.

---

## 4. Stage 1 - Image-VR live measurement *(Days 2-3, parallel with S2)*

Cheapest mechanism-bound probe. **No adapter work, no schema producer** - `bind_visual_reasoning_executor` already wires the real 461-LOC `VisualReasoningExecutor`.

| File | Deliverable | LOC |
|---|---|---:|
| `harness/qa_success.py` | `make_qa_success_fn(domain)` + `register_success_fn("visual_reasoning", ...)`. Predicate evaluators read the `_DerivationLog` produced by `VisualReasoningExecutor.run()` and the grounded-entities table; MCQ exact-match for VTB, LLM-judge for TIR-Bench open-ended | ~150 |
| `harness/few_shot_demos_vr.py` | Walks `Cold-start-out-visual-reasoning/<run>/{visual_toolbench,tir_bench}/` to produce `FewShotDemo[]`; `state = {image, question, choices}`; `bindings` extracted from the actor's emitted entity refs | ~150 |
| `labeling_supplement/_phase4_transfer_cycle.py` | Add `--target visual_reasoning` branch | ~40 |

**Acceptance** (measured 2026-05-02 via Stage 5/6 smoke runs against the shipped stub-tier executor):
- `2048 -> tir_bench` produces a non-empty admit/reject decision matrix (>=1 admit, >=1 reject) -- **PASS** (Stage 5's smoke produced 4 cells with verdicts each)
- Within-image-VR `vtb -> tir_bench` >=30% admit rate -- **FAIL** (0% via stub identity-pass; expected with the Stage 1 stub executor which returns identity-mapped predicates)
- Game->image-VR cell produces a measured rate in [10%, 50%]; flag if outside Stage 0's upper bound + slack -- **FAIL** (Stage 6's smoke measured 100% on game->tetris stub-pathology vs Stage 0 cap of 18%; the flagging mechanism G6 fired correctly)

Stage 1 ships a deterministic-stub executor at runtime: `bind_visual_reasoning_executor` is wired but the per-sample image-loading is not (per `labeling_supplement/_phase4_target_dispatch.py::_build_visual_reasoning_target`: *"Adapter is left on its inherited stub executor; bind_visual_reasoning_executor requires a per-sample PIL.Image we don't yet load."*), so the FAIL verdicts above are about stub-tier behaviour rather than infrastructure bugs. Stage 0's `cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv` row `tetris,game,visual_reasoning` caps the admit rate at 14.29% (slack-extended to ~24%); the stub's measured 100% is exactly the kind of upper-bound violation G6 was built to catch, and will resolve once a real per-sample image-loading executor lands.

**Total**: ~340 LOC, ~2 days.

---

## 5. Stage 2 - Video-VR live measurement *(Days 2-4, parallel with S1)*

Second-cheapest. Only missing piece is a `VideoExecutor` mirror - `tools_video_visual.build_video_visual_registry(frames=...)` ships today as a strict superset of the image registry.

| File | Deliverable | LOC |
|---|---|---:|
| `harness/video_executor.py` *(new)* | Port `VisualReasoningExecutor` against `build_video_visual_registry(frames, include_reasoning=True)`. Same `HopExecutor` interface; dispatches GROUND/RETRIEVE/CHECK/VERIFY/COMMIT/EXECUTE to video tool calls (`get_frame`, `find_moment`, `track_object`, `compare_frames` plus all image tools) | ~150-200 |
| `harness/adapters/video_adapter.py` | Add `bind_video_executor(adapter, *, frames)` helper analogous to `bind_visual_reasoning_executor` | ~20 |
| `harness/few_shot_demos_video.py` | Walks `Cold-start-out-visual-reasoning-video/<run>/{video_holmes,siv_bench}/` | ~150 |
| `harness/qa_success.py` *(extension from S1)* | `register_success_fn("video", ...)` reusing MCQ + LLM-judge helpers; adds video-specific predicates (`temporal_ordering_correct`, `frame_referent_grounded`) | ~50 |
| `labeling_supplement/_phase4_transfer_cycle.py` | Add `--target video` branch | ~40 |

**Acceptance** (measured 2026-05-02 via Stage 5/6 smoke runs against the shipped stub-tier executor):
- Within-video-VR `video_holmes -> siv_bench` >=30% admit rate -- **FAIL** (0% via stub identity-pass; expected with Stage 2's deterministic-stub executor which mirrors the image executor and does not decode video frames or call a VLM)
- Game->video-VR cross-cluster in [10%, 50%] -- **N-A in the 4x4 smoke** (Stage 6 G4 explicitly notes "N/A in the smoke (no video cells); evaluable when video corpora are passed"); will FAIL by the same stub-pathology as Stage 1's game->image-VR row when video corpora are exercised.

Stage 2 ships a deterministic-stub video executor (`harness/video_executor.py` -- the docstring frames it as keeping "the executor *deterministic* -- it does not actually decode video frames or call a VLM yet"), so the FAIL verdict above is about stub-tier behaviour rather than infrastructure bugs. Stage 0's `cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv` row `tetris,game,video` caps the admit rate at 14.29%; the [10%, 50%] criterion is a post-translation-layer aspirational target (cross-ref Section 11.5.4 of the rollout memo) that requires a real frame-decoding executor to evaluate.

**Dependencies**: shares `qa_success.py` scaffolding from S1 (start S1 first to avoid file conflict).

**Total**: ~410-460 LOC, ~3 days.

---

## 6. Stage 3 - OSWorld live measurement *(Days 5-7, parallel with S4)*

First real-env target. Heaviest of the four because the desktop ontology needs a fresh schema producer.

| File | Deliverable | LOC |
|---|---|---:|
| `harness/osworld_executor.py` | `make_osworld_executor(env)` translating typed hops to `pyautogui.{click, press, write, hotkey}` + window-manager queries. Wires into `harness/adapters/osworld_adapter.py::set_executor` | ~250 |
| `harness/osworld_schema_producer.py` | Emits `entity_label_count[window]`, `attribute_changed[focused_app]`, `phase=running\|saved\|aborted`, `entity_appeared[label=dialog]` from OSWorld step info + A11y tree | ~250 |
| `harness/few_shot_demos_osworld.py` | Walks `Cold-start-out-osworld/<run>/`; `state` from desktop schema producer; `bindings` from `pyautogui` action heads | ~150 |
| `harness/osworld_success.py` | `register_success_fn("osworld", ...)` with predicate evaluators for the desktop ontology | ~120 |
| `labeling_supplement/_phase4_transfer_cycle.py` | Add `--target osworld` branch | ~40 |

**Acceptance** (measured 2026-05-02 via Stage 6 smoke against the shipped stub-tier executor):
- `castlevania -> chrome` (or analogous game->osworld pair) produces a real admit rate, end-to-end live (no stub echoes) -- **FAIL** (directly violated by `harness/osworld_executor.py`'s shipped "deterministic-stub binding for the OsworldAdapter ... we do not actually invoke pyautogui or any real desktop tool"; the chain runs end-to-end but the rate is stub-echoed, not real)
- Within-osworld task transfer (`vlc -> chrome`) >=40% admit rate -- **FAIL** (0% via stub identity-pass; the executor returns `ok: True` plus a placeholder `EvidenceRef` regardless of action verb)
- Game->osworld in [20%, 60%] -- **FAIL** (Stage 0 caps every `<game>,game,osworld` cell at <=18% in `upper_bounds.csv`; the [20%, 60%] band is a post-translation-layer aspirational target, not a feasibility upper bound -- see Section 11.5.4 of the rollout memo)

Stage 3 ships a deterministic-stub OSWorld executor (`harness/osworld_executor.py` docstring: *"deterministic-stub binding for the OsworldAdapter ... Real OSWorld binding lands in a later cut via OsworldAdapter.set_executor"*), so the FAIL verdicts above are about stub-tier behaviour rather than infrastructure bugs. Stage 0's `cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv` rows for game->osworld already cap the feasibility upper bound at 0-18% across all 13 game corpora -- which is why the [20%, 60%] criterion needs Section 11.5.4's post-translation-layer reading rather than the Stage 0 oracle reading.

**Risk**: A11y tree dumps may not be in the cold-start trace; schema producer would then need to inspect live VM state on every step. Budget +1 day if so.

**Total**: ~810 LOC, ~3 days.

---

## 7. Stage 4 - BrowserGym live measurement *(Days 5-7, parallel with S3)*

Second real-env target. Slightly cheaper than OSWorld because AXTree state is structured.

| File | Deliverable | LOC |
|---|---|---:|
| `harness/browsergym_executor.py` | Translates typed hops to AXTree bid actions (`click(bid)`, `fill(bid, text)`, `scroll(dx, dy)`) | ~200 |
| `harness/browser_schema_producer.py` | Emits `entity_appeared[bid]`, `attribute_changed[focused_node]`, `phase_transitioned` from AXTree diffs | ~180 |
| `harness/few_shot_demos_browsergym.py` | Walks `Cold-start-out-browsergym/<run>/` | ~120 |
| `harness/browser_success.py` | `register_success_fn("browser", ...)` | ~100 |
| `labeling_supplement/_phase4_transfer_cycle.py` | Add `--target browsergym` branch | ~40 |

**Acceptance** (measured 2026-05-02 via Stage 6 smoke against the shipped stub-tier executor):
- Same shape as S3 (real admit rate end-to-end / within-cluster >=40%) -- **FAIL** (`harness/browsergym_executor.py` ships as "BrowserGym hop executor -- deterministic-stub stage-1 cut" and "does **not** drive a real browser via Playwright"; same stub-pathology as Stage 3)
- Game->browsergym cross-cluster in [15%, 45%] -- **FAIL** (Stage 0 caps `<game>,game,browser` at 0-18% in `upper_bounds.csv`; the [15%, 45%] band is a post-translation-layer aspirational target, not a feasibility upper bound -- see Section 11.5.4 of the rollout memo)

Stage 4 ships a deterministic-stub BrowserGym executor (`harness/browsergym_executor.py` docstring: *"deterministic-stub stage-1 cut"*; Playwright wiring is deferred to "later by replacing the closure `make_browsergym_executor` returns"), so the FAIL verdicts above are about stub-tier behaviour rather than infrastructure bugs. Same caveat as Stage 3: Stage 0's `cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv` already caps game-source rows on browser targets at 0-18%, well below the [15%, 45%] aspirational band; replacing the stub flips the gates.

**Total**: ~640 LOC, ~3 days.

---

## 8. Stage 5 - Within-VR/video 4x4 + cross-corpus archetype bank *(Day 8 - SHIPPED 2026-05-02)*

Closes Experiment B (declarative-reasoning transfer) and unblocks `TODO-1` simultaneously.

| File | Deliverable | Status | LOC |
|---|---|---|---:|
| `skill_transfer_test/extract/archetype_aggregator.py` *(closes TODO-1)* | Cluster per-sample skills into archetype banks for VR/video (4 corpora x archetype clusters); needed for the cross-corpus 4x4 cells | shipped | 376 |
| `labeling_supplement/_phase5_matrix.py` *(new)* | Cross-product driver over `(source_corpus, target_corpus)` from `{visual_toolbench, tir_bench, video_holmes, siv_bench}`. Reuses the Stage 1-4 dispatcher (`build_target`) + `_run_transfer` from `_phase4_transfer_cycle`. Loads source skills from `skill_transfer_test/skill_bank_local/<run>/<corpus>/{archetype,per_sample}/skill_bank.jsonl`. Emits `cross_domain_results/_phase5/<run_id>/{cells.json, cells.md, per_skill.jsonl}`. | shipped | 462 |
| `labeling_supplement/_harness_io_helpers.py::record_from_bank_entry` | Read both legacy (`eff_add`/`eff_del`) and cross-domain (`effects_add`/`effects_del`) contract spellings | shipped | +12 |

**Acceptance** (measured 2026-05-02 with `--bank-kind archetype --max-skills 5 --k 2`):
- All 16 within-cluster cells produce non-trivial admit rates -- **PASS** (5 verdicts each except VTB-as-source which contributes 2)
- Diagonal cells >=80% -- **FAIL** (0% across the board; expected with Stage 1+2 stub executors which return identity-mapped predicates)
- Off-diagonal within-cluster >=30% -- **FAIL** (same reason)
- Cross-cluster game-source cells >=15% (per the Section 11.5.6 revised floor) -- **N/A in Stage 5**; deferred to Stage 6 which adds the game-source rows.

The 0% admit rates do **not** indicate a bug in the matrix infrastructure -- they indicate that Stage 1+2 ship reproduction-quality stub executors (`harness/qa_success.py` + the `bind_visual_reasoning_executor` shim) which currently emit `pass_rate=0.00 diag='adaptation_overfitting'` for every cell. Stage 6 closes the loop by replacing the stubs with real executors, after which the Acceptance gates above flip to PASS.

**Within-corpus archetype counts** (measured 2026-05-02):
- `tir_bench`: 11 archetypes / 105 members -- PASS (≥3)
- `video_holmes`: 7 archetypes / 396 members -- PASS
- `siv_bench`: 10 archetypes / 220 members -- PASS
- `visual_toolbench`: 2 archetypes / 31 members -- FAIL (≥3 gate); LLM-clustered fallback deferred to Phase-2.

**Dependencies**: S1 + S2 (VR/video success_fn + demos must be live).

**Total**: ~850 LOC shipped (vs ~330 estimated -- the matrix driver was bigger than budgeted because it also handles markdown rendering + per-cell error capture; ships as a separate sibling module rather than inlined in `_phase4_transfer_cycle.py` to keep both CLIs single-purpose).

---

## 9. Stage 6 - Full NxN matrix + report *(Days 9-10 - SHIPPED 2026-05-02)*

| File | Deliverable | Status | LOC |
|---|---|---|---:|
| `labeling_supplement/_phase4_transfer_matrix.py` | Driver that runs every (source_bank x target_corpus) cell, writes per-cell PASS/FAIL/ADMIT_RATE to `cross_domain_results/_final/<run_id>/cells.json`. Generalises Stage 5 to span heterogeneous source banks (env_wrappers + gym_v + cross-domain) AND all 5 target_domains. Includes `--include-gym-v` flag for the 13 Temporal_*-v0 retro games. | shipped | 813 |
| `labeling_supplement/_phase4_transfer_report.py` | Generates the section 11.5.4 Experiment-A/B/C summary table + per-cluster heatmap; checks against Section 11.5.6 acceptance floors AND Stage 0 upper bounds. Loads `cells.json` + `upper_bounds.csv`, joins them on `(source_corpus, target_domain)`, emits 7-section markdown report. | shipped | 683 |
| `cross_domain_results/_final/<run_id>/{cells.json, cells.md, per_skill.jsonl, _report.md}` *(output)* | Auto-generated final outputs (gitignored). The `_report.md` carries: run metadata, Experiment-A 5xN table, Experiment-B 4x4 table, Experiment-C cross-cluster sub-tables, Stage 0 upper-bound comparison (per-cell delta + violation flag), 6 acceptance gates, top-10 admits + reject rationale. | shipped | -- |

**Acceptance** (memo section 11.5.6 floors -- evaluated 2026-05-02 on a 4x4 smoke run):
- **G1** Diagonal cells >=80% admit rate -- partial PASS for game targets (gymv stub returns 100%); FAIL for VR targets (image stubs return 0%)
- **G2** Within-cluster off-diagonal >=30% -- same pattern (game-cluster PASS via stub-pathology; image-cluster FAIL)
- **G3** Cross-cluster game<->image-VR in [15%, 35%] -- FAIL (50% game-source rows are 0% on image targets; 50% image-source rows are 100% on game targets via the stub pathology)
- **G4** Cross-cluster game<->video-VR in [15%, 30%] -- N/A in the smoke (no video cells); evaluable when video corpora are passed.
- **G5** Cross-cluster (QA-source -> game-target) <5% (informative) -- soft-FAIL (100% via stub pathology; per memo this is "expected near-zero" so the 100% is genuinely informative -- it marks the gymv stub executor as needing replacement before any cell here is meaningful)
- **G6** All measured rates <= Stage 0 upper bound + slack(0.10) -- FAIL (8/16 cells violate; ALL game-target cells are flagged because the gymv executor's identity-pass behaviour blows past the Stage 0 cap of 0-18%).

The G6 verdict is precisely the Stage 0 oracle catching the stub-pathological cells, exactly as designed. Stages 1-4 currently ship reproduction-quality stub executors; replacing them with reality-grounded implementations will flip G1/G2 to PASS for the diagonal, drop G3-G5 into the floor bands, and make G6 PASS by construction (real predicates respect the upper-bound cap).

See [`cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md) Section 11.5.0 for the Stage 0 oracle vs Section 11.5.4 runtime estimate asymmetry that explains why G6 fires on stub-pathological cells while Section 11.5.4 still projects 15-35% / 15-30% bands -- the two measure different quantities (vocab feasibility vs post-translation-layer runtime).

**Dependencies**: Stages 0-5 (all shipped).

**Total**: ~1500 LOC shipped (vs ~450 estimated -- the report generator was bigger than budgeted because it carries 7 markdown sections + Stage 0 join logic + 6 acceptance-gate evaluators each with diagnostic notes; the matrix driver was bigger because it spans 3 heterogeneous bank layouts (env_wrappers / gym_v aggregated / cross-domain `<bank-kind>`) instead of a single layout).

---

## 10. Total LOC and effort

| Stage | LOC | Days | Owner-phase from rollout memo |
|---|---:|---:|---|
| Stage 0 (audits) | ~530 | 1 | Phase 1.5b (`extract/audits/`) |
| Stage 1 (image-VR) | ~340 | 2 | Phase 3 (mostly already shipped) |
| Stage 2 (video-VR) | ~410-460 | 3 | Phase 4 measurement subset |
| Stage 3 (osworld) | ~810 | 3 | Phase 5 |
| Stage 4 (browsergym) | ~640 | 3 | Phase 2 |
| Stage 5 (within-VR + archetype) | ~850 *(shipped)* | 1 | Phase 1.5b (TODO-1) + Phase 6 |
| Stage 6 (matrix + report) | ~1500 *(shipped)* | 2 | Phase 6 |
| **Total** | **~3510-3560** est / ~5500 shipped | **~10** | - |

**LOC reconciliation note**: the ~5500 shipped vs ~3510-3560 estimated overage (~+1900 LOC, ~+55%) is concentrated in Stages 5 and 6, both of which carry inline per-stage explanations: Stage 5 (Section 8) explains the matrix-driver overage as "the matrix driver was bigger than budgeted because it also handles markdown rendering + per-cell error capture; ships as a separate sibling module rather than inlined in `_phase4_transfer_cycle.py` to keep both CLIs single-purpose"; Stage 6 (Section 9) explains the report-generator overage as "carries 7 markdown sections + Stage 0 join logic + 6 acceptance-gate evaluators each with diagnostic notes; the matrix driver was bigger because it spans 3 heterogeneous bank layouts (env_wrappers / gym_v aggregated / cross-domain `<bank-kind>`) instead of a single layout". Stages 0-4 shipped within their estimates. The total-row figure is intentionally non-rebudgeted -- the estimate column documents the original sprint plan; the shipped column documents reality after the per-stage scope expansions described in Section 8 and Section 9.

---

## 11. Anti-goals

- **Crafter / unified-skill-gate integration** - Phase 7 of the rollout memo. Out of scope.
- **LoRA training on admitted skills** - that's Phase 8 (coevolution). Run only after Stage 6 produces a non-trivial admitted set.
- **Schema producer / session module / dedicated test suite for video** - rollout memo Section 8.2 lists ~1100 LOC; this plan ships only the measurement-blocker subset (~310-380 LOC).
- **`extract/_unify.py` (TODO-2), test files (TODO-3/4/5)** - orthogonal Phase-1.5b housekeeping; doesn't block any stage.

---

## 12. Code-level implementation gaps post-Stage 6

Phase-5/6 Stages 0-6 all shipped 2026-05-02 -- the infrastructure runs
end-to-end (NxN matrix driver + report generator + 6 acceptance gates
across 5 target domains). However, post-ship analysis exposed that the
*actual transfer mechanism* is not yet measured: 4 of 5 target-domain
executors are deterministic stubs that identity-pass the rebound
contract's predicates rather than touching real envs / VLMs, so every
admit-rate number in `cross_domain_results/_final/run_*/_report.md` is
mechanism-trivial.

This section is the canonical, severity-ranked inventory of code-level
gaps that block §11.5.4 / §11.5.6 transferability bands of the sibling
memo [`cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md)
from becoming measurements rather than projections. Other docs back-link
to §12.X anchors below.

### 12.1 Tier 1 -- The 4 deterministic-stub executors (critical)

| # | Module | Status | What's missing | Acceptance impact |
|---|---|---|---|---|
| 1 | `harness/visual_reasoning` (Stage 1) | Bind helper + 461-LOC `VisualReasoningExecutor` ship; dispatcher does NOT call them | Per-sample `PIL.Image` loading is not wired ([`labeling_supplement/_phase4_target_dispatch.py:236-239`](../labeling_supplement/_phase4_target_dispatch.py): *"Adapter is left on its inherited stub executor; bind_visual_reasoning_executor requires a per-sample PIL.Image we don't yet load"*) | Stage 1 G3 cell measures 0% via stub identity-pass |
| 2 | [`harness/video_executor.py`](../harness/video_executor.py) (Stage 2) | Stub | Doesn't decode video frames or call a VLM. File docstring: *"deterministic -- does not actually decode video frames or call a VLM yet"* | Stage 2 within-video G2 = FAIL (0% across all 4 cells in Stage 5 4x4) |
| 3 | [`harness/osworld_executor.py`](../harness/osworld_executor.py) (Stage 3) | Stub | Doesn't invoke `pyautogui` against a real desktop. File docstring (line 1): *"deterministic-stub binding for the OsworldAdapter"*; line 71: *"deterministic stub doesn't actually step a desktop"* | Stage 3 G3 = 0% on game->osworld for all 13 game corpora |
| 4 | [`harness/browsergym_executor.py`](../harness/browsergym_executor.py) (Stage 4) | Stub | Doesn't call BrowserGym/Playwright. File docstring (line 1): *"deterministic-stub stage-1 cut"* | Stage 4 G3 = 0% on game->browser |

Empirical Stage 6 verdict on the latest run
(`cross_domain_results/_final/run_20260502T085239Z/_report.md`):

```
G1 (diagonal >=80%)              FAIL    4 violators (visual_toolbench->visual_toolbench=0%, tir_bench->tir_bench=0%, plus 2 game->game cells admitting 100% via stub identity-pass)
G2 (within-cluster off-diag>=30%) FAIL    4 violators
G3 (game<->image-VR in [15,35])  FAIL    8 violators
G4 (game<->video-VR in [15,30])  N-A     0 cells in smoke
G5 (QA->game <5%, informative)   soft-FAIL  max=100% (QA->game stub-pathology)
G6 (measured <= upper_bound+slack) FAIL  16 violators (game->game cells measure 100% vs Stage 0 upper bound of 0-18%)
```

All 6 G-gates either FAIL or are N-A on the latest Stage 6 smoke. The
infrastructure is not the bug; the stubs are.

### 12.2 Tier 2 -- Missing `vlm_wrapper/<domain>_adapter.py` files (critical)

The `harness/README.md` §"Suggested work-order" item 16 envisions wiring
real `vlm_wrapper/<domain>_adapter.py` executors into the harness via
`set_executor()`. Reality on disk (`ls vlm_wrapper/`):

```
vlm_wrapper/
+-- gymv_adapter.py       # ships
+-- osworld_adapter.py    # ships (real env)
+-- browser_adapter.py    # ships (real env)
+-- (no video_adapter.py, no visual_reasoning_adapter.py)
```

Even if Tier 1 stubs were swapped out tomorrow, **2 of 4 transfer target
domains have no real-env adapter to swap in**. Video and visual_reasoning
real adapters need to be authored from scratch in `vlm_wrapper/` before
Tier 1 work is meaningful for those targets.

Estimated effort: ~600-800 LOC each (mirroring the shape of
`vlm_wrapper/osworld_adapter.py` and `vlm_wrapper/browser_adapter.py`),
plus a VLM-quality eval harness for verification.

### 12.3 Tier 3 -- Per-domain runtime predicate-translators (critical, design-level)

Section 11.5.0 of the sibling memo
[`cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md)
introduces the concept and admits no implementation exists:

> §11.5.4's 15-35% / 15-30% bands assume the harness ships a per-domain
> runtime predicate-translator that bridges the predicate-name mismatch
> (e.g. `score_increased` becomes `answer_emitted` when the target is
> image-VR).

Without these translators, even a real (Tier 1 + Tier 2) executor will
reject every game->VR contract because `score_increased` isn't in
[`harness/qa_success.py`](../harness/qa_success.py)'s vocabulary.
Stage 0 records `upper_bound_admit_rate=0.0` for these cells precisely
because the static-vocab check fires before any executor runs.

**Status:** not designed, not specced. There is no module for predicate
translation anywhere in the tree as of 2026-05-02. The likely shape is
either (a) per-target-domain alias tables in
`harness/<domain>_success.py` or (b) a new `harness/predicate_translator.py`
module called as a pre-pass to `success_fn`. Tier 3 is a hard blocker
for §11.5.4's projected 15-35% / 15-30% bands becoming measured numbers
on game->VR cells -- without it, those cells continue to admit at 0%
even with real Tier 1+2 work shipped.

### 12.4 Tier 4 -- Open `skill_transfer_test/TODO.md` items (medium)

Per [`skill_transfer_test/TODO.md`](../skill_transfer_test/TODO.md):

- **TODO-2 -- `extract/_unify.py`** (open, ~50 LOC). Emits
  `_unified/{skill_index.jsonl, skill_catalog_all.json,
  skill_rag_index.json}` with all 6 corpora tagged distinctly. No blockers.
- **TODO-3 -- `extract/tests/test_corpus_specs.py`** (open, ~50 LOC).
  Validates each `CorpusSpec`. No blockers.
- **TODO-4 -- `extract/tests/test_single_shot_lift.py`** (open, ~150 LOC).
  Golden-file replay across 4 benchmarks. Needs fixtures.
- **TODO-5 -- `extract/tests/test_runner_smoke.py`** (open, ~120 LOC).
  End-to-end smoke. **Blocked on TODO-2.**

Closed by Phase-5/6 ship: TODO-1 (archetype_aggregator, Stage 5), TODO-6
(Stage 0 audits). Cancelled: TODO-7 (run_extract.sh, superseded by Python
CLI).

### 12.5 Tier 5 -- Stage 5 archetype-bank G3 partial-FAIL (low)

[`extract/archetype_aggregator.py`](../skill_transfer_test/extract/archetype_aggregator.py)
ships only the `direct` strategy. VTB has just 2 distinct `eval_focus`
values, so it ships 2 archetypes (acceptance gate requires >=3 -> FAIL
for VTB). The fallback strategy:

> **LLM-clustered** -- call `gpt-5.5` with the sample question + extracted
> protocol, ask for a topic tag, then cluster by tag. **Not shipped**;
> required only when `direct` produces fewer than 3 archetypes (currently
> only VTB).

Tracked as Phase-2 enhancement, gated on API access.
[`labeling_supplement/_phase5_matrix.py`](../labeling_supplement/_phase5_matrix.py)
handles 2-archetype VTB transparently (cells with VTB-as-source contribute
2 verdicts each), so this is non-blocking for the broader Stage 5/6
pipeline.

### 12.6 Tier 6 -- Workspace-wide "Not yet delivered" (pre-existing, unrelated to Phase-5/6)

These predate Phase-5/6 and are tracked in
[`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet
delivered" and `harness/README.md` §"Suggested work-order":

- **`HarnessSkillProvider`** --
  `decision_agents.skill_interface.SkillBankProvider` still queries the
  legacy bank directly. The "harness narrows + may veto" contract is not
  in force at runtime in the `decision_agents` library API. (Note: Day-10
  `SkillHarnessHook` already covers this for the trainer's co-evolution
  loop.)
- **`skill_bank/legacy_bridge.py`** -- closes the legacy `SkillBankMVP`
  ↔ new `SkillRepository` gap.
- **Numeric `fit_score / risk_score` LoRA head** -- Day-9+; per-check
  booleans shipped Day-8a, the numeric scoring head is still a placeholder.
- **Day-10+ first-class planner-context params** -- `intention /
  active_skill / local_reasoning_trace` plumbed via `state.extra`, not
  yet typed first-class params of `select_eligible_skills`.
- **`transfer_manager.py` shadow -> active quarantine** -- Phase-D row
  pending (per `harness/README.md`).
- **gymv real adapter wiring** -- `gymv_wrapper.set_executor()` exists
  but nothing in production calls it; deterministic-stub fallback hasn't
  been replaced with explicit ABORT.
- **Protocol lift (cold-start prose -> typed hops)** -- design locked +
  21-verb taxonomy + 92.5% coverage measured (per
  [`protocol-lift-design.md`](protocol-lift-design.md)),
  but implementation in `labeling/_decorate_skill_records.py` pending.
  Pre-requires gymv adapter wiring.

### 12.7 Severity-ranked summary

| Tier | Severity | Item count | Headline | Estimated effort |
|---|---|---|---|---|
| 1 | **critical** | 4 stub executors | every Stage 6 admit-rate number is currently mechanism-trivial | ~3-5 days per executor + per-domain real-env tests |
| 2 | critical | 2 missing `vlm_wrapper/` adapters | no real env to bind to for video / visual_reasoning | ~1 week per adapter + VLM eval harness |
| 3 | critical (design-level) | predicate translator | game->VR cells will admit at 0% without it even if Tier 1+2 land | design-and-spec sprint, then ~200-400 LOC |
| 4 | medium | TODO-2/3/4/5 | mostly missing tests; TODO-2 is a 50-LOC quality-of-life gap | ~2-3 days |
| 5 | low | LLM-clustered archetype fallback | VTB-only G3 FAIL; workaround in Stage 5 driver exists | ~150 LOC + API budget |
| 6 | medium-to-critical | pre-Phase-5/6 backlog | runtime "harness in-the-loop" rule still not enforced in `decision_agents` library API | tracked separately in `IMPLEMENTATION-STATUS.md` and `pre-training-readiness-audit.md` |

Tiers 1 + 2 + 3 collectively are why the Phase-5/6 *measurement*
infrastructure shipped in 4 commits today but **does not yet measure
anything**. Closing them is a multi-week effort: roughly 2-3 weeks for
real `vlm_wrapper` adapters + Tier 1 executor swaps + Tier 3 per-domain
predicate-translators + a few-hundred-sample re-run of Stage 6 to retire
G6.

---

## 13. Output layout

`cross_domain_results/` (gitignored alongside `skill_bank_local/`):

- `_phase0/<run_id>/` - Stage 0 outputs (vocab_jaccard.json/.md, predicate_firing_static.json, predicate_firing_per_skill.jsonl, slot_binding_feasibility.json, slot_binding_per_skill.jsonl, upper_bounds.csv)
- `_phase1_image_vr/<run_id>/` - Stage 1 outputs (cells.parquet, _report.md)
- `_phase2_video_vr/<run_id>/` - Stage 2 outputs
- ...
- `_final/<run_id>/` - Stage 6 unified report (cells.parquet, _report.md)

`cross_domain_results/` is gitignored; the README + this memo are the checked-in source of truth for what each output should look like.

---

## 14. TL;DR

Six stages over ~10 days. Stage 0 ships now (~530 LOC of static audits, no harness wiring). Sprint 1 (Days 2-4) parallelises image-VR and video-VR measurement (~750-800 LOC; cheapest because the tool registry is already the env). Sprint 2 (Days 5-7) parallelises OSWorld and BrowserGym (~1450 LOC; needs full executor + schema producer per target). Sprint 3 (Days 8-10) closes within-VR/video 4x4 + emits the final transfer matrix.

Stage 0's upper bounds are the oracle every later Stage's measured admit rate must respect - violations indicate either a wrong vocabulary table or an over-permissive success_fn.

**Post-ship reality check:** all 6 stages shipped on 2026-05-02 with deterministic-stub executors -- the infrastructure runs end-to-end but the underlying mechanism is not yet measured. See §12 for the canonical, severity-ranked inventory of what's still unimplemented (Tier 1: 4 stub executors; Tier 2: 2 missing `vlm_wrapper/<domain>_adapter.py`; Tier 3: per-domain runtime predicate-translators).
