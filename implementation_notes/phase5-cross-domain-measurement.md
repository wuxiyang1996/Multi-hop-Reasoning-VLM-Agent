# Phase-5/6 cross-domain transfer - measurement plan

> **Status:** plan written 2026-05-02. Stage 0 ships in this commit
> (`skill_transfer_test/extract/audits/`); Stages 1-6 are queued.
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
S0  [==]                                 pre-flight static audits
S1     [====]                            image-VR live measurement
S2     [======]                          video-VR live measurement
S3              [========]               osworld live measurement
S4              [========]               browsergym live measurement
S5                       [==]            within-VR/video 4x4 + TODO-1
S6                          [====]       full NxN matrix + report
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

**Acceptance**:
- `2048 -> tir_bench` produces a non-empty admit/reject decision matrix (>=1 admit, >=1 reject)
- Within-image-VR `vtb -> tir_bench` >=30% admit rate
- Game->image-VR cell produces a measured rate in [10%, 50%]; flag if outside Stage 0's upper bound + slack

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

**Acceptance**:
- Within-video-VR `video_holmes -> siv_bench` >=30% admit rate
- Game->video-VR cross-cluster in [10%, 50%]

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

**Acceptance**:
- `castlevania -> chrome` (or analogous game->osworld pair) produces a real admit rate, end-to-end live (no stub echoes)
- Within-osworld task transfer (`vlc -> chrome`) >=40% admit rate
- Game->osworld in [20%, 60%]

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

**Acceptance**: same shape as S3. Game->browsergym cross-cluster in [15%, 45%].

**Total**: ~640 LOC, ~3 days.

---

## 8. Stage 5 - Within-VR/video 4x4 + cross-corpus archetype bank *(Day 8)*

Closes Experiment B (declarative-reasoning transfer) and unblocks `TODO-1` simultaneously.

| File | Deliverable | LOC |
|---|---|---:|
| `skill_transfer_test/extract/archetype_aggregator.py` *(closes TODO-1)* | Cluster per-sample skills into archetype banks for VR/video (4 corpora x archetype clusters); needed for the cross-corpus 4x4 cells | ~250 |
| `labeling_supplement/_phase4_transfer_cycle.py` | Add `--source <vr_corpus> --target <vr_corpus>` matrix-mode loop; produces 4x4 within-cluster table | ~80 |

**Acceptance**:
- All 16 within-cluster cells produce non-trivial admit rates
- Diagonal cells >=80%
- Off-diagonal within-cluster >=30%
- Cross-cluster game-source cells >=15% (per the Section 11.5.6 revised floor)

**Dependencies**: S1 + S2 (VR/video success_fn + demos must be live).

**Total**: ~330 LOC, ~1 day.

---

## 9. Stage 6 - Full NxN matrix + report *(Days 9-10)*

| File | Deliverable | LOC |
|---|---|---:|
| `labeling_supplement/_phase4_transfer_matrix.py` | Driver that runs every (source_bank x target_corpus) cell, writes per-cell PASS/FAIL/ADMIT_RATE to `cross_domain_results/<run_id>/cells.parquet` | ~200 |
| `labeling_supplement/_phase4_transfer_report.py` | Generates the Section 11.5.4 Experiment-A/B/C summary table + per-cluster heatmap; checks against Section 11.5.6 acceptance floors AND Stage 0 upper bounds | ~250 |
| `cross_domain_results/<run_id>/_report.md` *(output)* | Auto-generated final report with: Experiment-A 5x5 table, Experiment-B 4x4 table, Experiment-C cross-cluster table, comparison to Stage 0 upper bounds, per-skill admit/reject rationale |

**Acceptance** (the Section 11.5.6 floors):
- Diagonal cells >=80% admit rate
- Within-cluster off-diagonal >=30%
- Cross-cluster (game <-> VR/video, **revised 2026-05-02**): 15-35% (image) / 15-30% (video)
- Cross-cluster (QA-source -> game-target): expected near-zero (genuine mismatch)
- All measured rates <= Stage 0 upper bound + slack(0.10)

**Total**: ~450 LOC, ~2 days.

---

## 10. Total LOC and effort

| Stage | LOC | Days | Owner-phase from rollout memo |
|---|---:|---:|---|
| Stage 0 (audits) | ~530 | 1 | Phase 1.5b (`extract/audits/`) |
| Stage 1 (image-VR) | ~340 | 2 | Phase 3 (mostly already shipped) |
| Stage 2 (video-VR) | ~410-460 | 3 | Phase 4 measurement subset |
| Stage 3 (osworld) | ~810 | 3 | Phase 5 |
| Stage 4 (browsergym) | ~640 | 3 | Phase 2 |
| Stage 5 (within-VR + archetype) | ~330 | 1 | Phase 1.5b (TODO-1) + Phase 6 |
| Stage 6 (matrix + report) | ~450 | 2 | Phase 6 |
| **Total** | **~3510-3560** | **~10** | - |

---

## 11. Anti-goals

- **Crafter / unified-skill-gate integration** - Phase 7 of the rollout memo. Out of scope.
- **LoRA training on admitted skills** - that's Phase 8 (coevolution). Run only after Stage 6 produces a non-trivial admitted set.
- **Schema producer / session module / dedicated test suite for video** - rollout memo Section 8.2 lists ~1100 LOC; this plan ships only the measurement-blocker subset (~310-380 LOC).
- **`extract/_unify.py` (TODO-2), test files (TODO-3/4/5)** - orthogonal Phase-1.5b housekeeping; doesn't block any stage.

---

## 12. Output layout

`cross_domain_results/` (gitignored alongside `skill_bank_local/`):

- `_phase0/<run_id>/` - Stage 0 outputs (vocab_jaccard.json/.md, predicate_firing_static.json, predicate_firing_per_skill.jsonl, slot_binding_feasibility.json, slot_binding_per_skill.jsonl, upper_bounds.csv)
- `_phase1_image_vr/<run_id>/` - Stage 1 outputs (cells.parquet, _report.md)
- `_phase2_video_vr/<run_id>/` - Stage 2 outputs
- ...
- `_final/<run_id>/` - Stage 6 unified report (cells.parquet, _report.md)

`cross_domain_results/` is gitignored; the README + this memo are the checked-in source of truth for what each output should look like.

---

## 13. TL;DR

Six stages over ~10 days. Stage 0 ships now (~530 LOC of static audits, no harness wiring). Sprint 1 (Days 2-4) parallelises image-VR and video-VR measurement (~750-800 LOC; cheapest because the tool registry is already the env). Sprint 2 (Days 5-7) parallelises OSWorld and BrowserGym (~1450 LOC; needs full executor + schema producer per target). Sprint 3 (Days 8-10) closes within-VR/video 4x4 + emits the final transfer matrix.

Stage 0's upper bounds are the oracle every later Stage's measured admit rate must respect - violations indicate either a wrong vocabulary table or an over-permissive success_fn.
