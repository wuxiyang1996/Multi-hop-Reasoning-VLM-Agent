# `skill_transfer_test/` — harness usability test, intra-gymv ablation runner

> **Status:** plan only. No code yet. Phase 0 is reversible and can start once
> [D1–D7](#decisions-locked-elsewhere) are confirmed.
> **Last reviewed:** 2026-04-30.
> **Cross-refs (design rationale lives there, not here):**
> [`implementation_notes/harness-usability-and-intra-gymv-transfer.md`](../implementation_notes/harness-usability-and-intra-gymv-transfer.md),
> [`plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) §20 (ablations),
> [`harness/README.md`](../harness/README.md) §16, §17, §21, §22 (audit + suggested work-order),
> [`labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py) (the driver this folder wraps),
> [`labeling_supplement/decide_promotion_gpt54.py`](../labeling_supplement/decide_promotion_gpt54.py) (the harness-free promotion path).

---

## 1. What this folder is

A **measurement layer** that wraps shipped drivers (`dump_harness_io_gpt54.py`,
`decide_skill_crafting_gpt54.py`, `decide_promotion_gpt54.py`) and emits the
three reports defined in [`PLAN-HARNESS.md` §20.6](../plans/05-harness/PLAN-HARNESS.md).
Five ablation cells × intra-gymv probe × four research questions × three
report templates — nothing more, nothing less.

It writes **none** of the audit-trail artefacts (`BankMutationProposal`,
`SkillEpisode`, `GateVerdictPayload`, `SkillEvaluationRecord`, `AuditRecord`,
`bank_snapshot_id`). Those are owned by Crafter / Harness / Orchestrator per
[`crafter-harness-orchestrator-roles.md` §3](../implementation_notes/crafter-harness-orchestrator-roles.md).
On-disk JSONL is the only API.

---

## 2. Folder layout (target)

```
skill_transfer_test/
├── README.md                          ← this file
├── conftest.py                        ← pytest discovery, shared fixtures
│
├── cell_configs/                      ← one YAML per §20.3 cell. No new code paths.
│   ├── a0_no_harness.yaml             ← Actor + bank retrieval only
│   ├── a1_harness_lite.yaml           ← + EligibilityFilter (G1 binding only)
│   ├── a2_harness_core.yaml           ← + G0 evidence + G2 adapter + veto + scoring
│   ├── a3_harness_transfer.yaml       ← + G3 replay + G4 shadow + G3a few-shot (task axis on)
│   └── a4_full_system.yaml            ← + G5 non-regression + promotion / rollback
│
├── runner.py                          ← CLI. Subprocess-invokes existing drivers.
│                                        --cells {a0,a1,a2,a3,a4,all} --probe intra_gymv
│                                        --max-episodes N --max-steps M --sources <list>
│
├── slices.py                          ← §20.5 axis builders.
│                                        in_domain_reuse / cross_domain_transfer / before_promotion
│                                        / after_promotion / easy / hard / per-game.
│
├── metrics/
│   ├── __init__.py
│   ├── validity.py                    ← Q1: invalid_invocation_rate, slot_binding_pass_rate, ...
│   ├── veto.py                        ← Q3: veto precision / recall (where ground truth exists)
│   ├── transfer.py                    ← Q2: transfer_pass_rate, regression_rate_after_transfer, ...
│   └── actor_quality.py               ← Q4: actor top-1 / top-k accuracy on Harness-eligible set
│
├── reports/
│   ├── __init__.py
│   ├── report_a_actor_decision.py     ← §20.6(a)  — per-cell × per-slice numbers
│   ├── report_b_harness_filtering.py  ← §20.6(b)  — needs G0/G2 active (Phase 2+)
│   ├── report_c_system_outcome.py     ← §20.6(c)  — overall reward / pass-rate by cell
│   └── render_summary.py              ← markdown roll-up consumed by humans
│
├── runs/                              ← gitignored. One subdir per invocation.
│   └── <ts>/
│       ├── _run_meta.json             ← argv + cell configs + input run paths (reproducibility contract)
│       ├── <cell>/<corpus>/<source>/  ← per-cell harness IO dumps (forwarded from dump driver)
│       └── reports/{a,b,c}.md         ← rendered §20.6 reports
│
└── tests/
    ├── test_cell_configs_load.py      ← every YAML parses + validates against a schema
    ├── test_metric_q1_validity.py     ← golden-file test on a single source pair
    ├── test_actor_quality_q4.py       ← golden-file test on a single source pair
    └── test_smoke_a0_a4_one_source.py ← Airstriker only, --max-episodes 2 --max-steps 5, end-to-end
```

---

## 3. CLI — what `runner.py` does

```bash
python -m skill_transfer_test.runner \
    --cells a0,a1 \
    --probe intra_gymv \
    --sources Airstriker-v0 \
    --bank-run    labeling/skill_bank_out/run_<ts> \
    --actions-run labeling/skill_actions_out/run_<ts> \
    --max-episodes 2 --max-steps 5 \
    --out-root runs/
```

What it does internally — every cell is a thin shell over **existing** drivers:

| Cell | Inner invocation chain |
|---|---|
| **A0** | read `<bank-run>/<corpus>/<source>/skill_bank.jsonl` directly + read `<actions-run>/.../episode_*.json` `skill_query.selected_skill_id`; compute Q1 / Q3 / Q4 from those. **No driver call.** |
| **A1** | `dump_harness_io_gpt54.py --surface online --disable-g0 --disable-g2 --disable-transfer` |
| **A2** | `dump_harness_io_gpt54.py --surface online` (default cell — all gates default-on except transfer) |
| **A3** | `dump_harness_io_gpt54.py --surface offline` over `crafter_proposals_out/`, with `--enable-g3a-task-axis` |
| **A4** | A3 + `decide_promotion_gpt54.py --gate-mode external --gate-verdicts-run <a3_out>` |

The **runner.py is a dispatcher**. It does not contain harness logic. Harness logic
lives in `harness/` and `decide_promotion_gpt54.py`. See
[`crafter-harness-orchestrator-roles.md` §8](../implementation_notes/crafter-harness-orchestrator-roles.md):
no driver under `skill_transfer_test/` may import another driver's code or write
into another driver's output directory.

---

## 4. Cell configs — schema

Every `cell_configs/*.yaml` is a flat dict. The runner translates it to
CLI flags for the underlying driver. Keep it boring.

```yaml
# cell_configs/a2_harness_core.yaml
cell_id: a2
human_label: harness-core
driver: dump_harness_io_gpt54
surface: online
gates:
  g0_evidence: true
  g1_binding:  true
  g2_adapter:  true
  g3_replay:   false      # A3+
  g3a_transfer: false     # A3+
  g4_shadow:   false      # A3+
  g5_non_regression: false  # A4+
scoring:
  fit_score:   true       # PLAN-HARNESS §1a.5 — Actor must still do work
  risk_score:  true
veto:
  enable: true            # invocation-time veto
notes: |
  A1→A2 delta = G0 evidence + G2 adapter + veto + advisory scoring.
  Phase 2 prerequisite: harness audit §21 (protocol lift) must land first
  or G0 is degenerate (zero hops in iter_hops()).
```

Cell-pair deltas that matter (canon §20.3):
A0→A1 = structural validation alone; A1→A2 = G0+veto contribution;
A2→A3 = transfer-safety contribution; A3→A4 = promotion/rollback contribution.

---

## 5. Phased rollout

Verbatim from
[`harness-usability-and-intra-gymv-transfer.md` §7](../implementation_notes/harness-usability-and-intra-gymv-transfer.md);
restated here as work items.

| Phase | Maps to | Deliverable here | Cells active | Blocking prereq |
|---|---|---|---|---|
| **0** Pre-investment check | Suggested-work-order #5 (shipped) | `runner.py --cells a0,a1` end-to-end on smoke slice (Airstriker, `--max-episodes 2 --max-steps 5`); reports §20.6(a) + (c) on `in_domain_reuse` | A0, A1 | none — runs today |
| **1** Protocol lift | Suggested-work-order #6 + harness/README §21 | (no work in this folder — landed in `labeling/_decorate_skill_records.py`-style transformer) | unblocks A2 | needs upstream lift |
| **2** Task axis | Suggested-work-order #7 + harness/README §22 | (no work in this folder — landed in `data_structure/extensions/skill_record.py` + `harness/eligibility.py` + `harness/few_shot_adapter.py`) | unblocks A3 | needs upstream additive contract change |
| **3** gymv real executor | Suggested-work-order #8 (first half) + harness/README §16.1 | smoke through `tests/test_smoke_a0_a4_one_source.py` once executor wired | A0, A1, A2 honest; A3 transferable | needs `GymvAdapter.set_executor` plumbed from `cold_start/generate_cold_start_actor_gymv.py` |
| **4** Stage 3a probe | Suggested-work-order #8 (second half) | gymv-shape `success_fn` + `FewShotDemo` builder over `labeling/skill_actions_out/.../episode_*.json` | A3 transfer cell active | depends on Phase 2 + Phase 3 |
| **5** Full sweep + reports | Suggested-work-order #5 (offline half) | `runner.py --cells all --sources <13 games>` + §20.6(a)(b)(c) reports. **This is the first offline promotion cycle** ([`harness/README.md` §17](../harness/README.md)). | A4 reference cell | depends on Phases 1–4 |
| **6** Cross-domain follow-up | Suggested-work-order #16 | swap `--probe intra_gymv` → `--probe cross_domain` once per-domain executors land | future arena | depends on Phase 5 + each transfer-target adapter |

---

## 6. Acceptance gates (don't start phase N+1 until phase N passes)

| Phase | Gate | How to check |
|---|---|---|
| **0** | A0 vs. A1 numbers differ on `in_domain_reuse`; both reports render | `cat runs/<ts>/reports/a.md` shows non-zero `delta_a0_a1` row |
| **3** | `tests/test_smoke_a0_a4_one_source.py` passes: 2 episodes × 5 steps × Airstriker, no crash, A2 produces non-stub `SkillEpisode`s | `pytest skill_transfer_test/tests/test_smoke_a0_a4_one_source.py -xvs` |
| **5** | All 5 cells × 13 games complete; `bank_snapshot/<id>/` for each `(corpus, source)` is non-empty after A4 | `runs/<ts>/_run_meta.json:n_promoted_skills > 0` |

---

## 7. Decisions locked elsewhere

These are **not re-litigated here**. The runner respects whatever was decided.

| ID | Decision | Pinned in |
|---|---|---|
| D1–D7 | bridge direction, status filter, K, N, compose-reject default, failure-synth home, reflection-builder home | [`implementation_notes/harness-usability-and-intra-gymv-transfer.md` §8](../implementation_notes/harness-usability-and-intra-gymv-transfer.md) |
| D8 | one-way `legacy_writeback.py` (Promotion → per-game `skill_bank.jsonl`) | trainer wire-up convo (next plan to write up) |
| D9 | Crafter alongside Stage 4 curator (not replacing yet) | trainer wire-up convo |

---

## 8. Limitations of this configuration (state in every report)

1. **Skills never reach `ACTIVE`** under `--gate-mode offline-synthetic`. Cap is `PROVISIONAL`.
2. **No invocation veto in real time.** Cells A2–A4 measure veto *as if* it had been live; they don't actually stop a bad call.
3. **No transfer probes** until Phase 2+3 land. Cells A3/A4 numbers are not meaningful before then.
4. **`EpisodeReflection.skill_episodes = []`** because no Harness emits them ([`§7.1` mismatch #1](../implementation_notes/crafter-harness-orchestrator-roles.md)). Q1/Q4 are computed against the synthesized `FailureTrace`s, not observed `SkillEpisode`s.
5. **`ROLLBACK` cannot fire** without batch metrics from a real gate stack. A4 reports promotion-precision only, not rollback-reactivity.
6. **Compose / Transfer / Generalize proposals are auto-rejected** in Phase 0 per D5 (cold-start `feasible_domains=["gymv"]` ⇒ Stage-0 fails). Don't read the zero count as a regression.

When Harness lands, every limit lifts mechanically. No re-architecting in this folder.

---

## 9. Anti-goals (mirrors `PLAN-HARNESS.md` §20.8)

- **Do not** build a parallel transfer framework here. The runner is a dispatcher.
  Adapter logic, success scorers, and proposal mints belong in `harness/`,
  `labeling/`, and `crafter/`.
- **Do not** redefine cell semantics, gate thresholds, or ablation metrics
  inside this folder. Those live in
  [`PLAN-HARNESS.md` §5.1, §10, §20](../plans/05-harness/PLAN-HARNESS.md) and
  [`PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md).
  If the runner needs different semantics, fix the upstream plan first.
- **Do not** skip Phase 0 because A2–A4 sound more interesting. Per
  [`PLAN-HARNESS.md` §20.7](../plans/05-harness/PLAN-HARNESS.md), if `A4 − A0 ≈ 0`
  on the smoke slice the rest of the suite is not worth running — Phase 0 is
  the cheapest way to find that out.
- **Do not** collapse Q1, Q2, Q3, Q4 into one number. The point of the suite
  is that they are separable.
- **Do not** write into `verified_domains` / `bank_snapshot_id` / `AuditRecord`.
  Those are the Orchestrator's exclusive surface
  ([`crafter-harness-orchestrator-roles.md` §3](../implementation_notes/crafter-harness-orchestrator-roles.md)).
  This folder reads them.

---

## 10. TL;DR

- **Folder = measurement layer.** No new harness logic, no new bank writes.
- **`runner.py` = dispatcher** over `dump_harness_io_gpt54.py` + `decide_promotion_gpt54.py`.
- **Phase 0 is dominated** (1–4 hr, $0 API spend, smoke slice). Run it before
  investing in Phases 1–4.
- **Phases 1–3 land outside this folder** — protocol lift, task axis, gymv
  executor are upstream changes consumed here.
- **Phase 5 = first offline promotion cycle.** Same execution graph; satisfies
  [`harness/README.md` §17](../harness/README.md) keystone (`bank.runnable()`
  becomes non-empty).
- **Limits at §8** must be stated in every report. Don't quietly outgrow them.

If you're opening this folder to start Phase 0, the next step is:

1. Confirm D1–D7 in
   [`implementation_notes/harness-usability-and-intra-gymv-transfer.md` §8](../implementation_notes/harness-usability-and-intra-gymv-transfer.md).
2. Skim
   [`labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py)
   to understand its existing CLI surface.
3. Implement `cell_configs/a0_no_harness.yaml` + `cell_configs/a1_harness_lite.yaml`
   + a 50-line `runner.py` dispatcher + `tests/test_smoke_a0_a4_one_source.py`
   in smoke mode (only A0/A1).
4. Run on Airstriker, eyeball `runs/<ts>/reports/a.md`, decide whether to invest in Phase 1.
