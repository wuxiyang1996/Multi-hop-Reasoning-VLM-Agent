# `skill_transfer_test/TODO.md` — Phase 1.5b deferred items

> **Scope:** open follow-ups from Phase 1.5 (LLM-free cross-corpus skill
> bank lift, shipped 2026-05-01 — `skill_transfer_test/extract/`,
> 1,083 records / 885 verified across 6 corpora).
>
> **Single source of truth for status only.** Each item below is described
> in detail in the design memo
> [`implementation_notes/cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md)
> §5.5.4 / §5.5.4a / §5.5.7a / §11.5.1. Treat the memo as authoritative for
> *what* each item is and *why*; treat this file as authoritative for
> *whether it has shipped*. When you ship one, flip the checkbox here and
> link the commit/PR — do **not** delete the row from §5.5.4a in the memo
> (the memo records the original spec; this file records execution state).
>
> **Status legend:** `[ ]` open · `[~]` in progress · `[x]` shipped · `[-]` cancelled (superseded).
>
> **Last updated:** 2026-05-02 (file created consolidating the seven items
> previously scattered across §5.5.4a, §5.5.7a, and §11.5.1 of the rollout
> memo; later same-day, the rollout memo §11.5.4 estimates were revised
> upward for game→VR/video transfer after a `visual_reasoning_wrapper`
> audit; later still, **TODO-6 closed** as part of Stage 0 of the
> [Phase-5/6 measurement plan](../implementation_notes/phase5-cross-domain-measurement.md)
> — three audit scripts shipped under `skill_transfer_test/extract/audits/`).

---

## Phase 1.5b — open

### `[ ]` TODO-1 — `extract/archetype_aggregator.py`

Emit the **archetype bank kind** for VR/video corpora (clusters of
per-sample skills). Two strategies, selectable per `CorpusSpec`:

- **direct** — group by `raw_sample.question_type` (Video-Holmes, SIV-Bench)
- **LLM-clustered** — call `gpt-5.5` with the sample question + extracted
  protocol, ask for a topic tag, then cluster by tag (VTB, TIR-Bench)

| Field | Value |
|---|---|
| **Output** | `skill_transfer_test/skill_bank_local/<run_id>/<corpus>/archetype/skill_bank.jsonl` |
| **Acceptance** | ≥ 3 archetypes per VR/video corpus; rollout memo §5.5.7a "Archetype bank size" row flips PASS |
| **Memo refs** | §5.5.2 (granularity), §5.5.4 (file table), §5.5.4a (deferred row), §5.5.7a row 4 |
| **Estimate** | ~180 LOC |
| **Blockers** | none |
| **Owner** | unassigned |

---

### `[ ]` TODO-2 — `extract/_unify.py`

Thin wrapper around
[`labeling.unify_skill_index.unify_roots`](../labeling/unify_skill_index.py)
configured for the `skill_transfer_test/skill_bank_local/` output root.
Emits `_unified/skill_index.jsonl` + `_unified/skill_catalog_all.json` +
`_unified/skill_rag_index.json` with all 6 corpora tagged distinctly.

| Field | Value |
|---|---|
| **Output** | `skill_transfer_test/skill_bank_local/<run_id>/_unified/{skill_index.jsonl,skill_catalog_all.json,skill_rag_index.json}` |
| **Acceptance** | 6 distinct `corpus` tags in the unified index; non-empty `skill_rag_index.json` for every corpus; rollout memo §5.5.7a "Unified index" row flips PASS |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 5 |
| **Estimate** | ~50 LOC |
| **Blockers** | none |
| **Owner** | unassigned |

---

### `[ ]` TODO-3 — `extract/tests/test_corpus_specs.py`

Validate every `CorpusSpec` in `_corpus_specs.py`: required fields populated,
`default_input_root` exists in the workspace, `action_parser_ref` (where
present) resolves to a callable, `archetype_cluster_field` (where present)
points at a real key in `raw_sample`.

| Field | Value |
|---|---|
| **Acceptance** | `pytest skill_transfer_test/extract/tests/test_corpus_specs.py` green; runs in <5 s |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 6 (partial) |
| **Estimate** | ~50 LOC |
| **Blockers** | none |
| **Owner** | unassigned |

---

### `[ ]` TODO-4 — `extract/tests/test_single_shot_lift.py`

Golden-file replay of one `correct=True` sample per benchmark (4 cases:
VTB, TIR-Bench, Video-Holmes, SIV-Bench). Asserts:

- Lifted protocol has 4 hops in `(GROUND, CHECK|RETRIEVE, VERIFY, COMMIT)` shape
- `e_N` entity references parsed correctly from `answer_reasoning`
- `verified_status` populated from the `correct` field
- Effects contract carries the v4 single-shot predicates
  (`answer_emitted`, `answer_matches_gold`, `entity_grounded`)

| Field | Value |
|---|---|
| **Fixtures** | one `sample_*.json` per benchmark, copied into `skill_transfer_test/extract/tests/fixtures/` |
| **Acceptance** | `pytest skill_transfer_test/extract/tests/test_single_shot_lift.py` green |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 6 (partial) |
| **Estimate** | ~150 LOC |
| **Blockers** | none |
| **Owner** | unassigned |

---

### `[ ]` TODO-5 — `extract/tests/test_runner_smoke.py`

End-to-end smoke test for `runner.py`: 2 episodes from browsergym + 2
episodes from osworld + 2 samples per visual benchmark, output structure
matches the §2 layout, `_unified/skill_index.jsonl` round-trips.

| Field | Value |
|---|---|
| **Fixtures** | small per-corpus subsets under `tests/fixtures/`; runs against tmpdir output |
| **Acceptance** | `pytest skill_transfer_test/extract/tests/test_runner_smoke.py` green; rollout memo §5.5.7a "All 3 unit-test files pass" row flips PASS |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 6 |
| **Estimate** | ~120 LOC |
| **Blockers** | TODO-2 (round-trip assertion needs `_unify.py`) |
| **Owner** | unassigned |

---

### `[x]` TODO-6 — `extract/audits/vocab_jaccard.py` *(shipped 2026-05-02)*

Reproducible audit script for the §11.5.1 (rollout memo) / §9.3.1
(`skill_transfer_test/README.md`) vocabulary-alignment Jaccard numbers.
Walks both the game banks
(`labeling/skill_bank_out/run_*/{env_wrappers,gym_v}/*`) and the
cross-domain banks (`skill_transfer_test/skill_bank_local/full_v5/*`);
emits a per-layer Jaccard table (protocol ops / slot-type ontology /
hop-level predicates / contract-level predicates / combined predicates).

| Field | Value |
|---|---|
| **Output** | `cross_domain_results/_phase0/<run_id>/vocab_jaccard.{json,md}` (gitignored). Run as `python -m skill_transfer_test.extract.audits.vocab_jaccard`. |
| **Acceptance** | Numbers reproduce §11.5.1 within ±0.05 Jaccard — **PASS** (measured: protocol_ops=0.82, slot_types=1.00, predicates_combined=0.00; identical to §11.5.1 reference within rounding). |
| **Memo refs** | §11.5.1 (caveat), `skill_transfer_test/README.md` §9.3.1, `implementation_notes/phase5-cross-domain-measurement.md` §3 (Stage 0) |
| **Shipped LOC** | 369 (vocab_jaccard.py) + 292 (`_loaders.py` shared) + 112 (`_target_vocabularies.py` shared) |
| **Shipped as part of** | Stage 0 of Phase-5/6 measurement plan (also closes the static predicate-firing + slot-binding feasibility audits — `audits/predicate_firing_static.py`, `audits/slot_binding_feasibility.py`, `audits/_runner.py`). The §11.5.1 / §9.3.1 caveat ("the Jaccard numbers above are an analytical estimate ...") can now be replaced with a `(generated by skill_transfer_test/extract/audits/vocab_jaccard.py on 2026-05-02)` attribution. |

---

## Cancelled / superseded

### `[-]` TODO-7 — `extract/run_extract.sh` (cancelled)

Originally specced as a shell wrapper:
`bash run_extract.sh --corpus all` runs the 6 corpora sequentially.
**Superseded by** the `python -m skill_transfer_test.extract.runner` CLI
which already supports `--corpora all|<list> --max-samples N
--output-root <path> --run-id <name>`. No follow-up needed; the §5.5.4a
"deferred" annotation is technically a *cancellation*, not a deferral.

| Memo refs | §5.5.4 (file table), §5.5.4a |
|---|---|

---

## Out of scope for Phase 1.5b

Not tracked in this file:

- **§7 v0 limitations of `extract/README.md`** — these are deliberate
  trade-offs (per-hop predicate mining is Phase-2; ontology-aware slot
  binder is design-deferred; canonical `SkillBankAgent` LLM path is API-
  budget-gated). They are *philosophical* limits, not action items.
  When the canonical path lands, those limits lift; until then they
  belong in the README, not here.
- **Phases 0, 1, 2, 3, 4, 5, 6 from the rollout memo** — these are
  *new* workstreams (executors, harness producers, transfer matrix),
  not Phase 1.5 follow-ups. They live in the design memo §4 / §5 /
  §6 / §7 / §8 / §9 / §10 with their own acceptance gates.
- **Phase-5/6 cross-domain measurement plan (game→{osworld, browser,
  image-VR, video-VR}, plus within-VR/video transfer)** — being
  designed in chat as of 2026-05-02; will land as a separate memo
  (`implementation_notes/phase5-cross-domain-measurement.md`). The
  measurement-blocker LOC subset is ~310 (image-VR) / ~310-380
  (video-VR) / ~500 (browser) / ~470 (osworld) per the revised rollout
  memo §11.5.5; image-VR has no adapter work because
  `bind_visual_reasoning_executor` already wires the real
  `VisualReasoningExecutor` (461 LOC). TODO-1 (`archetype_aggregator`)
  is a hard prerequisite for the cross-corpus VR/video transfer cells
  but NOT for the within-corpus or game→VR cells.
- **Workspace-wide "Not yet delivered"** — see
  [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) and
  [`harness/README.md`](../harness/README.md) §"Suggested work-order".

---

## Cross-refs

- **Design rationale:** [`implementation_notes/cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md) §5.5 (Phase 1.5), §11.5 (transferability)
- **Shipped feature doc:** [`skill_transfer_test/extract/README.md`](extract/README.md)
- **Top-level workstream:** [`skill_transfer_test/README.md`](README.md)
- **Previous audit rounds:** [`skill_transfer_test/extract/README.md`](extract/README.md) §6 (18 issues / 16 fixed / 1 not-a-bug / 1 v0-limit-doc)
