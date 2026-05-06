# Crafter → Promotion → Writeback → Phase A/B — end-to-end live-trainer pipeline

> **Status:** code captured. The per-step pipeline shipped in three
> commits on 2026-05-06: [`a540434`](https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent/commit/a540434)
> (Phase A inheritance + `LLM_CRAFTER_K_MAX=8`),
> [`c9fbe7b`](https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent/commit/c9fbe7b)
> (Phase B cold-start validation gate), and
> [`cd94335`](https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent/commit/cd94335)
> (README sweep + OOD-abstain).
> **Last reviewed:** 2026-05-06.
> **Cross-refs:**
> [`crafter-harness-orchestrator-roles.md`](crafter-harness-orchestrator-roles.md)
> (the role split this pipeline composes),
> [`harness-usability-and-intra-gymv-transfer.md`](harness-usability-and-intra-gymv-transfer.md)
> §3 D8 (one-way writeback contract),
> [`../../skill_bank/README.md`](../../skill_bank/README.md) §"Trainer-side
> writeback path",
> [`../../crafter/README.md`](../../crafter/README.md) §"Trainer
> post-writeback gate",
> [`../../trainer/coevolution/_post_writeback_inherit.py`](../../trainer/coevolution/_post_writeback_inherit.py)
> (canonical Phase A/B docstring).

This memo answers four questions that came up repeatedly during the
2026-05-06 audit of the live `Qwen3.5-9B_20260506_020501` run:

1. **Are we using the promotion gate?**
2. **Where does Crafter's output actually go after promotion?**
3. **Do Crafter-promoted skills land in the same bundle as the
   curator-mined skills the agent is already using?**
4. **If yes, why didn't the agent ever select them before Phase A
   landed?**

The short answers, in order: yes; into the per-game `skill_bank.jsonl`
via `writeback_promotion`; yes — the *exact same file*, by-id merged;
because the writeback projector left every Crafter skill with
`n_instances=0 / report=null / sub_episodes=[]`, which the actor's
top-K relevance filter silently rejected. Phase A patches that gap by
inheriting (discounted) evidence from the parent skill; Phase B then
cross-checks the patched contract against the SFT cold-start corpus
and physically removes contracts that contradict the data.

The rest of this note grounds those answers in the live code paths and
in the empirical bank trajectory recovered from
`runs/Qwen3.5-9B_20260506_020501/checkpoints/step_*/banks/`.

---

## 1. The five-stage flow, with code citations

Every `crafter_cycle_every_k_steps` (default `2`), the orchestrator
runs the following five steps **in the same Python process**, all
gated by `config.crafter_promotion_enabled and crafter_enabled`
(see [`trainer/coevolution/orchestrator.py`](../../trainer/coevolution/orchestrator.py)
lines 1278–1530):

```
[1] Crafter           → BankMutationProposal[]
        │              (PATCH / HYPOTHESIS / COMPOSE / GENERALIZE / RETIRE)
        ▼
[2] Promotion gate    → promotion_decisions.jsonl + bank_snapshots/snap-*.json
        │              (offline-with-llm-judge: 6 stages incl. 35B judge)
        ▼
[3] Writeback         → upsert promoted skills into per-game skill_bank.jsonl
        │              (the "materialize" step)
        ▼
[4] Phase A + Phase B → discounted inheritance + cold-start verification
        │              (filter / abstain / replace)
        ▼
[5] Hot-reload        → AsyncSkillBankPipeline + SkillQueryEngine
                        rebuild from the mutated skill_bank.jsonl,
                        actor sees new skills on next rollout step.
```

### 1.1 Stage [1] — Crafter

`run_crafter_step` in
[`trainer/coevolution/_crafter_hook.py`](../../trainer/coevolution/_crafter_hook.py)
takes the rollout failures (`rollout_results`), the *current* per-game
bank paths, and the harness hooks, and emits proposals into
`runs/<run>/crafter_proposals_out/step_NNNN/<corpus>/<game>/proposals.jsonl`.
Three sub-paths feed it:

* **Deterministic Path 1** — rule-based PATCH/HYPOTHESIS/RETIRE
  triggered by per-episode failure traces.
* **Hypothesizer fallthrough** — gated by
  `crafter_hypothesize_min_recurrences` (default 3) and
  `crafter_hypothesize_related_skill_jaccard` (default 0.30) so the
  bank does not fill up with placeholder hypotheses the actor never
  selects.
* **LLM Crafter (Path 2)** — when `llm_crafter_enabled=True`, calls
  the 35B judge with `LLM_CRAFTER_K_MAX` failure traces (bumped to
  `8` in `a540434`; was `2`).

The serializer used to project the `BankMutationProposal` into the
on-disk JSONL is `_to_offline_row` — fixed in `a540434` to retain
`patch_skill_id / hypothesis_payload / compose_payload /
generalize_payload`, which earlier dropped silently and made every
LLM-Crafter proposal arrive content-empty at the gate.

### 1.2 Stage [2] — Promotion gate

`run_promotion_step` in
[`trainer/coevolution/_promotion_hook.py`](../../trainer/coevolution/_promotion_hook.py)
forks `decide_promotion_gpt54.py` as a subprocess. Default mode in
the live trainer is **`offline-with-llm-judge`**, which exercises six
stages per proposal:

| Stage | Verdict source |
|---|---|
| `static` | proposal/skill consistency, content-hash, source_type matrix |
| `replay` | cached deterministic-replay seeds (limited_pass when none) |
| `shadow` | offline shadow log (limited_pass when none) |
| `transfer` | per-target offline mirror (limited_pass when no probes) |
| `non_regression` | baseline/post-score deltas (limited_pass when none) |
| `static (llm-judge)` | **`Qwen/Qwen3.5-35B-A3B`** — the live judge |

The gate writes one verdict envelope per proposal to
`promotion_decisions_out/step_NNNN/<corpus>/<game>/gate_verdicts.jsonl`,
one decision row to `promotion_decisions.jsonl`, and — when there is at
least one promote — a per-game snapshot
`bank_snapshots/snap-<ts>-<hash>.json` containing the promoted skills
in lifecycle envelope shape (`{skill: {...}, status, source_type, ...}`).

### 1.3 Stage [3] — Writeback (this is the "materialize" step)

`writeback_promotion` in
[`skill_bank/legacy_writeback.py`](../../skill_bank/legacy_writeback.py)
(invoked from inside `run_promotion_step` at lines 311–354) is what
projects the snapshot into the per-game bank:

```python
wb: WritebackReport = writeback_promotion(
    snapshot_path=snap,                 # bank_snapshots/snap-*.json from stage [2]
    legacy_bank_path=Path(bank_path),   # runs/<run>/skillbank/<game>/skill_bank.jsonl
    eligible_statuses={"active", "provisional", "shadow"},
)
```

It loads the existing `skill_bank.jsonl`, indexes by `skill_id`, and for
every snapshot skill whose `status` is in `eligible_statuses` it
**upserts an envelope** (insert or replace by id) into the same map,
then atomically writes the union back to `skill_bank.jsonl`. The
report carries `inserted_skill_ids` / `updated_skill_ids` so dashboards
can attribute origin downstream.

This is the **only** producer that mints rows tagged
`source: "promotion-writeback"` in `_writeback_status` /
`_writeback_source_type` envelope fields — see
`legacy_writeback.py:411–472`.

### 1.4 Stage [4] — Phase A + Phase B

After `writeback_promotion` returns, `run_promotion_step` calls
`inherit_evidence_per_game` from
[`trainer/coevolution/_post_writeback_inherit.py`](../../trainer/coevolution/_post_writeback_inherit.py).
Two passes run back-to-back over the freshly-inserted Crafter skills:

* **Phase A — discounted evidence inheritance.** For every just-inserted
  skill whose `source_type ∈ CRAFTER_SOURCE_TYPES` and whose
  `parent_skill_ids[0]` resolves to an existing skill in the bank, copy
  the parent's `sub_episodes` (capped at `MAX_INHERITED_SUB_EPISODES`),
  `strategic_description`, and `execution_hint` verbatim, and copy
  `n_instances / report.overall_pass_rate` *with discount*
  (`INHERIT_DISCOUNT=2`, `INHERIT_PASS_RATE_FACTOR=0.7`). HYPOTHESIS
  proposals (no parent) get no inheritance — they enter empty and
  must earn selection via UCB.

* **Phase B — cold-start validation gate.** When
  `COLD_START_VALIDATION_ROOT` resolves to a directory holding
  `frontier_distill_jsonl/run_*/skill_selection.jsonl`, the helper
  in [`_cold_start_validation_index.py`](../../trainer/coevolution/_cold_start_validation_index.py)
  parses every SFT prompt's state block into a `SegmentRecord`-like
  set of pre/post predicates plus inferred events. For each
  Crafter-inserted skill with a non-empty `contract.eff_*`,
  `verify_effects_contract` runs against the matched segments. The
  outcome decides what happens to that bank row:

  | Verdict | Action on `skill_bank.jsonl` |
  |---|---|
  | `pass_rate ≥ kind-specific threshold` (PATCH=0.6, HYPOTHESIS=0.4) on ≥ 5 segments | Replace the Phase A discounted `report.overall_pass_rate` with the real measured value; keep the row |
  | `pass_rate <` threshold on ≥ 5 segments | **Physically remove** the row — atomically rewrites the JSONL minus that line |
  | < 5 matched segments, *or* **any** contract literal is OOD relative to the union of all validation predicates | Abstain — keep the row with Phase A's discounted values intact |

  The OOD-abstain rule is the load-bearing safety net: the SFT
  cold-start corpus is heavily early-game-biased, so a legitimate
  `late:RECOVER/SURVIVE` skill whose contract references
  `world.phase=endgame` (a literal that simply does not appear in the
  SFT data) would otherwise be falsely rejected. We instead abstain
  and trust Phase A's discounted credibility, accepting that genuinely
  bogus contracts whose namespace is also OOD will pass through this
  same hole — they remain reachable to retire/rollback proposals on
  later cycles.

  Per-game outcomes are recorded in
  `_step_summary.json::inherit_per_game` for offline audit.

### 1.5 Stage [5] — Hot-reload

The orchestrator calls `sb_manager.refresh_caches()` (orchestrator.py
~1450) so that `AsyncSkillBankPipeline` re-reads the now-mutated
`skill_bank.jsonl` and `SkillQueryEngine` re-indexes. Without this
step the actor would keep serving the cached `SkillBankMVP._skills`
from before the writeback and the new rows would never surface.

---

## 2. The same-bundle invariant — empirical proof

The most actionable claim in this memo is that Crafter-promoted skills
and curator-mined skills physically **share one file per game**.
Three pieces of evidence confirm it:

### 2.1 Identical path objects, three call sites

```python
# orchestrator.py:1284
bank_paths = sb_manager.bank_paths(simple_only=True)

run_crafter_step(legacy_bank_paths=bank_paths, ...)    # Crafter reads prior bank
run_promotion_step(legacy_bank_paths=bank_paths, ...)  # Writeback target
# Hot-reload pulls from the same sb_manager → same path map
```

`bank_paths` is the *same dict* the actor uses for selection. There is
no shadow file and no per-source partition.

### 2.2 Identical envelope schema

`writeback_promotion._project_to_legacy_envelope` emits the schema the
curator already produces:

```jsonc
{
  "skill": {
    "skill_id": "...", "version": 2, "name": "...",
    "protocol": {...}, "contract": {...},
    "sub_episodes": [...], "report": {...},
    "status": "...", "source_type": "..."
  },
  "report": {...}
}
```

There is no `crafter_only` flag and no separate index; the actor's
`SkillQueryEngine` scans the union and runs the same top-K relevance
score regardless of origin.

### 2.3 The bank trajectory under
`runs/Qwen3.5-9B_20260506_020501/`

The promotion gate ran at trainer steps 2 / 4 / 6. The per-step
checkpointed banks for `gymv_thunder_force_iii` show the
Crafter-promoted skill `skill-6f2ad8348f` (promoted at step 2)
coexisting with the curator's 19–20 mined skills inside one file:

| checkpoint | total skills in file | contains `skill-6f2ad8348f` |
|---|---:|:---:|
| step_0000 (cold-start) | 20 | ✗ |
| step_0001 | 20 | ✗ |
| **step_0002** (gate fired) | 20 | **✓** |
| step_0003 | 20 | ✓ |
| step_0004 | **21** | ✓ (curator added one new mined skill that step) |
| step_0005 | 15 | ✗ (bank compaction culled it) |

The total ticking from 20 → 20 → 20 → 21 with the Crafter row
intermixed is the strict same-file proof: separate bundles would have
required `n_curator + n_crafter` totals, not by-id merged ones.

---

## 3. Why Phase A is structurally necessary

A naive reading of stages [1]–[3] suggests the pipeline is already
complete: gate verdict, writeback, done. The 2026-05-06 audit
showed it was not, and the failure mode is worth recording so it does
not recur.

The actor's `SkillBankSelector._build_index` derives relevance tokens
from `contract.eff_*`, `skill.name`, and `skill.strategic_description`.
`_compute_confidence` blends `pass_rate` (default 0.5 when no report
exists) with `n_instances` to assemble the top-K filter. Curator skills
arrive with `n_instances=15+`, `pass_rate≈0.8`, populated
`strategic_description`, and concrete `sub_episodes`. The skills
`writeback_promotion` projects from a Crafter PATCH/HYPOTHESIS arrive
with:

```
strategic_description = ""
sub_episodes          = []
n_instances           = 0
report                = null    →  pass_rate falls back to 0.5
```

Same file, same scorer, but the Crafter row's confidence ratio is
half a curator row's at best, and its query-token surface is half as
wide. The actor's top-K filter silently keeps the curator's tail and
discards every Crafter row before any rollout-time UCB exploration
can rescue it. Materialisation succeeded mechanically; selection
failed semantically.

Phase A's discounted inheritance closes the gap by giving the new row
a parent-credible confidence band and a non-empty token surface,
without claiming the parent's full track record. Phase B then makes
that loan honest by actually running the patched contract against
SFT-derived segments, replacing the discount with a real
measurement when the data supports it, removing the row when it
contradicts the data, and abstaining when the data is silent on the
literals in question.

---

## 4. What this pipeline does *not* do

Recording the negatives so the next reader does not chase them:

* It does not re-run the harness `GateRunner`. The promotion gate's
  `replay` / `transfer` stages are offline mirrors with `limited_pass`
  semantics whenever no shadow/probe artefacts exist for the
  proposal — see `gate_verdicts.jsonl` from any live run.
* It does not retrain or LoRA-update anything. The bank is a
  retrieval payload (lane (a), see
  [`skill-lane-decision.md`](skill-lane-decision.md)); Phase A/B only
  edits the per-game JSONL.
* It does not delete *curator-mined* skills — the writeback projector
  refuses to clobber upstream entries (`legacy_writeback.py:467–472`),
  and Phase B's removal path checks `source_type ∈
  CRAFTER_SOURCE_TYPES` before unlinking.
* It does not execute when `crafter_promotion_enabled=False` or
  `crafter_enabled=False`. Both ablation switches are wired through
  `CoEvolutionConfig` and surface in `step_log.jsonl` so an ablation
  run is auditable from the timeline alone.

---

## 5. Operator switches that change the pipeline's behaviour

All of these resolve via `os.environ` at hook time, with config-field
fallbacks. See [`scripts/README.md`](../../scripts/README.md)
"Crafter / Promotion environment switches" for the canonical list.

| Variable | Default | What it gates |
|---|---|---|
| `LLM_CRAFTER_K_MAX` | `8` (was `2` pre-`a540434`) | Failure-trace budget per Crafter cycle |
| `LLM_CRAFTER_ENABLED` | `0` / unset | Path 2 — 35B-judge LLM Crafter |
| `COLD_START_VALIDATION_ROOT` | `labeling/frontier_distill_jsonl/run_20260506_055632_with_labeled` (set in `run_phase1_curriculum.sh`) | Phase B validation corpus root; unset → Phase B disabled, Phase A still runs |
| `crafter_promotion_gate_mode` (config) | `offline-with-llm-judge` (live), `offline-synthetic` (CI) | Whether stage [2] consults the 35B judge |
| `promotion_bypass_mode` (config) | `gated` (live), `bypass-promote-all` (B3 ablation) | Disables stage [2] gating entirely |

---

## 6. Closure

The full chain landed in three commits on 2026-05-06 and was synced
into `Multi-hop-Reasoning-VLM-Agent/` for the next trainer restart;
the previously-running trainer (started 2026-05-06 02:05 UTC) ran the
pre-fix code and has been killed for an SFT re-run. Once that
re-run completes and `COLD_START_VALIDATION_ROOT` is repointed at the
new corpus, every Crafter cycle will exercise the full pipeline
end-to-end, and `runs/<run>/promotion_decisions_out/step_NNNN/
_step_summary.json::inherit_per_game` becomes the single audit
surface for "did the new Crafter skills actually become reachable to
the actor this step".
