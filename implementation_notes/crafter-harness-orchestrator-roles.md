# Skill Crafter / Harness / Orchestrator — roles, I/O contracts, and the offline `labeling_supplement` mirror

> **Status:** design captured. Live Crafter ships the two-tier trigger
> in [`crafter/service.py`](../crafter/service.py)
> (`reflect_on_episode` per-episode reactive pass + `cycle` per-batch
> reflective pass) plus a frozen [`BankView`](../crafter/_bank_view.py)
> snapshot for the wider read-scope. The Phase-1 rule-based mirror
> ships offline in
> [`labeling_supplement/decide_skill_crafting_gpt54.py`](../labeling_supplement/decide_skill_crafting_gpt54.py).
> Harness `GateRunner` mirror and Orchestrator `PromotionOrchestrator`
> mirror are the next two scripts in the same folder; their I/O
> contracts are fixed by this note so they can be implemented
> independently.
> **Last reviewed:** 2026-04-30.
> **Cross-refs:**
> [`plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md),
> [`plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md),
> [`plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md),
> [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md),
> [`crafter/README.md`](../crafter/README.md) (live code overview),
> [`labeling/readme.md`](../labeling/readme.md) (offline labelers it mirrors).

This memo records the design discussion behind splitting Skill Crafter,
Skill Harness, and Pipeline Orchestrator into three separate roles, why
the boundary between them is non-negotiable, and how each role is being
mirrored as a deterministic offline `labeling_supplement/*_gpt54.py`
script for cold-start production.

The goal is that the next person to open
`labeling_supplement/decide_skill_crafting_gpt54.py` and ask "what does
this script *not* do?" gets a precise answer without having to read four
plan documents.

---

## 1. Why one role is not enough

Three plan documents land in this repo (`04-skill-crafter`, `05-harness`,
`06-orchestrator`) plus a shared lifecycle gate spec (`07-skill-gate`).
On a first read the three plans look like they overlap — the Crafter
"validates" skills, the Harness "verifies" skills, the Orchestrator
"promotes" skills. They are not the same operation:

* The **Crafter** is *generative*. It owns *content production*: pick
  good skills out of a frozen teacher, hypothesize new ones, propose
  retiring stale ones. It never moves bank pointers.
* The **Harness** is *runtime*. Online it's the per-decision filter and
  veto layer; offline (the surface this note focuses on) it's the
  `GateRunner` that takes a Crafter proposal + a bank snapshot and
  emits per-stage `GateVerdictPayload`s. It never overwrites a skill's
  `status` or `verified_domains`.
* The **Orchestrator** is *transactional*. It is the only component
  allowed to write to a bank snapshot, mutate `status` /
  `verified_domains`, append `AuditRecord`s, or roll back.

Conflating any two of these breaks one of the four invariants below.

### The four-way ownership boundary (PLAN-PIPELINE-ORCHESTRATOR §2)

| Component       | Owns                                                                  | Does NOT own |
|-----------------|-----------------------------------------------------------------------|--------------|
| **Actor**       | next-token decoding from `(state, intention, retrieved_skills)`       | bank state, gate verdicts, evidence storage, promotion |
| **Harness**     | retrieval scoring, binding, validation, transfer probes, gate stages  | final action selection (Actor), bank state mutation (Orchestrator), per-batch promotion verdicts (Orchestrator) |
| **Skill Bank**  | persistence + retrieval of `SkillRecord`s; snapshot ids                | invocation, scoring, status mutation policy |
| **Orchestrator**| run-graph DAG; promotion / rollback transactions; bank-snapshot moves; `AuditRecord`s | running the gate stages itself (asks Harness); generating proposals (asks Crafter) |

The Crafter sits **outside this four-way table** as a content producer.
It writes to the `draft_store` (its own staging area), and that's all.
Promotion of a draft into the bank is the Orchestrator's job, gated by
the Harness.

This separation is what makes the pipeline *auditable*: every bank
mutation can be traced back to a Crafter proposal, a Harness gate
verdict roll-up, and an Orchestrator audit record — three independent
artefacts produced by three independent components.

---

## 2. Per-role scope

### 2.1 Skill Crafter (PLAN §1, §2.5, §6)

A frozen-teacher (32B/72B) **synthesis-reflection agent** running in
four modes:

| Mode | Trigger | Output |
|---|---|---|
| **Composer** | adjacent skills co-occur often → chain them | `ComposeProposal` |
| **Generalizer** | mature single-domain skill → propose K-shot transfer | `GeneralizeProposal` |
| **Hypothesizer** | gap in skill bank for a phenomenon seen in failures | `HypothesisProposal` (new draft) / `PatchProposal{protocol}` |
| **Failure-Reflector** | one or more failure traces on an existing skill | `RetireProposal` / `PatchProposal{precondition\|warrant-strengthen}` |

#### Two-tier trigger model

The Crafter has *two* invocation cadences. Each Crafter mode is wired
to one of them:

| Pass | Live entry point | Threshold | Modes that fire | Latency expectation |
|---|---|---|---|---|
| **Per-episode reactive** | [`SkillCrafterService.reflect_on_episode(EpisodeReflection)`](../crafter/service.py) | `min_count=1` (one signal is enough) | Failure-Reflector, per-episode Hypothesizer fall-through, **subsumption-retire** | one teacher call worst case; ms with rule path |
| **Per-batch reflective** | [`SkillCrafterService.cycle(new_failures=...)`](../crafter/service.py) | `hot_pattern_threshold` (default 3) | Composer, Generalizer, statistical retires/patches | scheduled by the orchestrator every K episodes |

The split exists because some Crafter modes have very different
statistical demands. Failure-Reflector and Hypothesizer can act on a
single trace; Composer needs adjacent-skill co-occurrence; Generalizer
needs multi-episode `n_instances` and `pass_rate`. Mixing them on one
trigger either floods the gate stack with noisy single-episode patches
(if cadence is per-episode) or leaves single-episode regressions
unfixed for the next batch (if cadence is per-batch). The two-tier
split fixes both ends.

Both surfaces share the *same* per-pattern dispatch chain
(`repair > retire > hypothesise`, PLAN-SKILL-CRAFTER §6.5) — only the
threshold and the post-dispatch surfaces (subsumption check, batch
stats) differ. This is mechanically enforced in
[`crafter/service.py::_run_failure_dispatch`](../crafter/service.py).

#### Read scope — wider than active-only

The per-episode pass needs to see *more* than the active store: it
reads candidate skills the Bank Agent just minted and recent
bank-mgmt actions for dedup. Live code exposes this through the frozen
[`BankView`](../crafter/_bank_view.py) snapshot built at the start of
every `reflect_on_episode` / `cycle` call:

| Store | Surface in `BankView` | Used by |
|---|---|---|
| Active | `BankView.actives` / `actives_iter()` | repair / retire base lookup, subsumption "did the candidate replace this active?" |
| Candidate | `BankView.candidates` / `candidates_iter()` | subsumption pair detection (`subsumed_pairs(candidate_ids=...)`) |
| Draft | `BankView.drafts` / `drafts_iter()` | dedup against the Crafter's own outstanding proposals |

`BankView` is a `frozen=True` dataclass — proposers receive it as a
parameter and cannot mutate it. The bank-store rule remains intact:
`SkillCrafterService` is the only crafter component that may build a
view (`_take_bank_view`); `Composer`, `Generalizer`, `Hypothesizer`,
`Repairer` accept a view as input and never re-fetch.

#### Subsumption-retire (per-episode only)

When the Skill Bank Agent emits a candidate skill whose
`parent_skill_ids` includes an active skill *and* whose contract
strictly covers the active's (`effects_add`, `effects_del`,
`expected_evidence_roles`, `success_criteria`), the Crafter
automatically emits a `RetireProposal{reason: "subsumed_by=<cand_id> ..."}`.
The check is conservative on purpose — false positives cost one
gate-rejected `RetireProposal`, and the active gets to keep running
until G3/G5 confirm it really is dominated. See
[`crafter/_bank_view.py::_subsumes`](../crafter/_bank_view.py).

#### Early-training noise filters: coalesce + cooldown

The per-episode reactive pass uses `min_count=1` (any single failure
fires Repair) — that is the *whole point* of the per-episode tier, but
in early training it produces severe write amplification. Empirically,
the offline mirror under [`labeling_supplement/reflect_per_episode_gpt54.py`](../labeling_supplement/reflect_per_episode_gpt54.py)
showed an **11.3× duplicate-mint factor** across 340 episodes
(260 PatchProposals collapsing to only 23 unique
`(base_skill_id, recovery_strategy)` tuples). To keep the gate stack,
the artifact store, and the audit trail bounded in early training,
[`SkillCrafterService`](../crafter/service.py) now applies two
deterministic filters to the failure-driven dispatch chain. Both are
on by default; both are observable on every `CrafterCycleResult`
(`n_patches_coalesced`, `n_patches_skipped_cooldown`).

| Filter | When it fires | Effect on the proposal | Cost |
|---|---|---|---|
| **Coalesce** | a still-DRAFT `PatchProposal` already covers `(base_skill_id, recovery_strategy)` | the new `failure_id`s are appended to that proposal's `seed_failure_ids`, the artifact-store JSON is overwritten in place, and an `audit{kind: patch_coalesced}` event is recorded | one in-memory dict lookup + one `put_proposal` overwrite — strictly cheaper than running the Repairer + ingesting a fresh DRAFT |
| **Cooldown** | no coalescable open patch *and* the same base was patched within the last `cooldown_passes` (default **5**) Crafter passes | the mint is skipped; the failure still landed in `FailureMemory.add` (so the per-batch `cycle()` can still pick it up later) and an `audit{kind: patch_skipped_cooldown}` event is recorded | one in-memory dict lookup; no Repairer call |

Ordering is fixed in `_run_failure_dispatch`: **coalesce always wins
over cooldown**. Appending evidence to an in-flight DRAFT is free *and*
is what the gate wants to see (richer `seed_failure_ids` = better
diagnosis quality at the gate). Cooldown gates only the *minting* of
parallel proposals — never the strengthening of an existing one.

A draft skill leaves DRAFT only via the orchestrator (gate decision:
promote → CANDIDATE / archive → ARCHIVE). The Crafter has no
notification channel for that, so `_lookup_open_patch` re-checks
`lifecycle.get(draft_skill_id).status` on every coalesce attempt and
lazily evicts the cache entry the moment the draft moves on. The next
failure on the same `(base, strategy)` then mints a fresh proposal —
this is correct: the gate has formed an opinion on the previous patch
and we should not be poisoning a closed proposal with new evidence.

The pass counter that drives cooldown is advanced **only on
productive Crafter passes** — `reflect_on_episode` short-circuits with
no signal does *not* increment it. Otherwise a few quiet episodes
after a patch would silently re-open the mint window without anyone
making a deliberate decision.

Verified end-to-end on the cold-start corpus
(`labeling_supplement/episode_reflections_out/run_<ts>/_run_summary.json`):

| Metric | Pre-fix | Post-fix |
|---|---|---|
| `n_proposals` (minted) | 260 | 23 |
| `n_patches_coalesced` | n/a | 237 |
| `n_patches_skipped_cooldown` | n/a | 0 (offline mirror has no gate, so DRAFTs never leave DRAFT and coalesce hits 100% — cooldown is exercised by `tests/test_crafter_coalesce_cooldown.py::TestCooldown` instead) |
| effective dedup factor | 11.3× | **1.0×** |
| unique `(base_skill_id, recovery_strategy)` | 23 | 23 |

#### Proposal payload (unchanged)

The Crafter never executes a skill. It writes typed
`BankMutationProposal` records (PLAN-SKILL-CRAFTER §2.5) into a
`draft_store`. Each proposal carries:

* `proposer` ∈ `{composer, generalizer, hypothesizer, reflector}`
* `evidence_role`, `evidence_interface` (mirrored from the source skill)
* `target_domains` — **must cover all five** (gymv, browser, osworld,
  video, visual_reasoning)
* `adapter_plan` — per-target binding strategy stub
* `replay_slice_ids` — segments the gate stages can replay
* free-form `rationale` — for human review and gate dashboard summaries

Phase-1 ships **rule-based + frozen-teacher only** — no trainable
update of the Crafter's parameters. The teacher is called only to draft
better names / rationales; the *decision* of which proposal to emit is
made by deterministic rules so the cold-start corpus is reproducible.

### 2.2 Skill Harness (PLAN-HARNESS §1, §3, §6)

The Harness is structurally one component with two surfaces:

**Online surface (per actor decision)**

```
Inputs  : (schema_state, intention, retrieved_skills, active_skill,
           local_reasoning_trace)
Outputs : (eligible_skills with scores, invocation_veto/permit, SkillEpisode)
```

Used inside `ActorAgent.step` to filter / score `retrieved_skills`
against the current state and (optionally) veto an invocation that
would violate `evidence_role` invariants. Fast, frozen, **never picks
the next action** — the Actor still chooses, the Harness just hands
over candidates with reasons.

**Offline surface (per Crafter proposal)**

```
Inputs  : (proposal: BankMutationProposal, bank_snapshot_id, replay_slice_ids,
           shadow_traces?, batch_metrics?)
Outputs : list[GateVerdictPayload]    # one per gate stage
          + roll-up SkillEvaluationRecord
```

This is the `GateRunner` per PLAN-SKILL-GATE / PLAN-HARNESS §6:

| Stage | Predicate (rough) | Diagnostic labels |
|---|---|---|
| **G0 evidence-driven** | does the proposal declare a non-empty `evidence_interface` consistent with `evidence_role`? | `opaque_skill_violation`, `evidence_starved` |
| **G1 binding** | does the protocol bind cleanly against the current schema (no missing slots)? | `binding_failed_<slot>` |
| **G2 adapter** | per-target adapter present & non-trivial? | `missing_adapter_<domain>` |
| **G3 replay** | running the proposed protocol on `replay_slice_ids` reproduces the observed evidence? | `replay_regression`, `replay_underperforms` |
| **G4 shadow** | shadow execution against current actor matches gate's expectation? | `shadow_disagreement` |
| **G5 non-regression** | promoting this proposal doesn't drop batch metrics on already-active skills? | `non_regression_failed_<metric>` |

The Harness emits *verdicts*. It does not promote, it does not retire,
it does not write `verified_domains`. Those mutations are exclusively
the Orchestrator's.

### 2.3 Pipeline Orchestrator (PLAN-PIPELINE-ORCHESTRATOR §2, §3a, §4)

Top-level loop closer. Two surfaces too:

**Online control plane** — wires
```
skill_bank.retrieve(state) -> harness.filter_and_score(...) ->
actor.step(...) -> harness.validate_invocation(...) -> (action, reward)
```
per outer-step, plus episode-local evidence/trace bookkeeping.

**Offline `PromotionOrchestrator`** — consumes the Harness's
`SkillEvaluationRecord` for each Crafter proposal and decides:

```
PROMOTE      | REJECT      | DEFER (more replay)  | ROLLBACK
```

It is the *only* component that:

* moves a skill between `draft_store → candidate_store → active_store`;
* mutates `status` / `applicable_domains` / `verified_domains`;
* takes a new `bank_snapshot_id`;
* writes `AuditRecord{who, when, rationale, linked_snapshot_ids}`.

If a promotion later regresses on actor batch metrics, the Orchestrator
is also the only component allowed to flip a snapshot pointer back
(`ROLLBACK`).

---

## 3. The I/O contract table

Synthesized from PLAN-04 / 05 / 06 / 07 + the on-disk `skill_actions_out`
and `skill_bank_out` shapes. This is the single source of truth for the
`labeling_supplement/` mirror scripts.

| Component / surface | Inputs (canonical names) | Outputs |
|---|---|---|
| **Crafter — per-episode reactive** ([`reflect_on_episode`](../crafter/service.py)) | [`EpisodeReflection`](../data_structure/extensions/episode_reflection.py) carrying `failure_traces`, `skill_episodes`, `new_candidate_skill_ids`, `bank_agent_actions`, `outcome_summary` + a frozen `BankView` snapshot (built by the service) | `PatchProposal` / `RetireProposal` / `HypothesisProposal` written to `draft_store` (plus subsumption `RetireProposal`s) |
| **Crafter — per-batch reflective** ([`cycle`](../crafter/service.py)) | accumulated `FailureTrace[]` over the last K episodes, current bank state via `BankView` | same proposal taxonomy + Composer / Generalizer outputs |
| **Crafter** (offline mirror) | `bank_snapshot @ snapshot_id`, `skill_actions_run` (usage stats), recent `bank_management_io.json` updates | `PatchProposal \| ComposeProposal \| GeneralizeProposal \| RetireProposal` written to `draft_store` |
| **Harness — online filter** | `(schema_state, intention, retrieved_skills, active_skill, local_trace)` | `eligible_skills (ranked)`, `invocation_veto/permit`, `SkillEpisode` |
| **Harness — `GateRunner` (offline)** | `proposal`, `bank_snapshot_id`, stage-specific args (`replay_slice_ids`, `shadow_traces`, `batch_metrics`) | per-stage `GateVerdictPayload{stage, verdict, metrics, diagnostic_labels}` + roll-up `SkillEvaluationRecord{final_decision, approved_domains, rejected_domains, bank_snapshot_id}` |
| **Orchestrator — online** | actor messages + harness verdicts each step | invocation log, episode artefacts |
| **Orchestrator — `PromotionOrchestrator`** | `SkillEvaluationRecord`, current `bank_snapshot_id`, batch-level actor metrics | `final_decision ∈ {PROMOTE, REJECT, DEFER, ROLLBACK}`, mutated bank state, new `bank_snapshot_id`, `AuditRecord` |

### "Who writes what" cheat sheet

| Artefact | Written by |
|---|---|
| `BankMutationProposal` (`Patch / Compose / Transfer / Retire`) | **Crafter** |
| `SkillEpisode` (per-skill execution record) | **Harness (online)** |
| `GateVerdictPayload` (per stage) | **Harness (`GateRunner`)** |
| `SkillEvaluationRecord` (roll-up) | **Harness (`GateRunner`)**; *consumed* by Orchestrator |
| `bank_snapshot_id` updates / `status` / `verified_domains` | **Orchestrator (`PromotionOrchestrator`)** |
| `AuditRecord` (rationale, who/when, linked snapshots) | **Orchestrator** |
| `SkillRecord` row content | originally Crafter (proposal); finalized + version-bumped by Orchestrator on promotion |

---

## 4. Why mirror these into `labeling_supplement/`?

The repo already has a successful pattern:
[`labeling/`](../labeling/) — frozen-teacher, deterministic, parallelised
offline labelers (`label_intentions_gpt54.py`,
`extract_skillbank_gpt54.py`, `extract_skillbank_gymv_gpt54.py`,
`label_skill_actions_gpt54.py`) that produce reproducible cold-start
SFT data without ever running the live Actor or live Harness.

For the Crafter / Harness-gate / Orchestrator-promotion loop we want
the **same property**: deterministic, parallel, replay-only, no live
env, so we can build / inspect / unit-test the lifecycle end-to-end on
the existing `skill_bank_out/run_<ts>` and `skill_actions_out/run_<ts>`
artefacts before any of these components gets wired into the live
agent.

That's what `labeling_supplement/` is. Its naming, defaults, and
dispatcher pattern follow `labeling/` 1-to-1. There will eventually be
three sibling drivers under it, mirroring the three roles:

| Sibling driver | Role mirrored | Status |
|---|---|---|
| `decide_skill_crafting_gpt54.py` (+ `run_decide_skill_crafting.sh`) | Crafter Phase-1 | **shipped** |
| `run_harness_gates_gpt54.py` (or `decide_gate_verdicts_gpt54.py`) | Harness `GateRunner` (G0–G5) | **pending** |
| `decide_promotion_gpt54.py` | Orchestrator `PromotionOrchestrator` | **pending** |

The contract between siblings is the same one the live components will
use: each driver consumes a previous run's `out/` directory (bank
snapshot → proposals → gate verdicts → promotion decisions). On-disk
JSONL is the only shared API. No driver imports another driver's code.

---

## 5. Crafter Phase-1 — `decide_skill_crafting_gpt54.py`

The shipped script is a faithful implementation of PLAN-SKILL-CRAFTER's
Phase-1 ("rule-based + frozen teacher"). It takes a `bank_run` snapshot
and (optionally) a `skill_actions_run` snapshot, and applies six
deterministic decision rules. Every decision is recorded with the
numeric inputs that triggered it, so a future Harness `GateRunner` can
re-run the same proposals and a human reviewer can trace any verdict.

### 5.1 Decision rules

| Rule | Predicate | Proposal | Proposer |
|---|---|---|---|
| **R1 — retire (evidence-starved)** | `(eff_add+eff_del+eff_event)==0 ∧ n_inst < min_inst_for_keep`, OR `usage_pct < retire_usage_pct_min` | `RetireProposal{retire_reason: "evidence-starved"}` | `reflector` |
| **R2 — patch (warrant-strengthen)** | `evidence_role==COMMIT ∧ eff_add+eff_event==0 ∧ n_inst ≥ min_inst_for_keep` | `PatchProposal{patch_kind: "warrant-strengthen"}` | `reflector` |
| **R3 — patch (precondition tighten)** | `pass_rate==1.0 ∧ mean_applicability` saturated near `0.5` over `min_usage_for_signal` selections | `PatchProposal{patch_kind: "precondition"}` | `reflector` |
| **R3b — patch (protocol)** | `protocol.steps < protocol_min_steps ∧ n_inst ≥ 2·min_inst_for_keep` | `PatchProposal{patch_kind: "protocol"}` | `hypothesizer` |
| **R4 — compose (sequence)** | `co_occurrence(A→B) ≥ compose_threshold · n_transitions`, neither retired | `ComposeProposal{compose_op: "sequence"}` | `composer` |
| **R5 — transfer** | `len(applicable_domains)==1 ∧ n_inst ≥ transfer_min_instances ∧ pass_rate ≥ transfer_min_pass_rate ∧ verified_domains==∅` | `TransferProposal` | `generalizer` |

### 5.2 Why these thresholds, why these rules

* **Phase-1 thresholds are conservative on purpose.** PLAN §10 sets a
  high-precision target so the downstream Harness gate stack is not
  flooded with junk proposals. False-positives on the Crafter side are
  expensive (they become G0/G3 replays); false-negatives are recoverable
  (a skill that should have been retired this batch will be retired the
  next batch when its `usage_pct` keeps dropping).
* **R1 has two branches** (bank-only `empty-contract+low-support`,
  usage-based `low-usage-pct`) so the rule fires both at cold-start
  (when no usage stats exist yet) and during steady-state usage.
* **R2 specifically targets COMMIT-role skills with no add/event
  effects.** Per PLAN-SKILL-GATE Gate G0, this is exactly the
  `opaque_skill_violation` shape — a skill that emits a decision
  without producing a measurable warrant. The patch is to require an
  `evidence_warrant` citation; if the Crafter can't synthesize one, the
  next batch will retire the skill.
* **R3 catches degenerate discriminators.** A skill whose mean
  applicability is exactly `0.5` across hundreds of selections is
  almost certainly being matched by relevance only (the applicability
  scorer isn't doing work). Patching the precondition is the cheap fix
  before retiring.
* **R4 (compose) only fires when usage stats are available.** Without
  them we have no transition counts and the rule cannot fire — the
  driver records this in the per-skill decision trace.
* **R5 (transfer) is gated on `verified_domains==∅`.** A skill that
  already has verified domains has been promoted by the Orchestrator
  before; re-proposing transfer would be a waste of the gate stack.

### 5.3 What the driver does NOT do

This is the critical list — every line is intentional:

1. **Never mutates the bank.** Output goes to a fresh
   `crafter_proposals_out/run_<ts>/` directory only. The
   `skill_bank_out/run_<ts>/` snapshot is read-only.
2. **Never runs gate stages.** No replay, no shadow, no binding check.
   It only decides *which proposal to emit*. G0–G5 are the Harness's
   job.
3. **Never sets `verified_domains` or `status`.** Even on
   `RetireProposal` we only mark `retire_reason: "evidence-starved"`;
   the actual `status: retired` mutation is a future Orchestrator's job.
4. **Never invokes a live Actor.** All inputs are frozen JSON files.
   This is what makes it cheap to re-run (full corpus completes in
   ~1 s on a single core).
5. **Never collapses `target_domains`.** Per PLAN-SKILL-CRAFTER §2.5,
   every proposal must enumerate all five target domains so the gate
   stack can verify general feasibility. Even a `RetireProposal`
   carries the full list — the Orchestrator decides whether to honor
   it per-domain.

### 5.4 Real-corpus statistics (sanity check)

Running the dispatcher across all 17 `(corpus, source)` pairs of the
real `skill_bank_out/run_20260430_030637` snapshot, with the 5%
`compose_threshold` and other defaults:

```
17/17 pairs ok, 114 skills in -> 224 proposals out
  by_kind     = {patch: 109, compose: 48, retire: 38, transfer: 29}
  by_proposer = {reflector: 147, composer: 48, generalizer: 29}
```

Distribution looks right for Phase-1: most proposals are patches
(low-cost, mostly shadowable in G3/G4); ~17% are retires (the cold-start
bank is noisy, this is expected); ~13% are transfers (most game banks
have at least one mature single-domain skill). If the share of retires
ever climbs above 30% the Crafter is too aggressive and the
Orchestrator will throttle promotion; this is the dial we tune in
Phase-2.

---

## 6. Open questions for the next two siblings

### 6.1 Harness-side mirror

Open design questions, in rough order of priority:

1. **Replay surface.** The Harness `GateRunner` G3 stage needs to
   "re-execute" a protocol on `replay_slice_ids`. For the offline
   mirror we don't run a live env, so re-execution = re-evaluating the
   skill's `success_criteria` against the segment's recorded
   `summary_state` deltas. This is enough for Phase-1; full live replay
   requires `cold_start/generate_cold_start_actor.py`-style env
   re-instantiation, which is Phase-2.
2. **Shadow trace input.** G4 requires a "shadow" run of the actor —
   for the offline mirror we substitute the rollout's recorded actions.
   PLAN-HARNESS §3a calls this `shadow_eq_recorded`. Document and
   commit to this approximation in the docstring.
3. **Per-stage early-exit.** A G0 fail blocks G1–G5 (no point binding
   an opaque skill). The mirror should support `--stop-at <stage>` for
   debugging.
4. **Output naming.** `gate_verdicts_out/run_<ts>/<corpus>/<source>/`
   matching the existing `out/run_<ts>` convention.

### 6.2 Orchestrator-side mirror

1. **Snapshot model.** The mirror does not need a real DB; a
   `bank_snapshots/<id>/` directory with copy-on-write semantics is
   enough for offline cold-start. `bank_snapshot_id` is just the
   directory name.
2. **Batch metrics input.** PLAN-PIPELINE-ORCHESTRATOR §4 `ROLLBACK`
   needs `actor_batch_metrics`. For the offline mirror these come from
   per-game `_run_summary.json` files in `skill_actions_out/`. No live
   eval is needed for the cold-start path.
3. **Audit-record format.** `AuditRecord` should be JSONL, one record
   per promotion / rejection / rollback, with the linked
   `SkillEvaluationRecord` ids. This is the file the gate dashboard
   reads.
4. **`DEFER` semantics.** A `DEFER` decision must produce a follow-on
   instruction for the next Crafter run (e.g. "regenerate proposal
   with more replay slices"). Encode this as a back-edge from the
   Orchestrator output into the Crafter input — `defer_followups.jsonl`
   feeding `--defer-followups` on the next Crafter dispatcher run.

### 6.3 Cross-cutting

* **No cycle in offline mirror.** Online the loop is closed: actor →
  harness → orchestrator → bank → crafter → … Offline we deliberately
  break the cycle: each driver runs exactly once per cold-start
  iteration. The "loop" is implemented by re-running drivers in
  sequence with new run-ts ids.
* **Reproducibility contract.** Every driver must write
  `_run_meta.json` with full argv + thresholds + input run paths. This
  is what makes "re-run with the same inputs and check the same outputs
  fall out" mechanical.
* **Diagnostic-label vocabulary.** PLAN-HARNESS §10a enumerates the
  agreed-upon `diagnostic_labels`. The Crafter's `decision_trace.json`
  already uses a subset (`opaque_skill_violation` for R2,
  `evidence_starved` for R1); the Harness mirror must use the same
  vocabulary verbatim so the dashboard groups them correctly.

---

## 7. TL;DR

* **Three roles, three responsibilities, three artefact families.** The
  Crafter writes proposals, the Harness writes gate verdicts, the
  Orchestrator writes promotion / audit records. Conflating any two
  breaks the audit trail.
* **The Crafter never touches the bank. The Harness never promotes.
  The Orchestrator never invents skills.** This is the rule the four
  plan documents converge on; the I/O contracts in §3 enforce it.
* **The Crafter has two cadences, not one.** `reflect_on_episode` runs
  every episode-end (`min_count=1`, Failure-Reflector + per-episode
  Hypothesizer + subsumption-retire); `cycle` runs every K episodes
  (`hot_pattern_threshold=3`, Composer / Generalizer / statistical
  retires/patches). Both share `_run_failure_dispatch` so the per-
  pattern decision is identical regardless of cadence — only which
  patterns reach it differs.
* **The Crafter reads the full bank, not just `active_store`.** A
  frozen `BankView` snapshot (active ∪ candidate ∪ draft) is built at
  the start of every pass and handed to proposers as a parameter. This
  enables subsumption-retire (a freshly-minted candidate strictly
  covering an existing active) without violating the no-direct-store
  invariant — only `SkillCrafterService` may build the view.
* **`labeling_supplement/` is the offline mirror** of the live three-
  role loop, modeled on `labeling/`. One driver per role, deterministic,
  parallel, replay-only. The Crafter Phase-1 driver is in; the Harness
  and Orchestrator drivers are next.
* **Phase-1 Crafter is rule-based + frozen teacher.** Six rules
  (R1–R5 + R3b) with conservative thresholds. On the real
  `skill_bank_out/run_20260430_030637` snapshot it produces
  ~2 proposals/skill across 17 sources, dominantly `patch`. Tune
  thresholds via CLI; do not weaken the rule structure.
* **Trap to avoid:** do not let any driver under `labeling_supplement/`
  import another driver's code or write into another driver's output
  directory. The on-disk JSONL is the only shared API. This is what
  makes the mirror replaceable — when the live Harness ships, swap
  the driver, keep the JSONL contract.
