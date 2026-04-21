# PLAN: Pipeline Orchestrator (End-to-End Harness)

**Scope:** Define the **single top-level runner** that closes the loop across [Visual Grounding](PLAN-VISUAL-GROUNDING.md), [Action Agent](PLAN-ACTION-AGENT.md), [Skill Bank](PLAN-SKILL-BANK.md), and [Skill Crafter](PLAN-SKILL-CRAFTER.md). The orchestrator is the **domain-general control plane** for grounding → reasoning → skill retrieval → action → verification → promotion → rollback, with **evidence & trace bookkeeping** (not a memory subsystem) plus budgets, artifacts, gates, and full-system evaluation.

**Scope boundaries (deliberate).** The orchestrator is domain-general across game / webagent / os-agent / video-understanding / visual reasoning. Its **first evaluation track** is **no-memory short-video evidence-grounded reasoning** (Video-Holmes-style). This repo does **not** include a memory subsystem and does **not** target long-horizon video; the orchestrator carries no memory APIs and makes no assumption that skills depend on a separate memory layer.

**Problem statement:** Sub-plans already specify module orchestrators (e.g., bank maintenance, grounding evaluation harness). What is missing is one **executable DAG** that repeatedly: collects rollouts → grounds → runs inner-hop reasoning → acts → logs traces → updates the bank → runs the crafter → **verifies** → promotes or rolls back → schedules training → re-evaluates — with explicit **acceptance gates**, **budget control**, and **observability**.

**Upstream:** All component plans; shared `<state>` schema ([README § Canonical `<state>`](README.md)); two-level MDP framing ([`LONG_HORIZON_REASONING.md`](../LONG_HORIZON_REASONING.md)).

**Downstream:** Implementations of runners, job queues, artifact stores, CI-style verification, and monitoring dashboards.

**Non-goals:** Replacing optional [Visual Skills](PLAN-VISUAL-SKILLS.md); duplicating milestone-level grounding SFT detail ([PLAN-VISUAL-GROUNDING-MILESTONES](PLAN-VISUAL-GROUNDING-MILESTONES.md)).

---

## 1. Rollout DAG

The orchestrator is a **directed acyclic graph of stages** with explicit inputs/outputs. Stages may run **inline** (same process), **queued** (async workers), or **scheduled** (cron / Airflow-style), but the **data dependencies** are fixed.

### 1.1 Online episode subgraph (hot path)

Every environment step (or batched chunk for video):

```
observe (pixels / DOM / API)
  → ground          # Visual Grounding: pixels → <state>
  → inner_mdp       # Action Agent: GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE
  → act             # environment action from valid set
  → log_step        # append to EpisodeTrace (see §2)
```

**Contract:** `inner_mdp` may recurse (re-ground on uncertainty) only through the **budget controller** (§7); the orchestrator records each sub-call as a child span.

### 1.2 End-of-episode subgraph

```
finalize_episode
  → segment_traces     # Skill Bank: boundaries on outer + inner traces
  → ingest_bank        # provisional skill updates / experience buffer
  → emit_signals       # success, failure class, stall, contract violations
```

### 1.3 Offline evolution subgraph (warm path)

Runs between episodes or on a timer; **never blocks** the hot path unless configured for shadow mode only.

```
select_batch
  → crafter_propose    # Skill Crafter: compose / hypothesize / patch
  → acceptance_gate    # §3 — contract checks, replay, non-regression
  → promote_or_rollback
  → schedule_train     # GRPO / LoRA jobs by timescale (§5)
  → re_evaluate        # §6 — full-system matrix
```

### 1.4 DAG invariants

| Invariant | Meaning |
|-----------|---------|
| **No bank promotion without gate** | Crafter outputs are **candidates** until acceptance completes. |
| **Replay before promote** | Any skill or protocol change that affects `skill_select` / `hop_select` must pass **held-out replay** on a frozen trace slice. |
| **Idempotent artifacts** | Every stage writes versioned artifacts (§2); re-runs do not corrupt prior versions. |
| **Explicit failure edges** | Gate failure → rollback or quarantine; never silent partial promotion. |

### 1.5 Optional: Visual Skills

If enabled, insert `visual_skill_retrieve` **after** `ground` and **before** `inner_mdp`, subject to the same budgets and logging. Default: **off** for pipeline-closure work.

---

## 2. Artifact / log schema

The harness is only as debuggable as its **unified telemetry**. Treat logs as **append-only event streams** with stable IDs.

### 2.1 Core identifiers

| Field | Description |
|-------|-------------|
| `run_id` | Orchestrator run (training or eval job). |
| `episode_id` | One task instance from reset to terminal. |
| `step_id` | Monotonic within episode. |
| `span_id` | Inner-hop or tool sub-call (tree under `step_id`). |
| `skill_id` / `skill_version` | Bank pointer at time of use. |
| `schema_hash` | Hash of `<state>` schema version + adapter. |

### 2.2 Required record types

1. **`EpisodeMeta`** — domain, task, goal, seed, model adapters, budget snapshot at episode start.
2. **`GroundingRecord`** — raw routing (Path A/B/C if applicable), latency, escalation reason, optional tool traces; **canonical `evidence_out`** for `GATHER`-role skills (see [PLAN-VISUAL-GROUNDING.md §3a](PLAN-VISUAL-GROUNDING.md#3a-shared-output-schema-both-domains-emit-this)): carries `evidence_id`, `source`, `kind`, `anchor`, `confidence`, `verified_by`.
3. **`InnerHopRecord`** — sequence of inner actions; slot coverage flags; uncertainty scores; re-ground triggers.
4. **`ActionRecord`** — chosen env action, parse path, valid-set constraints, **`evidence_warrant: List[EvidenceRef]`** (non-empty for any committed env action or final answer, per [PLAN-ACTION-AGENT.md §5.3-bis](PLAN-ACTION-AGENT.md)).
5. **`RewardRecord`** — `r_env`, `r_follow`, `r_cost`, and components.
6. **`SkillEpisode`** — per-skill-invocation record from the Harness (see [PLAN-HARNESS.md §5.1](PLAN-HARNESS.md#51-skillepisode)); carries `evidence_role`, `evidence_in`, `evidence_out`, `evidence_warrant`, `verify_verdict`, `reason_warrant`, `contract_progress`, `outcome`, and transfer-diagnostic labels. This is the record Gate G0 operates on.
7. **`BankMutationProposal`** — segmented spans, contract deltas, merge/split ops — **staged**, not live until gate. Typed subclasses: `PatchProposal | ComposeProposal | TransferProposal | RetireProposal` (see [PLAN-SKILL-CRAFTER.md §2.5](PLAN-SKILL-CRAFTER.md)); every subclass declares `evidence_role` and `evidence_interface`.
8. **`GateVerdict`** — pass/fail, failing checks list, replay diffs, non-regression metrics; carries zero or more typed diagnostic labels including `opaque_skill_violation`, `evidence_interface_mismatch`, `skill_role_mismatch` (see [PLAN-HARNESS.md §10a](PLAN-HARNESS.md#10a-transfer-failure-diagnostics-domain-specific)).
9. **`TrainJobSpec`** — which LoRA/timescale, data snapshot IDs, seed, cluster target.

### 2.3 Storage layout (logical)

```
artifacts/
  runs/{run_id}/
    episodes/{episode_id}/
      trace.jsonl          # stream of step + span records
      summary.json         # aggregates for eval
  bank/
    snapshots/{snapshot_id}/
    proposals/{proposal_id}/   # pre-promotion
  gates/
    {gate_run_id}/verdict.json
  train/
    {job_id}/spec.json
```

Physical backend (object store, DB, lakehouse) is an implementation choice; the **schema** is normative.

---

## 3. Promotion / rollback rules

The **acceptance gate** is the safety spine of self-evolution. Both Skill Bank and Skill Crafter depend on it; the orchestrator **centralizes** policy here.

### 3.1 Gate stages (ordered)

1. **Static contract check** — schemas, typed slots, effect families, adapter bindings; no execution.
2. **Symbolic consistency** — preconditions/effects compose without contradiction (bounded solver or LLM-judge + checks).
3. **Replay verification** — replay against **frozen traces** with deterministic settings; compare action distributions / outcomes within tolerance.
4. **Non-regression filter** — compare against prior `snapshot_id` on a **fixed eval slice** (§6); reject if regression beyond ε.
5. **Canary** (optional) — small live traffic or shadow scoring before full promotion.

### 3.2 Promotion

Promotion **creates a new bank snapshot**; pointers (`current_production`) move only after gate success. Previous snapshot remains addressable for rollback.

### 3.3 Rollback

| Trigger | Action |
|---------|--------|
| Gate failure | Discard proposal; optionally file **failure artifact** for crafter learning. |
| Post-promotion regression | Revert pointer to last good `snapshot_id`; quarantine offending skills. |
| Data corruption / schema drift | Halt scheduled training; require manual audit (§8). |

### 3.4 Asymmetric teacher outputs

Frozen 32B/72B proposals remain **candidates** until they pass the same gate stack as mined skills — no fast path based on model size.

---

## 4. Evidence & trace bookkeeping (no-memory contract)

This repo has **no memory subsystem** — no episodic/semantic store, no long-term write/query layer. What the orchestrator does maintain is a **within-episode evidence & trace contract** that is sufficient for short-video evidence-grounded reasoning and for web / os / game / visual-reasoning skills. Anything beyond within-episode is out of scope here.

The contract is richer than a plain "log all tool calls" contract because of the **evidence-driven invariant** ([PLAN-SKILL-BANK.md §0.3](PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)): every skill is required to either consume or produce addressable evidence. If `<state>.evidence_refs` and the claim–evidence link store below are not preserved episode-wide and rolled up into `SkillEpisode` records, Gate G0 cannot be evaluated, and skills would silently drift away from actually assisting reasoning. The bookkeeping below is the substrate that lets the Harness enforce "ALL skills are evidence-driven" mechanically rather than by convention.

### 4.1 What the orchestrator maintains per episode

| Slice | Contents | Lifetime |
|-------|----------|----------|
| **Current structured state** | Latest `<state>` from grounding (entities, attributes, relations, targets, uncertainty) | Replaced every outer step |
| **Short typed trace** | Ordered `InnerHopRecord` + `ActionRecord` from this episode | Cleared at episode boundary |
| **Evidence references** | Domain-appropriate pointers: clip/frame IDs for video, DOM-node + screenshot region IDs for web, desktop object + window IDs for os, tool-call IDs for visual reasoning | Episode-scoped |
| **Claim–evidence links** | `(claim_id → evidence_ref[])` produced by `CHECK` / `COMMIT` inner actions | Episode-scoped |
| **Transfer diagnostics** | For any reused skill: `(skill_id, source_domain, target_domain, binding_verdict, replay_pass)` | Episode-scoped, rolled up into §6a metrics |

No store is persisted across episodes as "memory". Skill-bank snapshots (§3) and the evidence logs in §2 are the only durable artifacts.

### 4.2 Evidence references, not memory reads

`RETRIEVE` inner actions retrieve **skills** from the bank (read-only against the active `bank_snapshot_id`). They do **not** query a memory store. Evidence that the agent wants to cite in subsequent hops must appear as an `evidence_ref` in the current state — which is produced by grounding or by a tool call, not by a memory API.

### 4.3 Grounding alignment

When grounding revises entities mid-episode (schema drift), the orchestrator:

1. Records a `schema_revision_notice` on the next `GroundingRecord` (see [PLAN-VISUAL-GROUNDING.md §`GroundingRecord`]).
2. Invalidates claim–evidence links whose entity anchors no longer resolve.
3. Re-issues `GROUND` for any affected slot before the next `COMMIT` / `EXECUTE`.

There is no memory re-anchoring step because there is no cross-episode memory to re-anchor.

### 4.4 Relation to other docs

An eventual long-horizon / multi-session memory layer is **out of scope** for this repo and intentionally deferred. If it is ever reintroduced it must arrive as a separate `PLAN-MEMORY-SUBSYSTEM.md` with its own acceptance-gate contract; until then the current evidence-and-trace contract above is the orchestrator's authoritative surface for "what the agent remembers within an episode."

---

## 5. Training cadence by timescale

Map jobs to the three-agent separation ([README § Three-agent role split](README.md)) with explicit **triggers** and **data dependencies**.

### 5.1 Fast (Actor — every iteration / continuous)

- **Targets:** `hop_select`, `skill_select`, action execution adapters tied to GRPO.
- **Inputs:** Fresh trajectories from `EpisodeTrace`, reward streams.
- **Trigger:** Buffer full, KL stable, or wall-clock micro-batch.
- **Blocking:** Must not wait on slow teacher; uses last **promoted** bank snapshot.

### 5.2 Medium (Skill Bank ops — every few iterations)

- **Targets:** segmentation, contract heads, curator policy.
- **Inputs:** Segmented traces + gate-verified labels where available.
- **Trigger:** N new episodes or drift in segmentation loss proxy.

### 5.3 Slow (Synthesis / reflection — batched)

- **Targets:** composition, hypothesis, counterfactuals, protocol patches (frozen teacher).
- **Inputs:** Failure clusters, gate failures, curated slices.
- **Trigger:** every N episodes **or** backlog threshold; always passes through **acceptance gate** before affecting production pointers.

### 5.4 Grounding (its own schedule)

- Visual grounding training runs on **grounding milestones** ([Milestones plan](PLAN-VISUAL-GROUNDING-MILESTONES.md)); orchestrator **coordinates** snapshots so the actor always declares which `schema_gen` checkpoint it assumes.

---

## 6. Evaluation matrix

Visual grounding has strong module metrics; the orchestrator adds **full-pipeline** metrics so regressions are attributable.

### 6.1 Actor + inner MDP

| Metric | Definition | Notes |
|--------|------------|-------|
| **Hop efficiency** | Inner actions to successful `EXECUTE` | Stratify by task difficulty. |
| **Action success @ budget** | Env success under fixed hop/token budgets | Primary outer outcome. |
| **Skill reuse quality** | Frequency + outcome of retrieved skills vs. baseline | Ablate retrieval off. |
| **Reasoning overhead** | Latency + tokens per outer step | Compare to budget policy. |
| **Re-ground rate** | `GROUND` insertions per episode | Should correlate with uncertainty, not explode. |

### 6.2 Bank + crafter

| Metric | Definition |
|--------|------------|
| **Contract validity rate** | Fraction of invoked skills whose contracts hold on replay |
| **Transfer lift** | Cross-domain slices: same skill_id usage → outcome delta |
| **Gate pass rate** | Proposals accepted / total proposals |
| **Bank churn** | Promotions + rollbacks per 1k episodes |

### 6.3 Evidence & trace quality (no-memory contract)

| Metric | Definition |
|--------|------------|
| **Evidence sufficiency** | Fraction of `CHECK` / `COMMIT` hops whose claim is backed by at least one valid `evidence_ref` |
| **Anchor consistency** | Fraction of claim–evidence links that still resolve after the next `GroundingRecord` (zero schema drift = 1.0) |
| **Short-video chain validity** | On Video-Holmes-style tasks: fraction of final answers whose full evidence chain replays successfully on frozen frames |
| **Cross-domain evidence coverage** | For reused skills: fraction of `evidence_refs` populated by the target-domain adapter (vs. empty / stubbed) |

### 6.4 Slices

Report all metrics by **domain** (game / web / video / embodied), **task length**, and **schema version**. Store in `summary.json` per `run_id`.

---

## 7. Budget controller

Centralize **limits** that are currently implicit across plans: one **policy object** per episode (and nested per inner MDP) that all stages consult.

### 7.1 Budget dimensions

| Dimension | Example knobs |
|-----------|----------------|
| **Token budget** | Max tokens per step, per hop, per episode |
| **Hop budget** | Max inner hops before forced `EXECUTE`. Default **0–2 hops**; **≤3 hops** under uncertainty. See [Action Agent §5.4](PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control). |
| **Grounding escalation budget** | Max tool/GPT calls; Path C limits |
| **Replay budget** | Max CPU time per gate stage |
| **Teacher budget** | Max 32B/72B calls per day; queue discipline |

### 7.2 Policy styles

- **Hard caps** — non-negotiable ceilings (safety).
- **Adaptive caps** — raise grounding budget when uncertainty spikes; **log trigger**.
- **Fairness across domains** — minimum replay and eval slices so rare domains are not starved.

### 7.3 Interface

```text
BudgetController.state(episode_id) → remaining budgets
BudgetController.consume(span_id, cost_vector) → ok | deny | degrade
```

`degrade` must map to **explicit** behaviors (e.g., skip optional `CHECK`, drop an `evidence_ref` enrichment, shorten the typed trace) — never silent omission without logging.

---

## 8. Failure escalation / human audit points

Continuous self-evolution requires **circuit breakers** and **human-readable audit trails**.

### 8.1 Escalation ladder

| Level | Condition | System response |
|-------|-----------|-----------------|
| **L0** | Single step failure | Log; continue with recovery inner actions if budget allows. |
| **L1** | Repeated gate failures on same proposal class | Quarantine proposal generator; switch to shadow-only. |
| **L2** | Domain-wide regression on eval slice | Block promotion; freeze bank pointer; alert. |
| **L3** | Safety / policy violation in traces | Halt rollouts; require human sign-off. |

### 8.2 Human audit points

- **First promotion** of a new skill family or effect chain.
- **Any rollback** of a previously stable snapshot.
- **Change to acceptance thresholds** (ε in non-regression).
- **Escalation of teacher budget** or disabling gate stages for speed (should be rare and logged).

### 8.3 Audit artifact

Each human decision produces a signed `AuditRecord` (who, when, rationale, linked `snapshot_id`s) stored beside gate verdicts.

---

## 9. Implementation checklist (Cursor-ready)

1. **Runner binary / service** — one entrypoint that loads config, wires stages, and writes `run_id` artifacts.
2. **Artifact schemas** — JSON Schema or pydantic models for §2 record types.
3. **Gate service** — pluggable checks + replay workers + report generation.
4. **Budget module** — shared library used by action agent and grounding routers.
5. **Eval driver** — runs §6 matrix on frozen checkpoints + bank snapshots.
6. **Dashboards** — gate pass rate, rollback count, budget denials, per-domain metrics.

---

## 10. Related documents

| Document | Relationship |
|----------|----------------|
| [README](README.md) | Pipeline overview and shared schema |
| [PLAN-VISUAL-GROUNDING.md](PLAN-VISUAL-GROUNDING.md) | Grounding module; feeds `GroundingRecord` |
| [PLAN-ACTION-AGENT.md](PLAN-ACTION-AGENT.md) | Inner MDP; feeds `InnerHopRecord` |
| [PLAN-SKILL-BANK.md](PLAN-SKILL-BANK.md) | Segmentation & bank; `BankMutationProposal` |
| [PLAN-SKILL-CRAFTER.md](PLAN-SKILL-CRAFTER.md) | Crafter proposals; teacher policies |
| [PLAN-VISUAL-GROUNDING-MILESTONES.md](PLAN-VISUAL-GROUNDING-MILESTONES.md) | Module-level training schedule (coordinate with §5) |

---

*This document does not duplicate module internals; it specifies what must exist **between** modules for a closed, safe, measurable pipeline.*
