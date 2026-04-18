# PLAN: Pipeline Orchestrator (End-to-End Harness)

**Scope:** Define the **single top-level runner** that closes the loop across [Visual Grounding](PLAN-VISUAL-GROUNDING.md), [Action Agent](PLAN-ACTION-AGENT.md), [Skill Bank](PLAN-SKILL-BANK.md), and [Skill Crafter](PLAN-SKILL-CRAFTER.md). This plan is **glue**: schedulers, artifacts, gates, budgets, memory contracts, and full-system evaluation — not a new research module.

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
  → inner_mdp       # Action Agent: GROUND | CHECK | RETRIEVE | CONCLUDE | EXECUTE
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
2. **`GroundingRecord`** — raw routing (Path A/B/C if applicable), latency, escalation reason, optional tool traces.
3. **`InnerHopRecord`** — sequence of inner actions; slot coverage flags; uncertainty scores; re-ground triggers.
4. **`ActionRecord`** — chosen env action, parse path, valid-set constraints.
5. **`RewardRecord`** — `r_env`, `r_follow`, `r_cost`, and components.
6. **`BankMutationProposal`** — segmented spans, contract deltas, merge/split ops — **staged**, not live until gate.
7. **`GateVerdict`** — pass/fail, failing checks list, replay diffs, non-regression metrics.
8. **`TrainJobSpec`** — which LoRA/timescale, data snapshot IDs, seed, cluster target.

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

## 4. Memory interfaces

Long-horizon video and social reasoning need a **first-class memory subsystem**; this section defines **orchestrator-facing interfaces** so memory is not an implicit side channel.

### 4.1 Stores (logical)

Align with existing three-store framing referenced across plans:

| Store | Role | Orchestrator hooks |
|-------|------|---------------------|
| **Episodic** | Time-indexed observations, events, dialogue | `memory.write_episode_span`, `memory.query_temporal` |
| **Semantic** | Facts, entity summaries, relational generalizations | `memory.assert_fact`, `memory.query_graph` |
| **State / working** | Current beliefs, open subgoals, slot fillers | `memory.snapshot`, `memory.apply_hop_update` |

### 4.2 Grounding alignment

Every memory read/write must record **`schema_hash` alignment**: if grounding revises entities, memory operations must either **re-anchor** or **version** entries. The harness logs **alignment events** for debugging drift.

### 4.3 Retrieval contract

`RETRIEVE` inner actions call a single **`MemoryRetrievalRequest`** shape: query type, scope (episode vs corpus), budget (tokens + latency), and **evidence pointers** returned for `CHECK`.

### 4.4 Compression & eviction

Policies (summarize, merge, drop low utility) run on **scheduled jobs**, not the hot path, unless the budget controller (§7) requests an emergency summarize. Eviction is **audited** — tombstone records remain in logs.

### 4.5 Relation to other docs

Detailed memory algorithms may live in a future `PLAN-MEMORY-SUBSYSTEM.md` or external design notes; until then, this section is the **integration contract** the orchestrator enforces.

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

### 6.3 Memory (when enabled)

| Metric | Definition |
|--------|------------|
| **Retrieval precision@k** | Human or automatic label on evidence usefulness |
| **Anchor consistency** | Contradiction rate after re-ground |

### 6.4 Slices

Report all metrics by **domain** (game / web / video / embodied), **task length**, and **schema version**. Store in `summary.json` per `run_id`.

---

## 7. Budget controller

Centralize **limits** that are currently implicit across plans: one **policy object** per episode (and nested per inner MDP) that all stages consult.

### 7.1 Budget dimensions

| Dimension | Example knobs |
|-----------|----------------|
| **Token budget** | Max tokens per step, per hop, per episode |
| **Hop budget** | Max inner actions before forced `CONCLUDE` or `EXECUTE` |
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

`degrade` must map to **explicit** behaviors (e.g., skip optional `CHECK`, summarize memory) — never silent omission without logging.

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
