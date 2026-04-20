# PLAN: Skill Harness — Unified Runtime for Skill Use, Validation, and Transfer

**Scope:** Define a **Skill Harness** that sits on top of the existing framework as a unified runtime orchestration layer for skill retrieval, binding, execution, validation, and cross-domain transfer. The Harness is **not a new agent** — it is a thin orchestration layer that wraps the [Action Agent](PLAN-ACTION-AGENT.md), the [Skill Bank](PLAN-SKILL-BANK.md), and the [Skill Crafter](PLAN-SKILL-CRAFTER.md), and provides a single entry point for every skill invocation in the system.

**Problem statement:** Today, skills live in the Skill Bank and are called by the Action Agent, but there is no shared execution surface that (a) normalizes state into slots, (b) attaches domain-specific adapters, (c) records standardized execution traces, and (d) gates transferred skills behind replay + shadow validation. Without this layer, skill transfer risks destabilizing rollouts, cross-domain metrics are inconsistent, and there is no uniform reward signal for later GRPO on skill-use decisions.

**Upstream:** Canonical `<state>` schema ([README § Canonical `<state>`](README.md)); shared slot names (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`); three-agent role split ([Action Agent §2](PLAN-ACTION-AGENT.md)); Skill Bank query/select API ([Skill Bank §6](PLAN-SKILL-BANK.md)); Skill Crafter transfer proposals ([Skill Crafter](PLAN-SKILL-CRAFTER.md)).

**Downstream:** [Pipeline Orchestrator](PLAN-PIPELINE-ORCHESTRATOR.md) acceptance gates (Harness emits `SkillEpisode` records consumed by the orchestrator's gate logic); unified reward signals for GRPO; evaluation harness for reuse and transfer benchmarks.

**Relation to Pipeline Orchestrator.** The Pipeline Orchestrator is the **system-level DAG** that runs grounding → action → bank → crafter → gates across many episodes and runs. The Skill Harness is the **per-invocation runtime** for a single skill call. Pipeline Orchestrator = macro scheduler; Skill Harness = micro runtime for skill use. They compose: the orchestrator calls the Harness at every `inner_mdp` step where a skill is invoked.

**Non-goals:** Replacing the Action Agent, Skill Bank, or Skill Crafter. Introducing a fourth agent. Making the 32B/72B teacher the default online controller. Adding new trainable models before the execution + validation loop works.

---

## 1. Goal

Make skills **executable units** — not static objects in the bank — that can be:

- retrieved,
- bound to the current task / domain state,
- executed with trace logging,
- validated before promotion,
- reused across tasks,
- transferred across domains safely.

The Harness should become the **default path** for all skill invocation and should produce standardized execution records (`SkillEpisode`) that feed the acceptance gates ([Pipeline Orchestrator §3](PLAN-PIPELINE-ORCHESTRATOR.md)) and the reward logger.

---

## 2. Design principle: semantic skill vs. domain adapter

The Harness enforces a strict separation between the **semantic skill** (domain-agnostic, shared across tasks) and the **domain adapter** (task-specific execution layer).

### 2.1 Semantic skill (shared, reusable)

The portable part of a skill:

- `summary`
- `preconditions`
- `plan` / `protocol` (inner-hop chain over shared primitives)
- `success_criteria`
- `abort_criteria`
- `contract` / expected effects
- `evidence_requirements`

This is what lives in the [Skill Bank](PLAN-SKILL-BANK.md) and is subject to composition / generalization by the [Skill Crafter](PLAN-SKILL-CRAFTER.md).

### 2.2 Domain adapter (task-specific)

The execution layer that maps abstract skill steps into concrete operations in:

- games (Gym-V)
- web (BrowserGym)
- desktop (OSWorld)
- video reasoning
- visual grounding / reasoning

Adapters are owned by the `AdapterRegistry` (§5.3) and looked up per `(skill_type, target_domain)`.

### 2.3 Transfer rule

Transfer is **not** "copy a skill end-to-end." Transfer is:

```
reuse semantic skill   +   rebind or synthesize target-domain adapter
```

This is the single most important invariant the Harness enforces.

---

## 3. Responsibilities

The Harness is responsible for exactly five things. Anything beyond this list belongs in the existing modules.

| # | Responsibility | Input | Output |
|---|----------------|-------|--------|
| 1 | **State normalization** | raw env / task state | shared structured state (typed slots) |
| 2 | **Skill retrieval + ranking** | normalized state, goal | ranked candidate skill list |
| 3 | **Skill binding + execution** | semantic skill + adapter | per-step actions + updates |
| 4 | **Transfer validation** | transferred skill candidate | replay verdict + shadow verdict |
| 5 | **Logging + reward hooks** | execution trace | `SkillEpisode` + reward signals |

---

## 4. Minimal runtime flow

The Harness wraps the actor loop:

```
raw state
  → normalize_state               (§5.2 / §5.3)
  → retrieve_candidates           (Skill Bank §6)
  → rank_candidates               (§7)
  → bind_skill                    (slot binding)
  → attach_adapter                (AdapterRegistry §5.3)
  → execute_step                  (protocol-aware)
  → update_episode                (trace / evidence / contract)
  → continue | switch | abort     (budget-aware)
  → finalize_episode              (reward + metrics)
  → return action(s) + reward signals
```

This becomes the **default path** for all skill usage in the Action Agent and the `skill_select` inner action path.

---

## 5. New core abstractions

### 5.1 `SkillEpisode`

Standardized execution record — one per skill invocation. This is the unit of evidence the acceptance gates consume.

```python
class SkillEpisode:
    episode_id: str
    step_id_start: int
    step_id_end: int | None
    skill_id: str
    skill_version: str
    skill_type: str              # reasoning | action | grounding | mixed
    source_domain: str
    target_domain: str
    adapter_id: str | None
    slot_bindings: dict          # typed slots → concrete entity IDs
    protocol_trace: list         # inner-hop sequence actually executed
    evidence_trace: list         # grounding / tool-call evidence
    contract_progress: dict      # per-effect progress in [0,1]
    outcome: str                 # success | fail | abort | stall
    abort_reason: str | None
    reward_components: dict      # r_env, r_follow, r_cost, transfer bonus
    reward_delta: float
    shadow: bool                 # True if running in shadow mode
    metadata: dict               # seeds, adapter_version, model_tier used
```

Purpose:

- unify execution logging,
- support replay-based validation,
- measure transfer success,
- generate reward signals for later GRPO on `skill_select` / `continue_vs_switch` / `accept_transfer`.

### 5.2 `SkillHarness`

Main orchestration entry point. Should expose the following methods; anything else is internal.

```python
class SkillHarness:
    def normalize_state(self, raw_state) -> NormalizedState: ...
    def retrieve_candidates(self, state, goal, k: int) -> list[SkillCandidate]: ...
    def rank_candidates(self, state, candidates) -> list[RankedSkill]: ...
    def bind_skill(self, skill, state) -> BoundSkill | BindFailure: ...
    def attach_adapter(self, skill, target_domain) -> AdapterHandle | AdapterMiss: ...
    def execute_step(self, bound_skill, state, trace) -> StepResult: ...
    def update_episode(self, episode, step_result) -> None: ...
    def decide_continue_switch_abort(self, episode, budget) -> Decision: ...
    def finalize_episode(self, episode, outcome) -> SkillEpisode: ...

    # entry points
    def run_active(self, state, goal, budget) -> (Action, SkillEpisode): ...
    def run_shadow(self, state, goal, candidate) -> SkillEpisode: ...
```

Runtime responsibilities:

- normalize input state,
- call retrieval engine,
- rank candidates (§7),
- bind slots,
- attach adapter,
- execute current skill,
- monitor progress,
- decide continue / switch / abort,
- emit execution logs,
- trigger shadow validation for transferred skills.

### 5.3 `AdapterRegistry`

Registry of domain-specific adapters.

```python
class AdapterRegistry:
    def register(self, skill_type: str, domain: str, adapter) -> None: ...
    def get(self, skill_type: str, domain: str) -> Adapter | None: ...
    def validate(self, adapter, skill, state) -> AdapterVerdict: ...
    def request_synthesis(self, skill, domain) -> AdapterProposal: ...  # calls 72B
```

Initial domains: `gymv`, `browser`, `osworld`, `video`, `visual_reasoning`.

Responsibilities:

- register adapters per domain,
- retrieve adapter by `(skill_type, target_domain)`,
- validate adapter availability / syntactic sanity,
- optionally call the **slow teacher** (32B/72B, frozen) to refine or propose a new adapter.

### 5.4 `TransferManager`

Transfer **does not** happen inside the Skill Bank. It happens inside the Harness, because it requires execution-level evidence that only the Harness can produce.

```python
class TransferManager:
    def propose_transfer(self, skill, target_domain) -> TransferProposal: ...
    def bind_to_target(self, skill, target_state) -> BoundSkill | BindFailure: ...
    def select_or_synthesize_adapter(self, skill, target_domain) -> Adapter: ...
    def dry_run_transfer(self, proposal, replay_slice) -> ReplayVerdict: ...
    def shadow_run_transfer(self, proposal, live_states) -> ShadowVerdict: ...
    def promote(self, proposal) -> None: ...
    def reject(self, proposal, reason) -> None: ...
```

Responsibilities:

- propose transfer,
- bind transferred skill to the target state,
- select or synthesize the target adapter,
- run replay-based dry checks,
- run shadow-mode online checks,
- promote or reject skills for active use.

### 5.5 `ReplayValidator`

Offline validation over logged transitions / held-out state slices ([Pipeline Orchestrator §3](PLAN-PIPELINE-ORCHESTRATOR.md)).

Checks:

- slot bindability on historical states,
- preconditions satisfied,
- expected effects match observed deltas,
- protocol does not contradict observed transitions,
- evidence-trace coherence.

### 5.6 `RewardLogger`

Central place for all skill-use reward signals. Existing components should not write reward directly anymore; they write through this.

Emits:

- `r_env` — environment reward
- `r_follow` — protocol following reward
- `r_cost` — latency / token / adapter cost penalty
- `r_transfer` — transfer success metrics
- `r_adapter` — adapter validation signals

---

## 6. Two-phase transfer protocol

This is the most load-bearing part of the plan. Transferred skills must never be allowed into the active policy before passing both phases.

### 6.1 Phase A — Shadow mode

Transferred skills **can** be:

- retrieved,
- slot-bound,
- adapter-attached,
- evaluated against current states and observed transitions.

Transferred skills **cannot**:

- control the active actor policy,
- change the environment,
- affect `r_env` or any reward that feeds training.

Shadow-mode checks per step:

- are slots bindable?
- are preconditions satisfied?
- does the protocol make sense in the target domain (type-check over primitives)?
- do expected effects align with observed transitions?
- is the evidence trace coherent?

### 6.2 Phase B — Active mode

Only after a transferred skill passes the promotion gates (§10) may it enter active policy execution.

Rationale: protect rollout quality while the bank and adapters are still evolving, and avoid feedback loops where a bad transferred skill pollutes the experience buffer that trains the actor.

---

## 7. Retrieval and ranking update

Current skill selection likely ranks by relevance and pass rate. Transfer-aware ranking must be added so transferred skills are not prematurely chosen on semantic similarity alone.

### 7.1 Suggested scoring

```
score =
    0.35 * retrieval_relevance
  + 0.25 * applicability            # preconditions satisfiable on current state
  + 0.20 * historical_pass_rate
  + 0.10 * adapter_validity
  + 0.10 * slot_binding_confidence
```

Weights are initial defaults; they should be tunable and logged per `run_id`.

### 7.2 Shadow-origin penalty

Skills that have not yet cleared §10 promotion gates receive a hard cap in `run_active` ranking; they are still retrievable in `run_shadow` at full weight.

---

## 8. Unified skill interface

Action skills, reasoning skills, and grounding skills must share a single interface so the Harness can treat them uniformly.

```python
class Skill:
    def preconditions(self, state) -> bool: ...
    def bind(self, state) -> SlotBindings | BindFailure: ...
    def step(self, state, trace) -> StepAction: ...
    def expected_effects(self) -> list[Effect]: ...
    def evidence_requirements(self) -> list[EvidenceSpec]: ...
    def get_adapter(self, domain) -> Adapter | None: ...
```

This lets the framework treat reasoning skills, grounding skills, and action skills as **different types of the same object**. It is a prerequisite for cross-domain transfer and for unified GRPO reward shaping.

---

## 9. Where 7B/8B and 72B fit

The Harness is **not** a 72B-only inference layer. Model assignment follows the [three-agent role split](README.md#three-agent-role-split--model-convention).

### 9.1 Fast loop — 7B/8B (Qwen3-8B)

Default model for:

- online state interpretation,
- skill selection,
- step-level execution,
- short reasoning traces,
- rollout-time continue/switch/abort decisions.

### 9.2 Slow loop — 32B/72B (frozen teacher)

Escalation only, used for:

- transfer proposal review,
- adapter refinement / synthesis,
- protocol rewriting,
- failure reflection,
- merge / split critique,
- hard-case judgment.

The Harness must default to 7B/8B and escalate to the teacher only when an explicit rule fires (e.g., repeated stall, adapter miss, shadow failure on a promising candidate).

---

## 10. Promotion gates

A transferred skill is only promoted to active use when it passes **all five** gate categories. Verdicts are recorded in `GateVerdict` ([Pipeline Orchestrator §2.2](PLAN-PIPELINE-ORCHESTRATOR.md)).

| Gate | Check | Source |
|------|-------|--------|
| **Binding** | target slots ground; abstract predicates map to target ontology | `SkillHarness.bind_skill` |
| **Adapter** | adapter exists (or synthesized adapter is valid); passes domain syntax / execution sanity | `AdapterRegistry.validate` |
| **Replay** | expected effects match held-out transitions; protocol does not contradict observed data | `ReplayValidator` |
| **Shadow** | shadow pass rate ≥ threshold; no severe instability / repeated stalls | `TransferManager.shadow_run_transfer` |
| **Non-regression** | enabling transfer does not degrade prior source-domain competence beyond tolerance | cross-run eval on frozen source slice |

Any failing gate → rejection with reason; candidate returns to the crafter for revision or is quarantined.

---

## 11. No new trainable model required at first

Phase 0 implementation is **inference-only**. It uses:

- the existing Qwen3-8B runtime model,
- the optional 32B/72B frozen slow teacher,
- the current retrieval and Skill Bank machinery,
- rule-based and replay-based validation.

Only after the Harness is working end-to-end should we consider adding trainable LoRAs for:

- `skill_select` (Harness-aware)
- `continue_vs_switch`
- `accept_transfer`
- `adapter_refine`

This mirrors the [Action Agent co-evolution schedule](PLAN-ACTION-AGENT.md) — the harness first, the learning later.

---

## 12. Suggested code structure

A new top-level module inside the repo:

```
harness/
  __init__.py
  skill_harness.py          # main orchestration
  skill_episode.py          # SkillEpisode dataclass + serialization
  transfer_manager.py       # two-phase transfer protocol
  adapter_registry.py       # per-domain adapter lookup + synthesis hooks
  replay_validator.py       # offline validation on held-out transitions
  reward_logger.py          # unified r_env / r_follow / r_cost / r_transfer
  eval_harness.py           # reuse + transfer benchmarks
  adapters/
    gymv.py
    browser.py
    osworld.py
    video.py
    visual_reasoning.py
```

### 12.1 File responsibilities

| File | Implements |
|------|------------|
| `skill_harness.py` | `normalize_state`, `retrieve_candidates`, `rank_candidates`, `bind_skill`, `attach_adapter`, `execute_step`, `update_episode`, `finalize_episode`, `run_active`, `run_shadow` |
| `skill_episode.py` | `SkillEpisode` record + JSONL serialization compatible with [Pipeline Orchestrator §2](PLAN-PIPELINE-ORCHESTRATOR.md) |
| `transfer_manager.py` | `propose_transfer`, `dry_run_transfer`, `shadow_run_transfer`, `promote`, `reject` |
| `adapter_registry.py` | adapter registration, lookup, validation, synthesis escalation to 72B |
| `replay_validator.py` | held-out replay checks (effects, protocol, evidence) |
| `reward_logger.py` | central reward emission + metric collation |
| `eval_harness.py` | reuse + transfer benchmark runner (metrics from §15) |

---

## 13. Integration points with the existing framework

The Harness **does not replace** any existing module. It sits on top.

### 13.1 What stays where

| Module | Keeps responsibility for |
|--------|--------------------------|
| Skill Bank | storage, retrieval backend, contract learning, curation |
| Skill Crafter | composition, abstraction, novel-skill hypothesis |
| Action Agent | actor policy inference, hop selection, action emission |
| Pipeline Orchestrator | full-system DAG, acceptance gates, training schedule |

### 13.2 What moves into the Harness

- runtime orchestration of skill invocation,
- transfer gating (shadow → active),
- replay + shadow validation driver,
- unified logging of `SkillEpisode`,
- evaluation hooks for reuse / transfer metrics.

### 13.3 Wiring pattern

```
ActionAgent
  └─ SkillHarness
       ├─ SkillBank.query (retrieval backend)
       ├─ AdapterRegistry
       ├─ TransferManager
       │    ├─ ReplayValidator
       │    └─ (shadow loop via SkillHarness.run_shadow)
       └─ RewardLogger
```

The Pipeline Orchestrator drives `ActionAgent` and, on end-of-episode / offline evolution, consumes `SkillEpisode` streams from the Harness.

---

## 14. Phased implementation plan

| Phase | Goal | Deliverables | Success criteria |
|-------|------|--------------|------------------|
| **0 — Harness MVP** | unify skill invocation before touching transfer | `SkillEpisode`, `SkillHarness`, basic retrieval + binding + execution wrapper, centralized logging | every skill invocation goes through the Harness; traces and outcomes recorded consistently |
| **1 — Adapterized execution** | separate semantic skill from domain adapter | `AdapterRegistry`, adapter attachment API, adapter validity checking, adapters for `gymv`, `browser` | same semantic skill attaches different adapters across domains |
| **2 — Shadow transfer validation** | support transfer safely before active use | `TransferManager`, `ReplayValidator`, shadow-mode runner | transferred skills can be tested without affecting active rollout |
| **3 — Promotion gates** | decide when a transferred skill becomes active | gate implementations (§10), non-regression checks, contract consistency checks | only validated transferred skills enter active policy |
| **4 — Evaluation harness** | measure reuse and transfer explicitly | `eval_harness.py`, in-domain reuse + cross-domain transfer benchmarks, metrics dashboard | metrics from §15 are logged centrally and comparable across runs |
| **5 — Trainable extensions (optional)** | add learning on skill-use decisions | `skill_select` / `continue_vs_switch` / `accept_transfer` / `adapter_refine` LoRAs | measurable gain over rule-based baselines |

**Immediate next implementation target:** **Phase 0 + Phase 1 only.** Build `SkillEpisode`, `SkillHarness`, and `AdapterRegistry`, and route all current skill usage through them **before** implementing transfer promotion.

---

## 15. Metrics

The Harness (not scattered modules) owns these metrics.

| Metric | Definition |
|--------|------------|
| `retrieval_hit_at_k` | fraction of invocations where the ground-truth useful skill appears in top-k retrieval |
| `slot_binding_success_rate` | fraction of candidate skills whose slots bind on current state |
| `adapter_validation_pass_rate` | fraction of `(skill, domain)` pairs with a valid adapter (existing or synthesized) |
| `shadow_transfer_pass_rate` | fraction of shadow episodes passing all shadow-mode checks |
| `promotion_rate` | fraction of transfer proposals reaching active mode |
| `active_transfer_success_rate` | success rate of transferred skills once in active mode |
| `avg_steps_to_success` | mean inner-hop count per successful skill invocation |
| `source_domain_non_regression` | Δ on frozen source-domain eval after enabling transfer |
| `target_domain_reward_delta` | Δ on target-domain reward vs. baseline without transfer |

These feed the [Pipeline Orchestrator §6](PLAN-PIPELINE-ORCHESTRATOR.md) evaluation matrix.

---

## 16. Implementation notes for Cursor

Priority edits, in order:

1. Create the `harness/` module and define `SkillEpisode` (`harness/skill_episode.py`).
2. Wrap the current `ActionAgent` skill-usage path with `SkillHarness` (`harness/skill_harness.py`) — Phase 0.
3. Refactor current skill invocation into: retrieval → slot binding → adapter attachment → execution → logging.
4. Add `AdapterRegistry` with two initial adapters (`gymv`, `browser`) — Phase 1.
5. Add `TransferManager` but keep it **shadow-only** initially — Phase 2.
6. Add `ReplayValidator` and the promotion gate (§10) — Phase 3.
7. Add `eval_harness.py` with the metrics from §15 — Phase 4.

All new code should write `SkillEpisode` records in a format compatible with the Pipeline Orchestrator's artifact schema ([§2](PLAN-PIPELINE-ORCHESTRATOR.md)).

---

## 17. What not to do

Do **not**:

- introduce a fourth agent just for the harness,
- make the 32B/72B teacher the default online controller,
- couple transfer directly to bank insertion (bank stays storage + retrieval; transfer lives in the Harness),
- allow transferred skills into active policy without shadow + replay validation,
- train new models before the execution / validation loop is working,
- scatter reward emission — all reward goes through `RewardLogger`.

---

## 18. Short design summary (for the repo)

We introduce a **Skill Harness** as a unified runtime and evaluation orchestration layer that mediates between the Action Agent, Skill Bank, and Skill Crafter. The Harness normalizes state into shared slots, retrieves candidate skills, binds semantic skills to target-domain adapters, executes them with protocol-aware tracing, and records evidence traces, contract progress, and outcomes in standardized `SkillEpisode` objects. For cross-domain transfer, the Harness uses a **two-stage protocol**: transferred skills are first evaluated in replay and shadow mode, and only promoted to active execution after passing **binding, adapter, replay, shadow, and non-regression** gates. This design lets action, reasoning, and grounding skills share a unified execution interface and makes skill reuse and transfer **measurable, safe, and extensible** to later GRPO-based decision learning.

---

## 19. Immediate next target

**Phase 0 + Phase 1.** Build `SkillEpisode`, `SkillHarness`, and `AdapterRegistry`, then route all current skill usage through them **before** implementing transfer promotion. That is the cleanest starting point; everything else (TransferManager, ReplayValidator, promotion gates, eval harness, trainable LoRAs) stacks cleanly on top once the Harness owns the execution surface.
