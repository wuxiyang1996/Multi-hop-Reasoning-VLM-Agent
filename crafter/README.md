# `crafter/` — Slow-timescale typed proposal layer

Spec: [`PLAN-SKILL-CRAFTER`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase C).

The crafter is the **creative** layer of the system: it proposes new skills (compositions, generalisations, novel hypotheses) and patches to existing ones (failure repairs, retirements). Every proposal is typed (`BankMutationProposal`) and is *only* a proposal — it lands in `draft_store/` and must pass the unified gate to be promoted (invariant 6 — crafter scope).

```python
from crafter import (
    SkillCrafterService,
    Composer, Generalizer, Hypothesizer,
    FailureDiagnoser,
    FailureMemory, FailurePattern,
)
```

---

## Architectural rules (mechanically enforced)

These are tested by `tests/test_invariants.py` and a static dependency rule in the package's `__init__.py`:

1. The crafter never imports `skill_bank.stores` directly.
2. The crafter never holds a `SkillLifecycleManager` reference.
3. The crafter writes proposals via `ArtifactStore.put_proposal` and ingests draft records via `SkillLifecycleManager.ingest_draft` **only** through `SkillCrafterService` — no other entry point exists.

The point: the crafter can dream up anything, but the lifecycle manager + the gate are the only path to runtime ACTIVE state.

---

## Module map

| File | Role |
|---|---|
| `failure_memory.py` | `FailureMemory` — append-only `FailureTrace[]` indexed by `(skill_id, failure_mode)`. Mines `FailurePattern`s (recurring (skill, mode) clusters with shared evidence signatures). The single source for "why is X failing in the wild?" |
| `failure_diagnoser.py` | `FailureDiagnoser` — given a `FailurePattern`, decides which `BankMutationProposal` family is appropriate. Outputs `FailureDiagnosis(family, reasoning, target_skills)`. The teacher (currently `gpt-4o`) lives here |
| `composer.py` | `Composer` — builds `ComposeProposal`s by chaining N existing ACTIVE skills whose evidence outputs match the next skill's evidence inputs. The chain itself becomes a typed inner-MDP protocol |
| `generalizer.py` | `Generalizer` — builds `GeneralizeProposal`s by lifting concrete slot bindings to slot variables. Often emitted when `FailureDiagnoser` flags `slot_binding_failed` patterns across multiple domains |
| `hypothesizer.py` | `Hypothesizer` — builds `HypothesisProposal`s for novel skills suggested by the teacher. Used when no compose / generalise route closes the failure pattern |
| `service.py` | `SkillCrafterService` — the **only** public entry point. Single-method facade `process(...)` that pulls failures from `FailureMemory`, runs `FailureDiagnoser`, dispatches to the right proposer, persists via `ArtifactStore.put_proposal`, and ingests the resulting draft `SkillRecord`s via `SkillLifecycleManager.ingest_draft` |

---

## Proposal taxonomy

Every proposer emits one of five typed `BankMutationProposal` subclasses (defined in [`data_structure/extensions/`](../data_structure/extensions/)):

| Family | When emitted | Touches |
|---|---|---|
| `ComposeProposal` | Two or more ACTIVE skills repeatedly co-occur in successful episodes with matching evidence handoff | New DRAFT chain skill |
| `GeneralizeProposal` | A skill succeeds across multiple domains with isomorphic slot bindings | New DRAFT generalised skill, parent-id pointer |
| `HypothesisProposal` | A failure pattern has no compose / generalise route; teacher proposes a novel skill | New DRAFT skill, marked `source_type=HYPOTHESIS` |
| `PatchProposal` | A known failure pattern has a clear recovery; produces an adapter / contract patch | Same skill_id, new `content_hash` (gate revalidates) |
| `RetireProposal` | A skill's success rate drops below `GateThresholds.retire_floor` over the eval window | DEPRECATED status (gate-bound) |

The crafter never decides the *outcome* — it only proposes. The orchestrator's `GateService` evaluates, and `PromotionOrchestrator` (with `SkillLifecycleManager`) applies.

---

## Failure-driven loop

```
runtime episodes ─────────────────► FailureTrace ─► FailureMemory
                                                         │
                                                         ▼ (clustered)
                                                    FailurePattern
                                                         │
                                                         ▼
                              ┌──────────► FailureDiagnoser (teacher = gpt-4o)
                              │                        │
                              │                        ▼
                              │           one of {compose, generalize,
                              │                    hypothesise, patch, retire}
                              │                        │
                              │                        ▼
                              │                 BankMutationProposal
                              │                        │
                              │                        ▼
                              │   SkillCrafterService.process
                              │     ├── ArtifactStore.put_proposal
                              │     └── SkillLifecycleManager.ingest_draft
                              │                        │
                              │                        ▼
                              │                  draft_store/  (DRAFT)
                              │                        │
                              │                        ▼ (offline gate)
                              │                  GateService → PromotionOrchestrator
                              │                                       │
                              ◄────────────────────────────────────────┘
                                            new ACTIVE skill enters runtime
```

---

## Phase boundaries

| Phase | What this package contains | Status |
|---|---|---|
| C (MVP) | `FailureMemory`, `FailureDiagnoser`, `Composer`, `Generalizer`, `Hypothesizer`, `SkillCrafterService` | **Delivered** — covered by `tests/test_smoke.py` (failure → DRAFT cycle) |
| D | `PatchProposal` repair plumbing exposed via `SkillCrafterService.propose_repair` | Pending — see root README |
| F | Replace teacher backbone (currently `gpt-4o` via `BACKBONE_TEACHER_MODEL`) with frozen Qwen3-VL-32B / 235B-A22B | Pending |

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the crafter sits in the four-stage pipeline.
- [`../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) — full proposal-family spec.
- [`../skill_bank/README.md`](../skill_bank/README.md) — the lifecycle authority that the crafter writes through (never around).
- [`../data_structure/extensions/README.md`](../data_structure/extensions/README.md) — the typed proposal records.
