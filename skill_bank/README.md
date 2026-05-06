# `skill_bank/` — Four-store, lifecycle-managed skill registry

Canonical implementation of [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §5 (storage split) and §6 (lifecycle ownership). This is the **target** layout that `harness/`, `orchestrator/`, and `crafter/` import from; the legacy [`skill_agents/skill_bank/`](../skill_agents/) package remains importable as the historical Stage-3 MVP.

Architectural rule (mechanically enforced — invariant 4 in the root README):

> **`SkillLifecycleManager` is the only symbol in the codebase that may write to a `SkillStore`.** Direct `store.put(...)` / `store.remove(...)` calls raise `StoreLockedError`.

```python
from skill_bank import (
    SkillLifecycleManager, LifecycleError,
    SkillRepository,
    SkillStore, StoreName, StoreLockedError,
)
```

---

## Module map

| File | Role |
|---|---|
| `stores.py` | The four physical `SkillStore` instances (`DRAFT`, `CANDIDATE`, `ACTIVE`, `ARCHIVE`) — JSON-backed, per-store directory layout, `_locked` sentinel that only the lifecycle manager knows how to clear. Mechanical isolation: a CANDIDATE record physically does not exist on disk inside `active_store/`, so the runtime cannot accidentally execute it as ACTIVE |
| `lifecycle.py` | `SkillLifecycleManager` — the **sole authority** for `SkillStatus` transitions. Owns the canonical state machine (`DRAFT → CANDIDATE → SHADOW → PROVISIONAL → ACTIVE → DEPRECATED → ROLLED_BACK`), physically migrates the JSON record between stores on each transition, and validates every invariant (G0 evidence-driven, ≥2 feasible domains, source/transfer asymmetry, content-hash stability) |
| `repository.py` | `SkillRepository` — read-only multi-store query surface used by the Harness, Actor, and Crafter. Exposes `get_by_id`, `iter_active`, `iter_by_domain`, `iter_by_status`, `find_by_content_hash`. Never writes |

---

## Storage split (§5)

```
_bank/
├── draft/        # DRAFT, REJECTED      — crafter writes; not eligible at runtime
├── candidate/    # CANDIDATE            — passed Stage 0/1; awaits Stage 2 shadow
├── active/       # SHADOW, PROVISIONAL, ACTIVE
└── archive/      # DEPRECATED, ROLLED_BACK   — read-only history
```

`SkillStore.put(...)` keys off the destination store name; the lifecycle manager picks the right store via `store_for_status(SkillStatus)` so transitions are file moves, not in-place edits. This makes "what skills are runnable right now" a single-directory `ls`.

---

## Lifecycle invariants (§6)

The lifecycle manager validates these on **every** transition. Failures raise `LifecycleError` and the JSON file is not touched.

| # | Invariant | Where it bites |
|---|---|---|
| 1 | **G0 evidence-driven** — `expected_evidence_roles` non-empty for ACTIVE promotion | `_validate_invariants` on transition into `ACTIVE` / `PROVISIONAL` / `SHADOW` |
| 3 | **General protocol** — `len(feasible_domains) ≥ 2` for any active state | same |
| 5 | **Gate-bound** — promotion to ACTIVE requires a passing `GateVerdictPayload` and matching `content_hash` | `promote(plan)` rejects FAIL verdicts and content-hash drift; refuses ACTIVE on `LIMITED_PASS` |
| 7 | **Source/transfer asymmetry** — every ACTIVE skill has ≥1 entry from `SOURCE_DOMAINS` (`{gymv}`) **and** ≥1 from `TRANSFER_TARGET_DOMAINS` (`{browser, osworld, video, visual_reasoning}`) | `_validate_invariants` on ACTIVE promotion |
| 8 | **`verified_domains` is gate-owned** — only `record_transfer_verification(skill_id, eligible_domains, gate_verdict_payload)` may mutate `verified_domains` / `adapter_history`. Any other write path is forbidden | The method itself; `PromotionOrchestrator._record_transfer_verifications` is the only caller |

The lifecycle manager also enforces the bank-write isolation invariant (#4) via `SkillStore`'s `_locked` sentinel — direct `store.put(...)` from any other module raises `StoreLockedError`.

---

## Wiring example

```python
from pathlib import Path
from skill_bank import (
    SkillLifecycleManager, SkillRepository,
    SkillStore, StoreName,
)

bank_root = Path("_bank")
repo = SkillRepository(
    draft_store     = SkillStore(StoreName.DRAFT,     bank_root / "draft"),
    candidate_store = SkillStore(StoreName.CANDIDATE, bank_root / "candidate"),
    active_store    = SkillStore(StoreName.ACTIVE,    bank_root / "active"),
    archive_store   = SkillStore(StoreName.ARCHIVE,   bank_root / "archive"),
)
lifecycle = SkillLifecycleManager(repo)

# Crafter ingests a DRAFT.
lifecycle.ingest_draft(skill_record)

# Orchestrator promotes after gate passes.
lifecycle.promote(promotion_plan)   # raises LifecycleError on any invariant violation
```

---

## Trainer-side writeback path (T2.18, 2026-05-06)

The trainer ships a parallel, lighter-weight writeback chain that lives
in `trainer/coevolution/_promotion_hook.py`. It mirrors the architectural
contract above (the lifecycle manager remains the only authority that
mints ACTIVE skills) but adds two post-writeback passes that are
specific to the Crafter loop:

1. **Phase A — `_post_writeback_inherit.inherit_evidence_for_inserted`**
   *(commit `a540434`).* When `writeback_promotion` materialises a
   PATCH / TRANSFER / COMPOSE skill, the new on-disk row is structurally
   correct but evidentially empty (`sub_episodes=[]`,
   `strategic_description=""`, `n_instances=0`, `report=null`). Without
   evidence the bank-cap-K selector silently drops the skill, so the
   Crafter loop produces zero-impact entries. This sweep inherits a
   *discounted* slice of the parent's evidence onto the child:

   * `sub_episodes` ← `parent.sub_episodes[:10]`
   * `strategic_description`, `execution_hint`, `expected_tag_pattern`
     ← parent verbatim
   * `n_instances` ← `max(1, parent.n_instances // 4)`
   * `report.overall_pass_rate` ← `parent.pass_rate * 0.7`

   HYPOTHESIS proposals (no parent) are passed through untouched —
   they enter the bank with empty fields and rely on UCB exploration to
   earn their first selection.

2. **Phase B — cold-start validation gate**
   *(commit `c9fbe7b`).* With `COLD_START_VALIDATION_ROOT` set (e.g. to
   `labeling/frontier_distill_jsonl/run_<...>_with_labeled/`), every
   Crafter PATCH / HYPOTHESIS skill with non-empty `contract.eff_*` is
   verified against teacher-derived `(B_start, B_end, eff_*)` segments
   parsed from the SFT corpus before the actor sees it.
   See `trainer/coevolution/_cold_start_validation_index.py` for the
   predicate-vocab translation
   (`state_flags.phase=early` → `world.phase=early`,
   score-entity ↔ `world.score`,
   step-pair delta ↔ `event.{phase,score}_changed`).

   * Threshold by `source_type`: `REPAIRED`=0.6, `TEACHER`=0.4, others=0.5.
   * Below 5 segments → abstain (Phase A discount remains).
   * ≥ threshold → real per-segment report replaces the discounted one
     (`validated_against=cold_start`).
   * < threshold → row is **physically removed** from the bank.

   Outcomes are recorded in `_step_summary.json`'s `inherit_per_game`
   block (`n_validation_attempted`, `n_validation_passed`,
   `n_validation_rejected`, `n_validation_abstained`,
   `rejected_skill_ids`).

Both passes are no-ops on curator-evolved skills (they already carry
their own evidence), and both rewrite `skill_bank.jsonl` atomically.
The lifecycle manager described in the rest of this README remains
unchanged — Phase A/B operate on the *legacy* per-game JSONL the
trainer reads, not on `_bank/active/`.

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Mechanically-enforced invariants" — the six (now eight) invariants this package owns.
- [`../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — the canonical lifecycle and store-split spec.
- [`../skill_agents/`](../skill_agents/) — legacy Stage-3 bank kept for back-compat; `skill_bank/legacy_bridge.py` (TODO) will provide a one-way migration of those records into `SkillRecord`.
- [`../trainer/coevolution/_post_writeback_inherit.py`](../trainer/coevolution/_post_writeback_inherit.py) — Phase A/B trainer-side writeback chain (this file's docstring is the canonical design rationale).
- [`../trainer/coevolution/_cold_start_validation_index.py`](../trainer/coevolution/_cold_start_validation_index.py) — SFT → `SegmentRecord` index used by Phase B.
- [`../tests/test_invariants.py`](../tests/test_invariants.py) — invariant tests that must stay green for any change to this package.
