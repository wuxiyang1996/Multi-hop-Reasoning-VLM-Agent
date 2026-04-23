# `common/` — Canonical types, IDs, schema, and backbone-model registry

Single import point for every cross-cutting type used by `harness/`, `orchestrator/`, `crafter/`, and `skill_bank/`. Every concrete definition lives in a sub-module so import graphs stay readable; this package only re-exports.

```python
from common import (
    # enums (state machines + categorical fields)
    SkillStatus, SkillType, SkillSourceType,
    GateStage, GateVerdict,
    InnerAction, RecoveryStrategy,
    DOMAINS, EVIDENCE_ROLES,
    # ID factories (UUID4 + 8-hex content hash)
    new_skill_id, new_episode_id, new_run_id,
    new_proposal_id, new_snapshot_id, new_span_id,
    schema_hash,
    # typed <state> schema (PLAN-SKILL-BANK §3)
    StateSchema, StateTargets, EvidenceRef,
    # backbone-model pin (root README §"Backbone model")
    BACKBONE_MODEL, BACKBONE_TEACHER_MODEL, BACKBONE_JUDGE_MODEL,
    DEFERRED_MODELS, assert_default_is_gpt4o, is_deferred,
    # type aliases
    JSONDict,
)
```

---

## Module map

| File | What lives here |
|---|---|
| `enums.py` | All `Enum` types: `SkillStatus` (DRAFT → CANDIDATE → SHADOW → PROVISIONAL → ACTIVE → DEPRECATED → ROLLED_BACK), `SkillType`, `SkillSourceType`, `GateStage` (the 7 canonical stages 0–6), `GateVerdict` (PASS / LIMITED_PASS / FAIL), `InnerAction` (`GROUND` / `CHECK` / `RETRIEVE` / `COMMIT` / `EXECUTE`), `RecoveryStrategy`. Also the frozen sets `DOMAINS`, `SOURCE_DOMAINS = {"gymv"}`, `TRANSFER_TARGET_DOMAINS = {browser, osworld, video, visual_reasoning}`, and `EVIDENCE_ROLES = {GATHER, VERIFY, REASON, COMMIT}` |
| `ids.py` | UUID4-based ID factories with stable prefixes (`skill_…`, `ep_…`, `run_…`, `prop_…`, `snap_…`, `span_…`) and `schema_hash(record)` for content-hash-bound promotion (invariant 5) |
| `state_schema.py` | The typed `<state>` schema produced by `vlm_wrapper`'s grounding cascade and consumed by every adapter: `StateSchema(domain, task, goal, step, entities, attributes, relations, state_flags, targets, uncertainty, evidence, actions/answer)` plus the `EvidenceRef` value type |
| `models.py` | Backbone-model registry (single source of truth for every API call). Pins `BACKBONE_MODEL = "gpt-4o"` for actor/harness, teacher, and judge. The deferred Qwen tracks (8B / 32B / 72B) appear in `DEFERRED_MODELS`; `is_deferred(name)` is the predicate the Qwen entrypoints check before booting. `assert_default_is_gpt4o()` is exercised by `tests/test_backbone_model.py` (13 tests) to fail loudly if a future edit accidentally ships a non-GPT-4o default. Override at process start with `VLM_AGENT_BACKBONE_MODEL` / `_TEACHER_MODEL` / `_JUDGE_MODEL` |
| `typing.py` | Type aliases (`JSONDict = dict[str, Any]`, …) shared by serialisation paths |

---

## Invariants this package owns

1. **Single backbone source.** Every component reads `BACKBONE_MODEL` from `common.models`; no module hardcodes a model name. Enforced by `tests/test_backbone_model.py`.
2. **No ad-hoc IDs.** Components must use `common.ids` factories so artefact provenance and content-hash drift detection (invariant 5 in the root README) keep working.
3. **One `<state>` schema, repo-wide.** `StateSchema` is imported by `vlm_wrapper`, `harness`, `crafter`, and `orchestrator`. Adding a field requires touching `state_schema.py` only — every consumer is automatically forward-compatible because `to_dict()` round-trips unknown keys.

---

## Cross-references

- Root [`readme.md`](../readme.md) — "Backbone model" and "Mechanically-enforced invariants".
- [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](../plans/03-skill-bank/PLAN-SKILL-BANK.md) §3 — `<state>` schema spec.
- [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §3 — `SkillStatus` / `GateStage` / `GateVerdict` state machine.
- [`vlm_wrapper/schema.py`](../vlm_wrapper/schema.py) — the parser-side counterpart that produces `<state>` blocks for `StateSchema.from_state_block(...)`.
