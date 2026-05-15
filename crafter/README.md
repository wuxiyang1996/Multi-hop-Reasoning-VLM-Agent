# `crafter/` — retired (soft-retirement shim)

The real package has moved to [`legacy/crafter/`](../legacy/crafter/README.md).
This directory now contains only a forwarding shim (`__init__.py`) that
keeps every existing import path working:

```python
from crafter import SkillCrafterService          # still works (DeprecationWarning)
from crafter.service import SkillCrafterService  # still works
from crafter._llm_runtime import install_llm_hooks  # still works
```

Under the hood the shim aliases each `crafter.X` submodule to the
corresponding `legacy.crafter.X` module via `sys.modules`, so module
identity (singletons, monkey-patches, `isinstance` checks) is preserved.

## Why retired

The post-update **skill enricher** at
[`trainer/coevolution/skill_enrichment.py`](../trainer/coevolution/skill_enrichment.py)
is the supported skill-evolution path going forward. See that module's
docstring for the "Old Crafter (failed) vs. New 35B Enricher" comparison
and [`frontier_data/SKILL_PARADIGM_COMPARISON.md`](../frontier_data/SKILL_PARADIGM_COMPARISON.md)
for the broader context.

## Migration

* **New code**: import from `legacy.crafter` directly, or — better —
  use the skill enricher (`trainer.coevolution.skill_enrichment`).
* **Existing call sites**: no action required; the shim keeps them
  green. Migrate opportunistically when you touch a call site for
  unrelated reasons.

The design memos in `legacy/crafter/README.md` and
[`implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md)
remain the canonical references for the retired design.
