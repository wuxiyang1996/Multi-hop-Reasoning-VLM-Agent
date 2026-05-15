"""Soft-retirement shim — the real package now lives at ``legacy.crafter``.

The original ``crafter`` package (slow-timescale typed proposal layer:
Composer / Generalizer / Hypothesizer / Repairer / FailureDiagnoser +
``SkillCrafterService``) has been moved to ``legacy/crafter/`` as part
of the co-evolution audit. The post-update **skill enricher**
(``trainer/coevolution/skill_enrichment.py``) is the supported skill-
evolution path going forward; see that module's docstring for the
"Old Crafter (failed) vs. New 35B Enricher" comparison.

This shim keeps every existing import path working without touching the
~40 consumer sites (production code, tests, scripts, docs):

    from crafter import SkillCrafterService            # OK
    from crafter.service import SkillCrafterService    # OK
    from crafter._llm_runtime import install_llm_hooks # OK
    import crafter; crafter.Composer                    # OK

Internally each access is routed to the corresponding ``legacy.crafter``
module via ``sys.modules`` aliasing — the import system sees one module
instance per submodule, so ``isinstance`` checks, monkey-patches, and
module-level singletons all behave identically to before the move.

To clear the deprecation warning, switch the import path to
``legacy.crafter`` (or migrate the call site to the skill enricher).
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "The top-level `crafter` package has been retired to `legacy.crafter`. "
    "Imports still work via this shim, but new code should either import "
    "from `legacy.crafter` directly or use the skill enricher at "
    "`trainer.coevolution.skill_enrichment` (the supported skill-evolution "
    "path going forward).",
    DeprecationWarning,
    stacklevel=2,
)

_legacy = _importlib.import_module("legacy.crafter")

# ── Submodule aliasing ────────────────────────────────────────────────
# Register every legacy.crafter submodule under the legacy `crafter.X`
# path so `from crafter.service import Y` (etc.) resolves to the *same*
# module object that `from legacy.crafter.service import Y` would.
_SUBMODULES = (
    "_bank_view",
    "_llm_runtime",
    "composer",
    "failure_diagnoser",
    "failure_memory",
    "generalizer",
    "hypothesizer",
    "repairer",
    "service",
)

for _name in _SUBMODULES:
    _mod = _importlib.import_module(f"legacy.crafter.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

# ── Public re-exports (mirror legacy/crafter/__init__.py) ─────────────
from legacy.crafter import (  # noqa: E402  (must follow sys.modules setup)
    BankView,
    Composer,
    CrafterCycleResult,
    FailureDiagnoser,
    FailureMemory,
    FailurePattern,
    Generalizer,
    Hypothesizer,
    Repairer,
    SkillCrafterService,
)

__all__ = [
    "BankView",
    "Composer",
    "CrafterCycleResult",
    "FailureDiagnoser",
    "FailureMemory",
    "FailurePattern",
    "Generalizer",
    "Hypothesizer",
    "Repairer",
    "SkillCrafterService",
]
