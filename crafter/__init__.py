"""Skill Crafter — slow-timescale proposal layer.

Spec: PLAN-SKILL-CRAFTER, PLAN-COMPONENTS-IMPLEMENTATION §4 (Phase C).

The crafter is the *creative* layer of the system: it proposes new skills
(compositions, generalizations, novel hypotheses) and patches to existing
ones (failure repairs, retirements). Every proposal is typed
(`BankMutationProposal`) and is *only* a proposal — it lands in the
draft store and must pass the unified gate to be promoted.

Architectural rules (mechanically enforced; see invariant tests):
  * The crafter never imports `skill_bank.stores` directly.
  * The crafter never holds a `SkillLifecycleManager` reference.
  * The crafter writes proposals via `ArtifactStore.put_proposal`
    and ingests draft records via `SkillLifecycleManager.ingest_draft`
    *only* through `crafter.service.SkillCrafterService`.

Public surface:

    from crafter import (
        SkillCrafterService,
        Composer,
        Generalizer,
        Hypothesizer,
        Repairer,
        FailureDiagnoser,
        FailureMemory,
    )
"""

from crafter.composer import Composer
from crafter.failure_diagnoser import FailureDiagnoser
from crafter.failure_memory import FailureMemory, FailurePattern
from crafter.generalizer import Generalizer
from crafter.hypothesizer import Hypothesizer
from crafter.repairer import Repairer
from crafter.service import SkillCrafterService

__all__ = [
    "Composer",
    "FailureDiagnoser",
    "FailureMemory",
    "FailurePattern",
    "Generalizer",
    "Hypothesizer",
    "Repairer",
    "SkillCrafterService",
]
