"""`Generalizer` — propose a domain-generalized version of an existing skill.

Spec: PLAN-SKILL-CRAFTER §5.2.

The MVP rule is structural: lift any hop whose `payload` mentions a
domain-specific symbol (e.g. `bbox`, `dom_id`, `cell`) into a typed slot
that adapters fill in at run time. The teacher model can replace this
later with a richer abstraction step.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional

from common.enums import DOMAINS
from data_structure.extensions.bank_mutation_proposal import GeneralizeProposal
from data_structure.extensions.skill_record import SkillContract, SkillRecord


_DOMAIN_TOKENS = re.compile(
    r"(?:bbox|dom_id|css_selector|xpath|cell|grid_xy|tile_id|frame_index)"
)


class Generalizer:
    def generalize(
        self,
        *,
        base: SkillRecord,
        new_domains: Iterable[str],
        rationale: str,
        teacher_model: Optional[str] = None,
    ) -> GeneralizeProposal:
        new_domains = sorted({d for d in new_domains if d in DOMAINS})
        if not new_domains:
            raise ValueError("Generalizer requires at least one new target domain.")
        target_domains = sorted(set(base.feasible_domains) | set(new_domains))

        abstracted = [self._abstract_hop(h) for h in base.protocol]
        contract = SkillContract(
            preconditions=list(base.contract.preconditions),
            effects_add=list(base.contract.effects_add),
            effects_del=list(base.contract.effects_del),
            belief_progress=list(base.contract.belief_progress),
            grounding_progress=list(base.contract.grounding_progress),
            expected_evidence_roles=list(base.contract.expected_evidence_roles),
            success_criteria=list(base.contract.success_criteria),
            abort_criteria=list(base.contract.abort_criteria),
        )
        return GeneralizeProposal(
            name=f"{base.name}__generalized",
            rationale=rationale,
            parent_skill_ids=[base.skill_id],
            target_domains=target_domains,
            teacher_model=teacher_model,
            base_skill_id=base.skill_id,
            abstracted_protocol=abstracted,
            contract=contract,
            proposed_at=time.time(),
        )

    @staticmethod
    def _abstract_hop(hop: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(hop)
        payload = dict(out.get("payload", {}))
        for k, v in list(payload.items()):
            if isinstance(v, str) and _DOMAIN_TOKENS.search(v):
                payload[k] = "${target}"
        out["payload"] = payload
        return out


__all__ = ["Generalizer"]
