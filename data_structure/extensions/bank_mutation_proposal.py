"""Typed crafter proposals (PLAN-SKILL-CRAFTER §4 + PLAN-UNIFIED-SKILL-GATE §3).

Crafter outputs are *always* proposals — every one of them must pass
through the unified gate. The crafter never writes to `active_store`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from common.enums import SkillSourceType
from common.ids import new_proposal_id
from data_structure.extensions.skill_record import SkillContract


@dataclass
class _ProposalBase:
    proposal_id: str = field(default_factory=new_proposal_id)
    rationale: str = ""
    parent_skill_ids: List[str] = field(default_factory=list)
    seed_failure_ids: List[str] = field(default_factory=list)
    target_domains: List[str] = field(default_factory=list)
    teacher_model: Optional[str] = None       # frozen-only, e.g. "qwen2.5-72b@v1"
    proposed_at: Optional[float] = None

    @property
    def source_type(self) -> SkillSourceType:
        raise NotImplementedError


@dataclass
class ComposeProposal(_ProposalBase):
    """Combine N existing skills into a new skill protocol."""

    name: str = ""
    component_skill_ids: List[str] = field(default_factory=list)
    composed_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.CRAFTED


@dataclass
class GeneralizeProposal(_ProposalBase):
    """Generalize a domain-specific skill to additional domains."""

    name: str = ""
    base_skill_id: str = ""
    abstracted_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.TRANSFERRED


@dataclass
class HypothesisProposal(_ProposalBase):
    """Net-new skill proposed from failure / rule reasoning."""

    name: str = ""
    novel_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)
    source_failure_pattern_ids: List[str] = field(default_factory=list)

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.TEACHER if self.teacher_model else SkillSourceType.CRAFTED


@dataclass
class PatchProposal(_ProposalBase):
    """A repair / patch to an existing skill (PLAN-SKILL-CRAFTER §6.5)."""

    base_skill_id: str = ""
    patched_protocol: List[Dict[str, Any]] = field(default_factory=list)
    patched_contract: Optional[SkillContract] = None
    recovery_strategy: str = ""

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.REPAIRED


@dataclass
class RetireProposal(_ProposalBase):
    """Mark a skill for deprecation."""

    target_skill_id: str = ""
    reason: str = ""

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.CRAFTED


BankMutationProposal = Union[
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
]


def proposal_to_json(p: BankMutationProposal) -> Dict[str, Any]:
    """Helper for `ArtifactStore`-friendly serialization."""
    out: Dict[str, Any] = {
        "type": type(p).__name__,
        "proposal_id": p.proposal_id,
        "rationale": p.rationale,
        "parent_skill_ids": list(p.parent_skill_ids),
        "seed_failure_ids": list(p.seed_failure_ids),
        "target_domains": list(p.target_domains),
        "teacher_model": p.teacher_model,
        "source_type": p.source_type.value,
        "proposed_at": p.proposed_at,
    }
    if isinstance(p, ComposeProposal):
        out.update(
            name=p.name,
            component_skill_ids=list(p.component_skill_ids),
            composed_protocol=list(p.composed_protocol),
            contract=p.contract.to_json(),
        )
    elif isinstance(p, GeneralizeProposal):
        out.update(
            name=p.name,
            base_skill_id=p.base_skill_id,
            abstracted_protocol=list(p.abstracted_protocol),
            contract=p.contract.to_json(),
        )
    elif isinstance(p, HypothesisProposal):
        out.update(
            name=p.name,
            novel_protocol=list(p.novel_protocol),
            contract=p.contract.to_json(),
            source_failure_pattern_ids=list(p.source_failure_pattern_ids),
        )
    elif isinstance(p, PatchProposal):
        out.update(
            base_skill_id=p.base_skill_id,
            patched_protocol=list(p.patched_protocol),
            patched_contract=p.patched_contract.to_json() if p.patched_contract else None,
            recovery_strategy=p.recovery_strategy,
        )
    elif isinstance(p, RetireProposal):
        out.update(target_skill_id=p.target_skill_id, reason=p.reason)
    return out


__all__ = [
    "BankMutationProposal",
    "ComposeProposal",
    "GeneralizeProposal",
    "HypothesisProposal",
    "PatchProposal",
    "RetireProposal",
    "proposal_to_json",
]
