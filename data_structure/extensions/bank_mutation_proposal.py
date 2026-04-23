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
    """Generalize a source-domain (game) skill to additional target domains
    via few-shot adaptation.

    PLAN-SKILL-BANK §0.4 / PLAN-UNIFIED-SKILL-GATE Stage 3a — the
    crafter no longer hand-asserts that a skill is "transferable"; it
    instead emits an explicit few-shot recipe (source → target domain,
    K-shot budget, slot remap, demo selection criteria) that the gate
    runs through `harness.FewShotAdapter` before any new
    `verified_domains` entry is granted. The crafter is therefore
    responsible for *proposing the binding*, not *asserting it works*.
    """

    name: str = ""
    base_skill_id: str = ""
    abstracted_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)
    # Few-shot adaptation recipe (PLAN-UNIFIED-SKILL-GATE §7 Stage 3a).
    source_domain: str = ""                                  # ⊆ SOURCE_DOMAINS
    target_domain: str = ""                                  # ⊆ TRANSFER_TARGET_DOMAINS
    slot_remap: Dict[str, str] = field(default_factory=dict)  # base_slot → target_slot
    demo_selection: Dict[str, Any] = field(default_factory=dict)  # criteria for picking demos
    demo_episode_ids: List[str] = field(default_factory=list)     # explicit demo seed IDs
    k_shot_budget: int = 5

    @property
    def source_type(self) -> SkillSourceType:
        # If the proposal carries a concrete source/target binding it
        # is a few-shot adaptation; otherwise it is a generic
        # cross-domain transfer (legacy path).
        if self.source_domain and self.target_domain:
            return SkillSourceType.FEW_SHOT_ADAPTED
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
            source_domain=p.source_domain,
            target_domain=p.target_domain,
            slot_remap=dict(p.slot_remap),
            demo_selection=dict(p.demo_selection),
            demo_episode_ids=list(p.demo_episode_ids),
            k_shot_budget=p.k_shot_budget,
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
