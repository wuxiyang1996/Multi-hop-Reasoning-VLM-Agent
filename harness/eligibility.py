"""Eligibility filtering — narrowing the bank's candidate list down to
skills that are actually runnable *now*.

PLAN-HARNESS §5.2 (`select_eligible_skills`).

The harness applies four kinds of filters, in order:

  F1  status       : only ACTIVE / SHADOW / PROVISIONAL skills are returned
                     to the Actor; CANDIDATE / DRAFT / DEPRECATED never are
                     (PLAN-UNIFIED-SKILL-GATE §6 GateRunner / "no shadow ⇒
                     no active" invariant).
  F2  domain       : `state.domain` must be in `skill.feasible_domains`.
  F3  adapter      : there must be a registered adapter for (domain, type).
  F4  applicability: adapter `can_handle()` returns True.

We never *score* skills here — that's the Actor's job. We only *narrow*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from common.enums import SkillStatus, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry


_RUNNABLE_STATUSES = frozenset(
    {SkillStatus.ACTIVE, SkillStatus.SHADOW, SkillStatus.PROVISIONAL}
)


@dataclass
class EligibleSkill:
    """A skill the harness deems runnable right now, with provenance."""

    skill: SkillRecord
    adapter_name: str
    reasons: List[str] = field(default_factory=list)
    shadow_only: bool = False    # SHADOW skills must not affect outer-env actions

    def to_json(self) -> dict:
        return {
            "skill_id": self.skill.skill_id,
            "skill_name": self.skill.name,
            "skill_status": self.skill.status.value,
            "adapter_name": self.adapter_name,
            "shadow_only": self.shadow_only,
            "reasons": list(self.reasons),
        }


class EligibilityFilter:
    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        allow_shadow: bool = True,
    ) -> None:
        self._registry = registry
        self._allow_shadow = allow_shadow

    def filter(
        self,
        candidates: Iterable[SkillRecord],
        state: StateSchema,
        *,
        skill_type_hint: Optional[SkillType] = None,
    ) -> List[EligibleSkill]:
        out: List[EligibleSkill] = []
        for skill in candidates:
            reasons: List[str] = []
            if skill.status not in _RUNNABLE_STATUSES:
                continue
            if skill.status == SkillStatus.SHADOW and not self._allow_shadow:
                continue
            if state.domain not in skill.feasible_domains:
                continue
            if skill_type_hint is not None and skill.skill_type != skill_type_hint and skill.skill_type != SkillType.MIXED:
                continue
            adapter = self._registry.get(state.domain, skill.skill_type)
            if adapter is None:
                continue
            try:
                ok = adapter.can_handle(skill, state)
            except Exception as exc:                        # noqa: BLE001
                reasons.append(f"adapter raised in can_handle: {exc!r}")
                continue
            if not ok:
                continue
            reasons.append(
                f"status={skill.status.value} domain={state.domain} "
                f"adapter={adapter.name} type={skill.skill_type.value}"
            )
            out.append(
                EligibleSkill(
                    skill=skill,
                    adapter_name=adapter.name,
                    reasons=reasons,
                    shadow_only=skill.status == SkillStatus.SHADOW,
                )
            )
        return out


__all__ = ["EligibilityFilter", "EligibleSkill"]
