"""Eligibility filtering — narrowing the bank's candidate list down to
skills that are actually runnable *now*.

PLAN-HARNESS §5.2 (`select_eligible_skills`).

The harness applies five kinds of filters, in order:

  F1  status       : only ACTIVE / SHADOW / PROVISIONAL skills are returned
                     to the Actor; CANDIDATE / DRAFT / DEPRECATED never are
                     (PLAN-UNIFIED-SKILL-GATE §6 GateRunner / "no shadow ⇒
                     no active" invariant).
  F2  domain       : `state.domain` must be in `skill.feasible_domains`.
  F2′ task         : if `skill.feasible_tasks` is non-empty, the task token
                     extracted from `state.task` must be in it. Skills with
                     `feasible_tasks=[]` are *task-agnostic* and pass this
                     filter unconditionally — back-compat for skills
                     decorated before the task axis landed.
                     See `harness/README.md` §22.
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


def task_id_from_state(state: StateSchema) -> Optional[str]:
    """Extract the bare task identifier from `state.task`.

    The cold-start corpus emits `state.task = "make_gaming_env/<game>"`
    (e.g. `"make_gaming_env/twenty_forty_eight"`). Other domains may
    use bare strings (`"<game>"`) or different prefixes. We take the
    last `/`-separated segment, which collapses both shapes onto the
    same canonical form. Returns `None` for unset or whitespace-only
    `state.task` so the F2′ filter degrades gracefully (admit, don't
    veto) rather than silently dropping every skill.
    """

    raw = (state.task or "").strip()
    if not raw:
        return None
    if "/" in raw:
        return raw.rsplit("/", 1)[-1].strip() or None
    return raw


@dataclass
class EligibleSkill:
    """A skill the harness deems runnable right now, with provenance."""

    skill: SkillRecord
    adapter_name: str
    reasons: List[str] = field(default_factory=list)
    shadow_only: bool = False    # SHADOW skills must not affect outer-env actions
    # F2′ task-axis classification (harness/README §22). Three values:
    #   "agnostic"  — skill.feasible_tasks is empty (back-compat)
    #   "same_task" — state task ∈ skill.feasible_tasks
    #   "verified"  — state task ∈ skill.verified_tasks (subset of same_task)
    # `task_match == "agnostic"` is admitted to preserve behaviour for
    # cold-start banks decorated before this field landed.
    task_match: str = "agnostic"

    def to_json(self) -> dict:
        return {
            "skill_id": self.skill.skill_id,
            "skill_name": self.skill.name,
            "skill_status": self.skill.status.value,
            "adapter_name": self.adapter_name,
            "shadow_only": self.shadow_only,
            "task_match": self.task_match,
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
        state_task = task_id_from_state(state)
        for skill in candidates:
            reasons: List[str] = []
            if skill.status not in _RUNNABLE_STATUSES:
                continue
            if skill.status == SkillStatus.SHADOW and not self._allow_shadow:
                continue
            if state.domain not in skill.feasible_domains:
                continue
            # F2′ task-axis veto (harness/README §22). When the skill
            # advertises a non-empty `feasible_tasks`, only states whose
            # task matches are admitted. Two cases pass:
            #   (a) `feasible_tasks=[]` (task-agnostic, back-compat),
            #   (b) state's bare task token ∈ skill.feasible_tasks.
            # If state has no task tag, we admit (degraded) rather than
            # veto blindly so single-step adapters / synthesised states
            # without a `state.task` continue to work.
            if skill.feasible_tasks:
                if state_task is None:
                    task_match = "agnostic"  # state-side missing → can't enforce
                elif state_task in skill.feasible_tasks:
                    task_match = (
                        "verified"
                        if state_task in skill.verified_tasks
                        else "same_task"
                    )
                else:
                    continue
            else:
                task_match = "agnostic"
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
                f"task={state_task or '<unset>'}/match={task_match} "
                f"adapter={adapter.name} type={skill.skill_type.value}"
            )
            out.append(
                EligibleSkill(
                    skill=skill,
                    adapter_name=adapter.name,
                    reasons=reasons,
                    shadow_only=skill.status == SkillStatus.SHADOW,
                    task_match=task_match,
                )
            )
        return out


__all__ = ["EligibilityFilter", "EligibleSkill", "task_id_from_state"]
