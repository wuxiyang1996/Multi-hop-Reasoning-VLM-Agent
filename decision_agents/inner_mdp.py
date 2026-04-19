"""Inner-MDP reasoning scaffold for the Actor Agent.

``plans/PLAN-ACTION-AGENT.md`` §5 reframes multi-hop visual reasoning as
a **two-level MDP**: the outer loop is ``env.step``; the inner loop is
an explicit sequence of reasoning hops whose last step exits the inner
loop and emits an environment action.  Each hop is one of:

* ``GROUND(query)`` — re-observe / find an entity / sharpen a position
* ``CHECK(predicate)`` — verify a relation or attribute
* ``RETRIEVE(key)`` — query skill bank / episodic memory
* ``CONCLUDE(subgoal)`` — commit an intermediate result to the trace
* ``EXECUTE(action)`` — exit the inner loop and perform an env action

This module exposes the data structures and a **heuristic default
policy**.  The long-term plan is to train a ``hop_select`` LoRA adapter
(Tier 2, Qwen3-8B); the :class:`HopPolicy` protocol is the seam where
that adapter plugs in.

Until the adapter exists, :class:`HeuristicHopPolicy` provides a
deterministic rule-based policy driven by schema uncertainty and skill
slot coverage — exactly the signals PLAN-ACTION-AGENT §10 calls out:

* ``skill.required_slot="target"`` missing ⇒ GROUND(target).
* ``skill.required_slot="blocker"`` missing but skill needs it ⇒ GROUND(scene).
* ``uncertainty.target=high`` ⇒ GROUND(target).
* ``candidate_set=[]`` and skill is a ``locate_filter_select`` family ⇒
  GROUND(candidates).
* All slots populated and uncertainty low ⇒ EXECUTE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .schema_parser import StateSchema
from .skill_interface import SkillGuidance


# ──────────────────────────────────────────────────────────────────────
# Inner-MDP action vocabulary (PLAN §5)
# ──────────────────────────────────────────────────────────────────────


class HopAction(str, Enum):
    """Inner-MDP action tokens.  Values match PLAN-VISUAL-GROUNDING
    ``INNER_MDP_OPS`` and the ``<evidence>`` hop.abstract_op enum."""

    GROUND = "GROUND"
    CHECK = "CHECK"
    RETRIEVE = "RETRIEVE"
    CONCLUDE = "CONCLUDE"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"


_CORE_HOP_ACTIONS = {
    HopAction.GROUND, HopAction.CHECK, HopAction.RETRIEVE,
    HopAction.CONCLUDE, HopAction.EXECUTE, HopAction.VERIFY,
}


def parse_hop_action(value: Optional[str]) -> Optional[HopAction]:
    """Best-effort parse of a string into a :class:`HopAction`."""
    if not value:
        return None
    v = value.strip().upper()
    for h in HopAction:
        if h.value == v:
            return h
    return None


# ──────────────────────────────────────────────────────────────────────
# Hop data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class HopStep:
    """One completed inner-MDP step.  Accumulated into :class:`HopTrace`."""

    action: HopAction
    arg: str = ""                          # free-form argument (entity eid, query, predicate)
    tool: str = ""                         # tool invoked for this hop (if any)
    result: Optional[Any] = None           # tool result or resolved env action
    note: str = ""                         # short explanation for logging
    confidence: Optional[str] = None       # high | medium | low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "arg": self.arg,
            "tool": self.tool,
            "result": _shallow_serialise(self.result),
            "note": self.note,
            "confidence": self.confidence,
        }


@dataclass
class HopTrace:
    """Sequence of :class:`HopStep` for one outer-MDP step.

    The trace is exposed on the :class:`Experience` so GRPO can use it
    as a trajectory-level reward signal (PLAN §5 "Reward for inner
    hops"), and so the Skill Bank / Visual Skills pipelines can mine
    recurring hop patterns.
    """

    steps: List[HopStep] = field(default_factory=list)
    terminated: bool = False               # True once EXECUTE has fired

    def __len__(self) -> int:
        return len(self.steps)

    def last(self) -> Optional[HopStep]:
        return self.steps[-1] if self.steps else None

    def append(self, step: HopStep) -> None:
        self.steps.append(step)
        if step.action is HopAction.EXECUTE:
            self.terminated = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "terminated": self.terminated,
            "n_steps": len(self.steps),
            "action_histogram": self._histogram(),
        }

    def _histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for s in self.steps:
            hist[s.action.value] = hist.get(s.action.value, 0) + 1
        return hist


def _shallow_serialise(v: Any) -> Any:
    """Best-effort JSON-safe conversion for logging the hop result."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_shallow_serialise(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _shallow_serialise(val) for k, val in v.items()}
    return repr(v)[:200]


# ──────────────────────────────────────────────────────────────────────
# Policy protocol
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class HopPolicy(Protocol):
    """Inner-MDP hop selector.

    One method: :meth:`select_next_hop`.  Given the current schema,
    the active skill, and the trace so far, return the next
    :class:`HopStep` to execute — or ``None`` to tell the actor "do
    nothing, go straight to EXECUTE".

    The returned step must be either:

    * A non-terminal hop (``GROUND``, ``CHECK``, ``RETRIEVE``,
      ``CONCLUDE``, ``VERIFY``) — the actor executes it and calls
      back for the next hop.
    * ``EXECUTE`` — the actor exits the inner loop and runs an env
      action.

    Implementations may return ``None`` which the actor interprets as
    an implicit ``EXECUTE``.  That is the contract the future LoRA
    ``hop_select`` adapter is expected to honour.
    """

    def select_next_hop(
        self,
        *,
        schema: Optional[StateSchema],
        guidance: Optional[SkillGuidance],
        trace: HopTrace,
        max_hops: int = 8,
    ) -> Optional[HopStep]:
        ...


# ──────────────────────────────────────────────────────────────────────
# Heuristic default policy
# ──────────────────────────────────────────────────────────────────────


class HeuristicHopPolicy:
    """Rule-based hop policy driven by schema uncertainty + slot coverage.

    Emits one hop at a time using a fixed priority list so the behaviour
    is debuggable and deterministic.  Replace with the LoRA ``hop_select``
    adapter once it's trained (see PLAN-ACTION-AGENT §5 "LoRA adapter
    layout").

    Priority (first satisfied rule wins):

    1. Trace length ≥ ``max_hops`` → EXECUTE (cap, PLAN §5).
    2. Active skill has required slot ``S`` not populated, and no prior
       hop grounded ``S``  → GROUND(S).
    3. ``target`` field of the schema has ``uncertainty.label=high`` or
       ``uncertainty.pos=high`` → GROUND(target).
    4. Active skill needs a ``blocker`` slot and ``blocker`` is null →
       GROUND(blocker).
    5. Active skill is a ``locate_filter_select``-family skill
       (name / tags / required_slots contain ``candidate_set``) and
       ``candidate_set=[]`` → GROUND(candidates).
    6. Otherwise → EXECUTE.
    """

    DEFAULT_MAX_HOPS: int = 4
    """Soft cap on inner-MDP hops per outer step (PLAN §5 "Inner loop length")."""

    def __init__(self, max_hops: int = DEFAULT_MAX_HOPS) -> None:
        self.max_hops = max(1, int(max_hops))

    # ── Protocol impl ────────────────────────────────────────────────

    def select_next_hop(
        self,
        *,
        schema: Optional[StateSchema],
        guidance: Optional[SkillGuidance],
        trace: HopTrace,
        max_hops: int = 8,
    ) -> Optional[HopStep]:
        effective_cap = min(max_hops, self.max_hops)

        if len(trace) >= effective_cap:
            return HopStep(
                action=HopAction.EXECUTE,
                arg="cap_reached",
                note=f"inner-loop cap={effective_cap} reached",
            )

        if schema is None:
            return HopStep(
                action=HopAction.EXECUTE,
                arg="",
                note="no schema available",
            )

        already_grounded = {
            s.arg for s in trace.steps if s.action is HopAction.GROUND and s.arg
        }

        # Rule 2 — skill-declared required slots
        if guidance is not None and guidance.required_slots:
            missing = schema.missing_slots(guidance.required_slots)
            for slot in missing:
                if slot in already_grounded:
                    continue
                return HopStep(
                    action=HopAction.GROUND,
                    arg=slot,
                    note=(
                        f"skill {guidance.skill_id!r} requires slot "
                        f"{slot!r} which is not populated"
                    ),
                )

        # Rule 3 — high uncertainty on target
        target_eid = schema.targets.target
        if target_eid is not None:
            ent = schema.get_entity(target_eid)
            if ent is not None and (
                ent.uncertainty.get("label") == "high"
                or ent.uncertainty.get("pos") == "high"
            ) and target_eid not in already_grounded:
                return HopStep(
                    action=HopAction.GROUND,
                    arg=target_eid,
                    note="target entity has high uncertainty",
                )

        # Rule 4 — blocker required by the active skill but null
        if guidance is not None and _skill_needs_blocker(guidance):
            if schema.targets.blocker is None and "blocker" not in already_grounded:
                return HopStep(
                    action=HopAction.GROUND,
                    arg="blocker",
                    note="skill requires a blocker but none was grounded",
                )

        # Rule 5 — empty candidate_set for locate-filter-select skills
        if guidance is not None and _skill_needs_candidates(guidance):
            if (not schema.targets.candidate_set
                    and "candidate_set" not in already_grounded):
                return HopStep(
                    action=HopAction.GROUND,
                    arg="candidate_set",
                    note="locate-filter-select skill needs a candidate_set",
                )

        # Rule 6 — default
        return HopStep(
            action=HopAction.EXECUTE,
            arg="",
            note="slot coverage OK, uncertainty low",
        )


def _skill_needs_blocker(guidance: SkillGuidance) -> bool:
    """Heuristic: does this skill need a blocker slot?"""
    if "blocker" in guidance.required_slots:
        return True
    name = (guidance.skill_name or guidance.skill_id or "").lower()
    if "blocker" in name or "unblock" in name or "clear" in name:
        return True
    return False


def _skill_needs_candidates(guidance: SkillGuidance) -> bool:
    """Heuristic: is this skill of the locate-filter-select family?"""
    if "candidate_set" in guidance.required_slots:
        return True
    name = (guidance.skill_name or guidance.skill_id or "").lower()
    if "filter" in name or "select" in name or "locate" in name or "pick" in name:
        return True
    return False


__all__ = [
    "HeuristicHopPolicy",
    "HopAction",
    "HopPolicy",
    "HopStep",
    "HopTrace",
    "parse_hop_action",
]
