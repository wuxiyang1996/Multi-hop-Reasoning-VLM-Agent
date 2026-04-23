"""Visual-reasoning harness — read-only image, scratchpad-mutating ops.

The image is fixed for the episode; the actor incrementally builds
evidence in a scratchpad through ``LOOK / CROP / COUNT / COMPARE /
READ_TEXT / RETRIEVE / NOTE`` ops, and finally commits with
``ANSWER(text)`` which terminates the episode.

This is where the legacy ``inner_mdp`` operators relocate (see the
"Migration of inner-MDP operators" table in
``decision_agents/README.md``):

* ``GROUND(slot)``   → :meth:`VRHarness.step` ``LOOK(region)``
* ``RETRIEVE(q)``    → :meth:`VRHarness.step` ``RETRIEVE(q)``
* ``CONCLUDE(text)`` → :meth:`VRHarness.step` ``NOTE(text)``
* ``EXECUTE(answer)``→ :meth:`VRHarness.step` ``ANSWER(text)``

Side effects mutate the :class:`~decision_agents.actor_agent.InnerScratchpad`
the harness was bound to — see :meth:`bind_actor`.  When no bind has
happened (e.g. unit tests that drive the harness in isolation) the
harness keeps a private scratchpad of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decision_agents.core.harness import (
    ACTION_KIND_PRIMITIVE,
    ACTION_KIND_VR_ANSWER,
    ACTION_KIND_VR_LOOK,
    ACTION_KIND_VR_NOTE,
    ACTION_KIND_VR_RETRIEVE,
    Harness,
    HarnessState,
    parse_op_call,
)
from decision_agents.core.multimodal import VisualInput
from decision_agents.schema_parser import StateSchema

_LOGGER = logging.getLogger(__name__)

MAX_VALID_ACTIONS_IN_PROMPT: int = 16


# ──────────────────────────────────────────────────────────────────────
# VR action vocabulary
# ──────────────────────────────────────────────────────────────────────

# Op tags surfaced to the LLM.  Kept short for prompt economy; the
# multi-strategy action parser tolerates the LLM swapping case or
# dropping the closing paren.  Order matters: this is the prompt's
# "numbered selection" order.
VR_OPS: Tuple[str, ...] = (
    "LOOK",
    "CROP",
    "COUNT",
    "COMPARE",
    "READ_TEXT",
    "RETRIEVE",
    "NOTE",
    "ANSWER",
)


# ──────────────────────────────────────────────────────────────────────
# Lightweight scratchpad fallback
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _LocalScratchpad:
    """Drop-in stand-in for :class:`InnerScratchpad` when no actor is bound.

    Mirrors the field names so the harness's mutation code works
    against either object — any future fields added to
    :class:`InnerScratchpad` should be mirrored here too.
    """

    grounded_slots: Dict[str, str] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# VR harness
# ──────────────────────────────────────────────────────────────────────


class VRHarness(Harness):
    """Read-only image + reasoning-op vocabulary.

    Parameters
    ----------
    image
        :class:`VisualInput` for the fixed scene the actor reasons over.
        May be ``None`` for offline tests; the harness still runs but
        no image is rendered into the actor's prompt.
    question
        Free-form question text.  Surfaced into the actor's task
        string and (optionally) the schema's ``goal``.
    gold_answer
        Optional reference answer.  When set, ``ANSWER(text)`` returns
        ``+1`` reward iff ``text.strip().lower() == gold.strip().lower()``;
        otherwise reward is ``0``.  ``None`` disables scoring.
    max_steps
        Hard cap on outer steps before the harness force-terminates
        with ``done=True``.  Prevents runaway VR rollouts when the
        actor never emits ``ANSWER``.
    candidate_args
        Optional dict mapping op → free-form arg suggestions used to
        pre-fill ``valid_actions`` (e.g. ``{"COUNT": ["cube", "sphere"]}``).
        When omitted, ``valid_actions`` enumerates entity-keyed ops
        from the schema and a generic ``ANSWER("<text>")`` placeholder.
    """

    def __init__(
        self,
        *,
        image: Optional[VisualInput] = None,
        question: str = "",
        gold_answer: Optional[str] = None,
        max_steps: int = 8,
        candidate_args: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.image = image
        self.question = question
        self.gold_answer = gold_answer
        self.max_steps = max(1, int(max_steps))
        self.candidate_args = candidate_args or {}

        # Episode-local state.
        self._t: int = 0
        self._done: bool = False
        self._last_obs: Any = ""
        self._answer: Optional[str] = None

        # Side-effect targets — bind via :meth:`bind_actor`.  Default
        # to a private scratchpad so tests that don't bind still work.
        self.scratchpad: Any = _LocalScratchpad()
        self._memory: Optional[Any] = None
        self._tracker: Optional[Any] = None

    # ── actor binding (called by ActorAgent before each episode) ─────

    def bind_actor(
        self,
        *,
        scratchpad: Any = None,
        memory: Optional[Any] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        """Wire the harness's side-effect channel to an actor.

        :class:`~decision_agents.actor_agent.ActorAgent` calls this in
        :meth:`reset` and again whenever the actor's scratchpad is
        rebuilt (e.g. on skill reselect).  Without a bind the harness
        falls back to its private scratchpad (useful for tests).
        """
        if scratchpad is not None:
            self.scratchpad = scratchpad
        if memory is not None:
            self._memory = memory
        if tracker is not None:
            self._tracker = tracker

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self._t = 0
        self._done = False
        self._last_obs = self.question or ""
        self._answer = None
        info: Dict[str, Any] = {
            "task": self.question,
            "image": self.image.to_dict() if self.image is not None else None,
            "harness": "VRHarness",
        }
        return self._last_obs, info

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Apply one VR op to the scratchpad.

        Unknown ops or malformed action strings are accepted but score
        ``0`` reward — the actor's parser already filters most of those
        upstream, but we stay permissive so a misbehaving LLM doesn't
        crash the rollout.
        """
        self._t += 1
        op, arg = parse_op_call(action)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"op": op, "arg": arg, "harness": "VRHarness"}

        if op in ("LOOK", "CROP", "READ_TEXT"):
            self._scratchpad_grounded(arg or "scene")
            if self._tracker is not None:
                try:
                    self._tracker.clear_ground_flag(None)
                except Exception:  # pragma: no cover — defensive
                    pass

        elif op in ("COUNT", "COMPARE"):
            # These are diagnostic ops — note the question on the
            # scratchpad without claiming a grounded slot.
            self._scratchpad_note(f"{op}({arg})")

        elif op == "RETRIEVE":
            hits = self._do_retrieve(arg)
            if hits:
                self._scratchpad_memory(query=arg, hits=hits)
            info["memory_hits"] = len(hits)

        elif op == "NOTE":
            self._scratchpad_note(arg)

        elif op == "ANSWER":
            self._answer = arg
            done = True
            reward = self._score(arg)
            info["answer"] = arg
            info["correct"] = reward > 0

        else:
            # Unknown op → no-op, but still consume a step.
            info["unknown_op"] = True

        if self._t >= self.max_steps and not done:
            done = True
            info["truncated"] = True

        self._done = done
        self._last_obs = self.question
        return self._last_obs, reward, done, info

    # ── action enumeration ───────────────────────────────────────────

    def valid_actions(self, state: HarnessState) -> List[str]:
        """Enumerate concrete VR ops + a few entity-keyed templates.

        The first action is always ``ANSWER("<text>")`` so the LLM has
        a clear "I'm done" path; the rest cycle through the standard
        VR ops with placeholder args (or schema-derived ones when the
        schema carries entities).
        """
        actions: List[str] = []
        seen: set[str] = set()

        # ANSWER first: it's the only action that terminates.
        ans = 'ANSWER("<text>")'
        actions.append(ans)
        seen.add(ans)

        # Schema-derived per-entity templates (LOOK / CROP / READ_TEXT).
        if state.schema is not None and state.schema.entities:
            for eid in state.schema.entity_order[:4]:
                for op in ("LOOK", "CROP", "READ_TEXT"):
                    rendered = f"{op}({eid})"
                    if rendered not in seen:
                        seen.add(rendered)
                        actions.append(rendered)

        # Caller-provided candidate args for COUNT/COMPARE.
        for op in ("COUNT", "COMPARE"):
            for arg in self.candidate_args.get(op, [f"<{op.lower()}>"])[:2]:
                rendered = f"{op}({arg})"
                if rendered not in seen:
                    seen.add(rendered)
                    actions.append(rendered)

        # Free-form ops.
        for fallback in (
            'LOOK(scene)',
            'RETRIEVE("<keywords>")',
            'NOTE("<text>")',
        ):
            if fallback not in seen:
                seen.add(fallback)
                actions.append(fallback)

        return actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    # ── optional cost lookup ─────────────────────────────────────────

    def action_kind(self, action: str) -> str:
        """Map an action string to the right ``RewardConfig`` cost field."""
        op, _ = parse_op_call(action)
        if op in ("LOOK", "CROP", "READ_TEXT", "COUNT", "COMPARE"):
            return ACTION_KIND_VR_LOOK
        if op == "RETRIEVE":
            return ACTION_KIND_VR_RETRIEVE
        if op == "NOTE":
            return ACTION_KIND_VR_NOTE
        if op == "ANSWER":
            return ACTION_KIND_VR_ANSWER
        return ACTION_KIND_PRIMITIVE

    # ── exposed accessors (read-only) ────────────────────────────────

    @property
    def t(self) -> int:
        return self._t

    @property
    def answer(self) -> Optional[str]:
        return self._answer

    # ── private helpers ──────────────────────────────────────────────

    def _scratchpad_grounded(self, slot: str) -> None:
        """Mark *slot* as observed on the bound scratchpad."""
        sp = self.scratchpad
        try:
            sp.grounded_slots.setdefault(slot, "observed")
        except AttributeError:  # pragma: no cover — exotic stand-in
            _LOGGER.debug("VRHarness scratchpad missing grounded_slots; ignoring")

    def _scratchpad_note(self, text: str) -> None:
        if not text:
            return
        try:
            self.scratchpad.notes.append(text[:140])
            self.scratchpad.notes = self.scratchpad.notes[-5:]
        except AttributeError:  # pragma: no cover
            _LOGGER.debug("VRHarness scratchpad missing notes; ignoring")

    def _scratchpad_memory(
        self, *, query: str, hits: Sequence[Any]
    ) -> None:
        try:
            self.scratchpad.memory_hits.extend(
                {"query": (query or "")[:80], "hit": _stringify(h)} for h in hits
            )
            self.scratchpad.memory_hits = self.scratchpad.memory_hits[-5:]
        except AttributeError:  # pragma: no cover
            _LOGGER.debug("VRHarness scratchpad missing memory_hits; ignoring")

    def _do_retrieve(self, arg: str) -> List[Any]:
        """Run the RETRIEVE query against the bound memory store.

        Falls back to an empty hit list when no memory is bound, when
        the query is empty, or when the memory store raises — keeping
        the rollout going matters more than perfect recall.
        """
        if self._memory is None or not arg:
            return []
        try:
            return list(self._memory.query(arg, k=3))
        except Exception as exc:  # pragma: no cover — defensive
            _LOGGER.warning("VRHarness memory.query failed: %s", exc)
            return []

    def _score(self, answer: Optional[str]) -> float:
        """Return ``+1`` for an exact match against ``gold_answer``."""
        if self.gold_answer is None or answer is None:
            return 0.0
        a = str(answer).strip().strip('"').strip("'").lower()
        g = str(self.gold_answer).strip().lower()
        return 1.0 if a == g else 0.0


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _stringify(hit: Any) -> str:
    """Render a memory hit compactly (mirrors ``actor_agent._stringify_memory_hit``)."""
    if isinstance(hit, dict):
        parts: List[str] = []
        for key in ("summary", "action", "outcome", "key"):
            v = hit.get(key)
            if v:
                parts.append(f"{key}={str(v)[:60]}")
        if not parts:
            parts = [f"{k}={str(v)[:60]}" for k, v in list(hit.items())[:3] if v]
        return " | ".join(parts) if parts else "(empty)"
    return str(hit)[:120]


__all__ = ["VRHarness", "VR_OPS"]
