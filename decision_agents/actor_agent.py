"""Actor Agent — Tier-2 decision agent for the COS-PLAY pipeline.

This module implements **Agent 1** from ``plans/PLAN-ACTION-AGENT.md``
§2.3: the trained actor that consumes a structured ``<state>…</state>``
schema from the Visual Grounding pipeline (``vlm_wrapper``), selects a
skill via an injected :class:`~decision_agents.skill_interface.SkillProvider`,
runs a short inner-MDP reasoning loop over the schema, and finally
emits an environment action.

Design goals (per the plan):

* **Schema-native** — the state representation is a parsed
  :class:`StateSchema`, not raw observation text (PLAN §7 Phase 2).
  Entity references like ``click(e5)`` resolve through the schema to
  real actionable handles (``click(bid)``) (PLAN §7 Phase 3).
* **Skill-agnostic** — skills are injected via the
  :class:`~decision_agents.skill_interface.SkillProvider` interface.
  The actor never imports from ``skill_agents`` directly.  That keeps
  Agent 1 and Agent 2 (Skill-Use) loosely coupled, per PLAN §2.3.
* **Inner MDP scaffold** — the per-step loop runs a short inner-MDP
  sequence (GROUND / CHECK / RETRIEVE / CONCLUDE / EXECUTE) governed by
  a pluggable :class:`~decision_agents.inner_mdp.HopPolicy`.  The
  default :class:`HeuristicHopPolicy` is rule-based; the long-term plan
  (§5) is to swap in a trained ``hop_select`` LoRA.
* **Reward-aware** — forwards through the existing
  :class:`~decision_agents.reward_func.RewardComputer` so the
  ``r_env + r_follow + r_cost`` shaping keeps working unchanged.

Relationship to :class:`~decision_agents.agent.VLMDecisionAgent`: the
older ``VLMDecisionAgent`` is a *text-observation* agent kept for
backward compatibility with ``scripts/qwen3_decision_agent.py`` and the
Pipeline A / B flows in :mod:`decision_agents.README`.  ``ActorAgent``
is its schema-native successor and the designated GRPO training target
(PLAN §6 Phase 1).
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from API_func import ask_model
except ImportError:  # pragma: no cover — optional dep for offline tests
    ask_model = None

_LOGGER = logging.getLogger(__name__)

from .agent_helper import (
    EpisodicMemoryStore,
    HARD_SUMMARY_CHAR_LIMIT,
    infer_intention,
)
from .inner_mdp import (
    HeuristicHopPolicy,
    HopAction,
    HopPolicy,
    HopStep,
    HopTrace,
)
from .reward_func import RewardComputer, RewardConfig, RewardResult
from .schema_parser import (
    Entity,
    ResolvedAction,
    StateSchema,
    parse_state_schema,
    resolve_entity_action,
)
from .skill_interface import (
    NullSkillProvider,
    SkillGuidance,
    SkillProvider,
)
from .skill_tracker import SkillTracker


# ──────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = "gpt-4o-mini"
MAX_LAST_ACTIONS: int = 5
MAX_PROGRESS_NOTES: int = 3
MAX_VALID_ACTIONS_IN_PROMPT: int = 16
MAX_ENTITIES_IN_PROMPT: int = 12
ANTI_REPETITION_WINDOW: int = 3
"""If the last N actions are identical with reward ≤ 0, pick an alternative."""


# ──────────────────────────────────────────────────────────────────────
# Per-step decision record
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ActorDecision:
    """Structured output of :meth:`ActorAgent.step`.

    The runner uses this to drive ``env.step`` and to build the
    :class:`Experience` payload.  Everything the actor did on this outer
    step is captured here so episodes can be replayed or scored offline.
    """

    action: str                                            # env-level action string
    resolved: Optional[ResolvedAction] = None               # entity-reference resolution
    intention: str = ""                                     # [TAG] subgoal phrase
    summary: str = ""                                       # compact key=value summary
    active_skill_id: Optional[str] = None
    reselected: bool = False
    reselect_reason: str = ""
    hop_trace: HopTrace = field(default_factory=HopTrace)
    anti_repetition_triggered: bool = False
    valid_actions: List[str] = field(default_factory=list)
    reasoning: str = ""                                     # raw LLM reasoning (best-effort)
    queried_skill: bool = False                             # this step ran skill_provider.select
    queried_mem: bool = False                               # this step ran memory.query (via RETRIEVE hop)
    parse_path: str = ""                                    # which strategy produced the action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "resolved_action": (
                self.resolved.resolved if self.resolved else self.action
            ),
            "entity_ref": self.resolved.eid if self.resolved else None,
            "intention": self.intention,
            "summary": self.summary,
            "active_skill_id": self.active_skill_id,
            "reselected": self.reselected,
            "reselect_reason": self.reselect_reason,
            "hop_trace": self.hop_trace.to_dict(),
            "anti_repetition_triggered": self.anti_repetition_triggered,
            "valid_actions": list(self.valid_actions),
            "reasoning": self.reasoning,
            "queried_skill": self.queried_skill,
            "queried_mem": self.queried_mem,
            "parse_path": self.parse_path,
        }


# ──────────────────────────────────────────────────────────────────────
# Actor internal state (owned by ActorAgent, not the runner)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class InnerScratchpad:
    """Option-A inner-MDP scratchpad (PLAN-ACTION-AGENT §5).

    Accumulated across the hops of a single outer step (and, for memory
    hits, across a whole skill period).  Hops share the same ``<state>``
    schema but deposit their side effects here so the action prompt can
    read them without calling vision tools.

    Fields
    ------
    pending_ground_slots
        Slots the tracker said were missing at activation time.  The
        actor uses this as a forced prefix of ``GROUND`` hops so the
        scaffold (not the LoRA) owns the slot-coverage guarantee from
        PLAN §10.
    grounded_slots
        Slots that were GROUND-ed in the current outer step, mapped to
        a best-effort resolved value.  Under Option A this value is a
        placeholder (``"best_effort"``) because the schema is already
        the grounding output — Option B will fold in actual tool
        results once the visual grounding handlers are wired.
    memory_hits
        Top-k hits from the last ``RETRIEVE`` hop.  Rendered into the
        action prompt and logged on the Experience so Skill-Bank mining
        can see which memories mattered.
    notes
        ``CONCLUDE(subgoal)`` arguments committed during this step —
        free-form strings the LLM can use as intermediate lemmas.
    """

    pending_ground_slots: List[str] = field(default_factory=list)
    grounded_slots: Dict[str, str] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.grounded_slots or self.memory_hits or self.notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending_ground_slots": list(self.pending_ground_slots),
            "grounded_slots": dict(self.grounded_slots),
            "memory_hits": list(self.memory_hits),
            "notes": list(self.notes),
        }


@dataclass
class ActorState:
    """Per-episode state the actor maintains across outer steps."""

    current_intention: str = ""
    last_actions: List[str] = field(default_factory=list)
    progress_notes: List[str] = field(default_factory=list)
    last_rewards: List[float] = field(default_factory=list)
    last_summary: str = ""
    last_schema: Optional[StateSchema] = None
    scratchpad: InnerScratchpad = field(default_factory=InnerScratchpad)


# ──────────────────────────────────────────────────────────────────────
# Actor agent
# ──────────────────────────────────────────────────────────────────────


class ActorAgent:
    """Schema-native decision agent.

    Parameters
    ----------
    model
        LLM used for intention inference and action selection.  Any
        model reachable through ``API_func.ask_model`` works (GPT-4o,
        Claude Sonnet, Qwen3-8B via vLLM).
    skill_provider
        Implementation of :class:`SkillProvider`.  Defaults to
        :class:`NullSkillProvider` (skill-free baseline).
    hop_policy
        Implementation of :class:`HopPolicy`.  Defaults to
        :class:`HeuristicHopPolicy`.  Plug in the trained ``hop_select``
        LoRA adapter here when it becomes available.
    reward_config
        Optional :class:`RewardConfig` — uses defaults if omitted.
    memory
        Optional :class:`EpisodicMemoryStore` for the ``RETRIEVE`` hop.
        Defaults to an in-memory store with auto-detected embedder.
    intention_model
        Optional separate model for ``infer_intention`` calls, e.g. if
        you want a cheaper model for the tag prediction.  Falls back to
        *model* when None.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        skill_provider: Optional[SkillProvider] = None,
        hop_policy: Optional[HopPolicy] = None,
        reward_config: Optional[RewardConfig] = None,
        memory: Optional[EpisodicMemoryStore] = None,
        intention_model: Optional[str] = None,
        max_hops_per_step: int = 4,
        stall_window: int = 4,
        anti_repetition_window: int = ANTI_REPETITION_WINDOW,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.intention_model = intention_model or self.model
        self.skill_provider: SkillProvider = skill_provider or NullSkillProvider()
        self.hop_policy: HopPolicy = hop_policy or HeuristicHopPolicy(
            max_hops=max_hops_per_step
        )
        self.reward_config = reward_config or RewardConfig()
        self.reward_computer = RewardComputer(self.reward_config)
        self.tracker = SkillTracker(stall_window=stall_window)
        self.memory = memory if memory is not None else _build_default_memory()
        self.max_hops_per_step = max(1, int(max_hops_per_step))
        self.anti_repetition_window = max(1, int(anti_repetition_window))
        self.state = ActorState()

    # ── Episode lifecycle ────────────────────────────────────────────

    def reset(self) -> None:
        """Forget episode-local state.  Call before each episode."""
        self.state = ActorState()
        self.tracker.reset()
        self.reward_computer.reset()

    # ── Main per-step entry point ────────────────────────────────────

    def step(
        self,
        *,
        observation: str,
        schema_text: Optional[str] = None,
        schema: Optional[StateSchema] = None,
        task: str = "",
        valid_actions: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> ActorDecision:
        """Run one outer-MDP step.

        The caller supplies the structured schema (either pre-parsed via
        *schema* or as text via *schema_text*); when neither is given the
        actor falls back to a raw-observation path so it still works for
        callers that have not yet integrated the VLM grounding head
        (PLAN §7 Phase 1).

        Returns an :class:`ActorDecision` the runner feeds to ``env.step``.
        The actor does NOT call the env — keeping it side-effect-free
        makes it trivially testable and GRPO-compatible.
        """
        info = info or {}

        # 1. Parse schema (Phase 2 input).
        parsed = schema or (parse_state_schema(schema_text) if schema_text else None)
        self.state.last_schema = parsed

        # 2. Compact summary — either from the schema or from raw text.
        summary = (
            parsed.compact_summary()
            if parsed is not None
            else _compact_from_text(observation)
        )
        self.state.last_summary = summary

        # 3. Intention inference — runs BEFORE reselect (PLAN-ACTION-AGENT §1
        #    step 2) so it feeds the RAG skill-query and the action prompt
        #    with a fresh subgoal instead of last step's stale tag.
        intention = self._infer_intention(summary, task)
        self.state.current_intention = intention

        # 4. Determine the valid-action list.  Prefer schema <actions>.
        valid = _resolve_valid_actions(valid_actions, parsed, observation)

        # 5. Re-selection check (PLAN §1 step 3).
        reselected = False
        reselect_reason = ""
        queried_skill = False
        should_reselect, reason = self.tracker.should_reselect(parsed)
        if should_reselect:
            # Report outcome of the skill we are about to replace.
            if self.tracker.active_skill_id is not None:
                record = self.tracker.finalize_active_skill(
                    outcome=reason or "switch"
                )
                if record:
                    self.skill_provider.record_outcome(
                        record["skill_id"],
                        outcome=record["outcome"],
                        reward=record["reward"],
                        steps_taken=record["steps_taken"],
                    )
            # Calling the provider at all counts as a QUERY_SKILL event
            # (PLAN §4 — reselect cost fires regardless of whether any
            # guidance comes back).
            queried_skill = True
            new_guidance = self._select_skill(parsed, summary, task)
            if new_guidance is not None:
                check = self.tracker.activate(new_guidance, parsed)
                reselected = True
                reselect_reason = reason or "reselect"
                # PLAN §10 — the tracker owns slot-coverage.  Seed the
                # scratchpad with its missing_slots so _run_inner_mdp can
                # treat them as a forced GROUND prefix.
                self.state.scratchpad = InnerScratchpad(
                    pending_ground_slots=list(check.missing_slots),
                )
        # On non-reselect steps, keep last step's scratchpad but forget
        # transient bookkeeping — memory hits are the one thing worth
        # carrying across steps so the LLM can keep citing them.
        else:
            self.state.scratchpad = InnerScratchpad(
                memory_hits=list(self.state.scratchpad.memory_hits[-3:]),
                notes=list(self.state.scratchpad.notes[-3:]),
            )

        # 6. Inner-MDP reasoning (PLAN §5).  Applies side effects against
        #    the scratchpad (GROUND → tracker.clear_ground_flag; RETRIEVE →
        #    memory.query; CONCLUDE → scratchpad.notes).
        trace = self._run_inner_mdp(parsed)
        queried_mem = any(s.action is HopAction.RETRIEVE for s in trace.steps)

        # 7. If the inner loop terminated with an EXECUTE, use its arg
        #    as the candidate action; otherwise prompt the LLM.
        candidate_from_hop = (
            trace.last().arg
            if trace.last() is not None
               and trace.last().action is HopAction.EXECUTE
               and trace.last().arg
            else ""
        )

        action_text, reasoning, parse_path = self._pick_action(
            schema=parsed,
            summary=summary,
            task=task,
            valid_actions=valid,
            candidate_from_hop=candidate_from_hop,
            observation=observation,
        )

        # 8. Resolve entity references in the action (PLAN §7 Phase 3).
        resolved = resolve_entity_action(action_text, parsed)
        final_action = resolved.resolved or action_text

        # 9. Anti-repetition guard.
        final_action, anti_rep = self._anti_repetition(final_action, valid)

        decision = ActorDecision(
            action=final_action,
            resolved=resolved,
            intention=intention,
            summary=summary,
            active_skill_id=self.tracker.active_skill_id,
            reselected=reselected,
            reselect_reason=reselect_reason,
            hop_trace=trace,
            anti_repetition_triggered=anti_rep,
            valid_actions=valid,
            reasoning=reasoning,
            queried_skill=queried_skill,
            queried_mem=queried_mem,
            parse_path=parse_path,
        )
        return decision

    # ── Update after env.step ────────────────────────────────────────

    def observe_result(
        self,
        decision: ActorDecision,
        *,
        reward: float,
        next_observation: str = "",
        next_schema_text: Optional[str] = None,
        next_schema: Optional[StateSchema] = None,
        done: bool = False,
    ) -> RewardResult:
        """Fold env feedback back into the actor state.

        Returns the :class:`RewardResult` for the step — the runner
        should stash it on the :class:`Experience` it builds.
        """
        schema_after = next_schema or (
            parse_state_schema(next_schema_text) if next_schema_text else None
        )

        # Update short history buffers.
        st = self.state
        st.last_actions.append(decision.action)
        if len(st.last_actions) > MAX_LAST_ACTIONS:
            st.last_actions = st.last_actions[-MAX_LAST_ACTIONS:]
        st.last_rewards.append(float(reward or 0.0))
        if len(st.last_rewards) > MAX_LAST_ACTIONS:
            st.last_rewards = st.last_rewards[-MAX_LAST_ACTIONS:]
        if reward and float(reward) > 0:
            st.progress_notes.append(
                f"+{float(reward):g} after {decision.action}"[:80]
            )
            if len(st.progress_notes) > MAX_PROGRESS_NOTES:
                st.progress_notes = st.progress_notes[-MAX_PROGRESS_NOTES:]

        # Tracker update + reward computation.
        self.tracker.record_step(
            reward=reward,
            schema_after=schema_after,
            advance_protocol=not _is_noop_action(decision.action),
        )

        # Derive the action_type for reward-cost bookkeeping.
        action_type = (
            "CALL_SKILL"
            if self.tracker.active_skill_id is not None
            else "primitive"
        )
        contract = self._active_skill_contract_view()
        rr = self.reward_computer.compute_reward(
            r_env=float(reward or 0.0),
            action_type=action_type,
            observation=next_observation or "",
            active_skill_id=self.tracker.active_skill_id,
            skill_contract=contract,
            queried_skill=bool(decision.queried_skill),
            queried_mem=bool(decision.queried_mem),
        )

        # Push to memory so RETRIEVE hops get a grounded store.
        if self.memory is not None:
            try:
                next_summary = (
                    schema_after.compact_summary()
                    if schema_after is not None
                    else _compact_from_text(next_observation)
                )
                self.memory.add_experience(
                    state_summary=decision.summary,
                    action=decision.action,
                    next_state_summary=next_summary,
                    done=done,
                )
            except Exception:
                pass

        # Episode termination: finalize the active skill with the right
        # outcome so the provider can update its statistics.
        if done and self.tracker.active_skill_id is not None:
            outcome = "success" if float(reward or 0.0) > 0 else "timeout"
            record = self.tracker.finalize_active_skill(outcome=outcome)
            if record:
                self.skill_provider.record_outcome(
                    record["skill_id"],
                    outcome=record["outcome"],
                    reward=record["reward"],
                    steps_taken=record["steps_taken"],
                )

        return rr

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _select_skill(
        self,
        schema: Optional[StateSchema],
        summary: str,
        task: str,
    ) -> Optional[SkillGuidance]:
        """Consult the injected :class:`SkillProvider` for guidance."""
        intention = self.state.current_intention or ""
        query_parts = [p for p in (task, intention, summary) if p]
        query = " | ".join(query_parts)[:1500] or summary or task

        predicates = _predicates_from_schema(schema)
        results = self.skill_provider.select(
            query,
            state_summary=summary,
            structured_state=schema,
            current_predicates=predicates,
            top_k=1,
        )
        return results[0] if results else None

    def _run_inner_mdp(
        self,
        schema: Optional[StateSchema],
    ) -> HopTrace:
        """Roll the inner-MDP until EXECUTE (or the cap).

        Unlike the earlier "hops are logged only" behaviour, each hop now
        produces a side effect on ``self.state.scratchpad`` so the next
        iteration sees an updated belief state.  This closes the PLAN §5
        Option-A gap where the plan promised a scratchpad but the code
        kept discarding it.

        Side effects per hop:

        * ``GROUND(slot)``  — record a best-effort resolution in the
          scratchpad and call ``tracker.clear_ground_flag(schema)`` so
          :class:`HeuristicHopPolicy` stops demanding the same slot on the
          next iteration (PLAN §10 slot-coverage insertion).
        * ``RETRIEVE(query)`` — call ``self.memory.query`` (when memory is
          wired) and attach hits to the scratchpad + ``HopStep.result``.
        * ``CONCLUDE(subgoal)`` — commit to the scratchpad's notes list.
        * ``CHECK`` / ``VERIFY`` — purely logged (the heuristic policy
          doesn't emit these yet; the LoRA can learn them later).
        * ``EXECUTE``       — terminates the loop, arg consumed by
          :meth:`_pick_action`.
        """
        trace = HopTrace()
        for _ in range(self.max_hops_per_step):
            step = self.hop_policy.select_next_hop(
                schema=schema,
                guidance=self.tracker.active_guidance,
                trace=trace,
                max_hops=self.max_hops_per_step,
            )
            if step is None:
                trace.append(
                    HopStep(
                        action=HopAction.EXECUTE,
                        arg="",
                        note="hop policy returned None",
                    )
                )
                break
            self._apply_hop_side_effect(step, schema)
            trace.append(step)
            if step.action is HopAction.EXECUTE:
                break
        return trace

    def _apply_hop_side_effect(
        self,
        step: HopStep,
        schema: Optional[StateSchema],
    ) -> None:
        """Mutate the actor scratchpad / tracker according to ``step``.

        Kept as a dedicated method so a future ``HopPolicy`` LoRA can
        emit any of {GROUND, RETRIEVE, CONCLUDE, CHECK, VERIFY} without
        changing the hop-execution contract.
        """
        sp = self.state.scratchpad

        if step.action is HopAction.GROUND:
            if step.arg:
                # Option A: the schema IS the grounding output, so we
                # only record that this slot has been considered.  Option B
                # (visual-tool call) will replace "best_effort" with the
                # actual tool result.
                sp.grounded_slots.setdefault(step.arg, "best_effort")
            # Deterministic rule: once we've emitted a GROUND hop for
            # an activation's pending slot, tell the tracker so it doesn't
            # re-trigger the reselect-on-missing-slot rule.
            if schema is not None:
                self.tracker.clear_ground_flag(schema)

        elif step.action is HopAction.RETRIEVE:
            if self.memory is None:
                step.note = (step.note + " | memory unavailable").strip(" |")
                return
            query = (
                step.arg
                or self.state.current_intention
                or self.state.last_summary
                or ""
            )
            if not query:
                return
            try:
                hits = self.memory.query(query, k=3)
            except Exception as exc:  # pragma: no cover — defensive
                _LOGGER.warning("memory.query failed: %s", exc)
                hits = []
            if hits:
                rendered = [_stringify_memory_hit(h) for h in hits]
                sp.memory_hits.extend(
                    {"query": query[:80], "hit": r} for r in rendered
                )
                # Keep scratchpad bounded — only the most recent 5 hits
                # are rendered into the prompt.
                sp.memory_hits = sp.memory_hits[-5:]
                step.result = rendered

        elif step.action is HopAction.CONCLUDE:
            if step.arg:
                sp.notes.append(step.arg[:140])
                sp.notes = sp.notes[-5:]

        # CHECK / VERIFY / EXECUTE: no scratchpad mutation.

    def _pick_action(
        self,
        *,
        schema: Optional[StateSchema],
        summary: str,
        task: str,
        valid_actions: List[str],
        candidate_from_hop: str,
        observation: str,
    ) -> Tuple[str, str, str]:
        """Return ``(action_text, reasoning, parse_path)``.

        Priority:

        1. Active skill's current protocol step (when it is a literal
           action and present in *valid_actions*).
        2. *candidate_from_hop* if it is in *valid_actions*.
        3. LLM over the compact prompt (when ``ask_model`` is available).
           The reply is decoded via the multi-strategy pipeline in
           :func:`_extract_action_from_reply` (exact → numbered →
           entity-ref → edit distance → token overlap).
        4. Deterministic fallback — first valid action or ``"no-op"``.

        ``parse_path`` is a short tag identifying which branch produced
        the action; it's logged on ``ActorDecision`` to let the GRPO
        pipeline quantify how often the LoRA actually chose a valid
        action without the fallback stack rescuing it.
        """
        # 1. Follow the active skill's protocol step when it matches.
        current_step = self.tracker.current_step()
        if current_step:
            exact = _match_valid_action(current_step, valid_actions)
            if exact is not None:
                return exact, f"protocol step: {current_step}", "protocol"

        # 2. Respect the inner-MDP EXECUTE argument when it's a valid action.
        if candidate_from_hop:
            exact = _match_valid_action(candidate_from_hop, valid_actions)
            if exact is not None:
                return exact, "inner-MDP EXECUTE", "hop_execute"

        # 3. LLM prompt path.
        if ask_model is not None:
            prompt = self._build_action_prompt(
                schema=schema,
                summary=summary,
                task=task,
                valid_actions=valid_actions,
                observation=observation,
            )
            try:
                reply = ask_model(
                    prompt,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=200,
                )
            except Exception:
                reply = ""
            reply = reply or ""
            action_text, parse_path = _extract_action_from_reply(reply, valid_actions)
            if action_text:
                return action_text, reply[:400], f"llm:{parse_path}"

        # 4. Deterministic fallback.
        if valid_actions:
            return valid_actions[0], "fallback: first valid action", "fallback_first"
        return "no-op", "fallback: no valid actions", "fallback_noop"

    def _build_action_prompt(
        self,
        *,
        schema: Optional[StateSchema],
        summary: str,
        task: str,
        valid_actions: List[str],
        observation: str,
    ) -> str:
        """Compose the action-selection prompt (compact, schema-aware).

        Renders — in order — task, intention, state summary, entity
        block, active-skill block, the inner-MDP scratchpad (memory hits
        + grounded slots + committed subgoals), recent actions / rewards,
        recent progress notes, and a numbered valid-action list.

        Falls back to ``schema.goal`` / ``schema.task`` when the caller
        passes an empty ``task`` so the prompt never renders
        ``"Task: (unspecified)"`` just because the external runner
        forgot to thread the task text through.
        """
        g = self.tracker.active_guidance
        skill_block = _format_skill_block(g, self.tracker.progress_marker())
        entity_block = _format_entity_block(schema)
        scratchpad_block = _format_scratchpad(self.state.scratchpad)
        recent = " | ".join(self.state.last_actions[-3:]) or "(none)"
        recent_rewards = (
            ", ".join(f"{r:+.2f}" for r in self.state.last_rewards[-3:]) or "-"
        )
        recent_notes = " | ".join(self.state.progress_notes[-3:]) or ""

        numbered_actions = "\n".join(
            f"  {i+1}. {a}" for i, a in enumerate(valid_actions)
        )

        effective_task = task or (
            (getattr(schema, "goal", "") or getattr(schema, "task", ""))
            if schema is not None
            else ""
        )

        parts: List[str] = [
            "You are an Actor Agent choosing ONE environment action.",
            "",
            f"Task: {effective_task or '(unspecified)'}",
            f"Intention: {self.state.current_intention or '(tbd)'}",
            f"State summary: {summary}",
        ]
        if schema is None and observation:
            parts.extend([
                "",
                "Observation (text fallback):",
                observation[:1500],
            ])
        if entity_block:
            parts.extend(["", "Entities:", entity_block])
        if skill_block:
            parts.extend(["", "Active skill:", skill_block])
        if scratchpad_block:
            parts.extend(["", "Inner reasoning so far:", scratchpad_block])
        parts.extend([
            "",
            f"Recent actions: {recent}",
            f"Recent rewards: {recent_rewards}",
        ])
        if recent_notes:
            parts.append(f"Recent progress: {recent_notes}")
        parts.extend([
            "",
            "Valid actions (pick EXACTLY one, copy the string verbatim; "
            "entity references like click(e5) are also accepted when "
            "the action list permits):",
            numbered_actions,
            "",
            "Output format (strict):",
            "THOUGHT: <at most 2 sentences>",
            "ACTION: <one of the valid actions, or its 1-based number>",
        ])
        return "\n".join(parts)

    def _infer_intention(self, summary: str, task: str) -> str:
        """Wrapper around :func:`agent_helper.infer_intention`."""
        if not summary or ask_model is None:
            return self.state.current_intention
        try:
            return infer_intention(
                summary,
                game=None,
                model=self.intention_model,
                context={
                    "last_actions": self.state.last_actions,
                    "progress_notes": self.state.progress_notes,
                    "task": task,
                },
            )
        except Exception:
            return self.state.current_intention

    def _anti_repetition(
        self,
        action: str,
        valid_actions: List[str],
    ) -> Tuple[str, bool]:
        """Return ``(action, triggered)``.

        PLAN §1 step 7.  If the last *anti_repetition_window* actions
        were identical AND all rewards in that window were ≤ 0, we swap
        to a different valid action (deterministically the next one in
        the list so tests are reproducible).
        """
        st = self.state
        window = self.anti_repetition_window
        if len(st.last_actions) < window or not valid_actions:
            return action, False
        recent = st.last_actions[-window:]
        recent_r = st.last_rewards[-window:] if st.last_rewards else []
        if len(recent_r) < window:
            return action, False
        if all(a == action for a in recent) and all(r <= 0 for r in recent_r):
            alternatives = [a for a in valid_actions if a != action]
            if alternatives:
                return alternatives[0], True
        return action, False

    def _active_skill_contract_view(self) -> Any:
        """Return a shim object exposing ``eff_add`` / ``eff_del`` for
        :class:`RewardComputer`.

        We don't want to pull in ``skill_agents.stage3_mvp.schemas``
        here — RewardComputer only calls ``getattr(contract,
        "eff_add", set())`` — so we build a tiny anonymous object.
        """
        g = self.tracker.active_guidance
        if g is None:
            return None

        class _ContractView:
            eff_add = set(g.eff_add)
            eff_del = set(g.eff_del)

        return _ContractView


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────


def run_actor_episode(
    env: Any,
    *,
    agent: Optional[ActorAgent] = None,
    model: Optional[str] = None,
    skill_provider: Optional[SkillProvider] = None,
    hop_policy: Optional[HopPolicy] = None,
    reward_config: Optional[RewardConfig] = None,
    task: str = "",
    max_steps: int = 200,
    schema_from_info: Callable[[Dict[str, Any]], Optional[str]] = None,
    verbose: bool = False,
) -> "Episode":
    """End-to-end runner that drives :class:`ActorAgent` against an env.

    Assumes the env returns ``(obs, info)`` from ``reset()`` and
    ``(obs, reward, term, trunc, info)`` from ``step()``.  The structured
    schema is expected under ``info["schema"]`` (override via
    *schema_from_info*).  When the schema is absent the actor falls back
    to the text-observation path so this runner works with both Phase 1
    (text) and Phase 2 (schema) environments (PLAN §7).
    """
    from data_structure.experience import Experience, Episode

    if agent is None:
        agent = ActorAgent(
            model=model,
            skill_provider=skill_provider,
            hop_policy=hop_policy,
            reward_config=reward_config,
        )
    agent.reset()

    if schema_from_info is None:
        schema_from_info = lambda info: (info or {}).get("schema") or \
            (info or {}).get("schema_text")

    obs, info = env.reset()
    observation = str(obs) if obs is not None else ""
    info = dict(info or {})
    episode_task = task or info.get("task", "")

    experiences: List[Experience] = []
    done = False
    step_count = 0

    while step_count < max_steps:
        schema_text = schema_from_info(info)
        valid_actions = info.get("valid_actions") or info.get("available_actions")

        decision = agent.step(
            observation=observation,
            schema_text=schema_text,
            task=episode_task,
            valid_actions=valid_actions,
            info=info,
        )

        env_action = (
            decision.resolved.resolved
            if decision.resolved is not None
            else decision.action
        )
        next_obs, reward, term, trunc, next_info = env.step(env_action)
        done = bool(term or trunc)

        next_observation = str(next_obs) if next_obs is not None else ""
        next_info = dict(next_info or {})
        next_schema_text = schema_from_info(next_info)

        rr = agent.observe_result(
            decision,
            reward=float(reward or 0.0),
            next_observation=next_observation,
            next_schema_text=next_schema_text,
            done=done,
        )

        exp = Experience(
            state=observation,
            action=decision.action,
            reward=float(reward or 0.0),
            next_state=next_observation,
            done=done,
            intentions=decision.intention or None,
            tasks=episode_task or None,
            sub_tasks=decision.active_skill_id,
        )
        exp.idx = step_count
        exp.summary_state = decision.summary or None
        exp.reward_details = rr.to_dict()
        exp.action_type = (
            "CALL_SKILL" if decision.active_skill_id else "primitive"
        )
        exp.available_actions = list(decision.valid_actions) or None
        extras = getattr(exp, "extras", None)
        if extras is None:
            extras = {}
        extras["hop_trace"] = decision.hop_trace.to_dict()
        extras["reselected"] = decision.reselected
        extras["reselect_reason"] = decision.reselect_reason
        extras["anti_repetition_triggered"] = decision.anti_repetition_triggered
        extras["queried_skill"] = decision.queried_skill
        extras["queried_mem"] = decision.queried_mem
        extras["parse_path"] = decision.parse_path
        extras["scratchpad"] = agent.state.scratchpad.to_dict()
        if decision.reasoning:
            extras["reasoning"] = decision.reasoning[:400]
        exp.extras = extras
        experiences.append(exp)

        if verbose:
            print(
                f"  step {step_count}: action={decision.action!r} "
                f"skill={decision.active_skill_id} hops={len(decision.hop_trace)} "
                f"{rr}"
            )

        observation = next_observation
        info = next_info
        step_count += 1
        if done:
            break

    cumulative = agent.reward_computer.cumulative
    episode = Episode(
        experiences=experiences,
        task=episode_task or "Unspecified task",
        env_name=info.get("env_name") or info.get("game"),
        game_name=info.get("game_name") or info.get("game"),
        metadata={
            "done": done,
            "steps": step_count,
            "cumulative_reward": cumulative.to_dict(),
            "skill_history": agent.tracker.state.history,
            "agent_state": {
                "current_intention": agent.state.current_intention,
                "last_actions": agent.state.last_actions,
                "progress_notes": agent.state.progress_notes,
                "active_skill_id": agent.tracker.active_skill_id,
            },
        },
    )
    episode.set_outcome()
    return episode


# ──────────────────────────────────────────────────────────────────────
# Module-private helpers
# ──────────────────────────────────────────────────────────────────────


_ACTION_LINE_RE = re.compile(r"ACTION\s*:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)


def _compact_from_text(observation: str) -> str:
    """Lightweight fallback when the actor has no parsed schema."""
    from .agent_helper import compact_text_observation
    return compact_text_observation(observation, max_chars=HARD_SUMMARY_CHAR_LIMIT)


def _resolve_valid_actions(
    valid_actions: Optional[List[str]],
    schema: Optional[StateSchema],
    observation: str,
) -> List[str]:
    """Pick the actor's action vocabulary for this step.

    Priority: explicit *valid_actions* argument → schema ``<actions>`` →
    regex over ``observation`` (legacy fallback for gymv).
    """
    if valid_actions:
        return [str(a) for a in valid_actions][:MAX_VALID_ACTIONS_IN_PROMPT]
    if schema and schema.actions:
        return schema.actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    m = re.search(
        r"[Vv]alid\s+actions?\s*[:\-]\s*(.+?)(?:\n|\.|$)", observation or ""
    )
    if m:
        raw = m.group(1).strip()
        return [a.strip() for a in re.split(r"[,;]", raw) if a.strip()]
    return []


def _match_valid_action(
    candidate: str,
    valid_actions: List[str],
) -> Optional[str]:
    """Return the valid-action string that matches *candidate*, or None."""
    if not candidate or not valid_actions:
        return None
    c = candidate.strip().strip(".").strip()
    for a in valid_actions:
        if a == c:
            return a
    lc = c.lower()
    for a in valid_actions:
        if a.lower() == lc:
            return a
    # prefix / substring tolerance
    for a in valid_actions:
        al = a.lower()
        if al in lc or lc in al:
            return a
    return None


_NUMBERED_RE = re.compile(r"^\s*(\d+)\s*[\.\)\-:]?\s*$")
_ENTITY_REF_RE = re.compile(r"\w+\s*\(\s*e\d+\s*\)")
_TOKEN_RE = re.compile(r"\w+")


def _extract_action_from_reply(
    reply: str,
    valid_actions: List[str],
) -> Tuple[Optional[str], str]:
    """Pull an action string from the LLM reply.

    Returns ``(action, parse_path)`` where ``parse_path`` names which
    strategy succeeded (``"exact"``, ``"numbered"``, ``"entity_ref"``,
    ``"edit_distance"``, ``"token_overlap"``, ``"loose"``) or ``""``
    when nothing matched.  This lets the pipeline log how often the
    LLM needed the fallback stack, which is directly actionable for
    GRPO reward shaping (PLAN-ACTION-AGENT §1 step 6).
    """
    if not reply:
        return None, ""

    # Strategy 0 — pull the ``ACTION:`` line if present.
    m = _ACTION_LINE_RE.search(reply)
    cand = m.group(1).strip().strip("`").strip('"').strip() if m else ""

    if cand:
        # 1. Exact / caseless / substring match against the valid list.
        matched = _match_valid_action(cand, valid_actions)
        if matched is not None:
            return matched, "exact"

        # 2. Numbered selection ("1", "1.", "1)", "1-", "1:").
        num = _NUMBERED_RE.match(cand)
        if num:
            idx = int(num.group(1)) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx], "numbered"

        # 3. Entity-reference pass-through (env runner resolves them).
        if _ENTITY_REF_RE.fullmatch(cand):
            return cand, "entity_ref"

        # 4. Edit-distance fallback — catches typos like "Righ" vs "[Right]".
        if valid_actions:
            close = difflib.get_close_matches(cand, valid_actions, n=1, cutoff=0.6)
            if close:
                return close[0], "edit_distance"

            # Same, but caseless, since difflib is case-sensitive.
            lowered = [a.lower() for a in valid_actions]
            close = difflib.get_close_matches(cand.lower(), lowered, n=1, cutoff=0.6)
            if close:
                return valid_actions[lowered.index(close[0])], "edit_distance"

        # 5. Token-overlap fallback — rank valid actions by shared tokens.
        c_tokens = {t for t in _TOKEN_RE.findall(cand.lower()) if len(t) >= 2}
        if c_tokens and valid_actions:
            scored: List[Tuple[int, int, str]] = []
            for a in valid_actions:
                a_tokens = {t for t in _TOKEN_RE.findall(a.lower()) if len(t) >= 2}
                overlap = len(a_tokens & c_tokens)
                if overlap > 0:
                    scored.append((-overlap, len(a), a))
            if scored:
                scored.sort()
                return scored[0][2], "token_overlap"

    # Strategy 6 — loose search over the whole reply.
    for a in valid_actions:
        if a and a in reply:
            return a, "loose"

    # Strategy 7 — a trailing digit in the reply is a numbered selection.
    trailing = re.search(r"(\d+)\s*$", reply.strip())
    if trailing:
        idx = int(trailing.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx], "numbered"

    return None, ""


def _predicates_from_schema(
    schema: Optional[StateSchema],
) -> Optional[Dict[str, float]]:
    """Convert schema slots into a ``{predicate: float}`` dict.

    The skill-selection engine scores skill effects against these; we
    emit conservative 1.0 indicators for flags that are clearly TRUE
    and leave anything uncertain / null absent.  See PLAN-SKILL-BANK §3
    for the contract.
    """
    if schema is None:
        return None
    predicates: Dict[str, float] = {}
    sf = schema.state_flags
    if sf.scene_type:
        predicates[f"scene_type={sf.scene_type}"] = 1.0
    if sf.phase:
        predicates[f"phase={sf.phase}"] = 1.0
    if sf.progress is not None:
        predicates["has_progress"] = 1.0
    if sf.error:
        predicates["error_present"] = 1.0
    if sf.dialog_open:
        predicates["dialog_open"] = 1.0
    if sf.input_pending:
        predicates["input_pending"] = 1.0
    for ent in schema.entities.values():
        if ent.ontology:
            predicates[f"has:{ent.ontology}"] = 1.0
    if schema.targets.target:
        predicates["has_target"] = 1.0
    if schema.targets.blocker:
        predicates["has_blocker"] = 1.0
    if schema.targets.candidate_set:
        predicates["has_candidate_set"] = 1.0
    return predicates or None


def _format_skill_block(
    g: Optional[SkillGuidance],
    progress: str,
) -> str:
    """Render a skill guidance package for the prompt."""
    if g is None:
        return ""
    parts: List[str] = [f"[{g.skill_id}] {g.skill_name or g.skill_id}"]
    if g.execution_hint:
        parts.append(f"  how: {g.execution_hint[:140]}")
    if g.strategic_description and not g.execution_hint:
        parts.append(f"  strategy: {g.strategic_description[:140]}")
    if g.protocol_steps:
        parts.append(f"  plan ({progress}):")
        for i, step in enumerate(g.protocol_steps[:6]):
            parts.append(f"    {i+1}. {step}")
    if g.preconditions:
        parts.append(f"  preconditions: {'; '.join(g.preconditions[:3])}")
    if g.termination_hint:
        parts.append(f"  done when: {g.termination_hint[:100]}")
    if g.failure_modes:
        parts.append(f"  watch for: {'; '.join(g.failure_modes[:2])}")
    return "\n".join(parts)


def _format_entity_block(schema: Optional[StateSchema]) -> str:
    """Render the most-relevant entities from the schema."""
    if schema is None or not schema.entities:
        return ""

    rows: List[str] = []
    ordered = [
        schema.entities[eid]
        for eid in schema.entity_order[:MAX_ENTITIES_IN_PROMPT]
        if eid in schema.entities
    ]
    for ent in ordered:
        pieces = [f"{ent.eid}"]
        if ent.label:
            pieces.append(ent.label)
        if ent.ontology:
            pieces.append(f"<{ent.ontology}>")
        if ent.affords:
            pieces.append("affords=" + ",".join(ent.affords[:4]))
        if ent.bid:
            pieces.append(f"bid={ent.bid}")
        rows.append("  " + " · ".join(pieces))
    if schema.targets.target:
        rows.append(f"  ** target={schema.targets.target} **")
    return "\n".join(rows)


_NOOP_PATTERNS = ("no-op", "noop", "stay", "wait")


def _is_noop_action(action: str) -> bool:
    a = (action or "").lower().strip()
    return any(p == a for p in _NOOP_PATTERNS)


def _stringify_memory_hit(hit: Any) -> str:
    """Render an :class:`EpisodicMemoryStore` result compactly.

    Memory hits are ``dict``s with keys like ``summary`` / ``action`` /
    ``outcome``; we emit a single ``key=value | key=value`` line that is
    cheap to tokenize and easy for the LLM to cite.  Non-dict hits fall
    back to ``str(hit)``.
    """
    if isinstance(hit, dict):
        ordered_keys = ("summary", "action", "outcome", "key")
        parts = []
        for k in ordered_keys:
            v = hit.get(k)
            if v:
                parts.append(f"{k}={str(v)[:60]}")
        if not parts:
            parts = [f"{k}={str(v)[:60]}" for k, v in list(hit.items())[:3] if v]
        return " | ".join(parts) if parts else "(empty)"
    return str(hit)[:120]


def _format_scratchpad(sp: Optional[InnerScratchpad]) -> str:
    """Render :class:`InnerScratchpad` for the action prompt.

    Returns an empty string when there's nothing to show so the prompt
    doesn't get polluted with ``Inner reasoning so far: (empty)``.
    """
    if sp is None or sp.is_empty():
        return ""
    rows: List[str] = []
    if sp.grounded_slots:
        slots = ", ".join(f"{k}={v}" for k, v in list(sp.grounded_slots.items())[:5])
        rows.append(f"  grounded: {slots}")
    for hit in sp.memory_hits[-3:]:
        rows.append(f"  memory: {hit.get('hit', '')}")
    for note in sp.notes[-3:]:
        rows.append(f"  subgoal: {note}")
    return "\n".join(rows)


def _build_default_memory() -> Optional[EpisodicMemoryStore]:
    try:
        from rag import get_text_embedder
        return EpisodicMemoryStore(embedder=get_text_embedder())
    except Exception as exc:
        _LOGGER.debug("default memory with embedder unavailable: %s", exc)
        try:
            return EpisodicMemoryStore(embedder=None)
        except Exception as exc2:
            _LOGGER.warning("EpisodicMemoryStore unavailable: %s", exc2)
            return None


__all__ = [
    "ActorAgent",
    "ActorDecision",
    "ActorState",
    "run_actor_episode",
]
