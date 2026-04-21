# Data structures for the no-memory, evidence-grounded skill runtime.
#
# This module implements the trajectory substrate described in
# ``PLAN-EXPERIENCE-REFACTOR.md`` and ``README.md`` (this directory).
#
# Three layered containers:
#   1. Experience           — one typed reasoning/control step
#                             (GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE | PRIMITIVE).
#   2. SubTask_Experience   — a segmented local trajectory; serves as a
#                             skill-candidate / verification / transfer unit.
#   3. Episode              — a full rollout container; carries the typed trace,
#                             the evidence ledger, and the final-answer support chain.
#
# Plus three buffers (Experience_Replay_Buffer, Episode_Buffer, Tool_Buffer)
# that support domain-aware filtering and named sampling modes.
#
# Design invariants honored here:
#   - **No-memory contract**: no episodic / semantic memory subsystem, no memory
#     retrieval API, no QUERY_MEM step kind.  All runtime context lives inside
#     these containers.
#   - **Evidence-driven**: every reasoning step can carry evidence references,
#     supported claims, and a claim status; ``validate_evidence_contract()``
#     exposes the lightweight invariant the Harness gate G0 enforces.
#   - **Skills are transferable**: SubTask_Experience carries source / target
#     domain, adapter id, transfer + verification status, and failure mode.
#   - **Episode-local**: nothing here grows beyond the current rollout.
#
# This file is data structures only; data management / processing pipelines
# (rollout collection, summarization, segmentation policies, training samplers)
# live elsewhere.

from __future__ import annotations  # Postponed evaluation of annotations.

import json
import random
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from API_func import ask_model

# Note: helper.py imports are commented out to avoid circular import.
# from data_structure.helper import *


# ---------------------------------------------------------------------------
# Constants — typed-step vocabulary and evidence/claim enums.
# ---------------------------------------------------------------------------

#: Canonical inner-loop step kinds.  ``PRIMITIVE`` is for raw low-level actions
#: when no typed step applies.  ``QUERY_MEM`` is intentionally absent — see
#: PLAN-EXPERIENCE-REFACTOR.md §2 (Principle A, no memory subsystem).
STEP_TYPES: Tuple[str, ...] = (
    "GROUND",
    "CHECK",
    "RETRIEVE",
    "COMMIT",
    "EXECUTE",
    "PRIMITIVE",
)

#: Domain tags shared across game / web / os / video / visual reasoning.
DOMAIN_TYPES: Tuple[str, ...] = (
    "game",
    "web",
    "os",
    "video",
    "visual_reasoning",
)

#: Lifecycle of a claim within an evidence chain.
CLAIM_STATUSES: Tuple[str, ...] = (
    "candidate",
    "verified",
    "contradicted",
    "insufficient",
)

#: Evidence-driven step kinds — the Harness G0 gate requires
#: ``evidence_refs ∪ evidence_items ≠ ∅`` for these.
_EVIDENCE_REQUIRED_STEPS: frozenset = frozenset({"GROUND", "CHECK", "COMMIT"})

#: Legacy ``action_type`` values that violate the no-memory contract.
_FORBIDDEN_ACTION_TYPES: frozenset = frozenset({"QUERY_MEM"})


# ---------------------------------------------------------------------------
# Experience — one typed reasoning/control step.
# ---------------------------------------------------------------------------
class Experience:
    """One atomic reasoning/control step in a trajectory.

    Beyond the classic ``state -> action -> next_state`` transition, an
    ``Experience`` is also one unit of evidence usage and one unit of claim
    progression.  See ``PLAN-EXPERIENCE-REFACTOR.md §4`` for the rationale
    behind every field.
    """

    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        intentions=None,
        tasks=None,
        sub_tasks=None,
        # ----- new domain-general runtime fields --------------------------
        domain_type: Optional[str] = None,
        task_type: Optional[str] = None,
        step_type: Optional[str] = None,
        active_skill: Optional[str] = None,
        skill_phase: Optional[str] = None,
        trace_parent_id: Optional[str] = None,
        # ----- new evidence-ledger fields ---------------------------------
        evidence_refs: Optional[List[str]] = None,
        evidence_items: Optional[List[dict]] = None,
        supports_claims: Optional[List[str]] = None,
        claim_status: Optional[str] = None,
        evidence_confidence: Optional[float] = None,
        # ----- new skill-execution framing (replace tasks/sub_tasks) ------
        goal: Optional[str] = None,
        subgoal: Optional[str] = None,
    ):
        # --- core transition (unchanged) ----------------------------------
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        # The sub-task done is for marking whether the current sub-task is completed.
        self.sub_task_done = None

        # Index of the experience in the episode (assigned by Episode).
        self.idx: Optional[int] = None

        # Raw environment state before any text/NL conversion.
        self.raw_state: Optional[Any] = None
        self.raw_next_state: Optional[Any] = None

        # Valid actions available at this step (list of action name strings).
        self.available_actions: Optional[List[str]] = None

        # Evaluation interface: holds evaluation function configs, criteria,
        # and results from external evaluators for this experience.
        self.interface: Optional[dict] = None

        # --- legacy long-task framing (kept for back-compat) --------------
        # New code should prefer ``goal`` / ``subgoal``.
        self.intentions = intentions
        self.tasks = tasks
        self.sub_tasks = sub_tasks

        # --- new skill-execution framing ----------------------------------
        # If new fields were not supplied, mirror the legacy ones so callers
        # that read ``goal`` / ``subgoal`` see something useful.
        self.goal: Optional[str] = goal if goal is not None else (
            tasks if isinstance(tasks, str) else None
        )
        self.subgoal: Optional[str] = subgoal if subgoal is not None else (
            sub_tasks if isinstance(sub_tasks, str) else None
        )

        # --- summarization (unchanged) ------------------------------------
        self.summary = None
        self.summary_state = None

        # Per-step reward breakdown (r_env, r_follow, r_cost, r_total).
        self.reward_details: Optional[dict] = None

        # --- typed-step vocabulary ----------------------------------------
        # ``step_type`` is the canonical typed inner-loop kind.
        # ``action_type`` is a legacy alias kept for back-compat; setting it
        # to ``QUERY_MEM`` raises a deprecation warning (no-memory contract).
        self._step_type: Optional[str] = None
        self.step_type = step_type  # uses property setter for validation
        self.action_type: Optional[str] = None  # legacy mirror; set below
        if step_type is not None:
            # Mirror the typed step into the legacy slot for downstream code
            # that still reads ``action_type``.
            self.action_type = step_type

        # --- domain-general runtime context -------------------------------
        self.domain_type: Optional[str] = domain_type
        self.task_type: Optional[str] = task_type
        self.active_skill: Optional[str] = active_skill
        self.skill_phase: Optional[str] = skill_phase
        self.trace_parent_id: Optional[str] = trace_parent_id

        # --- evidence ledger ----------------------------------------------
        self.evidence_refs: List[str] = list(evidence_refs) if evidence_refs else []
        self.evidence_items: List[dict] = list(evidence_items) if evidence_items else []
        self.supports_claims: List[str] = list(supports_claims) if supports_claims else []
        self.claim_status: Optional[str] = claim_status
        self.evidence_confidence: Optional[float] = evidence_confidence

    # ------------------------------------------------------------------
    # Typed-step property: validates and rejects forbidden legacy values.
    # ------------------------------------------------------------------
    @property
    def step_type(self) -> Optional[str]:
        return self._step_type

    @step_type.setter
    def step_type(self, value: Optional[str]) -> None:
        if value is None:
            self._step_type = None
            return
        if value in _FORBIDDEN_ACTION_TYPES:
            warnings.warn(
                f"step_type={value!r} is forbidden under the no-memory contract; "
                "use one of GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE | PRIMITIVE.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._step_type = None
            return
        if value not in STEP_TYPES:
            warnings.warn(
                f"Unknown step_type={value!r}; expected one of {STEP_TYPES}.",
                stacklevel=2,
            )
        self._step_type = value
        # Keep the legacy mirror in sync for downstream consumers.
        self.action_type = value

    # ------------------------------------------------------------------
    # Evidence-driven invariant.  The Harness G0 gate enforces this
    # globally; here we only expose the predicate.
    # ------------------------------------------------------------------
    def validate_evidence_contract(self) -> bool:
        """Return True iff this step satisfies the evidence-driven invariant.

        Steps with ``step_type ∈ {GROUND, CHECK, COMMIT}`` must have a
        non-empty ``evidence_refs`` *or* ``evidence_items`` set.  Other step
        kinds (``RETRIEVE``, ``EXECUTE``, ``PRIMITIVE``) are not required to
        carry evidence at this layer.
        """
        if self._step_type not in _EVIDENCE_REQUIRED_STEPS:
            return True
        return bool(self.evidence_refs) or bool(self.evidence_items)

    # ------------------------------------------------------------------
    # Summarization helpers — domain-aware where it matters.
    # ------------------------------------------------------------------
    def generate_summary(self):
        """Generate ``summary_state | note=<short note>`` for this experience.

        Branches on ``domain_type`` so the note is appropriate for game / web /
        os / video / visual_reasoning.  Falls back to the generic strategic
        note when ``domain_type`` is unset.
        """
        ss = self.summary_state
        if not ss:
            ss = self.generate_summary_state()

        state_text = (self.state or "")[:800]
        focus = _DOMAIN_SUMMARY_FOCUS.get(self.domain_type or "", _DEFAULT_SUMMARY_FOCUS)
        prompt = (
            f"Compress this {self.domain_type or 'agent'} step into a short note (max 12 words). "
            f"Focus on: {focus}.\n"
            f"Facts: {ss}\n"
            f"Action: {self.action}\n"
            f"State: {state_text}\n"
            "Note:"
        )
        raw = ask_model(prompt)
        if raw and not raw.startswith("Error"):
            note = raw.split("\n")[0].strip().strip('"').strip("'")[:80]
            self.summary = f"{ss} | note={note}" if ss else note
        else:
            self.summary = ss or ""
        return self.summary

    def generate_summary_state(self):
        """Generate a compact ``key=value`` state summary.

        Tries ``build_rag_summary`` first (deterministic, 0 LLM tokens), then
        falls back to an LLM call.  The prompt asks the model to expose
        candidate sets, constraints, uncertainty, and evidence anchors when
        present, so downstream consumers see a shared schema.
        """
        try:
            from decision_agents.agent_helper import build_rag_summary
            ss = build_rag_summary(self.state or "")
            if ss:
                self.summary_state = ss
                return self.summary_state
        except ImportError:
            pass

        state_text = (self.state or "")[:1500]
        prompt = (
            "Compress this state into a compact key=value summary. "
            "Max 240 characters. No prose. "
            "Format: key=value | key=value | ...\n"
            "When present, expose: candidates=, constraints=, uncertainty=, "
            "evidence_anchors=, entities=.\n\n"
            f"{state_text}"
        )
        raw = ask_model(prompt)
        if raw and not raw.startswith("Error"):
            self.summary_state = raw.split("\n")[0].strip()[:300]
        else:
            self.summary_state = ""
        return self.summary_state

    def generate_intentions(self, history: Optional[List["Experience"]] = None):
        """Generate a ``[TAG] subgoal phrase`` intention for this experience.

        Uses ``infer_intention`` when available, else asks an LLM with the
        domain-general reasoning/action tag set.
        """
        try:
            from decision_agents.agent_helper import infer_intention
            obs = self.summary_state or self.state or ""
            intent = infer_intention(obs)
            if intent:
                self.intentions = intent
                return self.intentions
        except ImportError:
            pass

        history = history or []
        prev_intents = [exp.intentions for exp in history[-3:] if exp.intentions]
        prev_line = f"Previous: {prev_intents[-1]}\n" if prev_intents else ""
        state_text = (self.state or "")[:800]
        prompt = (
            "What is the agent's immediate [TAG] subgoal? (max 12 words)\n"
            "Tags: LOCATE|VERIFY|DISAMBIGUATE|SELECT|PLAN|CHECK|GROUND|EXECUTE\n"
            f"State: {state_text}\n"
            f"Action: {self.action}\n"
            f"{prev_line}"
            "Examples:\n"
            "  [LOCATE] Find the suspect's hand in frame 42\n"
            "  [VERIFY] Confirm timestamp matches the alibi window\n"
            "Subgoal:"
        )
        raw = ask_model(prompt)
        if raw and not raw.startswith("Error"):
            self.intentions = raw.split("\n")[0].strip()[:150]
        else:
            self.intentions = f"[EXECUTE] {self.action}"
        return self.intentions

    def initialize_intentions_and_summary(self, history: Optional[List["Experience"]] = None):
        """Generate intentions and summary if not already provided."""
        if self.intentions is None:
            self.generate_intentions(history)
        if self.summary is None:
            self.generate_summary()
        return self.intentions, self.summary

    # ------------------------------------------------------------------
    # Serialization.  ``to_dict`` only emits new fields when they carry
    # information, keeping JSON output compact and back-compatible.
    # ``from_dict`` reads everything via ``.get`` so old payloads load fine.
    # ------------------------------------------------------------------
    def to_dict(self):
        d = {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "intentions": self.intentions,
            "tasks": self.tasks,
            "sub_tasks": self.sub_tasks,
            "summary": self.summary,
            "summary_state": self.summary_state,
            "idx": self.idx,
        }
        if self.raw_state is not None:
            d["raw_state"] = self.raw_state
        if self.raw_next_state is not None:
            d["raw_next_state"] = self.raw_next_state
        if self.available_actions is not None:
            d["available_actions"] = self.available_actions
        if self.interface is not None:
            d["interface"] = self.interface
        if self.reward_details is not None:
            d["reward_details"] = self.reward_details
        if self.action_type is not None:
            d["action_type"] = self.action_type

        # New fields (additive).
        for key, value in (
            ("step_type", self._step_type),
            ("domain_type", self.domain_type),
            ("task_type", self.task_type),
            ("active_skill", self.active_skill),
            ("skill_phase", self.skill_phase),
            ("trace_parent_id", self.trace_parent_id),
            ("goal", self.goal),
            ("subgoal", self.subgoal),
            ("claim_status", self.claim_status),
            ("evidence_confidence", self.evidence_confidence),
        ):
            if value is not None:
                d[key] = value
        if self.evidence_refs:
            d["evidence_refs"] = list(self.evidence_refs)
        if self.evidence_items:
            d["evidence_items"] = list(self.evidence_items)
        if self.supports_claims:
            d["supports_claims"] = list(self.supports_claims)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        """Construct an Experience from a dictionary (back-compat aware)."""
        exp = cls(
            state=d["state"],
            action=d["action"],
            reward=d["reward"],
            next_state=d["next_state"],
            done=d["done"],
            intentions=d.get("intentions"),
            tasks=d.get("tasks"),
            sub_tasks=d.get("sub_tasks"),
            domain_type=d.get("domain_type"),
            task_type=d.get("task_type"),
            step_type=d.get("step_type"),
            active_skill=d.get("active_skill"),
            skill_phase=d.get("skill_phase"),
            trace_parent_id=d.get("trace_parent_id"),
            evidence_refs=d.get("evidence_refs"),
            evidence_items=d.get("evidence_items"),
            supports_claims=d.get("supports_claims"),
            claim_status=d.get("claim_status"),
            evidence_confidence=d.get("evidence_confidence"),
            goal=d.get("goal"),
            subgoal=d.get("subgoal"),
        )
        exp.summary = d.get("summary")
        exp.summary_state = d.get("summary_state")
        exp.idx = d.get("idx")
        exp.raw_state = d.get("raw_state")
        exp.raw_next_state = d.get("raw_next_state")
        exp.available_actions = d.get("available_actions")
        exp.interface = d.get("interface")
        exp.reward_details = d.get("reward_details")

        # Back-compat: legacy ``action_type`` populates ``step_type`` when the
        # new field is absent and the legacy value is in our vocabulary.
        legacy_action_type = d.get("action_type")
        if legacy_action_type and exp.step_type is None:
            if legacy_action_type in STEP_TYPES:
                exp.step_type = legacy_action_type
            else:
                # Preserve the raw legacy value without polluting step_type.
                exp.action_type = legacy_action_type
        return exp


# Domain-aware focus phrases used by ``Experience.generate_summary``.
_DOMAIN_SUMMARY_FOCUS: Dict[str, str] = {
    "game": "the key threat or opportunity in the local board state",
    "web": "the UI interaction, blocker, or next control to act on",
    "os": "the active window / object and what interaction is pending",
    "video": "the temporal event and the evidence frame/moment it bears",
    "visual_reasoning": "the salient object/region clue and what it implies",
}
_DEFAULT_SUMMARY_FOCUS = "the most decision-relevant fact"


# ---------------------------------------------------------------------------
# Episode — full rollout container.
# ---------------------------------------------------------------------------
class Episode:
    """A full rollout / trajectory container.

    Carries the typed trace (ordered ``Experience``s), the answer + answer
    support chain, aggregated reward / outcome, and per-rollout metadata.
    """

    def __init__(
        self,
        experiences: List[Experience],
        task: str,
        metadata: Optional[dict] = None,
        episode_id: Optional[str] = None,
        env_name: Optional[str] = None,
        game_name: Optional[str] = None,
        # ----- new domain-level metadata ----------------------------------
        domain_type: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        input_modality: Optional[str] = None,
        output_modality: Optional[str] = None,
        episode_status: str = "running",
        # ----- new evidence-grounded answer bookkeeping -------------------
        final_answer: Optional[Any] = None,
        root_claims: Optional[List[str]] = None,
        answer_support_chain: Optional[List[Tuple[int, List[str], str]]] = None,
        episode_claim_graph: Optional[Dict[str, dict]] = None,
    ):
        self.experiences = experiences

        # Unique identifier for this episode (auto-generated if not provided).
        self.episode_id: str = episode_id or str(uuid.uuid4())

        # Legacy domain identifiers (kept for back-compat).
        self.env_name: Optional[str] = env_name
        self.game_name: Optional[str] = game_name

        # Task and outcome bookkeeping.
        self.task = task
        self.summary = None
        self.outcome = None

        # Arbitrary metadata (cumulative_reward, agent_state snapshot, etc.).
        self.metadata: Optional[dict] = metadata

        # --- domain-level metadata ----------------------------------------
        self.domain_type: Optional[str] = domain_type
        self.benchmark_name: Optional[str] = benchmark_name
        self.input_modality: Optional[str] = input_modality
        self.output_modality: Optional[str] = output_modality
        self.episode_status: str = episode_status

        # --- evidence-grounded answer chain -------------------------------
        self.final_answer: Optional[Any] = final_answer
        self.root_claims: List[str] = list(root_claims) if root_claims else []
        # Each entry: (experience_idx, evidence_refs_at_that_step, claim_id).
        self.answer_support_chain: List[Tuple[int, List[str], str]] = (
            list(answer_support_chain) if answer_support_chain else []
        )
        # claim_id -> {status, supported_by_refs, contradicted_by_refs, parent_claims}
        self.episode_claim_graph: Dict[str, dict] = (
            dict(episode_claim_graph) if episode_claim_graph else {}
        )

    # ------------------------------------------------------------------
    # Aggregation helpers.
    # ------------------------------------------------------------------
    def get_reward(self):
        """Sum of raw environment rewards (r_env) across all experiences."""
        return sum(experience.reward for experience in self.experiences)

    def get_total_reward(self):
        """Sum of shaped total rewards (r_total) from reward_details.

        Falls back to r_env (self.reward) when reward_details is absent.
        """
        total = 0.0
        for exp in self.experiences:
            if exp.reward_details and "r_total" in exp.reward_details:
                total += exp.reward_details["r_total"]
            else:
                total += exp.reward
        return total

    def get_length(self):
        return len(self.experiences)

    def set_outcome(self):
        # Set the outcome of the episode.
        self.outcome = self.experiences[-1].done
        return self.outcome

    def generate_summary(self):
        # Generate the summary of the episode.
        prompt = (
            f"Generate the summary of the episode. "
            f"The episode is {self.experiences}, the task is {self.task}, the outcome is {self.outcome}. "
        )
        self.summary = ask_model(prompt)
        return self.summary

    # ------------------------------------------------------------------
    # Metadata helper — validates the well-known keys without locking out
    # free-form ones.
    # ------------------------------------------------------------------
    _WELL_KNOWN_METADATA_KEYS: Tuple[str, ...] = (
        "model_version",
        "rollout_source",
        "adapter_ids",
        "transfer_mode",
        "budget_stats",
        "replay_split",
        "partition",
    )

    def set_metadata(self, **kwargs) -> None:
        """Merge keyword arguments into ``metadata``; warn on typos in the
        well-known key set, but allow free-form keys."""
        if self.metadata is None:
            self.metadata = {}
        for key, value in kwargs.items():
            if key not in self._WELL_KNOWN_METADATA_KEYS:
                # Free-form keys are allowed — only warn when the key looks
                # like a near-miss of a well-known one.
                lowered = key.lower()
                for wk in self._WELL_KNOWN_METADATA_KEYS:
                    if lowered != wk and lowered in wk:
                        warnings.warn(
                            f"metadata key {key!r} resembles {wk!r}; "
                            "consider using the well-known name.",
                            stacklevel=2,
                        )
                        break
            self.metadata[key] = value

    # ------------------------------------------------------------------
    # Answer-chain validation.
    # ------------------------------------------------------------------
    def validate_answer_chain(self) -> bool:
        """Return True iff every root claim resolves to ``verified`` in the
        episode claim graph and has a non-empty support chain."""
        if not self.root_claims:
            return False
        if not self.answer_support_chain:
            return False
        for claim_id in self.root_claims:
            node = self.episode_claim_graph.get(claim_id)
            if not node:
                return False
            if node.get("status") != "verified":
                return False
            if not node.get("supported_by_refs"):
                return False
        return True

    # ------------------------------------------------------------------
    # Two-stage segmentation.
    # ------------------------------------------------------------------
    def segment_boundaries(
        self,
        policy: str = "subgoal_change",
    ) -> List[Tuple[int, int, str]]:
        """Stage 1 — produce ``(start_idx, end_idx, segment_label)`` tuples.

        Supported policies:

        * ``"subgoal_change"`` — boundary whenever ``subgoal`` (or legacy
          ``sub_tasks``) changes.  Default; reproduces the previous
          ``separate_into_sub_episodes`` behavior on episodes that only used
          ``sub_tasks``.
        * ``"active_skill"`` — boundary on ``active_skill`` change; one
          segment per skill invocation.
        * ``"commit"`` — boundary at every ``step_type == "COMMIT"`` step.
        * ``"claim_resolution"`` — boundary at every transition from
          ``candidate`` to ``verified | contradicted``.
        """
        n = len(self.experiences)
        if n == 0:
            return []

        def _label_of(exp: Experience) -> Optional[str]:
            if policy == "subgoal_change":
                return exp.subgoal if exp.subgoal is not None else exp.sub_tasks
            if policy == "active_skill":
                return exp.active_skill
            if policy == "commit":
                return exp.step_type if exp.step_type == "COMMIT" else None
            if policy == "claim_resolution":
                return exp.claim_status
            raise ValueError(f"Unknown segmentation policy: {policy!r}")

        boundaries: List[Tuple[int, int, str]] = []

        if policy in ("subgoal_change", "active_skill"):
            start = 0
            current_label = _label_of(self.experiences[0])
            for i in range(1, n):
                lbl = _label_of(self.experiences[i])
                if lbl != current_label:
                    boundaries.append(
                        (start, i, str(current_label) if current_label is not None else "<none>")
                    )
                    start = i
                    current_label = lbl
            boundaries.append(
                (start, n, str(current_label) if current_label is not None else "<none>")
            )

        elif policy == "commit":
            # Each segment ends at (and includes) a COMMIT step.
            start = 0
            for i, exp in enumerate(self.experiences):
                if exp.step_type == "COMMIT":
                    boundaries.append((start, i + 1, f"commit@{i}"))
                    start = i + 1
            if start < n:
                boundaries.append((start, n, "tail"))

        elif policy == "claim_resolution":
            # Segment ends when claim_status transitions out of ``candidate``.
            start = 0
            for i, exp in enumerate(self.experiences):
                if exp.claim_status in ("verified", "contradicted"):
                    boundaries.append(
                        (start, i + 1, f"{exp.claim_status}@{i}")
                    )
                    start = i + 1
            if start < n:
                boundaries.append((start, n, "open"))

        else:
            raise ValueError(f"Unknown segmentation policy: {policy!r}")

        return boundaries

    def separate_into_sub_episodes(
        self,
        outcome_length: int = 5,
        policy: str = "subgoal_change",
    ) -> List["SubTask_Experience"]:
        """Two-stage segmentation: (1) compute boundaries, (2) build
        ``SubTask_Experience`` objects with segment-level evidence stats."""
        boundaries = self.segment_boundaries(policy=policy)
        n = len(self.experiences)
        sub_episodes: List[SubTask_Experience] = []

        for start_idx, end_idx, label in boundaries:
            segment_exps = self.experiences[start_idx:end_idx]
            outcome_end = min(end_idx + outcome_length, n)
            outcome_exps = (
                self.experiences[end_idx:outcome_end]
                if outcome_end > end_idx
                else None
            )

            ste = SubTask_Experience(
                sub_task=label,
                final_goal=self.task,
                experiences=segment_exps,
                outcome=outcome_exps,
                seg_id=None,
                episode_id=self.episode_id,
                rollout_source=(self.metadata or {}).get("rollout_source", ""),
                source_domain=self.domain_type,
                segment_label=label,
            )
            ste.recompute_evidence_stats()
            sub_episodes.append(ste)

        return sub_episodes

    # ------------------------------------------------------------------
    # Serialization.
    # ------------------------------------------------------------------
    def to_dict(self):
        d = {
            "episode_id": self.episode_id,
            "env_name": self.env_name,
            "game_name": self.game_name,
            "experiences": [exp.to_dict() for exp in self.experiences],
            "task": self.task,
            "outcome": self.outcome,
            "summary": self.summary,
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        for key, value in (
            ("domain_type", self.domain_type),
            ("benchmark_name", self.benchmark_name),
            ("input_modality", self.input_modality),
            ("output_modality", self.output_modality),
            ("episode_status", self.episode_status),
            ("final_answer", self.final_answer),
        ):
            if value is not None and value != "running":
                d[key] = value
            elif key == "episode_status":
                # Always emit episode_status — it has meaningful default.
                d[key] = value
        if self.root_claims:
            d["root_claims"] = list(self.root_claims)
        if self.answer_support_chain:
            d["answer_support_chain"] = list(self.answer_support_chain)
        if self.episode_claim_graph:
            d["episode_claim_graph"] = dict(self.episode_claim_graph)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        """Construct an Episode from a dictionary (back-compat aware)."""
        experiences = [Experience.from_dict(exp) for exp in d["experiences"]]
        ep = cls(
            experiences=experiences,
            task=d["task"],
            metadata=d.get("metadata"),
            episode_id=d.get("episode_id"),
            env_name=d.get("env_name"),
            game_name=d.get("game_name"),
            domain_type=d.get("domain_type"),
            benchmark_name=d.get("benchmark_name"),
            input_modality=d.get("input_modality"),
            output_modality=d.get("output_modality"),
            episode_status=d.get("episode_status", "running"),
            final_answer=d.get("final_answer"),
            root_claims=d.get("root_claims"),
            answer_support_chain=d.get("answer_support_chain"),
            episode_claim_graph=d.get("episode_claim_graph"),
        )
        ep.outcome = d.get("outcome")
        ep.summary = d.get("summary")
        return ep


# ---------------------------------------------------------------------------
# SubTask_Experience — segmented local trajectory; skill-candidate unit.
# ---------------------------------------------------------------------------
# Data-processing container for a sub-task's experiences.  This class holds
# the actual ``Experience`` objects during data labeling, summary generation,
# and quality assessment.  It is **not** for persistent storage in the skill
# bank — the skill bank stores only lightweight ``SubEpisodeRef`` pointers
# (see ``to_sub_episode_ref``).  Call that method after processing to produce
# the pointer that goes into ``Skill.sub_episodes``.
class SubTask_Experience:
    """A segmented local trajectory.

    Beyond holding the segment's experiences, this object carries the
    **skill-candidate metadata** (source / target domain, adapter, transfer
    status, failure mode) and the **segment-level evidence ledger**
    (evidence refs, supported claims, sufficiency).  It is the unit consumed
    by skill extraction, transfer validation, and failure-aware refinement.
    """

    def __init__(
        self,
        sub_task: str,
        final_goal: str,
        experiences: List[Experience],
        outcome: Optional[List[Experience]] = None,
        seg_id: Optional[str] = None,
        episode_id: str = "",
        rollout_source: str = "",
        # ----- skill-candidate metadata ----------------------------------
        source_domain: Optional[str] = None,
        candidate_target_domains: Optional[List[str]] = None,
        verified_domains: Optional[List[str]] = None,
        adapter_id: Optional[str] = None,
        transfer_status: str = "none",
        verification_status: str = "unverified",
        skill_candidate_type: Optional[str] = None,
        failure_mode: Optional[str] = None,
        # ----- segment-level evidence ------------------------------------
        segment_label: Optional[str] = None,
    ):
        # What this strategy or tool is used for.
        self.sub_task = sub_task
        self.final_goal = final_goal

        # Contents of the sub-task experience (for processing only, not
        # persisted in the skill bank).
        self.sub_task_experience = experiences

        # Outcome lookahead (a few steps after the segment ends).
        self.outcome_experiences = outcome

        # Link to the corresponding SegmentRecord from the skill pipeline.
        self.seg_id: Optional[str] = seg_id

        # Pointer fields — identify where the rollout data is stored.
        self.episode_id: str = episode_id
        self.rollout_source: str = rollout_source

        # Stable label assigned at segmentation time (from Stage 1 boundary).
        self.segment_label: Optional[str] = segment_label or sub_task

        # The summary for query of this strategy or tool.
        self.summary = None
        self.outcome_summary = None
        # Set by ``sub_task_labeling`` when run; declared up-front so
        # ``initialize_sub_task_experience`` can reference it safely.
        self.sub_task_label: Optional[str] = None

        # The length of the sub-task experience.
        self.length = len(experiences)

        # The cumulative reward of the sub-task experience.
        self.cumulative_reward = sum(exp.reward for exp in experiences)

        # Quality assessment fields (populated by the skill agent quality pipeline).
        self.quality_score: float = 0.0
        self.outcome_classification: str = "partial"  # "success" | "partial" | "failure"

        # ----- skill-candidate metadata -----------------------------------
        self.source_domain: Optional[str] = source_domain
        self.candidate_target_domains: List[str] = (
            list(candidate_target_domains) if candidate_target_domains else []
        )
        self.verified_domains: List[str] = (
            list(verified_domains) if verified_domains else []
        )
        self.adapter_id: Optional[str] = adapter_id
        self.transfer_status: str = transfer_status
        self.verification_status: str = verification_status
        self.skill_candidate_type: Optional[str] = skill_candidate_type
        self.failure_mode: Optional[str] = failure_mode

        # ----- segment-level evidence ledger ------------------------------
        self.segment_evidence_refs: List[str] = []
        self.segment_claims: List[str] = []
        self.segment_contract_progress: Dict[str, Any] = {}
        self.evidence_sufficiency: str = "insufficient"  # | "partial" | "sufficient"

    # ------------------------------------------------------------------
    # Recompute segment-level evidence stats from member experiences.
    # ------------------------------------------------------------------
    def recompute_evidence_stats(self) -> None:
        """Roll up ``evidence_refs``, claims, and a coarse sufficiency tag
        from the member ``Experience`` objects."""
        refs: List[str] = []
        claims: List[str] = []
        verified_required = 0
        verified_satisfied = 0

        for exp in self.sub_task_experience:
            refs.extend(exp.evidence_refs or [])
            claims.extend(exp.supports_claims or [])
            if exp.step_type in _EVIDENCE_REQUIRED_STEPS:
                verified_required += 1
                if exp.validate_evidence_contract():
                    verified_satisfied += 1

        # Deduplicate while preserving order.
        seen = set()
        self.segment_evidence_refs = [
            r for r in refs if not (r in seen or seen.add(r))
        ]
        seen = set()
        self.segment_claims = [
            c for c in claims if not (c in seen or seen.add(c))
        ]

        # Track active-skill phase progress as a simple count map.
        progress: Dict[str, int] = {}
        for exp in self.sub_task_experience:
            if exp.skill_phase:
                progress[exp.skill_phase] = progress.get(exp.skill_phase, 0) + 1
        self.segment_contract_progress = progress

        if verified_required == 0:
            # No evidence-required steps in the segment — neutral.
            self.evidence_sufficiency = "partial" if self.segment_evidence_refs else "insufficient"
        elif verified_satisfied == verified_required:
            self.evidence_sufficiency = "sufficient"
        elif verified_satisfied > 0:
            self.evidence_sufficiency = "partial"
        else:
            self.evidence_sufficiency = "insufficient"

    # ------------------------------------------------------------------
    # Summarization helpers.
    # ------------------------------------------------------------------
    def generate_summary(self):
        prev_summary_list = [exp.summary for exp in self.sub_task_experience]
        if self.sub_task is None:
            prompt = (
                f"Summarize the agent's strategy and motivation to achieve the final goal '{self.final_goal}'. "
                f"Experience history (time-ordered): {prev_summary_list}"
            )
        else:
            prompt = (
                f"Summarize why the agent chose sub-task '{self.sub_task}' to achieve the final goal '{self.final_goal}'. "
                f"Include the motivation and strategy. "
                f"Experience history (time-ordered): {prev_summary_list}"
            )
        self.summary = ask_model(prompt)
        return self.summary

    def generate_outcome_summary(self):
        if self.outcome_experiences is None:
            self.outcome_summary = ""
            return self.outcome_summary
        prev_outcome_summary_list = [
            getattr(exp, "outcome_summary", None) or exp.summary
            for exp in self.outcome_experiences
        ]
        if self.sub_task is None:
            prompt = (
                f"Evaluate whether the agent's actions contributed to the final goal '{self.final_goal}'. "
                f"Consider if they induced rewarding actions in subsequent steps. "
                f"Subsequent outcomes (time-ordered): {prev_outcome_summary_list}"
            )
        else:
            prompt = (
                f"Evaluate whether completing the sub-task '{self.sub_task}' contributed to the final goal '{self.final_goal}'. "
                f"Consider if it induced rewarding actions in the subsequent steps. "
                f"Subsequent outcomes (time-ordered): {prev_outcome_summary_list}"
            )
        self.outcome_summary = ask_model(prompt)
        return self.outcome_summary

    def sub_task_labeling(self):
        prev_summary_list = [exp.summary for exp in self.sub_task_experience]
        if self.outcome_experiences is not None:
            prev_outcome_summary_list = [
                getattr(exp, "outcome_summary", None) or exp.summary
                for exp in self.outcome_experiences
            ]
            prompt = (
                f"Create a concise one-sentence label for this sub-task that describes the strategy used. "
                f"Sub-task summary: {prev_summary_list}. "
                f"Outcome summary: {prev_outcome_summary_list}"
            )
        else:
            prompt = (
                f"Create a concise one-sentence label for this sub-task that describes the strategy used. "
                f"Sub-task summary: {prev_summary_list}. "
            )
        self.sub_task_label = ask_model(prompt)
        return self.sub_task_label

    def initialize_sub_task_experience(self):
        if self.summary is None:
            self.generate_summary()
        if self.outcome_summary is None:
            self.generate_outcome_summary()
        if self.sub_task_label is None:
            self.sub_task_labeling()
        return self.summary, self.outcome_summary, self.sub_task_label

    def _extract_intention_tags(self) -> List[str]:
        """Extract intention tags from each Experience in this sub-task."""
        tags = []
        for exp in self.sub_task_experience:
            intentions = getattr(exp, "intentions", None)
            if intentions and isinstance(intentions, str):
                tag = intentions.strip().split("]")[0].replace("[", "").strip()
                if tag:
                    tags.append(tag)
        return tags

    # ------------------------------------------------------------------
    # Skill-bank pointer.  Additive payload — older consumers ignore the
    # new keys; newer ones consume transfer / evidence metadata directly.
    # ------------------------------------------------------------------
    def to_sub_episode_ref(self):
        """Produce a lightweight SubEpisodeRef pointer for skill bank storage.

        The actual Experience data stays in this object (or the rollout file);
        only the pointer + cached summary goes into ``Skill.sub_episodes``.
        """
        from skill_agents.stage3_mvp.schemas import SubEpisodeRef

        ref = SubEpisodeRef(
            episode_id=self.episode_id,
            seg_start=0,
            seg_end=max(0, self.length - 1),
            rollout_source=self.rollout_source,
            summary=self.summary or "",
            intention_tags=self._extract_intention_tags(),
            outcome=self.outcome_classification,
            cumulative_reward=self.cumulative_reward,
            quality_score=self.quality_score,
        )

        # Attach optional transfer / evidence metadata as attributes when the
        # SubEpisodeRef schema doesn't accept them as constructor kwargs.
        for attr, value in (
            ("source_domain", self.source_domain),
            ("verified_domains", self.verified_domains),
            ("transfer_status", self.transfer_status),
            ("evidence_sufficiency", self.evidence_sufficiency),
            ("failure_mode", self.failure_mode),
        ):
            try:
                setattr(ref, attr, value)
            except Exception:
                # SubEpisodeRef may be a frozen dataclass; ignore silently.
                pass
        return ref

    # ------------------------------------------------------------------
    # Serialization.
    # ------------------------------------------------------------------
    def to_dict(self):
        d = {
            "sub_task": self.sub_task,
            "final_goal": self.final_goal,
            "sub_task_experience": [exp.to_dict() for exp in self.sub_task_experience],
            "seg_id": self.seg_id,
            "episode_id": self.episode_id,
            "rollout_source": self.rollout_source,
            "quality_score": self.quality_score,
            "outcome_classification": self.outcome_classification,
            "segment_label": self.segment_label,
            "transfer_status": self.transfer_status,
            "verification_status": self.verification_status,
            "evidence_sufficiency": self.evidence_sufficiency,
        }
        if self.outcome_experiences is not None:
            d["outcome_experiences"] = [exp.to_dict() for exp in self.outcome_experiences]
        else:
            d["outcome_experiences"] = None
        # Optional skill-candidate fields (omit when empty/default).
        for key, value in (
            ("source_domain", self.source_domain),
            ("adapter_id", self.adapter_id),
            ("skill_candidate_type", self.skill_candidate_type),
            ("failure_mode", self.failure_mode),
        ):
            if value is not None:
                d[key] = value
        if self.candidate_target_domains:
            d["candidate_target_domains"] = list(self.candidate_target_domains)
        if self.verified_domains:
            d["verified_domains"] = list(self.verified_domains)
        if self.segment_evidence_refs:
            d["segment_evidence_refs"] = list(self.segment_evidence_refs)
        if self.segment_claims:
            d["segment_claims"] = list(self.segment_claims)
        if self.segment_contract_progress:
            d["segment_contract_progress"] = dict(self.segment_contract_progress)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubTask_Experience":
        """Construct a SubTask_Experience from a dictionary."""
        sub_task_exps = [Experience.from_dict(exp) for exp in d["sub_task_experience"]]
        outcome_exps = None
        if d.get("outcome_experiences"):
            outcome_exps = [Experience.from_dict(exp) for exp in d["outcome_experiences"]]
        ste = cls(
            sub_task=d["sub_task"],
            final_goal=d["final_goal"],
            experiences=sub_task_exps,
            outcome=outcome_exps,
            seg_id=d.get("seg_id"),
            episode_id=d.get("episode_id", ""),
            rollout_source=d.get("rollout_source", ""),
            source_domain=d.get("source_domain"),
            candidate_target_domains=d.get("candidate_target_domains"),
            verified_domains=d.get("verified_domains"),
            adapter_id=d.get("adapter_id"),
            transfer_status=d.get("transfer_status", "none"),
            verification_status=d.get("verification_status", "unverified"),
            skill_candidate_type=d.get("skill_candidate_type"),
            failure_mode=d.get("failure_mode"),
            segment_label=d.get("segment_label"),
        )
        ste.quality_score = d.get("quality_score", 0.0)
        ste.outcome_classification = d.get("outcome_classification", "partial")
        ste.segment_evidence_refs = list(d.get("segment_evidence_refs") or [])
        ste.segment_claims = list(d.get("segment_claims") or [])
        ste.segment_contract_progress = dict(d.get("segment_contract_progress") or {})
        ste.evidence_sufficiency = d.get("evidence_sufficiency", "insufficient")
        return ste


# ---------------------------------------------------------------------------
# Filtering / sampling helpers shared by all buffers.
# ---------------------------------------------------------------------------
def _matches_criteria(item: Any, criteria: Dict[str, Any]) -> bool:
    """Return True iff ``item`` matches every key/value in ``criteria``.

    Supports a couple of structured operators:

    * scalar values:  ``getattr(item, key) == value``
    * tuple ``("ge", x)`` / ``("gt", x)`` / ``("le", x)`` / ``("lt", x)``
    * tuple ``("in", iterable)`` — set membership
    * tuple ``("contains", x)`` — ``x`` is in the attribute (list/set membership)
    """
    for key, value in criteria.items():
        attr = getattr(item, key, None)
        if isinstance(value, tuple) and len(value) == 2:
            op, ref = value
            if op == "ge" and not (attr is not None and attr >= ref):
                return False
            elif op == "gt" and not (attr is not None and attr > ref):
                return False
            elif op == "le" and not (attr is not None and attr <= ref):
                return False
            elif op == "lt" and not (attr is not None and attr < ref):
                return False
            elif op == "in" and attr not in ref:
                return False
            elif op == "contains":
                if attr is None or ref not in attr:
                    return False
        else:
            if attr != value:
                return False
    return True


def _safe_sample(pool: List[Any], batch_size: int) -> List[Any]:
    """Sample without replacement; returns the whole pool when too small."""
    if not pool:
        return []
    if batch_size >= len(pool):
        return list(pool)
    return random.sample(pool, batch_size)


# ---------------------------------------------------------------------------
# Experience Replay Buffer.
# ---------------------------------------------------------------------------
class Experience_Replay_Buffer:
    """FIFO replay container for individual ``Experience`` objects.

    Adds domain-aware ``filter`` / ``query`` and named ``sample_experience``
    modes (``uniform``, ``high_quality``, ``failure_replay``,
    ``transfer_success``, ``transfer_failure``, ``domain_balanced``).
    """

    def __init__(self, buffer_size: int):
        self.buffer: List[Experience] = []
        self.buffer_size = buffer_size

    def add_experience(self, experience):
        """Add one or more experiences (FIFO eviction when full)."""
        if isinstance(experience, list):
            self.buffer.extend(experience)
        elif isinstance(experience, Episode):
            self.buffer.extend(experience.experiences)
        else:
            self.buffer.append(experience)

        if len(self.buffer) > self.buffer_size:
            overflow = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[overflow:]

    def add_experiences(self, experiences: List[Experience]):
        """Add multiple experiences at once (convenience method)."""
        self.add_experience(experiences)

    def get_experience_summary(self, query: str):
        return [
            experience.summary
            for experience in self.buffer
            if experience.summary and query in experience.summary
        ]

    # ------------------------------------------------------------------
    # Filtering.
    # ------------------------------------------------------------------
    def filter(self, **criteria) -> List[Experience]:
        """Return experiences matching every key/value in ``criteria``.

        Common keys: ``domain_type``, ``step_type``, ``active_skill``.
        See ``_matches_criteria`` for supported operators (``("ge", x)`` etc.).
        """
        return [exp for exp in self.buffer if _matches_criteria(exp, criteria)]

    def query(self, **criteria) -> List[Experience]:
        """Alias of ``filter``."""
        return self.filter(**criteria)

    # ------------------------------------------------------------------
    # Sampling.
    # ------------------------------------------------------------------
    def sample_experience(
        self,
        batch_size: int,
        mode: str = "uniform",
        **criteria,
    ) -> List[Experience]:
        """Sample experiences using one of several named modes.

        ``criteria`` are extra filters AND-ed on top of the mode's own filter.
        """
        if mode == "uniform":
            pool = self.filter(**criteria) if criteria else self.buffer
        elif mode == "high_quality":
            threshold = criteria.pop("quality_threshold", 0.5)
            pool = [
                e for e in self.buffer
                if (getattr(e, "reward", 0) or 0) >= threshold
                and _matches_criteria(e, criteria)
            ]
        elif mode == "failure_replay":
            pool = [
                e for e in self.buffer
                if (e.reward_details or {}).get("failure_mode") is not None
                and _matches_criteria(e, criteria)
            ]
        elif mode == "transfer_success":
            pool = [
                e for e in self.buffer
                if (e.reward_details or {}).get("transfer_status") == "verified"
                and _matches_criteria(e, criteria)
            ]
        elif mode == "transfer_failure":
            pool = [
                e for e in self.buffer
                if (e.reward_details or {}).get("transfer_status") == "rejected"
                and _matches_criteria(e, criteria)
            ]
        elif mode == "domain_balanced":
            return self._sample_domain_balanced(batch_size, criteria)
        else:
            raise ValueError(f"Unknown sample mode: {mode!r}")
        return _safe_sample(pool, batch_size)

    def _sample_domain_balanced(
        self,
        batch_size: int,
        criteria: Dict[str, Any],
    ) -> List[Experience]:
        """Stratified sampling across ``domain_type``."""
        by_domain: Dict[str, List[Experience]] = {}
        for exp in self.buffer:
            if not _matches_criteria(exp, criteria):
                continue
            key = exp.domain_type or "<none>"
            by_domain.setdefault(key, []).append(exp)
        if not by_domain:
            return []
        per_domain = max(1, batch_size // len(by_domain))
        out: List[Experience] = []
        for pool in by_domain.values():
            out.extend(_safe_sample(pool, per_domain))
        return out[:batch_size]

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Episode Buffer.
# ---------------------------------------------------------------------------
class Episode_Buffer:
    """FIFO buffer of complete ``Episode`` objects with persistence helpers."""

    def __init__(self, buffer_size: int):
        self.buffer: List[Episode] = []
        self.buffer_size = buffer_size

    def add_episode(self, episode):
        """Add one or more episodes (FIFO eviction when full)."""
        if isinstance(episode, list):
            self.buffer.extend(episode)
        elif isinstance(episode, Episode):
            self.buffer.append(episode)
        else:
            raise TypeError(f"Expected Episode or list of Episodes, got {type(episode)}")

        if len(self.buffer) > self.buffer_size:
            overflow = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[overflow:]

    def add_episodes(self, episodes: List[Episode]):
        """Add multiple episodes at once (convenience method)."""
        self.add_episode(episodes)

    # ------------------------------------------------------------------
    # Filtering / sampling.
    # ------------------------------------------------------------------
    def filter(self, **criteria) -> List[Episode]:
        """Return episodes matching every key/value in ``criteria``.

        Common keys: ``domain_type``, ``benchmark_name``, ``episode_status``,
        ``input_modality``, ``output_modality``.
        """
        return [ep for ep in self.buffer if _matches_criteria(ep, criteria)]

    def query(self, **criteria) -> List[Episode]:
        """Alias of ``filter``."""
        return self.filter(**criteria)

    def sample_episode(
        self,
        batch_size: int,
        mode: str = "uniform",
        **criteria,
    ) -> List[Episode]:
        if mode == "uniform":
            pool = self.filter(**criteria) if criteria else self.buffer
        elif mode == "high_quality":
            threshold = criteria.pop("reward_threshold", 0.0)
            pool = [
                ep for ep in self.buffer
                if ep.get_total_reward() >= threshold
                and _matches_criteria(ep, criteria)
            ]
        elif mode == "failure_replay":
            pool = [
                ep for ep in self.buffer
                if ep.episode_status in ("aborted", "failed_contract")
                and _matches_criteria(ep, criteria)
            ]
        elif mode == "domain_balanced":
            return self._sample_domain_balanced(batch_size, criteria)
        else:
            raise ValueError(f"Unknown sample mode: {mode!r}")
        return _safe_sample(pool, batch_size)

    def _sample_domain_balanced(
        self,
        batch_size: int,
        criteria: Dict[str, Any],
    ) -> List[Episode]:
        by_domain: Dict[str, List[Episode]] = {}
        for ep in self.buffer:
            if not _matches_criteria(ep, criteria):
                continue
            key = ep.domain_type or ep.env_name or "<none>"
            by_domain.setdefault(key, []).append(ep)
        if not by_domain:
            return []
        per_domain = max(1, batch_size // len(by_domain))
        out: List[Episode] = []
        for pool in by_domain.values():
            out.extend(_safe_sample(pool, per_domain))
        return out[:batch_size]

    def get_episode_summary(self, query: str):
        return [
            episode.summary for episode in self.buffer
            if episode.summary and query in episode.summary
        ]

    def __len__(self):
        """Return the number of episodes in the buffer."""
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Serialization / persistence.
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "episodes": [episode.to_dict() for episode in self.buffer],
        }

    def from_dict(self, d: dict):
        self.buffer = [Episode.from_dict(ep) for ep in d["episodes"]]
        return self

    def save_to_json(self, filepath: str):
        """Save the episode buffer to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        buffer_dict = self.to_dict()
        buffer_dict["buffer_size"] = self.buffer_size
        buffer_dict["num_episodes"] = len(self.buffer)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(buffer_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filepath: str, buffer_size: Optional[int] = None):
        """Load an episode buffer from a JSON file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Episode buffer file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            buffer_dict = json.load(f)

        size = buffer_size
        if size is None:
            size = buffer_dict.get("buffer_size", 1000)

        buffer = cls(buffer_size=size)
        episodes_data = buffer_dict.get("episodes", [])
        buffer.buffer = [Episode.from_dict(ep_dict) for ep_dict in episodes_data]
        return buffer


# ---------------------------------------------------------------------------
# Tool Buffer — stores ``SubTask_Experience`` objects that act as skill
# candidates / extracted protocols.
# ---------------------------------------------------------------------------
class Tool_Buffer:
    """FIFO buffer of skill-candidate ``SubTask_Experience`` objects.

    Supports the same filter / query / sample-mode surface as the other
    buffers, oriented around transfer status, verified domains, and failure
    mode (the fields that matter for skill-bank curation).
    """

    def __init__(self, buffer_size: int):
        self.buffer: List[SubTask_Experience] = []
        self.buffer_size = buffer_size

    def add_tool(self, tool):
        """Add one or more tools / skill candidates (FIFO when full)."""
        if isinstance(tool, list):
            self.buffer.extend(tool)
        else:
            self.buffer.append(tool)

        if len(self.buffer) > self.buffer_size:
            overflow = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[overflow:]

    def add_tools(self, tools: List):
        """Add multiple tools at once (convenience method)."""
        self.add_tool(tools)

    def get_tool_summary(self, query: str):
        return [
            tool.summary for tool in self.buffer
            if getattr(tool, "summary", None) and query in tool.summary
        ]

    # ------------------------------------------------------------------
    # Filtering / sampling.
    # ------------------------------------------------------------------
    def filter(self, **criteria) -> List[SubTask_Experience]:
        """Return tools matching every key/value in ``criteria``.

        Common keys: ``source_domain``, ``transfer_status``,
        ``verification_status``, ``failure_mode``, ``outcome_classification``,
        ``evidence_sufficiency``, ``("ge", quality_score)``.
        """
        return [tool for tool in self.buffer if _matches_criteria(tool, criteria)]

    def query(self, **criteria) -> List[SubTask_Experience]:
        """Alias of ``filter``."""
        return self.filter(**criteria)

    def sample_tool(
        self,
        batch_size: int,
        mode: str = "uniform",
        **criteria,
    ) -> List[SubTask_Experience]:
        if mode == "uniform":
            pool = self.filter(**criteria) if criteria else list(self.buffer)
        elif mode == "high_quality":
            threshold = criteria.pop("quality_threshold", 0.5)
            pool = [
                t for t in self.buffer
                if t.quality_score >= threshold
                and _matches_criteria(t, criteria)
            ]
        elif mode == "failure_replay":
            pool = [
                t for t in self.buffer
                if t.failure_mode is not None
                and _matches_criteria(t, criteria)
            ]
        elif mode == "transfer_success":
            pool = [
                t for t in self.buffer
                if t.transfer_status == "verified" and t.verified_domains
                and _matches_criteria(t, criteria)
            ]
        elif mode == "transfer_failure":
            failure_modes = {"adapter_execution_mismatch", "slot_binding_failed"}
            pool = [
                t for t in self.buffer
                if (
                    t.transfer_status == "rejected"
                    or t.failure_mode in failure_modes
                )
                and _matches_criteria(t, criteria)
            ]
        elif mode == "domain_balanced":
            return self._sample_domain_balanced(batch_size, criteria)
        else:
            raise ValueError(f"Unknown sample mode: {mode!r}")
        return _safe_sample(pool, batch_size)

    def _sample_domain_balanced(
        self,
        batch_size: int,
        criteria: Dict[str, Any],
    ) -> List[SubTask_Experience]:
        by_domain: Dict[str, List[SubTask_Experience]] = {}
        for t in self.buffer:
            if not _matches_criteria(t, criteria):
                continue
            key = t.source_domain or "<none>"
            by_domain.setdefault(key, []).append(t)
        if not by_domain:
            return []
        per_domain = max(1, batch_size // len(by_domain))
        out: List[SubTask_Experience] = []
        for pool in by_domain.values():
            out.extend(_safe_sample(pool, per_domain))
        return out[:batch_size]

    def __len__(self):
        """Return the number of tools in the buffer."""
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Serialization.
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "tools": [tool.to_dict() for tool in self.buffer],
        }

    def from_dict(self, d: dict):
        self.buffer = [SubTask_Experience.from_dict(tool) for tool in d["tools"]]
        return self
