"""Unified skill selection + step tracking + GRPO record emission.

This module provides the **single decision pipeline** that ALL domains
(game, QA, web) use for skill selection and protocol step tracking.
The pipeline is domain-agnostic: the caller supplies a state summary
and gets back a skill guidance dict + GRPO records.  Domain-specific
differences (reward density, step-state extraction) are handled by
configuration, not by separate code paths.

Architecture
------------
::

    ┌────────────────────────────────────────────────────┐
    │               SkillDecisionCore                    │
    │                                                    │
    │  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
    │  │ StepTracker  │  │ SkillSelector│  │ RecordBuf │ │
    │  │ (per-domain  │  │ (shared LoRA │  │ (GRPORec  │ │
    │  │  step state) │  │  interface)  │  │  + offline │ │
    │  └─────────────┘  └──────────────┘  │  relabel)  │ │
    │                                      └───────────┘ │
    └────────────────────────────────────────────────────┘
         ▲                    ▲                    │
         │ state              │ candidates         │ records
    ─────┘                    └────────────────────┘
    Game env / QA HopExecutor / BrowserGym

Domain differences (configuration, not code paths):
    Game:  per-step env reward, state from parse_summary_state()
    QA:    episode-end reward, state from executed hop types
    Web:   episode-end reward, state from DOM changes
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain enum
# ---------------------------------------------------------------------------

DOMAIN_GAME = "game"
DOMAIN_QA = "qa"
DOMAIN_WEB = "web"


# ---------------------------------------------------------------------------
# Step tracker (extracted from episode_runner._SkillTracker, unified)
# ---------------------------------------------------------------------------

class StepTracker:
    """Protocol-aware skill lifecycle tracker, shared across all domains.

    Tracks which skill is active, which protocol step we are at, and
    whether it is time to reselect.  Step advancement comes from
    ``receive_step_assessment()`` (parsed from skill_selection LoRA
    output) — this works identically for games, QA hops, and web tasks.
    """

    def __init__(self, domain: str = DOMAIN_GAME):
        self.domain: str = domain
        self.active_skill_id: Optional[str] = None
        self.active_skill_name: str = ""
        self.steps_on_skill: int = 0
        self.reward_on_skill: float = 0.0
        self.max_skill_duration: int = 10
        self.skill_switches: int = 0
        self.hop_history: List[str] = []

        self._protocol: Optional[Dict[str, Any]] = None
        self._protocol_step_idx: int = 0
        self._success_criteria: List[str] = []
        self._abort_criteria: List[str] = []
        self._predicate_success: List[str] = []
        self._predicate_abort: List[str] = []
        self._prev_reward_on_skill: float = 0.0
        self._prev_steps_on_skill: int = 0
        self._just_switched: bool = False
        self._step_checks: List[str] = []
        self._reselect_reason: str = ""
        self._intrinsic_bonus: float = 0.0

    @property
    def protocol_step_idx(self) -> int:
        return self._protocol_step_idx

    @property
    def total_protocol_steps(self) -> int:
        if self._protocol and isinstance(self._protocol, dict):
            return len(self._protocol.get("steps", []))
        return 0

    @property
    def step_progress_ratio(self) -> float:
        total = self.total_protocol_steps
        if total <= 0:
            return 0.0
        return min(1.0, self._protocol_step_idx / total)

    # ── Criteria checking ──────────────────────────────────────────

    def _check_criteria(self, state_text: str, is_abort: bool) -> Optional[str]:
        from decision_agents.protocol_utils import (
            parse_summary_state, check_any_predicate, keyword_match,
        )
        preds = self._predicate_abort if is_abort else self._predicate_success
        texts = self._abort_criteria if is_abort else self._success_criteria
        label = "abort" if is_abort else "success"

        if preds:
            state_dict = parse_summary_state(state_text)
            if check_any_predicate(preds, state_dict):
                return f"{label}:predicate"

        for crit in texts:
            if keyword_match(crit, state_text):
                return f"{label}:{crit[:40]}"

        return None

    # ── Reselection logic ──────────────────────────────────────────

    def should_reselect(
        self,
        guidance: Optional[Dict[str, Any]],
        state_text: str = "",
    ) -> bool:
        """Determine whether the current skill should be reselected.

        Same logic for all domains.  For non-game domains where
        per-step reward is always 0, the ``zero_reward_stall`` trigger
        is effectively disabled (the LoRA learns SWITCH decisions
        instead).
        """
        self._reselect_reason = ""
        if guidance is None or not guidance.get("skill_id"):
            self._reselect_reason = "no_skill"
            return True
        new_id = guidance["skill_id"]
        if new_id != self.active_skill_id:
            return False
        if self.steps_on_skill >= self.max_skill_duration:
            self._reselect_reason = "duration_exceeded"
            return True
        if self.steps_on_skill >= 4 and self.reward_on_skill <= 0:
            self._reselect_reason = "zero_reward_stall"
            return True
        if state_text:
            abort_reason = self._check_criteria(state_text, is_abort=True)
            if abort_reason:
                self._reselect_reason = abort_reason
                return True
            if self.steps_on_skill >= 2:
                success_reason = self._check_criteria(state_text, is_abort=False)
                if success_reason:
                    self._reselect_reason = success_reason
                    return True
        return False

    # ── Step assessment (from LoRA output) ─────────────────────────

    def receive_step_assessment(self, completed: int, total: int):
        """Set protocol step index from skill_selection LoRA output.

        Monotonic constraint: only advances forward.
        """
        if total != self.total_protocol_steps or total <= 0:
            return
        clamped = max(0, min(completed, total - 1))
        self._protocol_step_idx = max(self._protocol_step_idx, clamped)

    # ── Update after each step ─────────────────────────────────────

    def update(
        self,
        skill_id: Optional[str],
        skill_name: str,
        reward: float,
        state_text: str = "",
        hop_type: Optional[str] = None,
    ):
        """Update tracker after one timestep (game action / QA hop / web action).

        ``hop_type`` is set for QA domains (e.g. "GROUND", "CHECK",
        "VERIFY", "COMMIT") and appended to ``hop_history`` for
        deterministic step-state tracking.
        """
        if hop_type:
            self.hop_history.append(hop_type)

        self._intrinsic_bonus = 0.0
        if skill_id != self.active_skill_id:
            self._prev_reward_on_skill = self.reward_on_skill
            self._prev_steps_on_skill = self.steps_on_skill
            self._just_switched = (
                self.active_skill_id is not None
                and self.steps_on_skill > 0
            )
            self.active_skill_id = skill_id
            self.active_skill_name = skill_name
            self.steps_on_skill = 1
            self.reward_on_skill = reward
            self.skill_switches += 1
            self._protocol_step_idx = 0
            self.hop_history = [hop_type] if hop_type else []
        else:
            self._just_switched = False
            self.steps_on_skill += 1
            self.reward_on_skill += reward

            if state_text and self.active_skill_id:
                success = self._check_criteria(state_text, is_abort=False)
                abort = self._check_criteria(state_text, is_abort=True)
                if success:
                    self._intrinsic_bonus += 0.3
                if abort:
                    self._intrinsic_bonus -= 0.1

    # ── Progress summary for prompt ───────────────────────────────

    def get_progress_summary(self, state_text: str = "") -> str:
        if not self._protocol or not isinstance(self._protocol, dict):
            return ""
        steps = self._protocol.get("steps", [])
        if not steps:
            return ""
        from decision_agents.protocol_utils import (
            build_progress_summary, parse_summary_state,
        )
        state_dict = parse_summary_state(state_text)
        return build_progress_summary(
            steps, self._step_checks, self._protocol_step_idx, state_dict,
        )

    # ── Protocol setup ─────────────────────────────────────────────

    def set_protocol(self, protocol: Optional[Dict[str, Any]]):
        self._protocol = protocol
        self._protocol_step_idx = 0
        self._success_criteria = []
        self._abort_criteria = []
        self._predicate_success = []
        self._predicate_abort = []
        self._step_checks = []
        if protocol and isinstance(protocol, dict):
            dur = protocol.get("expected_duration", 0)
            if isinstance(dur, (int, float)) and dur > 0:
                self.max_skill_duration = max(int(dur) + 3, 5)
            else:
                self.max_skill_duration = 10
            self._success_criteria = protocol.get("success_criteria", []) or []
            self._abort_criteria = protocol.get("abort_criteria", []) or []
            self._predicate_success = protocol.get("predicate_success", []) or []
            self._predicate_abort = protocol.get("predicate_abort", []) or []
            self._step_checks = protocol.get("step_checks", []) or []
        else:
            self.max_skill_duration = 10


# ---------------------------------------------------------------------------
# Prompt / parse helpers (domain-agnostic)
# ---------------------------------------------------------------------------

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are an expert strategist. "
    "Given the current state and a set of candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits the current state>\n"
    "STEP: <completed>/<total>\n"
    "DECISION: <CONTINUE|SWITCH>\n"
    "SKILL: <number>\n"
)


def format_candidates_for_selection(candidates: List[Dict[str, Any]]) -> str:
    """Render candidate skills for the skill_selection prompt."""
    lines: List[str] = []
    for i, c in enumerate(candidates, 1):
        name = c.get("skill_name") or c.get("skill_id", f"strategy_{i}")
        hint = c.get("execution_hint", "")
        protocol = c.get("protocol", {})
        steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
        lines.append(f"  {i}. {name}")
        if hint:
            lines.append(f"     Strategy: {hint[:150]}")
        if steps:
            step_text = " -> ".join(steps[:4])
            if len(steps) > 4:
                step_text += " -> ..."
            lines.append(f"     Plan: {step_text}")
        confidence = c.get("confidence")
        if confidence is not None:
            lines.append(f"     Confidence: {confidence:.2f}")
        adapt = c.get("_harness_adaptation_score")
        if isinstance(adapt, (int, float)):
            lines.append(f"     Adaptation: {float(adapt):.2f}")
        deboost = c.get("_harness_deboost")
        if isinstance(deboost, (int, float)) and float(deboost) < 0.95:
            lines.append(f"     Recent veto rate: {1.0 - float(deboost):.2f}")
    return "\n".join(lines)


_DECISION_RE = re.compile(
    r"DECISION\s*:\s*(CONTINUE|SWITCH|SKIP)",
    re.IGNORECASE,
)


def parse_skill_selection(
    reply: str,
    n_candidates: int,
    candidates: Optional[List[Dict[str, Any]]] = None,
    strip_think_tags: Optional[Callable[[str], str]] = None,
) -> Tuple[int, Optional[str], Optional[Tuple[int, int]], str]:
    """Parse skill selection reply.

    Returns ``(chosen_idx, reasoning, step_progress, decision)`` where
    ``step_progress`` is ``(completed, total)`` or ``None``, and
    ``decision`` is one of ``"CONTINUE"`` / ``"SWITCH"`` / ``"SKIP"``.
    """
    if not reply:
        return 0, None, None, "SWITCH"

    cleaned = reply
    if strip_think_tags is not None:
        cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    reasoning = None
    reasoning_m = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nSTEP|\nDECISION|\nSKILL|\Z)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    step_progress: Optional[Tuple[int, int]] = None
    step_m = re.search(r"STEP\s*:\s*(\d+)\s*/\s*(\d+)", cleaned, re.IGNORECASE)
    if step_m:
        completed = int(step_m.group(1))
        total = int(step_m.group(2))
        if 0 <= completed <= total <= 20 and total > 0:
            step_progress = (completed, total)

    decision = "SWITCH"
    decision_m = _DECISION_RE.search(cleaned)
    if decision_m:
        decision = decision_m.group(1).upper()

    skill_m = re.search(r"SKILL\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if skill_m:
        idx = int(skill_m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning, step_progress, decision

    tail = cleaned[-100:]
    nums = re.findall(r"\b(\d+)\b", tail)
    for n_str in reversed(nums):
        idx = int(n_str) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning, step_progress, decision

    if candidates:
        cleaned_lower = cleaned.lower()
        for i, c in enumerate(candidates):
            name = (c.get("skill_name") or "").lower()
            if name and len(name) >= 4 and name in cleaned_lower:
                return i, reasoning, step_progress, decision

    return 0, reasoning, step_progress, decision


# ---------------------------------------------------------------------------
# Skill selection prompt builder (domain-agnostic)
# ---------------------------------------------------------------------------

def build_skill_selection_prompt(
    state_text: str,
    intention: str,
    candidates: List[Dict[str, Any]],
    tracker: StepTracker,
    profile_prefix: str = "",
) -> str:
    """Build the skill_selection prompt, identical for all domains.

    The state_text is domain-specific (game summary / evidence chain /
    DOM state), but the prompt structure is always the same.
    """
    candidates_text = format_candidates_for_selection(candidates)

    proto_ctx = ""
    if tracker._protocol:
        p_steps = tracker._protocol.get("steps", [])
        p_idx = tracker.protocol_step_idx
        if p_steps:
            step_list = " → ".join(s[:50] for s in p_steps)
            proto_ctx = (
                f"Current protocol progress: step {p_idx + 1}/{len(p_steps)}\n"
                f"Protocol: {step_list}\n"
            )

    hop_ctx = ""
    if tracker.hop_history:
        hop_ctx = f"Executed hops: {' → '.join(tracker.hop_history)}\n"

    user_content = (
        f"Current state:\n{state_text[:3500]}\n\n"
        f"Current intention: {intention[:500]}\n"
        f"{proto_ctx}"
        f"{hop_ctx}\n"
        f"Available strategies (pick ONE by number):\n{candidates_text}\n\n"
        f"Assess protocol progress and choose a strategy.\n"
        f"Output format:\n"
        f"REASONING: <why this strategy fits + progress assessment>\n"
        f"STEP: <completed>/<total>\n"
        f"DECISION: <CONTINUE|SWITCH>\n"
        f"SKILL: <number>"
    )
    return profile_prefix + SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user_content


# ---------------------------------------------------------------------------
# GRPO record for skill_selection (domain-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class SkillSelectionRecord:
    """One skill_selection decision point, ready for GRPO training.

    Created at each skill selection call.  The ``reward`` field is
    initially set to a placeholder (clamped env reward for games,
    0.0 for non-game).  After the episode completes, the offline
    reward labeler overwrites it with the trajectory-level signal.
    """

    domain: str
    task: str
    episode_id: str
    step: int
    prompt: str
    completion: str
    reward: float = 0.0
    candidates: List[str] = field(default_factory=list)
    chosen_skill_id: Optional[str] = None
    chosen_idx: int = 0
    decision: str = "SWITCH"
    step_progress: Optional[Tuple[int, int]] = None
    reasoning: Optional[str] = None
    reselect_reason: str = ""
    hop_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Offline reward labeler
# ---------------------------------------------------------------------------

def relabel_skill_rewards(
    records: List[SkillSelectionRecord],
    episode_reward: float,
    episode_success: bool = False,
    gamma: float = 0.95,
) -> List[SkillSelectionRecord]:
    """Relabel skill_selection rewards using full trajectory information.

    For games: per-step env reward is already known, so we use
    discounted cumulative future reward from each decision point.

    For non-game (QA/web): episode_reward is binary (correct/incorrect)
    or task-completion score.  All skill decisions in the episode share
    the outcome, weighted by recency (later decisions get more credit).

    This replaces the noisy inline reward computation and works
    identically across all domains.
    """
    n = len(records)
    if n == 0:
        return records

    for i, rec in enumerate(records):
        steps_remaining = n - i
        recency_weight = gamma ** (n - i - 1)

        base_reward = episode_reward * recency_weight

        if episode_success:
            base_reward += 0.3

        if rec.decision == "SWITCH" and i < n - 1:
            next_rec = records[i + 1]
            if next_rec.chosen_skill_id == rec.chosen_skill_id:
                base_reward -= 0.1

        rec.reward = max(0.0, min(1.0, base_reward))

    return records


__all__ = [
    "DOMAIN_GAME",
    "DOMAIN_QA",
    "DOMAIN_WEB",
    "SKILL_SELECTION_SYSTEM_PROMPT",
    "SkillSelectionRecord",
    "StepTracker",
    "build_skill_selection_prompt",
    "format_candidates_for_selection",
    "parse_skill_selection",
    "relabel_skill_rewards",
]
