"""Unified skill selection + effect-tag tracking + GRPO record emission.

This module provides the **single decision pipeline** that ALL domains
(game, QA, web) use for skill selection and progress tracking.

Progress tracking uses **effect tags** instead of step counters:
instead of "we are on step 3/5", we track "we have achieved
{state_observed, merge_executed} and still need {board_transformed}".

This is robust because:
  - One env step can achieve multiple tags at once (cognitive steps
    like "observe" and "choose" happen within a single LLM call)
  - Tags are unordered — no fragile ordinal alignment needed
  - Skill completion = required tag set is satisfied
  - 9B LoRA outputs EFFECTS (easy) not STEP numbers (hard)
  - Tags are domain-agnostic: games emit "merge_executed", QA emits
    "evidence_cited", web emits "form_filled"

Architecture
------------
::

    ┌───────────────────────────────────────────────────────┐
    │               SkillDecisionCore                       │
    │                                                       │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
    │  │ EffectTracker │  │ SkillSelector│  │  RecordBuf  │ │
    │  │ (tag accum,   │  │ (shared LoRA │  │  (GRPORec   │ │
    │  │  completion)  │  │  interface)  │  │  + offline)  │ │
    │  └──────────────┘  └──────────────┘  └─────────────┘ │
    └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain enum
# ---------------------------------------------------------------------------

DOMAIN_GAME = "game"
DOMAIN_QA = "qa"
DOMAIN_WEB = "web"


# ---------------------------------------------------------------------------
# Effect/tag tracker (replaces step counter)
# ---------------------------------------------------------------------------

class StepTracker:
    """Skill lifecycle tracker using effect-tag accumulation.

    Instead of tracking "step 3/5", tracks a set of achieved effect
    tags.  The skill is progressing as long as new tags are being
    achieved, and is complete when all required tags are satisfied.

    Tags come from two sources:
      1. **StateEffectObserver** (deterministic): compares prev/curr
         game state and emits tags like ``board_transformed``,
         ``merge_executed``.  Zero LLM cost, runs as fallback.
      2. **9B LoRA EFFECTS output** (learned): the skill_selection
         LoRA outputs ``EFFECTS: merge_executed, tile_promoted``
         as part of its existing call.  Trained via 35B offline
         annotations.  Zero extra inference cost.

    Both sources feed into the same cumulative tag set.
    """

    def __init__(self, domain: str = DOMAIN_GAME, game_name: str = ""):
        self.domain: str = domain
        self.game_name: str = game_name
        self.active_skill_id: Optional[str] = None
        self.active_skill_name: str = ""
        self.steps_on_skill: int = 0
        self.reward_on_skill: float = 0.0
        self.max_skill_duration: int = 10
        self.skill_switches: int = 0
        self.hop_history: List[str] = []

        self._protocol: Optional[Dict[str, Any]] = None
        self._achieved_effects: Set[str] = set()
        self._required_effects: Set[str] = set()
        self._completion_effect: str = ""
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

        self._deterministic_effects: Set[str] = set()
        self._prev_deterministic_ratio: float = 0.0
        self._new_effects_this_step: bool = False

        from decision_agents.protocol_utils import StateEffectObserver
        self._effect_observer = StateEffectObserver()

    # ── Effect tag properties ─────────────────────────────────────

    @property
    def achieved_effects(self) -> FrozenSet[str]:
        return frozenset(self._achieved_effects)

    @property
    def remaining_effects(self) -> FrozenSet[str]:
        return frozenset(self._required_effects - self._achieved_effects)

    @property
    def effects_complete(self) -> bool:
        """True when all required effects have been achieved."""
        if not self._required_effects:
            return False
        return self._required_effects.issubset(self._achieved_effects)

    @property
    def completion_ratio(self) -> float:
        """Fraction of required effects achieved (0.0 to 1.0)."""
        if not self._required_effects:
            return 0.0
        return len(self._required_effects & self._achieved_effects) / len(self._required_effects)

    @property
    def deterministic_completion_ratio(self) -> float:
        """Like completion_ratio but only counts StateEffectObserver tags.

        LoRA-reported effects are excluded to prevent the reward signal
        from being inflated by hallucinated tags.
        """
        if not self._required_effects:
            return 0.0
        return len(self._required_effects & self._deterministic_effects) / len(self._required_effects)

    # ── Backward-compat properties (for code that still reads step idx) ──

    @property
    def protocol_step_idx(self) -> int:
        total = self.total_protocol_steps
        if total <= 0:
            return 0
        return min(int(self.completion_ratio * total), total - 1)

    @property
    def total_protocol_steps(self) -> int:
        if self._protocol and isinstance(self._protocol, dict):
            return len(self._protocol.get("steps", []))
        return 0

    @property
    def step_progress_ratio(self) -> float:
        return self.completion_ratio

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
            # Merge the cumulative effect tags from StateEffectObserver
            # so predicates like ``score_increased=true`` can resolve
            # (parse_summary_state only has keys like ``game``, ``score``).
            state_dict.update(self._effect_observer._cumulative_effects)
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

        Triggers:
          - no_skill: no guidance or skill_id
          - effects_complete: all required effect tags achieved
          - duration_exceeded: steps_on_skill >= max_skill_duration
          - zero_reward_stall: 4+ steps with no reward (games only)
          - abort/success criteria from protocol
        """
        self._reselect_reason = ""
        if guidance is None or not guidance.get("skill_id"):
            self._reselect_reason = "no_skill"
            return True
        new_id = guidance["skill_id"]
        if new_id != self.active_skill_id:
            return False

        if self.effects_complete:
            self._reselect_reason = "effects_complete"
            return True

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

    # ── Receive effects from LoRA output ──────────────────────────

    def receive_lora_effects(self, effects: List[str]):
        """Accept effect tags reported by the 9B skill_selection LoRA.

        The LoRA outputs ``EFFECTS: merge_executed, tile_promoted``
        as part of its existing call.  Tags are added to the
        cumulative set (never removed).
        """
        for tag in effects:
            tag = tag.strip().lower()
            if tag:
                self._achieved_effects.add(tag)

    def receive_step_assessment(self, completed: int, total: int):
        """Backward-compat: accept STEP: n/m from LoRA output.

        Converts the ordinal step index into approximate effect
        completion.  Kept for transition period while SFT data
        still uses STEP format.
        """
        pass

    # ── Observe state effects (deterministic fallback) ────────────

    def observe_state_effects(
        self,
        curr_facts: Dict[str, Any],
        reward: float = 0.0,
        action: str = "",
    ):
        """Observe state changes after env.step() and accumulate tags.

        Deterministic tags go into both ``_achieved_effects`` (prompt
        context) and ``_deterministic_effects`` (reward-safe subset).
        ``_new_effects_this_step`` is set when at least one new
        deterministic tag appears — used for CONTINUE reward.
        """
        prev_count = len(self._deterministic_effects)
        effects = self._effect_observer.observe(
            curr_facts, reward=reward, action=action,
            game_name=self.game_name,
        )
        for key, val in effects.items():
            if val == "true":
                self._achieved_effects.add(key)
                self._deterministic_effects.add(key)
        self._new_effects_this_step = len(self._deterministic_effects) > prev_count

    # ── Update after each step ─────────────────────────────────────

    def update(
        self,
        skill_id: Optional[str],
        skill_name: str,
        reward: float,
        state_text: str = "",
        hop_type: Optional[str] = None,
    ):
        """Update tracker after one timestep."""
        if hop_type:
            self.hop_history.append(hop_type)
            self._achieved_effects.add(f"hop_{hop_type.lower()}")

        self._intrinsic_bonus = 0.0
        if skill_id != self.active_skill_id:
            self._prev_reward_on_skill = self.reward_on_skill
            self._prev_steps_on_skill = self.steps_on_skill
            self._prev_deterministic_ratio = self.deterministic_completion_ratio
            self._just_switched = (
                self.active_skill_id is not None
                and self.steps_on_skill > 0
            )
            self.active_skill_id = skill_id
            self.active_skill_name = skill_name
            self.steps_on_skill = 1
            self.reward_on_skill = reward
            self.skill_switches += 1
            self._achieved_effects = set()
            self._deterministic_effects = set()
            self._new_effects_this_step = False
            # Keep StateEffectObserver alive across skill switches so
            # _cumulative_effects (and _prev_facts for delta computation)
            # persist across the whole episode.  Resetting here was the
            # root cause of 70%+ key_missing on gymv games: every switch
            # wiped the effects dict, so the very next predicate check
            # saw an empty dict and all effect-type predicates failed.
            self.hop_history = [hop_type] if hop_type else []
        else:
            self._just_switched = False
            self.steps_on_skill += 1
            self.reward_on_skill += reward

            success = None
            abort = None
            if state_text and self.active_skill_id:
                success = self._check_criteria(state_text, is_abort=False)
                abort = self._check_criteria(state_text, is_abort=True)
                if success:
                    self._intrinsic_bonus += 0.3
                if abort:
                    self._intrinsic_bonus -= 0.1

            if self._step_checks and not (success or abort):
                from decision_agents.protocol_utils import check_predicate
                effects = self._effect_observer._cumulative_effects
                idx = min(self.steps_on_skill - 1,
                          len(self._step_checks) - 1)
                check = self._step_checks[idx] if idx >= 0 else ""
                if check and check_predicate(check, effects):
                    self._intrinsic_bonus += 0.2

    # ── Progress summary for prompt ───────────────────────────────

    def get_progress_summary(self, state_text: str = "") -> str:
        """Build a tag-based progress summary for prompt injection."""
        if not self._required_effects:
            return ""
        achieved = sorted(self._achieved_effects & self._required_effects)
        remaining = sorted(self._required_effects - self._achieved_effects)
        parts = []
        if achieved:
            parts.append(f"Achieved: {', '.join(achieved)}")
        if remaining:
            parts.append(f"Remaining: {', '.join(remaining)}")
        return " | ".join(parts)

    # ── Protocol setup ─────────────────────────────────────────────

    def set_protocol(self, protocol: Optional[Dict[str, Any]]):
        self._protocol = protocol
        self._achieved_effects = set()
        self._deterministic_effects = set()
        self._new_effects_this_step = False
        self._required_effects = set()
        self._completion_effect = ""
        # Do NOT reset _effect_observer here — same reasoning as
        # the update() skill-switch path: _cumulative_effects and
        # _prev_facts must persist across the whole episode so
        # predicate checks can find effect keys like score_increased,
        # enemy_hit, etc.  Resetting here caused 40-81% key_missing
        # on all gymv games.
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

            # Runtime predicate filtering: drop cross-domain predicates
            # that can never fire for this game.  The bank-update pipeline
            # (skillbank_pipeline._sanitize_skill_in_place) already filters
            # at write time, but pre-existing / cold-start-seeded protocols
            # loaded from disk bypass that path.  Without runtime filtering,
            # every evaluation of a stale cross-domain predicate increments
            # key_missing (1000-2700 per step for gymv games), leaving
            # r_progress ~0 and GRPO without fine-grained reward signal.
            if self.game_name:
                self._step_checks, self._predicate_success, self._predicate_abort = (
                    _filter_protocol_predicates_for_game(
                        self._step_checks,
                        self._predicate_success,
                        self._predicate_abort,
                        protocol.get("steps", []) or [],
                        self.game_name,
                    )
                )

            req = protocol.get("required_effects", []) or []
            if req:
                self._required_effects = set(req)
                if self.game_name:
                    self._required_effects = _filter_required_effects(
                        self._required_effects, self.game_name,
                    )
            else:
                self._required_effects = _infer_required_effects(
                    protocol, game_name=self.game_name,
                    filtered_step_checks=self._step_checks,
                )

            self._completion_effect = protocol.get("completion_effect", "")
        else:
            self.max_skill_duration = 10


def _filter_required_effects(
    required: Set[str],
    game_name: str,
) -> Set[str]:
    """Drop required effects that cannot fire for *game_name*.

    If the game has a registered closed effect set, only keep effects
    that appear in it.  If no set is registered, return unchanged.
    """
    from decision_agents.protocol_utils import (
        canonicalize_game_key,
        TASK_EFFECT_SUBSET,
    )
    canonical = canonicalize_game_key(game_name)
    if canonical not in TASK_EFFECT_SUBSET:
        return required
    allowed = set(TASK_EFFECT_SUBSET[canonical])
    filtered = {e for e in required if e in allowed}
    if not filtered and required:
        return required
    return filtered


def _filter_protocol_predicates_for_game(
    step_checks: List[str],
    predicate_success: List[str],
    predicate_abort: List[str],
    protocol_steps: List[str],
    game_name: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Filter all protocol predicates against the game's closed effect set.

    Called at protocol-load time inside :meth:`StepTracker.set_protocol`
    so that even pre-existing / cold-start-seeded protocols don't poison
    runtime evaluation with cross-domain predicates (the write-time
    sanitizer in ``skillbank_pipeline`` only runs on freshly-mined
    protocols, not on loaded ones).

    Returns ``(filtered_step_checks, filtered_pred_success, filtered_pred_abort)``.
    """
    from decision_agents.protocol_utils import (
        repair_step_checks_against_registry,
        filter_predicates_against_registry,
    )

    new_checks, _ = repair_step_checks_against_registry(
        list(step_checks), list(protocol_steps), game_name=game_name,
    )
    new_ps, _ = filter_predicates_against_registry(
        list(predicate_success), game_name=game_name,
    )
    new_pa, _ = filter_predicates_against_registry(
        list(predicate_abort), game_name=game_name,
    )
    return new_checks, new_ps, new_pa


def _infer_required_effects(
    protocol: Dict[str, Any],
    game_name: str = "",
    filtered_step_checks: Optional[List[str]] = None,
) -> Set[str]:
    """Infer required_effects from existing protocol fields.

    Bridges old protocol format (step_checks, action_vocab,
    template_signature) to the new effect-tag model.

    When ``filtered_step_checks`` is provided (from runtime predicate
    filtering), use those instead of the protocol's raw step_checks to
    avoid pulling cross-domain effect keys into the required set.
    """
    effects: Set[str] = set()

    step_checks = (
        filtered_step_checks if filtered_step_checks is not None
        else (protocol.get("step_checks", []) or [])
    )
    for check in step_checks:
        if not check:
            continue
        key = check.split("=")[0].strip()
        if key:
            effects.add(key)

    if not effects:
        from decision_agents.protocol_utils import (
            generate_step_checks_from_effects,
        )
        steps = protocol.get("steps", [])
        if steps:
            generated = generate_step_checks_from_effects(steps, game_name=game_name)
            for check in generated:
                if check:
                    key = check.split("=")[0].strip()
                    if key:
                        effects.add(key)

    if not effects:
        vocab = protocol.get("action_vocab", []) or []
        from decision_agents.protocol_utils import OPERATOR_TO_EFFECT
        for op in vocab:
            eff = OPERATOR_TO_EFFECT.get(op.upper(), "")
            if eff:
                effects.add(eff)

    return effects


# ---------------------------------------------------------------------------
# Prompt / parse helpers (domain-agnostic)
# ---------------------------------------------------------------------------

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are a skill selector. Output exactly 3 lines:\n"
    "EFFECTS: <comma-separated effects achieved so far from the valid set>\n"
    "DECISION: CONTINUE or SWITCH\n"
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
        req_effects = protocol.get("required_effects", []) if isinstance(protocol, dict) else []
        lines.append(f"  {i}. {name}")
        if hint:
            lines.append(f"     Strategy: {hint[:150]}")
        if req_effects:
            lines.append(f"     Required effects: {', '.join(req_effects[:6])}")
        elif steps:
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

_EFFECTS_RE = re.compile(
    r"EFFECTS?\s*:\s*(.+?)(?=\nDECISION|\nSKILL|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _fuzzy_match_tag(tag: str, valid_set: set) -> str:
    """Best-effort recovery for misspelled/shortened effect tags."""
    for valid in valid_set:
        if tag in valid or valid in tag:
            return valid
    tag_parts = set(tag.split("_"))
    best, best_overlap = "", 0
    for valid in valid_set:
        valid_parts = set(valid.split("_"))
        overlap = len(tag_parts & valid_parts)
        if overlap > best_overlap:
            best, best_overlap = valid, overlap
    return best if best_overlap >= 1 else ""


# ── Skill-selection parse telemetry ──────────────────────────────────
# Tracks which parse path produced each ``(chosen_idx, effects,
# decision)`` tuple so we can distinguish "LoRA emitted a valid
# ``SKILL: N`` tag" from "LoRA emitted garbage and we silently fell
# back to candidate 0".  Pre-May-2026 the legacy parser had 4
# unguarded fallback layers (empty reply, trailing-number heuristic,
# name-substring heuristic, default-to-zero) with no logging — the
# reward log recorded ``chosen_skill_id=candidates[0].skill_id`` for
# both intelligent selections AND silent fallbacks, so the GRPO
# learning signal couldn't penalise unparseable LoRA output.  The
# counters here surface fallback rates to monitoring; the new
# ``parse_path`` field on ``SkillSelectionResult`` lets callers
# include it in GRPO metadata.

PARSE_PATH_SKILL_TAG       = "skill_tag"        # ✅ Correct SFT format
PARSE_PATH_TAIL_NUMBER     = "tail_number"      # ⚠️ Heuristic recovery
PARSE_PATH_NAME_SUBSTRING  = "name_substring"   # ⚠️ Heuristic recovery
PARSE_PATH_FALLBACK_ZERO   = "fallback_zero"    # 🚨 Unparseable → 0
PARSE_PATH_EMPTY_REPLY     = "empty_reply"      # 🚨 No reply → 0

_PARSE_STATS: Dict[str, int] = {
    PARSE_PATH_SKILL_TAG:      0,
    PARSE_PATH_TAIL_NUMBER:    0,
    PARSE_PATH_NAME_SUBSTRING: 0,
    PARSE_PATH_FALLBACK_ZERO:  0,
    PARSE_PATH_EMPTY_REPLY:    0,
}


def reset_parse_stats() -> None:
    """Reset the module-level parse-path counters (called by the
    orchestrator at the start of each co-evolution step so per-step
    fallback rates surface in the step report)."""
    for k in list(_PARSE_STATS):
        _PARSE_STATS[k] = 0


def get_parse_stats() -> Dict[str, int]:
    """Snapshot of the parse-path counters since the last reset."""
    return dict(_PARSE_STATS)


def parse_skill_selection(
    reply: str,
    n_candidates: int,
    candidates: Optional[List[Dict[str, Any]]] = None,
    strip_think_tags: Optional[Callable[[str], str]] = None,
    task_name: str = "",
    *,
    return_parse_path: bool = False,
) -> Tuple[int, List[str], str]:
    """Parse skill selection reply (3-line ``EFFECTS / DECISION / SKILL``
    format) into ``(chosen_idx, effects, decision)``.

    ``decision`` is ``"CONTINUE"`` / ``"SWITCH"`` / ``"SKIP"``.

    Parse-path telemetry
    --------------------
    The function tries 4 progressively-less-trustworthy strategies to
    recover ``chosen_idx`` when the LoRA doesn't emit a clean
    ``SKILL: N`` tag.  Each strategy increments a module-level counter
    (:data:`_PARSE_STATS`) so callers can monitor *how often* the
    parser is falling back versus reading a valid LoRA emission:

      1. ``skill_tag``      — clean ``SKILL: N`` match (preferred)
      2. ``tail_number``    — any number in the last 100 chars
      3. ``name_substring`` — any candidate skill name appears as
                              substring of the reply
      4. ``fallback_zero``  — none of the above → return ``idx=0``

    A non-trivial ``tail_number`` / ``name_substring`` / ``fallback_zero``
    rate is a SIGNAL THAT THE LoRA IS BROKEN, not a benign recovery.
    Each fallback path also emits a ``logger.warning`` (rate-limited
    via the counter) so the issue surfaces in real-time logs.

    When ``return_parse_path=True`` the tuple is extended with the
    chosen parse path string (one of the ``PARSE_PATH_*`` constants).
    Default ``False`` preserves the 3-element tuple for all existing
    callers.
    """
    parse_path: str

    if not reply:
        _PARSE_STATS[PARSE_PATH_EMPTY_REPLY] += 1
        if _PARSE_STATS[PARSE_PATH_EMPTY_REPLY] in (1, 10, 100, 1000):
            logger.warning(
                "parse_skill_selection: empty reply, falling back to "
                "candidate 0 (count=%d)",
                _PARSE_STATS[PARSE_PATH_EMPTY_REPLY],
            )
        parse_path = PARSE_PATH_EMPTY_REPLY
        if return_parse_path:
            return 0, [], "SWITCH", parse_path  # type: ignore[return-value]
        return 0, [], "SWITCH"

    cleaned = reply
    if strip_think_tags is not None:
        cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    effects: List[str] = []
    effects_m = _EFFECTS_RE.search(cleaned)
    if effects_m:
        raw = effects_m.group(1).strip()
        parsed = [t.strip().lower() for t in raw.split(",") if t.strip()]
        from decision_agents.protocol_utils import EFFECT_REGISTRY
        valid_set = set(EFFECT_REGISTRY.keys())
        for tag in parsed:
            if tag in valid_set:
                effects.append(tag)
            else:
                closest = _fuzzy_match_tag(tag, valid_set)
                if closest:
                    effects.append(closest)
                else:
                    logger.debug("Dropping unknown effect tag: %s", tag)

    decision = "SWITCH"
    decision_m = _DECISION_RE.search(cleaned)
    if decision_m:
        decision = decision_m.group(1).upper()

    # ── Path 1 (preferred): clean ``SKILL: N`` match ─────────────────
    skill_m = re.search(r"SKILL\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if skill_m:
        idx = int(skill_m.group(1)) - 1
        if 0 <= idx < n_candidates:
            _PARSE_STATS[PARSE_PATH_SKILL_TAG] += 1
            parse_path = PARSE_PATH_SKILL_TAG
            if return_parse_path:
                return idx, effects, decision, parse_path  # type: ignore[return-value]
            return idx, effects, decision
        if n_candidates > 0:
            clamped = max(0, min(idx, n_candidates - 1))
            _PARSE_STATS[PARSE_PATH_SKILL_TAG] += 1
            parse_path = PARSE_PATH_SKILL_TAG
            if return_parse_path:
                return clamped, effects, decision, parse_path  # type: ignore[return-value]
            return clamped, effects, decision

    # ── Path 2: trailing-number heuristic (LoRA forgot the SKILL: tag
    # but ended with "...so I pick 3.").  Already a degradation signal.
    tail = cleaned[-100:]
    nums = re.findall(r"\b(\d+)\b", tail)
    for n_str in reversed(nums):
        idx = int(n_str) - 1
        if 0 <= idx < n_candidates:
            _PARSE_STATS[PARSE_PATH_TAIL_NUMBER] += 1
            if _PARSE_STATS[PARSE_PATH_TAIL_NUMBER] in (1, 10, 100, 1000):
                logger.warning(
                    "parse_skill_selection: no SKILL: tag, recovered idx=%d "
                    "from trailing number heuristic (count=%d, reply tail=%r)",
                    idx, _PARSE_STATS[PARSE_PATH_TAIL_NUMBER], tail[-60:],
                )
            parse_path = PARSE_PATH_TAIL_NUMBER
            if return_parse_path:
                return idx, effects, decision, parse_path  # type: ignore[return-value]
            return idx, effects, decision

    # ── Path 3: candidate-name substring (LoRA wrote the skill name
    # instead of an index — also a degradation signal).
    if candidates:
        cleaned_lower = cleaned.lower()
        for i, c in enumerate(candidates):
            name = (c.get("skill_name") or "").lower()
            if name and len(name) >= 4 and name in cleaned_lower:
                _PARSE_STATS[PARSE_PATH_NAME_SUBSTRING] += 1
                if _PARSE_STATS[PARSE_PATH_NAME_SUBSTRING] in (1, 10, 100, 1000):
                    logger.warning(
                        "parse_skill_selection: no SKILL: tag, recovered "
                        "idx=%d from candidate-name substring match "
                        "(count=%d, name=%r)",
                        i, _PARSE_STATS[PARSE_PATH_NAME_SUBSTRING], name,
                    )
                parse_path = PARSE_PATH_NAME_SUBSTRING
                if return_parse_path:
                    return i, effects, decision, parse_path  # type: ignore[return-value]
                return i, effects, decision

    # ── Path 4 (silent fallback): unparseable reply → candidate 0.
    # This is the bug-of-record: pre-May-2026 the legacy parser took
    # this path with zero telemetry, so the reward log recorded
    # ``chosen_skill_id=candidates[0]`` for both intelligent selections
    # and total LoRA failure.  Now counted + logged at exponential
    # checkpoints so the rate surfaces without log spam.
    _PARSE_STATS[PARSE_PATH_FALLBACK_ZERO] += 1
    if _PARSE_STATS[PARSE_PATH_FALLBACK_ZERO] in (1, 10, 100, 1000, 10000):
        logger.warning(
            "parse_skill_selection: UNPARSEABLE reply → fallback to "
            "candidate 0 (count=%d, reply=%r)",
            _PARSE_STATS[PARSE_PATH_FALLBACK_ZERO], cleaned[:200],
        )
    parse_path = PARSE_PATH_FALLBACK_ZERO
    if return_parse_path:
        return 0, effects, decision, parse_path  # type: ignore[return-value]
    return 0, effects, decision


# ---------------------------------------------------------------------------
# Skill selection prompt builder (domain-agnostic)
# ---------------------------------------------------------------------------

def build_skill_selection_prompt(
    state_text: str,
    intention: str,
    candidates: List[Dict[str, Any]],
    tracker: StepTracker,
    profile_prefix: str = "",
    recent_actions: Optional[List[str]] = None,
    recent_rewards: Optional[List[float]] = None,
    task_name: str = "",
) -> str:
    """Build the skill_selection prompt, identical for all domains.

    The prompt gives the LoRA everything it needs for CONTINUE vs SWITCH:
      - Current state (what the world looks like now)
      - Closed-set valid effect tags for this task (constrained vocabulary)
      - Current skill + what effects it still needs (CONTINUE context)
      - Historical actions/rewards (is the skill making progress?)
      - Candidate skills to switch to (SWITCH context)
    """
    from decision_agents.protocol_utils import get_valid_effects

    candidates_text = format_candidates_for_selection(candidates)

    # Closed-set effect vocabulary for this task
    tn = task_name or tracker.game_name
    valid_tags = get_valid_effects(tn)
    valid_tags_str = ", ".join(valid_tags)

    # Current skill context (for CONTINUE branch)
    skill_ctx = ""
    if tracker.active_skill_name:
        skill_ctx = f"Active skill: {tracker.active_skill_name}"
        skill_ctx += f" (step {tracker.steps_on_skill}, reward so far: {tracker.reward_on_skill:.1f})\n"

    # Effect progress (for CONTINUE branch — what tags achieved / remaining)
    progress_ctx = ""
    achieved = tracker.achieved_effects
    remaining = tracker.remaining_effects
    if achieved or remaining:
        parts = []
        if achieved:
            parts.append(f"Achieved effects: {', '.join(sorted(achieved))}")
        if remaining:
            parts.append(f"Still needed: {', '.join(sorted(remaining))}")
        progress_ctx = "\n".join(parts) + "\n"

    # Recent action/reward history (is the skill progressing?)
    history_ctx = ""
    if recent_actions:
        rr = recent_rewards or [0.0] * len(recent_actions)
        tail_a = recent_actions[-5:]
        tail_r = rr[-5:]
        history_lines = [f"  {a} → reward {r:.1f}" for a, r in zip(tail_a, tail_r)]
        history_ctx = "Recent history:\n" + "\n".join(history_lines) + "\n"
        if sum(tail_r) <= 0 and len(tail_a) >= 3:
            history_ctx += "  (no reward in recent steps — consider SWITCH)\n"

    hop_ctx = ""
    if tracker.hop_history:
        hop_ctx = f"Executed hops: {' → '.join(tracker.hop_history)}\n"

    user_content = (
        f"Current state:\n{state_text[:3500]}\n\n"
        f"Intention: {intention[:500]}\n"
        f"Valid effect tags: [{valid_tags_str}]\n"
        f"{skill_ctx}"
        f"{progress_ctx}"
        f"{history_ctx}"
        f"{hop_ctx}\n"
        f"Strategies:\n{candidates_text}\n\n"
        f"EFFECTS: <list achieved effects from the valid set>\n"
        f"DECISION: CONTINUE or SWITCH\n"
        f"SKILL: <number>"
    )
    return profile_prefix + SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user_content


# ---------------------------------------------------------------------------
# GRPO record for skill_selection (domain-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class SkillSelectionRecord:
    """One skill_selection decision point, ready for GRPO training."""

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
    effects: List[str] = field(default_factory=list)
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
    """Relabel skill_selection rewards using full trajectory information."""
    n = len(records)
    if n == 0:
        return records

    for i, rec in enumerate(records):
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
