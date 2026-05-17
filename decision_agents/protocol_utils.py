"""Shared utilities for protocol-aware skill lifecycle management.

Provides predicate checking against parsed ``summary_state`` dicts and
progress tracking helpers.  Used by ``_SkillTracker`` in both
``scripts/qwen3_decision_agent.py`` and
``trainer/coevolution/episode_runner.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


_CMP_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*([<>=!]+)\s*(.+)$"
)


# ── Predicate-check telemetry ────────────────────────────────────────
# Distinguishes "predicate is false" from "key is missing from state"
# so the operator can spot when a Protocol references state fields the
# runtime never produces (e.g. LoRA hallucinated ``shield_buff`` for
# TF3 — both bug classes return False but mean very different things).
# Counters reset at each co-evolution step boundary via
# ``reset_predicate_stats()``.

PREDICATE_RESULT_MATCH       = "match"         # ✅ pred holds
PREDICATE_RESULT_MISMATCH    = "mismatch"      # ✅ pred false (legit)
PREDICATE_RESULT_KEY_MISSING = "key_missing"   # 🚨 state lacks the key
PREDICATE_RESULT_PARSE_ERROR = "parse_error"   # 🚨 malformed predicate

_PREDICATE_STATS: Dict[str, int] = {
    PREDICATE_RESULT_MATCH:       0,
    PREDICATE_RESULT_MISMATCH:    0,
    PREDICATE_RESULT_KEY_MISSING: 0,
    PREDICATE_RESULT_PARSE_ERROR: 0,
}

# Set of unique keys observed to be missing from state.  Useful for
# post-hoc diagnosis ("which Protocols reference state fields the
# runtime never produces?") without per-call log spam.
_MISSING_PREDICATE_KEYS: Set[str] = set()


def reset_predicate_stats() -> None:
    """Reset the predicate-evaluation counters and missing-key set.

    Called by the orchestrator at the start of each co-evolution step
    so per-step ``key_missing`` rates land in ``step_log.jsonl``.
    """
    for k in list(_PREDICATE_STATS):
        _PREDICATE_STATS[k] = 0
    _MISSING_PREDICATE_KEYS.clear()


def get_predicate_stats() -> Dict[str, Any]:
    """Snapshot ``{result_kind: count, missing_keys: [...]}`` for the
    current step.  The list of missing keys is bounded to 50 entries so
    badly-misconfigured protocols don't blow up the step log.
    """
    return {
        **_PREDICATE_STATS,
        "missing_keys": sorted(_MISSING_PREDICATE_KEYS)[:50],
    }


def parse_summary_state(state_str: str) -> Dict[str, str]:
    """Parse a ``key=value | key=value`` summary_state string into a dict."""
    result: Dict[str, str] = {}
    if not state_str:
        return result
    for part in state_str.split("|"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.strip()
    return result


def check_predicate(pred: str, state: Dict[str, str]) -> bool:
    """Check a single predicate against a parsed summary_state dict.

    Supported formats:
      ``key=value``      — exact match
      ``key!=value``     — not equal
      ``key>N``          — numeric greater-than
      ``key<N``          — numeric less-than
      ``key>=N``         — numeric greater-or-equal
      ``key<=N``         — numeric less-or-equal

    Returns False if the key is missing from state or parsing fails.
    Records the outcome in module-level telemetry
    (:data:`_PREDICATE_STATS`) so the operator can distinguish "the
    Protocol references a state field the runtime never produces"
    from "the predicate is legitimately false".
    """
    result, _ = check_predicate_with_telemetry(pred, state)
    return result


def check_predicate_with_telemetry(
    pred: str,
    state: Dict[str, str],
) -> Tuple[bool, str]:
    """Like :func:`check_predicate` but also returns a result-kind tag
    (one of :data:`PREDICATE_RESULT_MATCH`,
    :data:`PREDICATE_RESULT_MISMATCH`,
    :data:`PREDICATE_RESULT_KEY_MISSING`,
    :data:`PREDICATE_RESULT_PARSE_ERROR`).

    The tag distinguishes the four cases that the legacy
    boolean-return collapsed onto False:

      * ``mismatch``    — predicate is well-formed, key exists, value
                          comparison failed.  This is the SAFE False.
      * ``key_missing`` — predicate is well-formed but the runtime
                          state dict doesn't carry that key.  This is
                          the DANGEROUS False: the Protocol may be
                          referencing a field the runtime never
                          produces (the May-2026 contamination bug).
      * ``parse_error`` — predicate string is malformed.  Should be
                          rare; logs at exponential checkpoints.

    Empty / blank predicates (``""``) return ``(False, "parse_error")``
    — caller must filter those out beforehand if it cares.
    """
    m = _CMP_RE.match(pred.strip())
    if not m:
        _PREDICATE_STATS[PREDICATE_RESULT_PARSE_ERROR] += 1
        if _PREDICATE_STATS[PREDICATE_RESULT_PARSE_ERROR] in (1, 10, 100):
            logger.warning(
                "check_predicate: malformed predicate %r (count=%d)",
                pred, _PREDICATE_STATS[PREDICATE_RESULT_PARSE_ERROR],
            )
        return False, PREDICATE_RESULT_PARSE_ERROR

    key, op, expected = m.group(1), m.group(2), m.group(3).strip()
    actual = state.get(key)
    if actual is None:
        _PREDICATE_STATS[PREDICATE_RESULT_KEY_MISSING] += 1
        if key not in _MISSING_PREDICATE_KEYS:
            _MISSING_PREDICATE_KEYS.add(key)
            logger.warning(
                "check_predicate: key %r referenced by predicate %r is "
                "MISSING from runtime state (first-seen; total "
                "key_missing count=%d).  This is the silent-zero "
                "intrinsic_bonus failure mode — either the Protocol "
                "references a hallucinated field or the runtime "
                "summary_state generator dropped the key.",
                key, pred, _PREDICATE_STATS[PREDICATE_RESULT_KEY_MISSING],
            )
        return False, PREDICATE_RESULT_KEY_MISSING

    matched: bool
    if op == "==" or op == "=":
        matched = actual == expected
    elif op == "!=":
        matched = actual != expected
    else:
        try:
            a_num = float(actual)
            e_num = float(expected)
        except (ValueError, TypeError):
            _PREDICATE_STATS[PREDICATE_RESULT_PARSE_ERROR] += 1
            return False, PREDICATE_RESULT_PARSE_ERROR

        if op == ">":
            matched = a_num > e_num
        elif op == "<":
            matched = a_num < e_num
        elif op == ">=":
            matched = a_num >= e_num
        elif op == "<=":
            matched = a_num <= e_num
        else:
            _PREDICATE_STATS[PREDICATE_RESULT_PARSE_ERROR] += 1
            return False, PREDICATE_RESULT_PARSE_ERROR

    if matched:
        _PREDICATE_STATS[PREDICATE_RESULT_MATCH] += 1
        return True, PREDICATE_RESULT_MATCH
    _PREDICATE_STATS[PREDICATE_RESULT_MISMATCH] += 1
    return False, PREDICATE_RESULT_MISMATCH


def check_predicates(preds: List[str], state: Dict[str, str]) -> bool:
    """Return True if ALL predicates pass (AND semantics)."""
    if not preds:
        return False
    return all(check_predicate(p, state) for p in preds)


def check_any_predicate(preds: List[str], state: Dict[str, str]) -> bool:
    """Return True if ANY predicate passes (OR semantics)."""
    if not preds:
        return False
    return any(check_predicate(p, state) for p in preds)


def keyword_match(criteria_text: str, state_text: str) -> bool:
    """Legacy keyword matching (fallback when no predicates available).

    Checks if at least 3-char tokens from *criteria_text* all appear in
    *state_text*.  This is the old behavior from ``_SkillTracker``.
    """
    if not criteria_text or not state_text:
        return False
    state_lower = state_text.lower()
    tokens = [t for t in criteria_text.lower().split() if len(t) >= 3]
    return bool(tokens) and all(tok in state_lower for tok in tokens[:3])


def compute_step_advancement(
    current_idx: int,
    step_checks: List[str],
    state: Dict[str, str],
    total_steps: int,
) -> int:
    """Determine the protocol step index after one timestep.

    If ``step_checks`` are available and the current step's check passes,
    advance.  If no checks are defined **or the current check is empty**,
    stay at the current step (no free advancement).
    Returns the new step index (clamped to ``total_steps - 1``).
    """
    if total_steps <= 0:
        return 0

    if not step_checks or current_idx >= len(step_checks):
        return current_idx

    check = step_checks[current_idx]
    if not check:
        return current_idx

    if check_predicate(check, state):
        return min(current_idx + 1, total_steps - 1)

    return current_idx


def build_progress_summary(
    steps: List[str],
    step_checks: List[str],
    current_idx: int,
    state: Dict[str, str],
) -> str:
    """Build a short progress summary for prompt injection.

    Returns a string like:
      ``Steps 1-2 done. Current: step 3 — Shift piece to target column.``
    """
    if not steps:
        return ""

    completed = []
    for i in range(min(current_idx, len(steps))):
        completed.append(i + 1)

    parts = []
    if completed:
        if len(completed) == 1:
            parts.append(f"Step {completed[0]} done.")
        else:
            parts.append(f"Steps {completed[0]}-{completed[-1]} done.")

    if current_idx < len(steps):
        parts.append(f"Current: step {current_idx + 1} — {steps[current_idx][:80]}")

    return " ".join(parts)


def compute_expected_duration(
    sub_episode_lengths: List[int],
    protocol_steps: int = 0,
) -> int:
    """Compute a reasonable expected_duration from sub-episode statistics.

    Uses the median length (robust to outliers), capped between
    ``max(protocol_steps, 3)`` and 30.  Falls back to ``protocol_steps``
    or 10 if no data.
    """
    min_dur = max(protocol_steps, 3) if protocol_steps > 0 else 3
    if not sub_episode_lengths:
        return max(min_dur, protocol_steps) if protocol_steps > 0 else 10

    sorted_lens = sorted(sub_episode_lengths)
    n = len(sorted_lens)
    if n % 2 == 0:
        median = (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2
    else:
        median = sorted_lens[n // 2]

    return max(min_dur, min(int(median), 30))


# ── Step-check generation from Layer-C operator types ────────────────

OPERATOR_TO_EFFECT: Dict[str, str] = {
    "PERCEIVE": "evidence_cited",
    "RECALL":   "evidence_cited",
    "COMPARE":  "options_compared",
    "FILTER":   "candidates_eliminated",
    "DECIDE":   "answer_selected",
    "COMMIT":   "answer_emitted",
    "VERIFY":   "answer_confirmed",
}


def build_step_checks_from_signature(
    template_signature: str,
    action_vocab: Optional[List[str]] = None,
    n_steps: int = 0,
) -> List[str]:
    """Generate step_checks from a Layer-C template_signature.

    ``template_signature`` is like ``"PERCEIVE → COMPARE → FILTER → DECIDE → VERIFY"``.
    ``action_vocab`` is ``["COMPARE", "DECIDE", "FILTER", "PERCEIVE", "VERIFY"]``.

    Returns a list of predicate strings (one per step) that
    ``compute_step_advancement`` can evaluate against a state dict.
    """
    if template_signature:
        ops = [op.strip().upper() for op in template_signature.split("→")]
    elif action_vocab:
        ops = [op.strip().upper() for op in action_vocab]
    else:
        return [""] * max(n_steps, 1)

    checks: List[str] = []
    for op in ops:
        effect = OPERATOR_TO_EFFECT.get(op, "")
        if effect:
            checks.append(f"{effect}=true")
        else:
            checks.append("")

    if n_steps > 0 and len(checks) < n_steps:
        checks.extend([""] * (n_steps - len(checks)))
    elif n_steps > 0 and len(checks) > n_steps:
        checks = checks[:n_steps]

    return checks


# ── State-effect observer (replaces LoRA self-report for step tracking) ──

class StateEffectObserver:
    """Observes state deltas between env steps and emits effect predicates.

    Instead of relying on the LoRA to self-report ``STEP: 3/5``, this
    observer watches what actually changed in the environment and
    produces a cumulative effect dict like::

        {"board_changed": "true", "score_increased": "true",
         "highest_increased": "true", "reward_positive": "true"}

    These effects are matched against ``step_checks`` predicates by
    the existing ``compute_step_advancement()`` function, giving us
    grounded, deterministic step tracking for games.

    The protocol step ↔ env step mapping is inherently many-to-many:
    one env step can satisfy multiple protocol steps (cognitive steps
    like "observe" and "choose" happen within a single LLM call), and
    one protocol step can span multiple env steps (a complex action
    sequence).  The observer handles this by accumulating effects and
    advancing through steps greedily.
    """

    def __init__(self):
        self._prev_facts: Dict[str, str] = {}
        self._cumulative_effects: Dict[str, str] = {}
        self._step_rewards: List[float] = []

    def reset(self):
        self._prev_facts = {}
        self._cumulative_effects = {}
        self._step_rewards = []

    def observe(
        self,
        curr_facts: Dict[str, str],
        reward: float = 0.0,
        action: str = "",
        game_name: str = "",
    ) -> Dict[str, str]:
        """Compare current facts against previous, emit effects.

        Call this after each ``env.step()``.  Returns the cumulative
        effect dict (grows monotonically — effects are never removed).
        """
        prev = self._prev_facts
        self._step_rewards.append(reward)

        effects = dict(self._cumulative_effects)

        if reward > 0:
            effects["reward_positive"] = "true"
        if sum(self._step_rewards) > 0:
            effects["cumulative_reward_positive"] = "true"

        if action:
            effects["action_taken"] = "true"

        if not prev:
            effects["state_observed"] = "true"
            self._prev_facts = dict(curr_facts)
            self._cumulative_effects = effects
            return effects

        for key, curr_val in curr_facts.items():
            prev_val = prev.get(key)
            if prev_val is None:
                continue

            try:
                cv = float(curr_val)
                pv = float(prev_val)
                if cv != pv:
                    effects[f"{key}_changed"] = "true"
                if cv > pv:
                    effects[f"{key}_increased"] = "true"
                if cv < pv:
                    effects[f"{key}_decreased"] = "true"
            except (ValueError, TypeError):
                if curr_val != prev_val:
                    effects[f"{key}_changed"] = "true"

        game_effects = _compute_game_specific_effects(
            game_name, prev, curr_facts, action, reward,
        )
        effects.update(game_effects)

        self._prev_facts = dict(curr_facts)
        self._cumulative_effects = effects
        return effects

    def advance_step(
        self,
        current_idx: int,
        step_checks: List[str],
        total_steps: int,
    ) -> int:
        """Advance through as many protocol steps as the effects satisfy.

        Unlike ``compute_step_advancement`` which checks one step,
        this greedily advances through consecutive steps whose checks
        all pass — handling the case where one env step satisfies
        multiple protocol steps (e.g., "observe" + "choose" + "act"
        all within one LLM call + env.step).
        """
        if total_steps <= 0 or not step_checks:
            return current_idx

        idx = current_idx
        effects = self._cumulative_effects

        while idx < total_steps and idx < len(step_checks):
            check = step_checks[idx]
            if not check:
                break
            if check_predicate(check, effects):
                idx = min(idx + 1, total_steps - 1)
                if idx >= total_steps - 1:
                    break
            else:
                break

        return idx

    @property
    def effects(self) -> Dict[str, str]:
        return dict(self._cumulative_effects)


def _compute_game_specific_effects(
    game_name: str,
    prev_facts: Dict[str, str],
    curr_facts: Dict[str, str],
    action: str,
    reward: float,
) -> Dict[str, str]:
    """Compute game-specific semantic effects from state deltas.

    Each game defines what state changes constitute meaningful
    "effects" for protocol step tracking.  These are higher-level
    than raw key_changed predicates — they capture game semantics.
    """
    gn = game_name.lower().replace(" ", "_")
    fn = _GAME_EFFECT_MAP.get(gn)
    if fn is not None:
        return fn(prev_facts, curr_facts, action, reward)
    return {}


def _effects_2048(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    effects: Dict[str, str] = {}
    p_highest = _safe_int(prev.get("highest", "0"))
    c_highest = _safe_int(curr.get("highest", "0"))
    p_empty = _safe_int(prev.get("empty", "0"))
    c_empty = _safe_int(curr.get("empty", "0"))
    p_merges = _safe_int(prev.get("merges", "0"))
    c_merges = _safe_int(curr.get("merges", "0"))

    if c_highest > p_highest:
        effects["tile_promoted"] = "true"
    if p_merges > 0 and reward > 0:
        effects["merge_executed"] = "true"
    if c_empty != p_empty:
        effects["board_transformed"] = "true"
    if c_empty < 4:
        effects["board_crowded"] = "true"
    if action:
        effects["direction_applied"] = "true"
    return effects


def _effects_tetris(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    effects: Dict[str, str] = {}
    p_stack = _safe_int(prev.get("stack_h", "0"))
    c_stack = _safe_int(curr.get("stack_h", "0"))
    p_holes = _safe_int(prev.get("holes", "0"))
    c_holes = _safe_int(curr.get("holes", "0"))
    p_piece = prev.get("piece", "")
    c_piece = curr.get("piece", "")

    if reward > 0:
        effects["line_cleared"] = "true"
    if c_piece != p_piece:
        effects["piece_changed"] = "true"
    if c_stack > p_stack:
        effects["piece_placed"] = "true"
    if c_holes < p_holes:
        effects["holes_reduced"] = "true"
    if c_holes > p_holes:
        effects["holes_created"] = "true"
    if action:
        effects["move_applied"] = "true"
    return effects


def _effects_candy_crush(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    effects: Dict[str, str] = {}
    p_score = _safe_int(prev.get("score", "0"))
    c_score = _safe_int(curr.get("score", "0"))
    p_moves = _safe_int(prev.get("moves", "0"))
    c_moves = _safe_int(curr.get("moves", "0"))
    p_pairs = _safe_int(prev.get("pairs", "0"))
    c_pairs = _safe_int(curr.get("pairs", "0"))

    if c_score > p_score:
        effects["match_scored"] = "true"
    if c_moves < p_moves:
        effects["move_spent"] = "true"
    if c_pairs != p_pairs:
        effects["board_reshuffled"] = "true"
    if action:
        effects["swap_applied"] = "true"
    return effects


def _effects_mario(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    effects: Dict[str, str] = {}
    p_pos = prev.get("mario", "")
    c_pos = curr.get("mario", "")
    if p_pos and c_pos and p_pos != c_pos:
        effects["mario_moved"] = "true"
    if reward > 0:
        effects["progress_made"] = "true"
    if reward < 0:
        effects["damage_taken"] = "true"
    if action:
        effects["action_executed"] = "true"
    return effects


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


# ── Gym-V Temporal games (shooter / brawler / platformer / puzzle) ────

def _effects_gymv_shooter(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    """Airstriker, SpaceHarrierII, ThunderForceIII."""
    effects: Dict[str, str] = {}
    if reward > 0:
        effects["enemy_hit"] = "true"
    if reward < 0:
        effects["damage_taken"] = "true"
    p_score = _safe_int(prev.get("score", "0"))
    c_score = _safe_int(curr.get("score", "0"))
    if c_score > p_score:
        effects["score_increased"] = "true"
    if action:
        effects["action_executed"] = "true"
    if "fire" in action.lower() or "b" == action.lower():
        effects["projectile_fired"] = "true"
    return effects


def _effects_gymv_brawler(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    """AlteredBeast, StreetsOfRage2, Strider, DynamiteHeaddy."""
    effects: Dict[str, str] = {}
    if reward > 0:
        effects["enemy_hit"] = "true"
    if reward < 0:
        effects["damage_taken"] = "true"
    p_score = _safe_int(prev.get("score", "0"))
    c_score = _safe_int(curr.get("score", "0"))
    if c_score > p_score:
        effects["score_increased"] = "true"
    if action:
        effects["action_executed"] = "true"
    act_low = action.lower()
    if any(k in act_low for k in ("attack", "punch", "kick", "a", "b", "c")):
        effects["attack_landed"] = "true"
    if any(k in act_low for k in ("left", "right", "up", "down")):
        effects["position_changed"] = "true"
    return effects


def _effects_gymv_columns(
    prev: Dict[str, str], curr: Dict[str, str],
    action: str, reward: float,
) -> Dict[str, str]:
    """Columns (puzzle/match game)."""
    effects: Dict[str, str] = {}
    if reward > 0:
        effects["match_scored"] = "true"
    p_score = _safe_int(prev.get("score", "0"))
    c_score = _safe_int(curr.get("score", "0"))
    if c_score > p_score:
        effects["score_increased"] = "true"
    if action:
        effects["action_executed"] = "true"
    act_low = action.lower()
    if "rotate" in act_low or "cycle" in act_low:
        effects["piece_rotated"] = "true"
    if "down" in act_low or "drop" in act_low:
        effects["piece_placed"] = "true"
    return effects


_GAME_EFFECT_MAP: Dict[str, Any] = {
    # Classic games
    "twenty_forty_eight": _effects_2048,
    "2048": _effects_2048,
    "tetris": _effects_tetris,
    "candy_crush": _effects_candy_crush,
    "candy": _effects_candy_crush,
    "super_mario": _effects_mario,
    "mario": _effects_mario,
    # Gym-V Temporal — shooters
    "temporal_airstriker-v0": _effects_gymv_shooter,
    "temporal_spaceharrierii-v0": _effects_gymv_shooter,
    "temporal_thunderforceiii-v0": _effects_gymv_shooter,
    # Gym-V Temporal — brawlers / platformers
    "temporal_alteredbeast-v0": _effects_gymv_brawler,
    "temporal_streetsofrage2-v0": _effects_gymv_brawler,
    "temporal_strider-v0": _effects_gymv_brawler,
    "temporal_dynamiteheaddy-v0": _effects_gymv_brawler,
    # Gym-V Temporal — puzzle
    "temporal_columns-v0": _effects_gymv_columns,
}

# Gym-V wrapper aliases (episode_runner passes ``gymv_thunder_force_iii``
# but _GAME_EFFECT_MAP was keyed only by canonical gym env IDs like
# ``temporal_thunderforceiii-v0``, so game-specific effects such as
# ``enemy_hit``, ``score_increased``, ``damage_taken`` were never
# computed for any gymv game).
_GAME_EFFECT_MAP["gymv_thunder_force_iii"] = _effects_gymv_shooter
_GAME_EFFECT_MAP["gymv_airstriker"] = _effects_gymv_shooter
_GAME_EFFECT_MAP["gymv_space_harrier_ii"] = _effects_gymv_shooter
_GAME_EFFECT_MAP["gymv_altered_beast"] = _effects_gymv_brawler
_GAME_EFFECT_MAP["gymv_streets_of_rage_2"] = _effects_gymv_brawler
_GAME_EFFECT_MAP["gymv_strider"] = _effects_gymv_brawler
_GAME_EFFECT_MAP["gymv_dynamite_headdy"] = _effects_gymv_brawler
_GAME_EFFECT_MAP["gymv_columns"] = _effects_gymv_columns


# ── Auto step_check generation from trajectories ────────────────────

def generate_step_checks_from_effects(
    protocol_steps: List[str],
    game_name: str = "",
) -> List[str]:
    """Generate step_checks for game skills based on step descriptions.

    Parses each protocol step description and maps keywords to known
    game effect predicates.  This replaces the empty step_checks in
    game skill banks with observable, deterministic predicates.
    """
    gn = game_name.lower().replace(" ", "_")
    keyword_to_effect = dict(_STEP_KEYWORD_TO_EFFECT.get(gn, {}))
    keyword_to_effect.update(_UNIVERSAL_KEYWORD_TO_EFFECT)

    checks: List[str] = []
    for step_desc in protocol_steps:
        desc_lower = (step_desc or "").lower()
        matched_effect = ""
        for keyword, effect in keyword_to_effect.items():
            if keyword in desc_lower:
                matched_effect = effect
                break
        checks.append(matched_effect)

    return checks


# ── Game-name canonicalization for TASK_EFFECT_SUBSET lookup ────────────
# Runtime code refers to gym-v shooters/brawlers using the wrapper
# naming convention (e.g. ``gymv_thunder_force_iii``) while
# ``TASK_EFFECT_SUBSET`` is keyed by the canonical gym env id
# (``temporal_thunderforceiii-v0``).  Without this alias table the
# fuzzy substring match in ``get_valid_effects`` /
# ``repair_step_checks_against_registry`` falls through to the global
# ``EFFECT_REGISTRY`` and silently accepts cross-domain predicates
# (``dom_changed``, ``element_clicked``, ``board_transformed``…) that
# can never fire in a Sega Genesis shooter, killing
# ``intrinsic_bonus`` for every action step.

_GAMEV_KEY_ALIASES: Dict[str, str] = {
    # Gym-V Temporal shooters
    "gymv_airstriker":         "temporal_airstriker-v0",
    "gymv_space_harrier_ii":   "temporal_spaceharrierii-v0",
    "gymv_thunder_force_iii":  "temporal_thunderforceiii-v0",
    # Gym-V Temporal brawlers / platformers
    "gymv_altered_beast":      "temporal_alteredbeast-v0",
    "gymv_streets_of_rage_2":  "temporal_streetsofrage2-v0",
    "gymv_strider":            "temporal_strider-v0",
    "gymv_dynamite_headdy":    "temporal_dynamiteheaddy-v0",
    # Gym-V Temporal puzzle
    "gymv_columns":            "temporal_columns-v0",
}


def canonicalize_game_key(game_name: str) -> str:
    """Return the ``TASK_EFFECT_SUBSET`` lookup key for *game_name*.

    Resolution order:
      1. Exact alias hit in ``_GAMEV_KEY_ALIASES`` (handles all the
         ``gymv_*`` wrapper names → canonical gym env ids).
      2. Exact key hit in ``TASK_EFFECT_SUBSET``.
      3. Substring fuzzy match against ``TASK_EFFECT_SUBSET`` keys
         (preserves the legacy behaviour for non-wrapper names like
         ``2048``, ``twenty_forty_eight``).
      4. Returns the input unchanged if nothing matches; callers
         then fall back to the global ``EFFECT_REGISTRY``.
    """
    if not game_name:
        return ""
    gn = game_name.lower().replace(" ", "_")
    aliased = _GAMEV_KEY_ALIASES.get(gn)
    if aliased and aliased in TASK_EFFECT_SUBSET:
        return aliased
    if gn in TASK_EFFECT_SUBSET:
        return gn
    for key in TASK_EFFECT_SUBSET:
        if key in gn or gn in key:
            return key
    return gn


# ── Closed-set step_check validation for newly mined protocols ──────────
# Used by skill_agents.pipeline._llm_synthesize_protocol to repair LoRA-
# generated step_checks that hallucinate predicates outside the game's
# closed effect registry (e.g. "shield_buff=active" for TF3).

def _extract_predicate_key(check: str) -> str:
    """Pull the key part of a 'key=value' / 'key>N' / 'key<N' predicate."""
    if not check:
        return ""
    s = check.strip()
    for op in ("=", ">=", "<=", ">", "<", "!="):
        if op in s:
            return s.split(op, 1)[0].strip().lower()
    return s.lower()


def repair_step_checks_against_registry(
    step_checks: List[str],
    protocol_steps: List[str],
    game_name: str = "",
) -> Tuple[List[str], bool]:
    """Validate ``step_checks`` against the game's closed effect set.

    Returns ``(repaired_checks, was_repaired)``.

    A predicate is considered valid if its key belongs to either
    ``TASK_EFFECT_SUBSET[<game_key>]`` (preferred) or the global
    ``EFFECT_REGISTRY``.  Empty strings are allowed (= "no check for
    this step").  If *any* check has an off-registry key, the entire
    list is regenerated via ``generate_step_checks_from_effects`` so
    that runtime ``compute_step_advancement`` can actually evaluate it.

    Rationale: ``contract`` / ``curator`` LoRAs synthesise protocols
    free-form and frequently produce predicates that don't exist in the
    game state dict (e.g. ``shield_buff``, ``target_health``).  Without
    repair these checks never fire, leaving ``r_progress=0`` for every
    newly-mined skill.
    """
    if not step_checks:
        return step_checks, False

    canonical = canonicalize_game_key(game_name or "")
    allowed: set = set()
    if canonical in TASK_EFFECT_SUBSET:
        allowed = set(TASK_EFFECT_SUBSET[canonical])
    if not allowed:
        # No game-specific subset → can't safely validate. Fall back
        # to the global registry, but mark every off-subset predicate
        # as "needs repair" so the regeneration path is exercised
        # (the legacy code accepted *anything* in the global registry
        # which silently passed cross-domain predicates like
        # ``dom_changed`` and ``element_clicked`` into TF3 protocols).
        allowed = set(EFFECT_REGISTRY.keys())

    # Predicates valid only if their key is in the game-specific
    # subset (or if we had to fall back to EFFECT_REGISTRY because the
    # game key was unknown).  Cross-domain predicates whose keys are
    # in EFFECT_REGISTRY but NOT in the game subset must trigger
    # repair — that was the silent-pass bug for gymv_*.
    needs_repair = False
    for chk in step_checks:
        if not chk:
            continue
        k = _extract_predicate_key(chk)
        if k and k not in allowed:
            needs_repair = True
            break

    if not needs_repair:
        return step_checks, False

    repaired = generate_step_checks_from_effects(protocol_steps, game_name)
    while len(repaired) < len(protocol_steps):
        repaired.append("")
    return repaired[:len(protocol_steps)], True


def filter_predicates_against_registry(
    predicates: List[str],
    game_name: str = "",
) -> Tuple[List[str], int]:
    """Drop cross-domain predicates from ``predicate_success`` /
    ``predicate_abort`` lists.

    Returns ``(filtered_predicates, n_dropped)``.

    A predicate is *kept* iff its key (the LHS of ``key=value`` /
    ``key>N`` / etc.) appears in
    ``TASK_EFFECT_SUBSET[canonicalize_game_key(game_name)]``.  If no
    game-specific subset is registered the predicate is kept (legacy
    behaviour for unknown games); otherwise off-subset predicates are
    dropped so they can't poison ``StepTracker`` (which would never
    fire them in this game).

    Unlike ``repair_step_checks_against_registry`` this function does
    NOT regenerate replacements — the goal/abort gates carry the
    intent of the LLM-synthesised protocol and we'd rather have a
    short valid list than a regenerated one.  Empty result is allowed
    and downstream code must tolerate it (StepTracker treats empty
    ``predicate_success`` as "no intrinsic-bonus gate", which is
    correct in this context).
    """
    if not predicates:
        return predicates, 0

    canonical = canonicalize_game_key(game_name or "")
    if canonical not in TASK_EFFECT_SUBSET:
        # Unknown game — can't safely filter without risking dropping
        # everything.  Leave the list intact.
        return list(predicates), 0

    allowed = set(TASK_EFFECT_SUBSET[canonical])
    kept: List[str] = []
    dropped = 0
    for pred in predicates:
        if not pred:
            continue
        k = _extract_predicate_key(pred)
        if k and k in allowed:
            kept.append(pred)
        else:
            dropped += 1
    return kept, dropped


_UNIVERSAL_KEYWORD_TO_EFFECT: Dict[str, str] = {
    "observe": "state_observed=true",
    "inspect": "state_observed=true",
    "look": "state_observed=true",
    "identify": "state_observed=true",
    "assess": "state_observed=true",
    "confirm": "reward_positive=true",
    "verify": "reward_positive=true",
    "validate": "reward_positive=true",
}

_STEP_KEYWORD_TO_EFFECT: Dict[str, Dict[str, str]] = {
    "twenty_forty_eight": {
        "enumerate": "state_observed=true",
        "discard": "state_observed=true",
        "choose": "action_taken=true",
        "apply": "board_transformed=true",
        "transform": "board_transformed=true",
        "merge": "merge_executed=true",
        "confirm": "board_transformed=true",
        "promote": "tile_promoted=true",
        "corner": "direction_applied=true",
    },
    "2048": {
        "enumerate": "state_observed=true",
        "discard": "state_observed=true",
        "choose": "action_taken=true",
        "apply": "board_transformed=true",
        "transform": "board_transformed=true",
        "merge": "merge_executed=true",
        "confirm": "board_transformed=true",
    },
    "tetris": {
        "position": "state_observed=true",
        "rotate": "move_applied=true",
        "shift": "move_applied=true",
        "place": "piece_placed=true",
        "drop": "piece_placed=true",
        "clear": "line_cleared=true",
        "minimize": "holes_reduced=true",
        "stack": "piece_placed=true",
    },
    "candy_crush": {
        "scan": "state_observed=true",
        "find": "state_observed=true",
        "swap": "swap_applied=true",
        "match": "match_scored=true",
        "cascade": "match_scored=true",
        "chain": "match_scored=true",
    },
    "super_mario": {
        "scan": "state_observed=true",
        "jump": "mario_moved=true",
        "move": "mario_moved=true",
        "run": "mario_moved=true",
        "avoid": "action_executed=true",
        "collect": "progress_made=true",
    },
    # ── Gym-V Temporal shooters (Airstriker, SpaceHarrierII, ThunderForceIII)
    # Effect tags drawn from TASK_EFFECT_SUBSET[temporal_<game>-v0]:
    # state_observed, action_taken, action_executed, projectile_fired,
    # enemy_hit, damage_taken, score_increased, reward_positive,
    # cumulative_reward_positive.
    "temporal_airstriker-v0": {
        "observe": "state_observed=true", "inspect": "state_observed=true",
        "scan": "state_observed=true", "identify": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "compare": "state_observed=true", "explore": "state_observed=true",
        "recover": "state_observed=true",
        "choose": "action_taken=true", "decide": "action_taken=true",
        "select": "action_taken=true", "filter": "action_taken=true",
        "execute": "action_executed=true", "issue": "action_executed=true",
        "apply": "action_executed=true", "commit": "action_executed=true",
        "move": "action_executed=true", "navigate": "action_executed=true",
        "position": "action_executed=true",
        "shoot": "projectile_fired=true", "fire": "projectile_fired=true",
        "attack": "enemy_hit=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "advance": "score_increased=true", "progress": "score_increased=true",
    },
    "temporal_spaceharrierii-v0": {
        "observe": "state_observed=true", "inspect": "state_observed=true",
        "scan": "state_observed=true", "identify": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "compare": "state_observed=true", "explore": "state_observed=true",
        "recover": "state_observed=true",
        "choose": "action_taken=true", "decide": "action_taken=true",
        "select": "action_taken=true",
        "execute": "action_executed=true", "issue": "action_executed=true",
        "apply": "action_executed=true", "commit": "action_executed=true",
        "move": "action_executed=true", "navigate": "action_executed=true",
        "position": "action_executed=true",
        "shoot": "projectile_fired=true", "fire": "projectile_fired=true",
        "attack": "enemy_hit=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "advance": "score_increased=true", "progress": "score_increased=true",
    },
    "temporal_thunderforceiii-v0": {
        "observe": "state_observed=true", "inspect": "state_observed=true",
        "scan": "state_observed=true", "identify": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "compare": "state_observed=true", "explore": "state_observed=true",
        "recover": "state_observed=true",
        "choose": "action_taken=true", "decide": "action_taken=true",
        "select": "action_taken=true", "filter": "action_taken=true",
        "execute": "action_executed=true", "issue": "action_executed=true",
        "apply": "action_executed=true", "commit": "action_executed=true",
        "move": "action_executed=true", "navigate": "action_executed=true",
        "position": "action_executed=true",
        "shoot": "projectile_fired=true", "fire": "projectile_fired=true",
        "attack": "enemy_hit=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "advance": "score_increased=true", "progress": "score_increased=true",
    },
    # ── Gym-V Temporal brawlers (AlteredBeast, StreetsOfRage2, Strider, DynamiteHeaddy)
    # Effect tags: state_observed, action_taken, action_executed, attack_landed,
    # enemy_hit, damage_taken, position_changed, score_increased,
    # reward_positive, cumulative_reward_positive.
    "temporal_alteredbeast-v0": {
        "observe": "state_observed=true", "inspect": "state_observed=true",
        "scan": "state_observed=true", "look": "state_observed=true",
        "assess": "state_observed=true", "explore": "state_observed=true",
        "recover": "state_observed=true",
        "choose": "action_taken=true", "decide": "action_taken=true",
        "select": "action_taken=true",
        "execute": "action_executed=true", "issue": "action_executed=true",
        "apply": "action_executed=true", "commit": "action_executed=true",
        "attack": "attack_landed=true", "strike": "attack_landed=true",
        "punch": "attack_landed=true", "kick": "attack_landed=true",
        "hit": "enemy_hit=true",
        "shoot": "attack_landed=true", "fire": "attack_landed=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "block": "damage_taken=false",
        "navigate": "position_changed=true", "move": "position_changed=true",
        "advance": "position_changed=true", "position": "position_changed=true",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "progress": "score_increased=true",
    },
    "temporal_streetsofrage2-v0": {
        "observe": "state_observed=true", "scan": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "explore": "state_observed=true", "recover": "state_observed=true",
        "choose": "action_taken=true", "select": "action_taken=true",
        "execute": "action_executed=true", "apply": "action_executed=true",
        "commit": "action_executed=true",
        "attack": "attack_landed=true", "strike": "attack_landed=true",
        "punch": "attack_landed=true", "kick": "attack_landed=true",
        "hit": "enemy_hit=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "block": "damage_taken=false",
        "navigate": "position_changed=true", "move": "position_changed=true",
        "advance": "position_changed=true", "position": "position_changed=true",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "progress": "score_increased=true",
    },
    "temporal_strider-v0": {
        "observe": "state_observed=true", "scan": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "explore": "state_observed=true", "recover": "state_observed=true",
        "choose": "action_taken=true", "select": "action_taken=true",
        "execute": "action_executed=true", "apply": "action_executed=true",
        "commit": "action_executed=true",
        "attack": "attack_landed=true", "strike": "attack_landed=true",
        "slash": "attack_landed=true", "hit": "enemy_hit=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "navigate": "position_changed=true", "move": "position_changed=true",
        "advance": "position_changed=true", "jump": "position_changed=true",
        "climb": "position_changed=true", "position": "position_changed=true",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "progress": "score_increased=true",
    },
    "temporal_dynamiteheaddy-v0": {
        "observe": "state_observed=true", "scan": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "explore": "state_observed=true", "recover": "state_observed=true",
        "choose": "action_taken=true", "select": "action_taken=true",
        "execute": "action_executed=true", "apply": "action_executed=true",
        "commit": "action_executed=true",
        "attack": "attack_landed=true", "hit": "enemy_hit=true",
        "throw": "attack_landed=true", "punch": "attack_landed=true",
        "evade": "damage_taken=false", "dodge": "damage_taken=false",
        "navigate": "position_changed=true", "move": "position_changed=true",
        "advance": "position_changed=true", "jump": "position_changed=true",
        "position": "position_changed=true",
        "survive": "cumulative_reward_positive=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "progress": "score_increased=true",
    },
    # ── Gym-V Temporal puzzle (Columns)
    # Effect tags: state_observed, action_taken, action_executed,
    # piece_placed, piece_rotated, match_scored, board_transformed,
    # score_increased, reward_positive, cumulative_reward_positive.
    "temporal_columns-v0": {
        "observe": "state_observed=true", "scan": "state_observed=true",
        "look": "state_observed=true", "assess": "state_observed=true",
        "inspect": "state_observed=true", "compare": "state_observed=true",
        "explore": "state_observed=true", "recover": "state_observed=true",
        "choose": "action_taken=true", "select": "action_taken=true",
        "decide": "action_taken=true",
        "execute": "action_executed=true", "apply": "action_executed=true",
        "commit": "action_executed=true",
        "drop": "piece_placed=true", "place": "piece_placed=true",
        "stack": "piece_placed=true",
        "rotate": "piece_rotated=true",
        "match": "match_scored=true", "clear": "match_scored=true",
        "transform": "board_transformed=true",
        "verify": "reward_positive=true", "confirm": "reward_positive=true",
        "advance": "score_increased=true", "progress": "score_increased=true",
        "survive": "cumulative_reward_positive=true",
    },
}

# ── Runtime aliases: per_task_banks dir name → canonical TASK_EFFECT_SUBSET key
# Co-evolution receives game_name in the gymv_* form; we point those at the
# canonical Temporal_<game>-v0 keyword map above.
_STEP_KEYWORD_TO_EFFECT["gymv_thunder_force_iii"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_thunderforceiii-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_airstriker"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_airstriker-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_space_harrier_ii"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_spaceharrierii-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_altered_beast"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_alteredbeast-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_streets_of_rage_2"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_streetsofrage2-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_strider"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_strider-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_dynamite_headdy"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_dynamiteheaddy-v0"
]
_STEP_KEYWORD_TO_EFFECT["gymv_columns"] = _STEP_KEYWORD_TO_EFFECT[
    "temporal_columns-v0"
]


# ── QA (multi-hop) step state computation ────────────────────────────

_QA_HOP_TO_PHASE: Dict[str, str] = {
    "GROUND":   "grounding",
    "RETRIEVE": "retrieval",
    "CHECK":    "verification",
    "VERIFY":   "verification",
    "COMMIT":   "answering",
    "COMPARE":  "comparison",
    "FILTER":   "filtering",
    "PERCEIVE": "perception",
    "RECALL":   "retrieval",
    "DECIDE":   "decision",
}

_QA_PHASE_ORDER = [
    "perception", "grounding", "retrieval",
    "comparison", "filtering", "verification",
    "decision", "answering",
]

_QA_HOP_TO_EXTRA_EFFECT: Dict[str, str] = {
    "PERCEIVE": "state_observed",
    "GROUND":   "evidence_cited",
    "RETRIEVE": "context_retrieved",
    "RECALL":   "context_retrieved",
    "COMPARE":  "hypothesis_formed",
    "FILTER":   "candidates_eliminated",
    "CHECK":    "answer_confirmed",
    "VERIFY":   "answer_confirmed",
    "DECIDE":   "answer_selected",
    "COMMIT":   "answer_emitted",
}


def compute_qa_step_state(
    hop_history: List[str],
    protocol_steps: List[str],
) -> Tuple[int, Dict[str, str]]:
    """Compute the current protocol step index for a QA multi-hop task.

    Uses a deterministic mapping from executed hop types to protocol
    step phases, replacing the regex-based approach.

    Parameters
    ----------
    hop_history : list[str]
        Ordered list of executed hop action types, e.g.
        ``["GROUND", "CHECK", "VERIFY", "COMMIT"]``.
    protocol_steps : list[str]
        The protocol step descriptions from the skill bank,
        e.g. ``["Perceive visual elements", "Compare options", ...]``.

    Returns
    -------
    step_idx : int
        The highest protocol step that has been reached (0-based).
    state_dict : dict[str, str]
        A state dict compatible with ``check_predicate``, including
        keys like ``evidence_cited=true``, ``answer_emitted=true``.
    """
    if not hop_history or not protocol_steps:
        return 0, {}

    state_dict: Dict[str, str] = {}
    reached_phases: List[str] = []
    for hop in hop_history:
        hop_upper = hop.strip().upper()
        phase = _QA_HOP_TO_PHASE.get(hop_upper)
        if phase and phase not in reached_phases:
            reached_phases.append(phase)
        effect = OPERATOR_TO_EFFECT.get(hop_upper, "")
        if effect:
            state_dict[effect] = "true"
        extra = _QA_HOP_TO_EXTRA_EFFECT.get(hop_upper, "")
        if extra:
            state_dict[extra] = "true"

    if hop_history:
        state_dict["action_taken"] = "true"
        state_dict["state_observed"] = "true"

    state_dict["hops_completed"] = str(len(hop_history))

    step_idx = 0
    for i, step_desc in enumerate(protocol_steps):
        desc_lower = step_desc.lower()
        for phase in reached_phases:
            if phase in desc_lower or _phase_keyword_in(phase, desc_lower):
                step_idx = max(step_idx, i)

    return step_idx, state_dict


def _phase_keyword_in(phase: str, text: str) -> bool:
    """Check if a phase's characteristic keywords appear in text."""
    phase_keywords: Dict[str, List[str]] = {
        "perception":   ["perceive", "observe", "look", "identify", "visual"],
        "grounding":    ["ground", "locate", "find", "detect", "bbox"],
        "retrieval":    ["retrieve", "recall", "fetch", "search", "gather"],
        "comparison":   ["compare", "contrast", "differ", "similar"],
        "filtering":    ["filter", "eliminate", "narrow", "exclude", "discard"],
        "verification": ["verify", "check", "confirm", "validate"],
        "decision":     ["decide", "choose", "select", "pick"],
        "answering":    ["answer", "commit", "output", "respond", "final"],
    }
    kws = phase_keywords.get(phase, [])
    return any(kw in text for kw in kws)


# ── Web (BrowserGym) step state computation ──────────────────────────

def compute_web_step_state(
    action_history: List[str],
    dom_change_flags: List[bool],
    protocol_steps: List[str],
) -> Tuple[int, Dict[str, str]]:
    """Compute the current protocol step index for a web task.

    Parameters
    ----------
    action_history : list[str]
        Ordered list of browsergym action types executed so far,
        e.g. ``["click", "fill", "click", "scroll"]``.
    dom_change_flags : list[bool]
        Whether each action produced a meaningful DOM change.
    protocol_steps : list[str]
        The protocol step descriptions from the skill bank.

    Returns
    -------
    step_idx : int
        The highest protocol step that has been reached (0-based).
    state_dict : dict[str, str]
        A state dict compatible with ``check_predicate``.
    """
    if not action_history or not protocol_steps:
        return 0, {}

    state_dict: Dict[str, str] = {}
    state_dict["actions_taken"] = str(len(action_history))
    meaningful = sum(1 for f in dom_change_flags if f)
    state_dict["dom_changes"] = str(meaningful)

    fill_count = sum(1 for a in action_history if "fill" in a.lower() or "type" in a.lower())
    click_count = sum(1 for a in action_history if "click" in a.lower())
    nav_count = sum(1 for a in action_history if "goto" in a.lower() or "navigate" in a.lower())
    search_count = sum(1 for a in action_history if "search" in a.lower())
    state_dict["fills"] = str(fill_count)
    state_dict["clicks"] = str(click_count)
    state_dict["navigations"] = str(nav_count)

    step_idx = 0
    n_steps = len(protocol_steps)
    if n_steps > 0 and meaningful > 0:
        step_idx = min(n_steps - 1, meaningful - 1)

    return step_idx, state_dict


def compute_web_effects(
    action_history: List[str],
    dom_change_flags: List[bool],
) -> Dict[str, str]:
    """Compute effect tags for web tasks from action history.

    Produces the same tag vocabulary used in ``TASK_EFFECT_SUBSET``
    for miniwob / webshop, so the LoRA can be trained on grounded
    observations.
    """
    effects: Dict[str, str] = {}
    if not action_history:
        return effects

    effects["action_taken"] = "true"

    for act in action_history:
        act_l = act.lower()
        if "click" in act_l:
            effects["element_clicked"] = "true"
        if "fill" in act_l or "type" in act_l:
            effects["form_filled"] = "true"
        if "goto" in act_l or "navigate" in act_l:
            effects["page_navigated"] = "true"
        if "search" in act_l:
            effects["search_performed"] = "true"

    if dom_change_flags and any(dom_change_flags):
        effects["dom_changed"] = "true"

    return effects


# ══════════════════════════════════════════════════════════════════════
# CANONICAL EFFECT TAG REGISTRY
# ──────────────────────────────────────────────────────────────────────
# Closed-set vocabulary for the skill-selection LoRA's EFFECTS output.
# The LoRA picks a subset of these per step; no free-form tags allowed.
#
# Design criteria
# ───────────────
# 1. Every tag MUST be independently observable (no LoRA self-report).
# 2. Tag names are short, verb_noun style, all lowercase + underscores.
# 3. The set is domain-agnostic: each game / QA / web task uses a
#    subset; the LoRA learns the mapping from (state, task_type) to
#    the relevant subset.
# ══════════════════════════════════════════════════════════════════════

EFFECT_REGISTRY: Dict[str, str] = {
    # ══ Universal (every domain) ══════════════════════════════════════
    "state_observed":            "Agent has perceived / inspected current state",
    "action_taken":              "Agent executed at least one action this turn",
    "action_executed":           "A domain-specific action was performed",
    "reward_positive":           "Positive reward received this step",
    "cumulative_reward_positive":"Sum of rewards across skill so far > 0",
    "score_increased":           "Numeric score went up",

    # ══ Reasoning / QA ════════════════════════════════════════════════
    # (video_holmes, siv_bench, tir_bench, visual_toolbench)
    "evidence_cited":            "Relevant visual or textual evidence extracted",
    "hypothesis_formed":         "A candidate hypothesis / interpretation stated",
    "context_retrieved":         "External or temporal context recalled / fetched",
    "options_compared":          "Multiple candidate answers compared",
    "candidates_eliminated":     "At least one wrong option ruled out",
    "answer_selected":           "A single best answer chosen",
    "answer_emitted":            "Final answer committed / output produced",
    "answer_confirmed":          "Answer cross-checked against evidence",

    # ══ Board / puzzle (2048, tetris, candy_crush, Columns) ═══════════
    "board_transformed":         "Board layout changed from previous state",
    "board_crowded":             "Board is near capacity / few open cells",
    "board_reshuffled":          "Board was reshuffled or cascaded",
    "tile_promoted":             "Highest tile value increased (2048)",
    "merge_executed":            "A merge / combine occurred (2048)",
    "direction_applied":         "A directional move was applied (2048)",
    "piece_placed":              "A piece was placed on the board (tetris/columns)",
    "piece_changed":             "Active piece changed / new piece spawned (tetris)",
    "piece_rotated":             "A piece was rotated (columns)",
    "line_cleared":              "One or more lines cleared (tetris)",
    "holes_reduced":             "Board holes decreased (tetris)",
    "holes_created":             "Board holes increased (tetris, negative signal)",
    "move_applied":              "A move/shift/rotate was executed (tetris)",
    "match_scored":              "A match / cascade scored points (candy/columns)",
    "swap_applied":              "A swap action performed (candy crush)",
    "move_spent":                "A move resource was consumed (candy crush)",

    # ══ Platformer / action (mario, Strider, AlteredBeast, …) ═════════
    "position_changed":          "Agent position moved to a new location",
    "mario_moved":               "Mario character moved (super_mario specific)",
    "progress_made":             "Forward progress toward goal",
    "damage_taken":              "Agent took damage / lost health",
    "obstacle_cleared":          "An obstacle or hazard was successfully avoided",
    "collectible_obtained":      "A coin / power-up / item was collected",

    # ══ Shooter / combat (Airstriker, SpaceHarrier, ThunderForce, …) ══
    "enemy_hit":                 "An enemy was hit / destroyed",
    "projectile_fired":          "Agent fired a projectile",
    "attack_landed":             "A melee / ranged attack connected",

    # ══ Web interaction (miniwob, webshop) ════════════════════════════
    "page_navigated":            "Browser navigated to a new URL / page",
    "form_filled":               "A text input / form field was filled",
    "element_clicked":           "A UI element was clicked",
    "dom_changed":               "Page DOM changed meaningfully after action",
    "item_found":                "Target item / element located on page",
    "search_performed":          "A search query was submitted",
    "product_selected":          "A product / option was chosen from results",
    "cart_updated":              "Shopping cart was modified (webshop)",
}

EFFECT_TAGS: List[str] = sorted(EFFECT_REGISTRY.keys())

TASK_EFFECT_SUBSET: Dict[str, List[str]] = {
    # ── Classic games ─────────────────────────────────────────────────
    "twenty_forty_eight": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "board_transformed", "board_crowded",
        "tile_promoted", "merge_executed", "direction_applied",
    ],
    "tetris": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "board_transformed", "piece_placed", "piece_changed",
        "line_cleared", "holes_reduced", "holes_created", "move_applied",
    ],
    "candy_crush": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "board_transformed", "board_reshuffled",
        "match_scored", "swap_applied", "move_spent",
    ],
    "super_mario": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "position_changed", "mario_moved", "progress_made",
        "damage_taken", "obstacle_cleared", "collectible_obtained",
    ],
    # ── Gym-V Temporal — shooters ─────────────────────────────────────
    "temporal_airstriker-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "projectile_fired", "damage_taken",
    ],
    "temporal_spaceharrierii-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "projectile_fired", "damage_taken",
    ],
    "temporal_thunderforceiii-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "projectile_fired", "damage_taken",
    ],
    # ── Gym-V Temporal — brawlers / platformers ───────────────────────
    "temporal_alteredbeast-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "attack_landed", "position_changed", "damage_taken",
    ],
    "temporal_streetsofrage2-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "attack_landed", "position_changed", "damage_taken",
    ],
    "temporal_strider-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "attack_landed", "position_changed", "damage_taken",
    ],
    "temporal_dynamiteheaddy-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "enemy_hit", "attack_landed", "position_changed", "damage_taken",
    ],
    # ── Gym-V Temporal — puzzle ───────────────────────────────────────
    "temporal_columns-v0": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "board_transformed", "match_scored",
        "piece_placed", "piece_rotated",
    ],
    # ── QA reasoning ──────────────────────────────────────────────────
    "video_holmes": [
        "state_observed", "action_taken",
        "evidence_cited", "hypothesis_formed", "context_retrieved",
        "options_compared", "candidates_eliminated",
        "answer_selected", "answer_emitted", "answer_confirmed",
    ],
    "siv_bench": [
        "state_observed", "action_taken",
        "evidence_cited", "hypothesis_formed", "context_retrieved",
        "options_compared", "candidates_eliminated",
        "answer_selected", "answer_emitted", "answer_confirmed",
    ],
    "tir_bench": [
        "state_observed", "action_taken",
        "evidence_cited", "hypothesis_formed", "context_retrieved",
        "options_compared", "candidates_eliminated",
        "answer_selected", "answer_emitted", "answer_confirmed",
    ],
    "visual_toolbench": [
        "state_observed", "action_taken",
        "evidence_cited", "hypothesis_formed", "context_retrieved",
        "options_compared", "candidates_eliminated",
        "answer_selected", "answer_emitted", "answer_confirmed",
    ],
    # ── Web interaction ───────────────────────────────────────────────
    "miniwob": [
        "state_observed", "action_taken",
        "evidence_cited", "options_compared",
        "page_navigated", "form_filled", "element_clicked",
        "dom_changed", "item_found", "answer_emitted",
    ],
    "webshop": [
        "state_observed", "action_taken",
        "evidence_cited", "candidates_eliminated",
        "page_navigated", "form_filled", "element_clicked",
        "dom_changed", "item_found", "answer_emitted",
        "search_performed", "product_selected", "cart_updated",
    ],
}


def get_valid_effects(task_name: str) -> List[str]:
    """Return the closed set of valid effect tags for a given task.

    Uses :func:`canonicalize_game_key` to map wrapper names like
    ``gymv_thunder_force_iii`` onto the registry key
    ``temporal_thunderforceiii-v0``.  Falls back to ``EFFECT_TAGS``
    (the full global set) only if no game-specific subset can be
    identified.
    """
    canonical = canonicalize_game_key(task_name)
    if canonical in TASK_EFFECT_SUBSET:
        return TASK_EFFECT_SUBSET[canonical]
    return EFFECT_TAGS
