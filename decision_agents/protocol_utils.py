"""Shared utilities for protocol-aware skill lifecycle management.

Provides predicate checking against parsed ``summary_state`` dicts and
progress tracking helpers.  Used by ``_SkillTracker`` in both
``scripts/qwen3_decision_agent.py`` and
``trainer/coevolution/episode_runner.py``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_CMP_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*([<>=!]+)\s*(.+)$"
)


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
    """
    m = _CMP_RE.match(pred.strip())
    if not m:
        return False
    key, op, expected = m.group(1), m.group(2), m.group(3).strip()
    actual = state.get(key)
    if actual is None:
        return False

    if op == "==" or op == "=":
        return actual == expected
    if op == "!=":
        return actual != expected

    try:
        a_num = float(actual)
        e_num = float(expected)
    except (ValueError, TypeError):
        return False

    if op == ">":
        return a_num > e_num
    if op == "<":
        return a_num < e_num
    if op == ">=":
        return a_num >= e_num
    if op == "<=":
        return a_num <= e_num
    return False


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
    state_dict["fills"] = str(fill_count)
    state_dict["clicks"] = str(click_count)
    state_dict["navigations"] = str(nav_count)

    step_idx = 0
    n_steps = len(protocol_steps)
    if n_steps > 0 and meaningful > 0:
        step_idx = min(n_steps - 1, meaningful - 1)

    return step_idx, state_dict
