"""QA / visual-reasoning / browser-task vocabulary extension.

Extends ``decision_agents.agent_helper`` with operators and subgoals
that don't make sense for action-game rollouts but DO show up in:

* QA reasoning chains (video_holmes, siv_bench, tir_bench,
  visual_toolbench): the model has to **REASON** over evidence,
  **TOOL_USE** to compute / lookup intermediate facts, and finally
  **COMMIT** an ANSWER.
* MiniWob browser rollouts: largely already covered by the canonical
  vocab (NAVIGATE / EXPLORE / EXECUTE), but a few subgoals like
  FORM_FILL, SUBMIT, LOOKUP fit better under explicit names.

The extended axes are written into ``[OPERATOR/SUBGOAL] note`` exactly
like the canonical labeler — downstream skill_query / SFT readers
(``labeling/label_skill_actions_gpt54.py`` /
``scripts/build_multimodal_decision_sft.py``) treat the bracketed
prefix as opaque, so adding new tags here does NOT break anything.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from decision_agents.agent_helper import (
    INTENT_OPERATORS as _CANON_OPERATORS,
    UNIFIED_SUBGOALS as _CANON_SUBGOALS,
    OPERATOR_TO_SUBGOAL as _CANON_OP2SG,
    SUBGOAL_TO_OPERATOR as _CANON_SG2OP,
)

# QA / web specific operators.  ``REASON`` and ``TOOL_USE`` are the two
# we add; everything else (INSPECT/COMPARE/COMMIT/VERIFY/TRACK/RECOVER)
# we re-use verbatim.
QA_EXTRA_OPERATORS: Tuple[str, ...] = (
    "REASON",   # logical inference, arithmetic, rule-out, deduction
    "TOOL_USE", # invoke a tool: search, calculator, OCR, code-exec, frame-pick, ...
)

QA_EXTRA_SUBGOALS: Tuple[str, ...] = (
    # QA-flavoured cognitive subgoals
    "EVIDENCE",   # ground a claim in a frame / passage / element
    "IDENTIFY",   # name a person / object / scene / element
    "TIMELINE",   # order events / scenes / steps
    "COUNT",      # enumerate how many of X
    "MEASURE",    # quantify duration / size / distance / score
    "LOOKUP",     # external retrieval (search, doc, KG)
    "DEDUCE",     # logical inference from premises
    "RULE_OUT",   # eliminate candidates
    "ANSWER",     # commit the final answer
    # MiniWob / browser-flavoured subgoals
    "FORM_FILL",  # fill an input/textbox/dropdown
    "SUBMIT",     # press submit/confirm/ok button
)

INTENT_OPERATORS_QA: Tuple[str, ...] = _CANON_OPERATORS + QA_EXTRA_OPERATORS
UNIFIED_SUBGOALS_QA: Tuple[str, ...] = _CANON_SUBGOALS + QA_EXTRA_SUBGOALS

# Default subgoal for the new operators (used when LLM returns operator
# only; mirrors ``OPERATOR_TO_SUBGOAL`` upstream).
OPERATOR_TO_SUBGOAL_QA: Dict[str, str] = {
    **_CANON_OP2SG,
    "REASON": "DEDUCE",
    "TOOL_USE": "LOOKUP",
}

SUBGOAL_TO_OPERATOR_QA: Dict[str, str] = {
    **_CANON_SG2OP,
    "EVIDENCE": "INSPECT",
    "IDENTIFY": "INSPECT",
    "TIMELINE": "COMPARE",
    "COUNT": "REASON",
    "MEASURE": "REASON",
    "LOOKUP": "TOOL_USE",
    "DEDUCE": "REASON",
    "RULE_OUT": "REASON",
    "ANSWER": "COMMIT",
    "FORM_FILL": "COMMIT",
    "SUBMIT": "COMMIT",
}

_OP_VALID: FrozenSet[str] = frozenset(INTENT_OPERATORS_QA)
_SG_VALID: FrozenSet[str] = frozenset(UNIFIED_SUBGOALS_QA)

# Common synonym mapping (loose — the LLM tends to drift on near-synonyms).
_OP_SYN: Dict[str, str] = {
    "INFER": "REASON", "DEDUCE": "REASON", "CALCULATE": "REASON",
    "ANALYZE": "REASON", "ANALYSE": "REASON",
    "OBSERVE": "INSPECT", "READ": "INSPECT", "WATCH": "INSPECT",
    "SCAN": "INSPECT", "EXAMINE": "INSPECT", "VIEW": "INSPECT",
    "EVALUATE": "COMPARE", "WEIGH": "COMPARE", "CHOOSE": "COMPARE",
    "DECIDE": "COMMIT", "ACT": "COMMIT", "ANSWER": "COMMIT",
    "SUBMIT": "COMMIT", "CLICK": "COMMIT", "TYPE": "COMMIT",
    "TOOL": "TOOL_USE", "USE_TOOL": "TOOL_USE", "INVOKE": "TOOL_USE",
    "CALL": "TOOL_USE", "QUERY": "TOOL_USE", "SEARCH": "TOOL_USE",
    "RETRIEVE": "TOOL_USE", "LOOKUP": "TOOL_USE",
    "WAIT": "TRACK", "MONITOR": "TRACK",
    "CHECK": "VERIFY", "CONFIRM": "VERIFY", "VALIDATE": "VERIFY",
    "RECOVER": "RECOVER", "REDO": "RECOVER", "RETRY": "RECOVER",
}

_SG_SYN: Dict[str, str] = {
    "EVIDENCE_GATHER": "EVIDENCE", "GROUND": "EVIDENCE",
    "RECOGNIZE": "IDENTIFY", "RECOGNISE": "IDENTIFY", "NAME": "IDENTIFY",
    "ORDER": "TIMELINE", "SEQUENCE": "TIMELINE",
    "ENUMERATE": "COUNT", "TALLY": "COUNT",
    "QUANTIFY": "MEASURE", "ESTIMATE": "MEASURE",
    "RETRIEVE": "LOOKUP", "FETCH": "LOOKUP",
    "INFER": "DEDUCE", "CONCLUDE": "DEDUCE",
    "ELIMINATE": "RULE_OUT", "REJECT": "RULE_OUT",
    "FINAL": "ANSWER", "COMMIT_ANSWER": "ANSWER", "RESPOND": "ANSWER",
    "FILL": "FORM_FILL", "TYPE_INTO": "FORM_FILL",
    "CLICK_SUBMIT": "SUBMIT", "CONFIRM": "SUBMIT",
}


def normalize_operator_qa(raw: str) -> str:
    """Map a free-form operator string to ``INTENT_OPERATORS_QA``."""
    s = (raw or "").strip().upper().strip("[]")
    if not s:
        return "COMMIT"
    if s in _OP_VALID:
        return s
    if s in _OP_SYN:
        return _OP_SYN[s]
    if s in _SG_VALID:
        return SUBGOAL_TO_OPERATOR_QA.get(s, "COMMIT")
    return "COMMIT"


def normalize_subgoal_qa(raw: str, operator: str = "") -> str:
    """Map a free-form subgoal string to ``UNIFIED_SUBGOALS_QA``."""
    s = (raw or "").strip().upper().strip("[]")
    if not s:
        return OPERATOR_TO_SUBGOAL_QA.get(operator, "EXECUTE")
    if s in _SG_VALID:
        return s
    if s in _SG_SYN:
        return _SG_SYN[s]
    if s in _OP_VALID:
        return OPERATOR_TO_SUBGOAL_QA.get(s, "EXECUTE")
    return OPERATOR_TO_SUBGOAL_QA.get(operator, "EXECUTE")


def normalize_dual_tag_qa(obj: Dict) -> Tuple[str, str]:
    """Return canonical ``(operator, subgoal)`` from a parsed JSON object."""
    op_raw = obj.get("operator") or obj.get("op") or ""
    sg_raw = obj.get("subgoal") or obj.get("sg") or ""
    op = normalize_operator_qa(str(op_raw))
    if str(op_raw).strip().upper() in _SG_VALID and not sg_raw:
        sg_raw = op_raw
    sg = normalize_subgoal_qa(str(sg_raw), operator=op)
    return op, sg
