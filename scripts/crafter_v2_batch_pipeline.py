#!/usr/bin/env python3
"""Crafter v2 batch pipeline — produce candidate skills from a completed
phase's rollout artifacts and emit them as injection-ready SkillRecord
JSONL.

Pipeline (offline, read-only against the live run; writes to
``<run_dir>/crafter_v2_offline/``):

  1. Load all action_taking + skill_selection grpo_data + reward_log
     for the target game across every phase step.
  2. Detect enriched failures (USELESS_ACTION_WASTE, ZERO_REWARD_STREAK,
     EARLY_DEATH, SHARP_NEGATIVE) → ``enriched_failures/all_failures.jsonl``.
  3. Bucket failures by (failure_class, in-episode phase prefix) and
     send each bucket (≤16 failures) to the 35B proposer with the
     existing skill bank rendered as "do-not-duplicate" context.
  4. Aggregate proposals across buckets, dedup by name / cosine
     similarity against existing bank descriptions (sentence-bag model
     fallback when no embedding model is configured).
  5. Lift each surviving proposal into a full Skill record
     (``confidence_tag="crafter_v2"``, ``feasible_tasks=[game]``,
     ``derived_from=null``) and write to
     ``proposals/candidate_skills.jsonl``.

The output JSONL is in the SAME format as
``skillbank/<game>/skill_bank.jsonl`` so ``phase1_finalize.py`` can
``cat`` it onto the live bank without further surgery.

Usage::

    python scripts/crafter_v2_batch_pipeline.py \\
        --run-dir runs/Qwen3.5-9B_<ts> \\
        --game gymv_thunder_force_iii \\
        --bucket-size 16 --max-buckets 8

When ``--max-buckets`` is hit early the pipeline stops calling 35B but
still writes the partial proposal set + a summary file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the prototype's failure extraction (load dynamically so we don't
# duplicate parser logic).  We DO NOT import ``PROPOSER_SYSTEM`` from the
# prototype anymore — that prompt asked the LLM for *concrete* button-
# level output ("name specific buttons B, A, C, UP…"), which the
# 2026-05-08 abstractness audit (`scripts/audit_skill_abstractness.py`)
# showed produced skills with **0.5% abstract-share** vs **92% for
# foundry-mined skills**.  The new prompt below enforces game-agnostic
# operator / predicate / tag vocabularies via an explicit JSON schema.
from scripts.crafter_v2_extract_and_probe import (
    detect_failures,
    load_action_taking,
    load_episode_outcomes,
    load_reward_log,
    load_skill_bank,
    load_skill_selection,
    render_skill_bank_for_prompt,
    render_failures_for_prompt,
    call_35b_proposer,
)

# Canonical vocabularies — proposals MUST stay within these.
try:
    from decision_agents.agent_helper import (  # type: ignore
        INTENT_OPERATORS as _OPERATORS_CANON,
        UNIFIED_SUBGOALS as _SUBGOALS_CANON,
    )
    from labeling.qa_vocab import (  # type: ignore
        QA_EXTRA_OPERATORS, QA_EXTRA_SUBGOALS,
    )
    OPERATORS: List[str] = list(_OPERATORS_CANON) + list(QA_EXTRA_OPERATORS)
    SUBGOALS: List[str] = list(_SUBGOALS_CANON) + list(QA_EXTRA_SUBGOALS)
except Exception:                                                 # noqa: BLE001
    OPERATORS = ["INSPECT", "TRACK", "COMPARE", "COMMIT", "RECOVER",
                 "VERIFY", "REASON", "TOOL_USE"]
    SUBGOALS = ["SETUP", "NAVIGATE", "POSITION", "CLEAR", "MERGE",
                "COLLECT", "BUILD", "ATTACK", "DEFEND", "EVADE",
                "OPTIMIZE", "SURVIVE", "EXPLORE", "EXECUTE",
                "EVIDENCE", "IDENTIFY", "TIMELINE", "COUNT", "MEASURE",
                "LOOKUP", "DEDUCE", "RULE_OUT", "ANSWER",
                "FORM_FILL", "SUBMIT"]


CONFIDENCE_TAG_CRAFTER_V2 = "crafter_v2"


# -------------------------- abstract-only LLM contract ------------------


# Tokens that NEVER belong in an abstract skill. If any field contains one
# of these (case-insensitive whole-word match), the proposal is dropped.
# Curated from the 2026-05-08 audit's leak vocabulary.
_BANNED_TOKENS_BUTTON = {
    # Genesis / generic console buttons + glyphs
    "press", "tap", "hold", "release", "joystick", "joypad", "dpad",
    "d-pad", "trigger", "stick",
    # Single-letter button glyphs (whole-word; we use \b boundaries)
    "a", "b", "c", "x", "y", "z", "l", "r",
    "up", "down", "left", "right",
    "start", "select",
}
_BANNED_TOKENS_UI = {
    "menu", "screen", "panel", "dialog", "popup", "dropdown",
    "form", "textbox", "checkbox", "toolbar", "sidebar", "modal",
    "input field", "submit button",
}
_BANNED_TOKENS_GAMENAMES = {
    "thunder force", "tf3", "altered beast", "ab2", "columns",
    "dynamite headdy", "headdy", "candy crush", "tetris",
    "sega", "genesis", "mega drive", "megadrive", "arcade",
}
BANNED_TOKENS: set = (_BANNED_TOKENS_BUTTON | _BANNED_TOKENS_UI
                      | _BANNED_TOKENS_GAMENAMES)

# Predicate-form regex — required for preconditions / eff_* / predicate_*
# fields.  Accepts:
#   world.foo, world.foo=bar, event.baz, predicate.qux=true,
#   has:target, score=Δ, score>=10, score+=Δ, op(arg1, arg2), op(arg).
# The relational op slot accepts:  =  ==  !=  <  <=  >  >=  +=  -=  *=
_PREDICATE_RE = re.compile(
    r"""^\s*(?:
        (?:world|event|state|predicate|score|game|player)\.[\w_\.]+
            (?:\s*(?:[=!<>]+|[+\-*]?=)\s*[\w\.\+\-Δ_]+)?
      | (?:event\.[\w_]+)
      | has[:_][\w_]+
      | affords[=:][\w_,]+
      | [\w_]+\s*(?:[=!<>]+|[+\-*]?=)\s*[\w\.\+\-Δ_]+
      | [a-z_][\w_]*\([^)]*\)
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_predicate_form(s: str) -> bool:
    return bool(_PREDICATE_RE.match((s or "").strip()))


def _has_banned_token(text: str) -> Tuple[bool, List[str]]:
    """Return (had_banned, list of which) using whole-word matching for
    short button glyphs and substring matching for multi-word phrases."""
    if not text:
        return False, []
    low = " " + text.lower() + " "
    found: List[str] = []
    # Multi-word phrases — substring is fine
    for term in BANNED_TOKENS:
        if " " in term:
            if term in low:
                found.append(term)
            continue
        # Single-word — whole-word boundary
        if re.search(rf"\b{re.escape(term)}\b", low):
            found.append(term)
    return bool(found), found


# JSON schema for vLLM's ``guided_json`` enforcement. We define the
# proposal *structure* but NOT the per-string content (the LLM still
# composes content; we reject + re-validate post-hoc).
PROPOSAL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": [
                    "operator", "subgoal", "name",
                    "preconditions", "predicate_success", "predicate_abort",
                    "eff_add", "eff_del", "expected_tag_pattern",
                    "rationale_evidence", "non_redundant_reason",
                    "source_failure_ids",
                ],
                "properties": {
                    "operator": {"type": "string", "enum": OPERATORS},
                    "subgoal": {"type": "string", "enum": SUBGOALS},
                    "name": {"type": "string", "maxLength": 80},
                    # All predicate-list fields are arrays of short strings
                    # we'll re-validate after parse.
                    "preconditions": {
                        "type": "array", "maxItems": 5,
                        "items": {"type": "string", "maxLength": 80},
                    },
                    "predicate_success": {
                        "type": "array", "maxItems": 4,
                        "items": {"type": "string", "maxLength": 80},
                    },
                    "predicate_abort": {
                        "type": "array", "maxItems": 4,
                        "items": {"type": "string", "maxLength": 80},
                    },
                    "eff_add": {
                        "type": "array", "maxItems": 5,
                        "items": {"type": "string", "maxLength": 80},
                    },
                    "eff_del": {
                        "type": "array", "maxItems": 5,
                        "items": {"type": "string", "maxLength": 80},
                    },
                    "expected_tag_pattern": {
                        "type": "array",
                        "minItems": 2, "maxItems": 6,
                        # Tag = OPERATOR or SUBGOAL — checked enum-wise
                        # post-parse, schema only enforces presence.
                        "items": {"type": "string"},
                    },
                    "rationale_evidence": {"type": "string", "maxLength": 400},
                    "non_redundant_reason": {"type": "string", "maxLength": 300},
                    "source_failure_ids": {
                        "type": "array", "minItems": 1, "maxItems": 5,
                        "items": {"type": "string", "maxLength": 60},
                    },
                },
                "additionalProperties": False,
            },
        },
        "no_novel_skills": {"type": "boolean"},
        "review_notes": {"type": "string", "maxLength": 600},
    },
    "required": ["proposals", "no_novel_skills", "review_notes"],
    "additionalProperties": False,
}


PROPOSER_SYSTEM_V2 = f"""You are a SKILL-EXTRACTION expert for a multi-game agent.
You review FAILURE EVIDENCE from one game and propose new SKILL CONTRACTS
that the agent can reuse — IDEALLY ACROSS MULTIPLE GAMES.

CRITICAL DESIGN RULE — abstract format only
============================================

Skills are stored in a SHARED, GAME-AGNOSTIC schema.  The actor at
runtime grounds them into the current game's button vocabulary.  Your
job is therefore to write the ABSTRACT POLICY ONLY:

* operator + subgoal — pick from these CANONICAL vocabularies:
  - operator ∈ {sorted(OPERATORS)}
  - subgoal  ∈ {sorted(SUBGOALS)}
* preconditions / predicate_success / predicate_abort / eff_add / eff_del
  — write ONLY in PREDICATE FORM, e.g.:
    world.threat_count>0
    event.score_changed
    predicate.in_range=true
    has:target
    has:candidate_set
    score>=old_score+Δ
    player.hp<old.hp
* expected_tag_pattern — list of 2-6 tags from {{OPERATOR ∪ SUBGOAL}}.

ABSOLUTELY FORBIDDEN in any field:

* Button names: A, B, X, Y, Z, L, R, UP, DOWN, LEFT, RIGHT, START, SELECT
  (or "press X", "hold L", etc.)
* UI words: menu, button, screen, panel, dialog, dropdown, form, textbox,
  toolbar, modal
* Game names: Thunder Force, TF3, Altered Beast, Columns, Headdy,
  Tetris, Candy Crush, Sega, Genesis, Mega Drive, Arcade, BrowserGym

If the failure pattern only makes sense if you reference a button or
specific game mechanic, that means it's not yet a SKILL — skip it.
The agent's per-game grounding layer handles button-level execution.

Good name examples (abstract, reusable across games):
  "Engage Threat When Target In Range"
  "Recover Position After Damage"
  "Verify Predicate Before Commit"
  "Optimize Layout Before Clear"

Bad name examples (concrete, DO NOT EMIT):
  "Press B Rapidly On Enemy Formation"   ← button name
  "Navigate Score Adjustment Menu"        ← UI word
  "Counter Sega Genesis Boss Pattern"     ← game name

OUTPUT FORMAT — STRICT JSON only (no prose, no fences):

{{
  "proposals": [
    {{
      "operator": "<OPERATOR>",
      "subgoal":  "<SUBGOAL>",
      "name": "<≤80 chars, abstract>",
      "preconditions":      ["<predicate-form>", ...],   // ≤5
      "predicate_success":  ["<predicate-form>", ...],   // ≤4
      "predicate_abort":    ["<predicate-form>", ...],   // ≤4
      "eff_add":            ["<predicate-form>", ...],   // ≤5
      "eff_del":            ["<predicate-form>", ...],   // ≤5
      "expected_tag_pattern": ["<OP-or-SG>", ...],       // 2-6 entries
      "rationale_evidence": "<which fail_id supports this>",
      "non_redundant_reason": "<why distinct from existing skill>",
      "source_failure_ids": ["fail_xxx", ...]            // ≥1
    }}
  ],
  "no_novel_skills": <bool>,
  "review_notes": "<≤200 words on patterns observed>"
}}

If after reviewing the failures you find NO truly novel skill (all
patterns are already covered), set `proposals: []`,
`no_novel_skills: true`, and explain in `review_notes` which existing
skill covers each pattern.

CONSTRAINTS:
* Maximum 4 proposals per call.
* Each proposal MUST cite ≥1 failure_id from the input.
* Output ONLY the JSON object."""


# -------------------------- 35B proposer (v2 abstract-only) -------------


def call_35b_proposer_v2(
    system: str, user: str, *, max_tokens: int = 3500,
    judge_url: Optional[str] = None,
) -> Tuple[str, dict]:
    """Like ``call_35b_proposer`` but adds vLLM ``guided_json`` so the
    response respects ``PROPOSAL_JSON_SCHEMA`` structurally.  We still
    do post-hoc semantic validation (predicate-form, banned-token
    filter, canonical operator/subgoal enums) downstream."""
    import openai                                                  # noqa: WPS433

    client = openai.OpenAI(
        base_url=judge_url or os.environ.get("PROBE_JUDGE_URL", "http://localhost:8001/v1"),
        api_key="dummy",
    )
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model="Qwen/Qwen3.5-35B-A3B",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        # vLLM-specific: enforce JSON schema.  Falls back gracefully —
        # if the server doesn't support it the request still works,
        # just without guidance.
        extra_body={
            "guided_json": PROPOSAL_JSON_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    dt = time.monotonic() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    meta = {
        "model": resp.model,
        "wall_s": dt,
        "finish_reason": resp.choices[0].finish_reason,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    return content, meta


# -------------------------- proposal validation -------------------------


def _check_predicate_list(items: List[Any], field: str) -> Tuple[List[str], List[str]]:
    """Return (kept, dropped_with_reason).  Drops any item that fails
    predicate-form OR contains a banned token."""
    kept: List[str] = []
    dropped: List[str] = []
    for it in items or []:
        s = str(it).strip()
        if not s:
            continue
        had_ban, terms = _has_banned_token(s)
        if had_ban:
            dropped.append(f"{field}:'{s[:40]}' banned={terms[:3]}")
            continue
        if not _is_predicate_form(s):
            dropped.append(f"{field}:'{s[:40]}' not predicate-form")
            continue
        kept.append(s)
    return kept, dropped


def validate_proposal(p: dict) -> Tuple[bool, dict]:
    """Apply abstract-format gate.  Returns (ok, validated_proposal_or_reason).

    Mutates ``p`` to drop banned-token / non-predicate items rather than
    rejecting the whole proposal — but if the abstract layer drops
    below a quality floor (≥1 eff_add OR ≥1 predicate_success AND
    valid operator+subgoal+tag_pattern) we reject it.
    """
    op = (p.get("operator") or "").strip().upper()
    sg = (p.get("subgoal") or "").strip().upper()
    if op not in OPERATORS:
        return False, {"reject_reason": f"operator '{op}' not in canonical set"}
    if sg not in SUBGOALS:
        return False, {"reject_reason": f"subgoal '{sg}' not in canonical set"}

    name = (p.get("name") or "").strip()
    name_banned, name_terms = _has_banned_token(name)
    if name_banned:
        return False, {"reject_reason": f"name contains banned tokens: {name_terms}"}

    rationale_banned, rationale_terms = _has_banned_token(
        f"{p.get('rationale_evidence','')} {p.get('non_redundant_reason','')}"
    )
    # rationale leaks are warnings — we sanitize but don't reject.
    sanitization_log: List[str] = []
    if rationale_banned:
        sanitization_log.append(
            f"rationale contained banned tokens (kept proposal): {rationale_terms[:3]}"
        )

    # Predicate-form audit on each list field.
    p["preconditions"], dp1 = _check_predicate_list(
        p.get("preconditions") or [], "precond")
    p["predicate_success"], dp2 = _check_predicate_list(
        p.get("predicate_success") or [], "pred_succ")
    p["predicate_abort"], dp3 = _check_predicate_list(
        p.get("predicate_abort") or [], "pred_abort")
    p["eff_add"], dp4 = _check_predicate_list(
        p.get("eff_add") or [], "eff_add")
    p["eff_del"], dp5 = _check_predicate_list(
        p.get("eff_del") or [], "eff_del")

    sanitization_log.extend(dp1 + dp2 + dp3 + dp4 + dp5)

    # Tag pattern: must be canonical OP or SG only.
    raw_tags = list(p.get("expected_tag_pattern") or [])
    valid_tags = [
        t.upper().strip() for t in raw_tags
        if t.upper().strip() in OPERATORS or t.upper().strip() in SUBGOALS
    ]
    if len(valid_tags) < 2:
        return False, {"reject_reason": f"tag_pattern has <2 canonical tags: {raw_tags}"}
    p["expected_tag_pattern"] = valid_tags

    # Quality floor: must keep at least 1 abstract-grounded effect AND
    # 1 source failure.
    if not (p["eff_add"] or p["predicate_success"]):
        return False, {"reject_reason": "no abstract eff_add or predicate_success after sanitization"}
    if not (p.get("source_failure_ids") or []):
        return False, {"reject_reason": "missing source_failure_ids"}

    p["_sanitization_log"] = sanitization_log
    p["_skill_id_canonical"] = f"{op}/{sg}"
    return True, p


# -------------------------- bucketing ------------------------------------


def bucket_failures(failures: List[dict], bucket_size: int = 16) -> List[List[dict]]:
    """Group failures by ``failure_class``, chunk each into ``bucket_size``
    lists, then INTERLEAVE across classes so an early ``--max-buckets``
    cap still gives the proposer balanced coverage of all failure
    modes (not just whichever class iterated first).
    """
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for f in failures:
        by_class[f["failure_class"]].append(f)

    per_class_buckets: Dict[str, List[List[dict]]] = {}
    for cls, items in by_class.items():
        rng = random.Random(hash(cls) & 0xFFFFFFFF)
        rng.shuffle(items)
        per_class_buckets[cls] = [
            items[i:i + bucket_size]
            for i in range(0, len(items), bucket_size)
        ]

    # Round-robin interleave: bucket[0] of each class, then bucket[1], ...
    interleaved: List[List[dict]] = []
    max_depth = max((len(v) for v in per_class_buckets.values()), default=0)
    for depth in range(max_depth):
        for cls in per_class_buckets:
            if depth < len(per_class_buckets[cls]):
                interleaved.append(per_class_buckets[cls][depth])
    return interleaved


# -------------------------- novelty gate ---------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def _bag(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text)} - {
        # very common stop-tokens that don't carry meaning
        "the", "a", "an", "of", "to", "and", "in", "is", "are", "for",
        "with", "this", "skill", "phase", "game", "agent", "score",
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def is_novel(
    proposal: dict,
    existing_bank: List[dict],
    accepted_so_far: List[dict],
    threshold: float = 0.55,
) -> Tuple[bool, str]:
    """Bag-of-tokens novelty gate. Returns (is_novel, why)."""
    name = (proposal.get("name") or "").strip()
    desc = (
        proposal.get("rationale_evidence") or
        proposal.get("non_redundant_reason") or
        proposal.get("action_pattern") or
        ""
    )
    blob = f"{name} {desc} {' '.join(proposal.get('preconditions') or [])} {proposal.get('action_pattern') or ''}"
    new_bag = _bag(blob)

    # Compare against existing bank skills' descriptions.
    for s in existing_bank:
        s_blob = f"{s.get('name','')} {s.get('desc','')} {' '.join(s.get('preconds') or [])}"
        sim = jaccard(new_bag, _bag(s_blob))
        if sim >= threshold:
            return False, f"jaccard={sim:.2f} vs existing skill_id={s['skill_id']}"

    # Compare against already accepted v2 proposals (de-dup within batch).
    for s in accepted_so_far:
        s_blob = f"{s.get('name','')} {s.get('rationale_evidence','')} {s.get('action_pattern','')}"
        sim = jaccard(new_bag, _bag(s_blob))
        if sim >= threshold:
            return False, f"jaccard={sim:.2f} vs accepted v2 skill_name={s.get('name')}"

    return True, ""


# -------------------------- skill record builder -------------------------


def build_skill_record(
    *, proposal: dict, game: str, source_failure_ids: List[str],
) -> dict:
    """Lift a *validated* abstract proposal into the SkillBank record
    envelope.  The proposal MUST have already passed
    ``validate_proposal`` so all list fields are predicate-form and the
    operator/subgoal/tag-pattern are canonical.

    Note: ``protocol.steps`` is left empty.  Pre-curator skills should
    not carry hallucinated executable plans — the live curator/contract
    LoRAs fill these in from real episode evidence after admission.
    The abstract layer (preconditions/predicates/effects/expected_tag_
    pattern) is enough for the actor's skill-selection LoRA to surface
    the skill, and for the action_taking LoRA to ground it into the
    current game's button vocab at execution time.

    Returns a dict matching the on-disk schema of
    ``skillbank/<game>/skill_bank.jsonl``.
    """
    now = time.time()
    op = proposal["_skill_id_canonical"].split("/")[0]
    sg = proposal["_skill_id_canonical"].split("/")[1]
    canonical_id = f"{op}/{sg}"
    # Append a short uuid suffix so multiple v2 skills with the same
    # canonical (op/sg) pair don't collide in-bank with each other or
    # with foundry-mined skills.
    skill_id = f"{canonical_id}#v2:{uuid.uuid4().hex[:8]}"
    name = (proposal.get("name") or canonical_id).strip()

    skill = {
        "skill_id": skill_id,
        "version": 1,
        "name": name,
        "strategic_description": (
            proposal.get("rationale_evidence")
            or proposal.get("non_redundant_reason")
            or ""
        )[:400],
        "tags": ["crafter_v2", op.lower(), sg.lower()],
        "protocol": {
            "preconditions": list(proposal.get("preconditions") or [])[:5],
            # CRITICAL: leave steps empty.  Curator/contract LoRAs fill
            # this in from real episode evidence at admission time.
            # Hallucinated steps are what produced the 0.5%-abstract-share
            # output in the v1 audit.
            "steps": [],
            "abort_criteria": [],
            "success_criteria": [],
            "predicate_success": list(proposal.get("predicate_success") or [])[:4],
            "predicate_abort": list(proposal.get("predicate_abort") or [])[:4],
            "step_checks": [],
            "expected_duration": None,
            "source": "crafter_v2",
        },
        "contract": {
            "skill_id": skill_id,
            "name": name,
            "version": 1,
            "n_instances": 0,
            "support": {},
            "eff_add": list(proposal.get("eff_add") or [])[:5],
            "eff_del": list(proposal.get("eff_del") or [])[:5],
            "eff_event": [],
            "description": (proposal.get("non_redundant_reason") or "")[:300],
            "created_at": now,
            "updated_at": now,
        },
        "sub_episodes": [],
        "expected_tag_pattern": list(proposal.get("expected_tag_pattern") or [])[:6],
        "execution_hint": {
            # Short abstract sentence — actor reads this; we keep it
            # brief and predicate-grounded so it doesn't poison the
            # prompt with concrete narrative.
            "execution_description": (
                f"Operator={op}, Subgoal={sg}. "
                f"Apply when preconds satisfied; succeeds when predicate_success holds."
            ),
            "common_preconditions": list(proposal.get("preconditions") or [])[:3],
            "common_target_objects": [],
            "common_failure_modes": [],
            "termination_cues": list(proposal.get("predicate_abort") or [])[:2],
            "state_transition_pattern": "",
            "n_source_segments": len(source_failure_ids),
            "updated_at": now,
        },
        "protocol_history": [],
        "n_instances": 0,
        "retired": False,
        "created_at": now,
        "updated_at": now,
        # Cross-game metadata.  Empty feasible_tasks = "any game" — the
        # whole point of the abstract format is cross-game reuse.  The
        # per-game harness still validates at admission time before
        # promoting to ``stable``.
        "feasible_tasks": [],
        "verified_tasks": [],
        "derived_from": None,
        "confidence_tag": CONFIDENCE_TAG_CRAFTER_V2,
    }
    report = {
        "skill_id": skill_id,
        "n_instances": 0,
        "eff_add_success_rate": {},
        "eff_del_success_rate": {},
        "eff_event_rate": {},
        "overall_pass_rate": 0.0,
        "worst_segments": [],
        "failure_signatures": {},
    }
    return {"skill": skill, "report": report}


# -------------------------- main ----------------------------------------


def run_pipeline(
    run_dir: Path, game: str,
    bucket_size: int = 16, max_buckets: int = 8,
    novelty_threshold: float = 0.55,
    judge_url: str = "http://localhost:8001/v1",
) -> Dict[str, Any]:
    """End-to-end pipeline. Returns a summary dict and writes outputs."""
    out_dir = run_dir / "crafter_v2_offline"
    (out_dir / "enriched_failures").mkdir(parents=True, exist_ok=True)
    (out_dir / "proposals").mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading rollout artifacts for game={game}…")
    rows = load_action_taking(run_dir, game)
    skill_sel = load_skill_selection(run_dir, game)
    reward_log = load_reward_log(run_dir, game)
    outcomes = load_episode_outcomes(run_dir, game)
    bank = load_skill_bank(run_dir, game)
    print(f"   rows={len(rows)}  skill_sel={len(skill_sel)}  reward_log={len(reward_log)}  episodes={len(outcomes)}  bank={len(bank)}")

    print(f"\n[2/5] detecting enriched failures…")
    failures = detect_failures(rows, reward_log, outcomes)
    by_kind = Counter(f["failure_class"] for f in failures)
    print(f"   total={len(failures)}  classes={dict(by_kind)}")

    fpath = out_dir / "enriched_failures" / "all_failures.jsonl"
    with open(fpath, "w") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"   → {fpath} ({fpath.stat().st_size:,} bytes)")

    print(f"\n[3/5] bucketing into {bucket_size}-failure batches…")
    buckets = bucket_failures(failures, bucket_size=bucket_size)
    print(f"   {len(buckets)} buckets total; will run at most {max_buckets}")

    os.environ["PROBE_JUDGE_URL"] = judge_url

    bank_block = render_skill_bank_for_prompt(bank)

    print(f"\n[4/5] calling 35B proposer (v2 abstract-only, guided_json)…")
    all_proposals: List[dict] = []
    bucket_meta: List[dict] = []
    n_calls = min(len(buckets), max_buckets)
    for i, bucket in enumerate(buckets[:n_calls]):
        fail_block = render_failures_for_prompt(bucket)
        user_msg = f"{bank_block}\n\n{fail_block}\n\nReview the failures and emit your JSON."
        t0 = time.monotonic()
        try:
            raw, meta = call_35b_proposer_v2(
                PROPOSER_SYSTEM_V2, user_msg, max_tokens=3500,
                judge_url=judge_url,
            )
        except Exception as exc:
            print(f"   bucket {i+1}/{n_calls}: 35B call FAILED: {exc}")
            bucket_meta.append({"i": i, "err": str(exc), "n_failures": len(bucket)})
            continue
        dt = time.monotonic() - t0
        try:
            cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
        except Exception as e:
            print(f"   bucket {i+1}/{n_calls}: parse fail ({e})")
            bucket_meta.append({"i": i, "parse_err": str(e), "n_failures": len(bucket)})
            continue
        bk_props = parsed.get("proposals") or []
        # Tag with bucket index for traceability
        for p in bk_props:
            p["_bucket_i"] = i
            p["_bucket_classes"] = [f["failure_class"] for f in bucket]
        all_proposals.extend(bk_props)
        bucket_meta.append({
            "i": i, "n_failures": len(bucket), "n_proposals": len(bk_props),
            "no_novel": parsed.get("no_novel_skills"),
            "wall_s": dt, **meta,
        })
        print(f"   bucket {i+1}/{n_calls}: {len(bk_props)} proposals  ({dt:.1f}s)  classes={set([f['failure_class'] for f in bucket])}")

    print(f"\n   total raw proposals across buckets: {len(all_proposals)}")

    # ── Stage A: ABSTRACT FORMAT VALIDATION ───────────────────────
    print(f"\n[5/5a] abstract-format validation (banned-token / predicate-form)…")
    validated: List[dict] = []
    rejected_format: List[dict] = []
    for p in all_proposals:
        ok, info = validate_proposal(p)
        if ok:
            validated.append(p)
        else:
            rejected_format.append({
                "name": p.get("name", "?"),
                "operator": p.get("operator"),
                "subgoal": p.get("subgoal"),
                "why": info.get("reject_reason", "?"),
            })
    print(f"   passed format gate: {len(validated)} / {len(all_proposals)}")
    if rejected_format:
        print(f"   rejected (sample first 5):")
        for r in rejected_format[:5]:
            print(f"      - {r['name'][:40]:40s} op={r['operator']} sg={r['subgoal']}: {r['why']}")

    # ── Stage B: NOVELTY GATE (against existing bank + within batch) ──
    print(f"\n[5/5b] novelty filter (jaccard threshold={novelty_threshold})…")
    accepted: List[dict] = []
    rejected_log: List[dict] = list(rejected_format)
    for p in validated:
        ok, why = is_novel(p, bank, accepted, threshold=novelty_threshold)
        if ok:
            accepted.append(p)
        else:
            rejected_log.append({"name": p.get("name"), "why": why})
    print(f"   accepted: {len(accepted)} / {len(validated)} (after format+novelty)")
    if len(rejected_log) > len(rejected_format):
        print(f"   novelty-rejected (sample first 5):")
        novel_rej = [r for r in rejected_log if r not in rejected_format][:5]
        for r in novel_rej:
            print(f"      - {r['name']}: {r['why']}")

    # Build skill records ready for bank injection.
    skill_records = []
    for p in accepted:
        rec = build_skill_record(
            proposal=p, game=game,
            source_failure_ids=p.get("source_failure_ids") or [],
        )
        skill_records.append(rec)

    # Persist
    candidate_path = out_dir / "proposals" / "candidate_skills.jsonl"
    with open(candidate_path, "w") as fh:
        for r in skill_records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\n   → {candidate_path} ({candidate_path.stat().st_size:,} bytes; {len(skill_records)} skills)")

    raw_path = out_dir / "proposals" / "raw_proposals.json"
    with open(raw_path, "w") as fh:
        json.dump({
            "all_proposals": all_proposals,
            "bucket_meta": bucket_meta,
            "rejected": rejected_log,
        }, fh, ensure_ascii=False, indent=2, default=str)

    summary = {
        "game": game,
        "n_failures": len(failures),
        "by_class": dict(by_kind),
        "n_buckets_total": len(buckets),
        "n_buckets_run": n_calls,
        "n_raw_proposals": len(all_proposals),
        "n_passed_format_gate": len(validated),
        "n_rejected_format": len(rejected_format),
        "n_accepted": len(accepted),
        "n_rejected_redundant": len(rejected_log) - len(rejected_format),
        "candidate_path": str(candidate_path),
        "skills": [{"skill_id": s["skill"]["skill_id"], "name": s["skill"]["name"]}
                   for s in skill_records],
        "prompt_version": "v2_abstract_only_2026_05_08",
    }
    summary_path = out_dir / "proposals" / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"   → {summary_path}")

    print(f"\n=== DONE === {len(skill_records)} candidate skills ready for injection")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--bucket-size", type=int, default=16)
    ap.add_argument("--max-buckets", type=int, default=8)
    ap.add_argument("--novelty-threshold", type=float, default=0.55)
    ap.add_argument("--judge-url", default="http://localhost:8001/v1")
    args = ap.parse_args()

    summary = run_pipeline(
        Path(args.run_dir), args.game,
        bucket_size=args.bucket_size,
        max_buckets=args.max_buckets,
        novelty_threshold=args.novelty_threshold,
        judge_url=args.judge_url,
    )
    return 0 if summary["n_accepted"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
