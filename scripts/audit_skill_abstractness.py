#!/usr/bin/env python3
"""Skill abstractness audit — quantify how game-agnostic our skill bank
entries currently are.

Motivation
----------
The design intent is for skills to be representable in a *general
format* that can be reused across games and tasks (PLAN-COEVO §2,
2026-05-08 conversation).  The current ``Skill`` schema has both
abstract slots (``contract.eff_add/del``, ``required_slots``,
``predicate_success/abort``, ``expected_tag_pattern``) and concrete
slots (``strategic_description``, ``protocol.steps``, free-text
``preconditions``).  At runtime the actor's prompt (see
``decision_agents/actor_agent.py::_format_skill_block``) renders the
*concrete* slots — so even when the abstract layer is well-grounded,
the actor experiences a game-specific narrative.

This script quantifies the situation per phase / per skill / per
field, so we can answer:

* What % of our skills have meaningfully populated abstract slots?
* What % of their concrete prose contains game-leaking phrases
  (button names, UI elements, Sega-Genesis-specific vocabulary)?
* Which TF3 protocol_steps would be nonsensical if surfaced to AB?

Outputs both a markdown report and a JSON dump.

Usage::

    python scripts/audit_skill_abstractness.py \\
        --run-dir runs/Qwen3.5-9B_<ts> \\
        --out /tmp/skill_abstractness.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Canonical abstract vocabularies (game-agnostic) ────────────────────
try:
    from decision_agents.agent_helper import (  # type: ignore
        INTENT_OPERATORS as CANON_OPERATORS,
        UNIFIED_SUBGOALS as CANON_SUBGOALS,
    )
    from labeling.qa_vocab import (  # type: ignore
        QA_EXTRA_OPERATORS, QA_EXTRA_SUBGOALS,
    )
    OPERATORS = set(CANON_OPERATORS) | set(QA_EXTRA_OPERATORS)
    SUBGOALS = set(CANON_SUBGOALS) | set(QA_EXTRA_SUBGOALS)
except Exception:                                                 # noqa: BLE001
    OPERATORS = {
        "INSPECT", "TRACK", "COMPARE", "COMMIT", "RECOVER", "VERIFY",
        "REASON", "TOOL_USE",
    }
    SUBGOALS = {
        "SETUP", "NAVIGATE", "POSITION", "CLEAR", "MERGE", "COLLECT",
        "BUILD", "ATTACK", "DEFEND", "EVADE", "OPTIMIZE", "SURVIVE",
        "EXPLORE", "EXECUTE",
        "EVIDENCE", "IDENTIFY", "TIMELINE", "COUNT", "MEASURE",
        "LOOKUP", "DEDUCE", "RULE_OUT", "ANSWER", "FORM_FILL", "SUBMIT",
    }

# Predicate-shaped patterns — count as abstract.
PREDICATE_PATTERNS = [
    re.compile(r"^\s*(world|event|state|predicate|score|game)\.[a-z_0-9]+", re.IGNORECASE),
    re.compile(r"^\s*[a-z_]+\s*[=<>]+\s*[\w\.\+\-]+", re.IGNORECASE),
    re.compile(r"^\s*[a-z_]+\([^)]*\)\s*$", re.IGNORECASE),  # operator(args)
    re.compile(r"^\s*has[:_][a-z_]+", re.IGNORECASE),         # has:target
    re.compile(r"^\s*affords[=:]", re.IGNORECASE),
]

# ── Concrete-leak vocabulary (probable game-specific narrative) ────────
# These are heuristics; they over-flag a bit, but the false-positive rate
# is acceptable for an audit metric.
LEAK_BUTTON_WORDS = {
    # Genesis / generic console buttons
    "button", "press", "tap", "hold", "release", "joystick", "joypad",
    "dpad", "d-pad", "trigger", "stick",
    # Direction keywords used as concrete actions
    "left arrow", "right arrow", "up arrow", "down arrow",
    # Specific button glyphs that should never appear in an abstract skill
    " a button", " b button", " c button", " x button", " y button",
    " z button", " start button", " select button",
}
LEAK_UI_WORDS = {
    "menu", "screen", "panel", "dialog", "popup", "dropdown",
    "form", "textbox", "input field", "checkbox", "radio button",
    "toolbar", "sidebar", "modal",
}
LEAK_GAME_NAMES = {
    "thunder force", "tf3", "altered beast", "ab2", "columns",
    "dynamite headdy", "headdy", "candy crush", "tetris",
    "sega", "genesis", "mega drive", "megadrive", "arcade",
    "browsergym", "miniwob",
}
LEAK_KEY_GLYPHS = {
    # Single-letter button glyphs — only count if surrounded by non-letters
    " a ", " b ", " x ", " y ", " z ", " l ", " r ",
}
ALL_LEAK_TERMS = sorted(
    LEAK_BUTTON_WORDS | LEAK_UI_WORDS | LEAK_GAME_NAMES,
    key=len, reverse=True,
)


# ── Abstractness scoring helpers ───────────────────────────────────────


def is_abstract_token(s: str) -> bool:
    """A short token (single line / list element) is 'abstract' if it
    matches the operator / subgoal vocab OR the predicate patterns."""
    if not s or not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    # Tag pattern: one of OPERATORS or SUBGOALS (possibly inside [])
    upper_clean = t.upper().strip("[]() ")
    if upper_clean in OPERATORS or upper_clean in SUBGOALS:
        return True
    # operator(arg) or world.x or score=...
    for pat in PREDICATE_PATTERNS:
        if pat.match(t):
            return True
    # bracketed operator/subgoal pair
    bracket = re.match(r"^\s*\[([A-Z_]+)/([A-Z_]+)\]", t)
    if bracket and bracket.group(1) in OPERATORS and bracket.group(2) in SUBGOALS:
        return True
    return False


def text_leak_score(s: str) -> Tuple[int, List[str]]:
    """Return (n_leaks, leak_terms_found) in free-text ``s``.
    A high leak score means concrete game-specific phrases appear."""
    if not s or not isinstance(s, str):
        return 0, []
    low = s.lower()
    found: List[str] = []
    for term in ALL_LEAK_TERMS:
        if term in low:
            found.append(term.strip())
    # single-letter button glyphs
    for glyph in LEAK_KEY_GLYPHS:
        if glyph in low:
            found.append(glyph.strip())
    return len(found), found


def list_abstractness(items: Iterable[Any]) -> Tuple[int, int]:
    """Return (n_abstract, n_total) for a list of strings."""
    items = list(items or [])
    n_total = len(items)
    n_abstract = sum(1 for it in items if is_abstract_token(str(it)))
    return n_abstract, n_total


# ── Per-skill audit ────────────────────────────────────────────────────


def audit_skill(d: Dict[str, Any]) -> Dict[str, Any]:
    s = d.get("skill") or {}
    contract = s.get("contract") or {}
    protocol = s.get("protocol") or {}

    # ── Abstract layer presence (boolean per field) ────────────────
    has_abstract_id = bool(s.get("skill_id")) and (
        s.get("skill_id", "").upper().split("/")[-1] in OPERATORS
        or s.get("skill_id", "").upper().split("/")[-1] in SUBGOALS
        or any(part in OPERATORS or part in SUBGOALS
               for part in re.split(r"[/_:]+", s.get("skill_id", "").upper()))
    )

    eff_add_n_abs, eff_add_n = list_abstractness(contract.get("eff_add"))
    eff_del_n_abs, eff_del_n = list_abstractness(contract.get("eff_del"))
    eff_event_n_abs, eff_event_n = list_abstractness(contract.get("eff_event"))

    pred_succ_n_abs, pred_succ_n = list_abstractness(protocol.get("predicate_success"))
    pred_abort_n_abs, pred_abort_n = list_abstractness(protocol.get("predicate_abort"))
    step_checks_n_abs, step_checks_n = list_abstractness(protocol.get("step_checks"))

    tag_pat_n_abs, tag_pat_n = list_abstractness(s.get("expected_tag_pattern") or [])
    req_slots_n_abs, req_slots_n = list_abstractness(s.get("required_slots") or [])
    opt_slots_n_abs, opt_slots_n = list_abstractness(s.get("optional_slots") or [])

    abstract_total = (
        eff_add_n_abs + eff_del_n_abs + eff_event_n_abs
        + pred_succ_n_abs + pred_abort_n_abs + step_checks_n_abs
        + tag_pat_n_abs + req_slots_n_abs + opt_slots_n_abs
    )
    abstract_filled = (
        eff_add_n + eff_del_n + eff_event_n
        + pred_succ_n + pred_abort_n + step_checks_n
        + tag_pat_n + req_slots_n + opt_slots_n
    )

    # ── Concrete-leak audit on actor-visible free-text fields ──────
    leak_steps_n, leak_steps_terms = 0, []
    for st in (protocol.get("steps") or []):
        n, terms = text_leak_score(str(st))
        leak_steps_n += n
        leak_steps_terms.extend(terms)

    leak_precond_n, leak_precond_terms = 0, []
    for p in (protocol.get("preconditions") or []):
        n, terms = text_leak_score(str(p))
        leak_precond_n += n
        leak_precond_terms.extend(terms)

    leak_succ_n, leak_succ_terms = 0, []
    for c in (protocol.get("success_criteria") or []):
        n, terms = text_leak_score(str(c))
        leak_succ_n += n
        leak_succ_terms.extend(terms)

    leak_abort_n, leak_abort_terms = 0, []
    for c in (protocol.get("abort_criteria") or []):
        n, terms = text_leak_score(str(c))
        leak_abort_n += n
        leak_abort_terms.extend(terms)

    leak_strategy_n, leak_strategy_terms = text_leak_score(s.get("strategic_description") or "")
    leak_exec_hint_n, leak_exec_hint_terms = text_leak_score(s.get("execution_hint") or "")

    # ── Field-presence flags (does the skill HAVE the slot?) ───────
    has_proto_steps = bool(protocol.get("steps"))
    has_pred_succ = bool(protocol.get("predicate_success"))
    has_pred_abort = bool(protocol.get("predicate_abort"))
    has_step_checks = bool(protocol.get("step_checks"))
    has_eff_add = bool(contract.get("eff_add"))
    has_eff_del = bool(contract.get("eff_del"))
    has_tag_pat = bool(s.get("expected_tag_pattern"))
    has_req_slots = bool(s.get("required_slots"))

    # Composite metrics
    abstract_share = (
        abstract_total / abstract_filled if abstract_filled else 0.0
    )
    n_protocol_step_lines = len(protocol.get("steps") or [])
    leak_density_steps = (
        leak_steps_n / n_protocol_step_lines if n_protocol_step_lines else 0.0
    )

    return {
        "skill_id": s.get("skill_id"),
        "confidence_tag": s.get("confidence_tag", "stable"),
        "version": s.get("version", 0),
        # presence
        "has_abstract_id": has_abstract_id,
        "has_proto_steps": has_proto_steps,
        "has_pred_succ": has_pred_succ,
        "has_pred_abort": has_pred_abort,
        "has_step_checks": has_step_checks,
        "has_eff_add": has_eff_add,
        "has_eff_del": has_eff_del,
        "has_tag_pat": has_tag_pat,
        "has_req_slots": has_req_slots,
        # abstract layer fill rate
        "abstract_filled_count": abstract_filled,
        "abstract_passing_count": abstract_total,
        "abstract_share": round(abstract_share, 3),
        # concrete leak counts (these are bad — game-specific narrative)
        "leak_proto_steps_n": leak_steps_n,
        "leak_proto_steps_unique": sorted(set(leak_steps_terms)),
        "leak_strategy_n": leak_strategy_n,
        "leak_strategy_unique": sorted(set(leak_strategy_terms)),
        "leak_precond_n": leak_precond_n,
        "leak_succ_n": leak_succ_n,
        "leak_abort_n": leak_abort_n,
        "leak_exec_hint_n": leak_exec_hint_n,
        "leak_density_steps": round(leak_density_steps, 3),
        # raw counts for aggregation
        "n_proto_step_lines": n_protocol_step_lines,
        "n_eff_add": len(contract.get("eff_add") or []),
        "n_eff_del": len(contract.get("eff_del") or []),
    }


# ── Aggregation ────────────────────────────────────────────────────────


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_skills": 0}

    pres = lambda key: round(100.0 * sum(1 for r in rows if r[key]) / n, 1)

    abs_filled = [r["abstract_filled_count"] for r in rows]
    abs_pass = [r["abstract_passing_count"] for r in rows]
    abs_share = [r["abstract_share"] for r in rows]

    leak_steps = [r["leak_proto_steps_n"] for r in rows]
    leak_density = [r["leak_density_steps"] for r in rows]

    skills_with_any_leak = sum(
        1 for r in rows if r["leak_proto_steps_n"] + r["leak_strategy_n"]
        + r["leak_precond_n"] + r["leak_succ_n"] + r["leak_abort_n"]
        + r["leak_exec_hint_n"] > 0
    )

    # Top leaking phrases across all skills
    leak_counter: Counter = Counter()
    for r in rows:
        for t in r["leak_proto_steps_unique"]:
            leak_counter[t] += 1
        for t in r["leak_strategy_unique"]:
            leak_counter[t] += 1

    return {
        "n_skills": n,
        "presence_pct": {
            "abstract_id": pres("has_abstract_id"),
            "proto_steps": pres("has_proto_steps"),
            "pred_succ": pres("has_pred_succ"),
            "pred_abort": pres("has_pred_abort"),
            "step_checks": pres("has_step_checks"),
            "eff_add": pres("has_eff_add"),
            "eff_del": pres("has_eff_del"),
            "tag_pat": pres("has_tag_pat"),
            "req_slots": pres("has_req_slots"),
        },
        "abstract_layer": {
            "mean_filled_count": round(sum(abs_filled) / n, 1),
            "mean_passing_count": round(sum(abs_pass) / n, 1),
            "mean_passing_share_pct": round(100.0 * sum(abs_share) / n, 1),
            "share_with_zero_abstract_fields": round(
                100.0 * sum(1 for x in abs_pass if x == 0) / n, 1,
            ),
        },
        "concrete_leaks": {
            "skills_with_any_leak_pct": round(100.0 * skills_with_any_leak / n, 1),
            "mean_leaks_in_proto_steps": round(sum(leak_steps) / n, 2),
            "mean_leak_density_per_step": round(sum(leak_density) / n, 3),
            "top_leaking_phrases": leak_counter.most_common(15),
        },
    }


# ── Report ────────────────────────────────────────────────────────────


def render_md(by_bucket: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Skill Abstractness Audit\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

    lines.append("## Methodology\n")
    lines.append("Each skill is scored on two axes:\n")
    lines.append("1. **Abstract layer fill rate** — what fraction of our schema's "
                 "designed-to-be-abstract slots (`eff_add/del/event`, `predicate_success/abort`, "
                 "`step_checks`, `expected_tag_pattern`, `required_slots`) are populated AND "
                 "actually pass the abstract-token check (operator/subgoal vocab OR "
                 "predicate pattern).\n")
    lines.append("2. **Concrete leak count** — number of game-specific phrases "
                 "(button names, UI elements, console/game names) appearing in "
                 "actor-visible free-text fields (`protocol.steps`, "
                 "`strategic_description`, `preconditions`, `success_criteria`, "
                 "`abort_criteria`, `execution_hint`).\n")
    lines.append("A perfectly cross-game-general skill would have **abstract layer "
                 "fill ≫ 0** AND **leak count = 0**.\n")
    lines.append("")

    lines.append("## Headline\n")
    lines.append("| game | tag | n | abstract-share % | %skills with leaks | mean leaks/skill | leak density per step |")
    lines.append("|---|---|---|---|---|---|---|")
    for (game, tag), s in sorted(by_bucket.items()):
        a = s["abstract_layer"]
        c = s["concrete_leaks"]
        lines.append(
            f"| `{game}` | {tag} | {s['n_skills']} | "
            f"{a['mean_passing_share_pct']}% | {c['skills_with_any_leak_pct']}% | "
            f"{c['mean_leaks_in_proto_steps']} | {c['mean_leak_density_per_step']} |"
        )
    lines.append("")

    lines.append("## Field-presence breakdown\n")
    lines.append("| game | tag | abstract_id | proto_steps | pred_succ | pred_abort | step_checks | eff_add | eff_del | tag_pat | req_slots |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (game, tag), s in sorted(by_bucket.items()):
        p = s["presence_pct"]
        lines.append(
            f"| `{game}` | {tag} | {p['abstract_id']}% | {p['proto_steps']}% | "
            f"{p['pred_succ']}% | {p['pred_abort']}% | {p['step_checks']}% | "
            f"{p['eff_add']}% | {p['eff_del']}% | {p['tag_pat']}% | {p['req_slots']}% |"
        )
    lines.append("")

    lines.append("## Top concrete-leak phrases\n")
    for (game, tag), s in sorted(by_bucket.items()):
        leaks = s["concrete_leaks"]["top_leaking_phrases"]
        if not leaks:
            continue
        lines.append(f"### `{game}` ({tag})\n")
        for term, cnt in leaks[:10]:
            lines.append(f"- `{term}` — appears in {cnt} skill(s)")
        lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- `abstract-share %` < 30 → most schema slots are empty or "
                 "filled with free-text prose; cross-game transfer at the "
                 "skill-content level cannot work because there's no abstract "
                 "layer to ground.")
    lines.append("- `%skills with leaks` > 50 → actor's prompt is dominated by "
                 "game-specific narrative; even if a skill is conceptually "
                 "general, it'll feel TF3-flavoured (or AB-flavoured) to the "
                 "actor on the next game.")
    lines.append("- `top concrete-leak phrases` — concrete fix-list: "
                 "any skill containing these in `protocol.steps` should be "
                 "regenerated with an abstract-only prompt before exporting "
                 "across games.")
    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="",
                    help="Markdown output path; default = "
                         "<run_dir>/skill_abstractness_audit.md")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run-dir not found: {run_dir}")
        return 2

    bank_root = run_dir / "skillbank"
    if not bank_root.is_dir():
        print(f"ERROR: skillbank not found at {bank_root}")
        return 3

    by_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    sample_concrete: List[Dict[str, Any]] = []
    sample_leakiest: List[Dict[str, Any]] = []

    for game_dir in sorted(bank_root.iterdir()):
        if not game_dir.is_dir():
            continue
        bank_path = game_dir / "skill_bank.jsonl"
        if not bank_path.is_file():
            continue
        n_loaded = 0
        for L in open(bank_path):
            try:
                d = json.loads(L)
            except Exception:                                     # noqa: BLE001
                continue
            row = audit_skill(d)
            tag = row["confidence_tag"] or "stable"
            by_bucket[(game_dir.name, tag)].append(row)
            n_loaded += 1
            if row["leak_proto_steps_n"] >= 3:
                sample_leakiest.append(row)
        print(f"  loaded {n_loaded} skills from {game_dir.name}/{tag}")

    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {
        k: aggregate(rows) for k, rows in by_bucket.items()
    }

    md = render_md(aggregated)

    out_path = Path(args.out) if args.out else run_dir / "skill_abstractness_audit.md"
    out_path.write_text(md)
    print(f"\nwrote {out_path}")

    json_path = Path(args.json_out) if args.json_out else out_path.with_suffix(".json")
    json_payload = {
        f"{g}::{t}": v for (g, t), v in aggregated.items()
    }
    # Top-N leakiest skills for inspection
    sample_leakiest.sort(key=lambda r: -r["leak_proto_steps_n"])
    json_payload["__sample_leakiest_skills"] = sample_leakiest[:10]
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2, default=str))
    print(f"wrote {json_path}")

    # Also print the headline table to stdout for quick CLI consumption.
    print()
    print("=" * 80)
    print("HEADLINE")
    print("=" * 80)
    for (game, tag), s in sorted(aggregated.items()):
        a = s["abstract_layer"]
        c = s["concrete_leaks"]
        print(f"  {game[:30]:30s} {tag[:12]:12s}  n={s['n_skills']:3d}  "
              f"abs_share={a['mean_passing_share_pct']:5.1f}%  "
              f"leak_skills={c['skills_with_any_leak_pct']:5.1f}%  "
              f"leak_density={c['mean_leak_density_per_step']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
