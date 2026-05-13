#!/usr/bin/env python3
"""Smoke test: render skill prompts under three paradigms side-by-side.

Reads the per-task skill bank and renders each skill as the agent would
see it under:
  A — Subgoal Tag (game-style: tag + one-line objective)
  B — Multi-step Protocol (Layer-C style: 5-step plan with >> marker)
  C — Hybrid (archetype name + direction + exemplar from protocol_raw)

No LLM calls, no training — purely a prompt-rendering diagnostic.

Usage:
    python frontier_data/scripts/smoke_test_paradigms.py
    python frontier_data/scripts/smoke_test_paradigms.py --task siv_bench
    python frontier_data/scripts/smoke_test_paradigms.py --task video_holmes --skill-index 0
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
BANK_ROOT = REPO / "frontier_data" / "output" / "per_task_banks"

AVAILABLE_TASKS = sorted(
    p.name for p in BANK_ROOT.iterdir()
    if p.is_dir() and (p / "skill_bank.jsonl").exists()
) if BANK_ROOT.exists() else []


def load_skills(task: str) -> List[Dict[str, Any]]:
    path = BANK_ROOT / task / "skill_bank.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _tag_from_skill_id(skill_id: str) -> str:
    """archetype.video_holmes.CTI -> CTI"""
    parts = skill_id.split(".")
    return parts[-1] if parts else skill_id


def _wrap(text: str, indent: int = 4, width: int = 80) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


# ── Paradigm A: Subgoal Tag ─────────────────────────────────────────

def render_paradigm_a(skill: Dict[str, Any]) -> str:
    sk = skill.get("skill", skill)
    tag = _tag_from_skill_id(sk.get("skill_id", ""))
    desc = sk.get("strategic_description", "")
    if len(desc) > 60:
        desc = desc[:57] + "..."

    return (
        f"Assigned subgoal: [{tag}] {desc}"
    )


# ── Paradigm B: Multi-step Protocol ─────────────────────────────────

def render_paradigm_b(
    skill: Dict[str, Any], protocol_step_idx: int = 0,
) -> str:
    sk = skill.get("skill", skill)
    name = sk.get("name", sk.get("skill_id", "unknown"))
    proto = sk.get("protocol", {})
    steps = proto.get("steps", []) if isinstance(proto, dict) else []
    checks = proto.get("step_checks", []) if isinstance(proto, dict) else []
    success = proto.get("success_criteria", []) if isinstance(proto, dict) else []
    abort = proto.get("abort_criteria", []) if isinstance(proto, dict) else []

    parts = [f"--- Active Skill: {name} ---"]

    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7]):
            marker = ">>" if i == protocol_step_idx else "  "
            parts.append(f"  {marker} {i+1}. {step}")

    empty_checks = all(not c for c in checks) if checks else True
    if checks and not empty_checks:
        parts.append(f"  Step checks: {checks}")
    elif checks:
        parts.append(f"  Step checks: ALL EMPTY (intrinsic bonus is free)")

    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")
    parts.append("--- end skill ---")
    return "\n".join(parts)


# ── Paradigm C: Hybrid (archetype + direction + exemplar) ───────────

def render_paradigm_c(skill: Dict[str, Any]) -> str:
    sk = skill.get("skill", skill)
    report = skill.get("report", {})
    tag = _tag_from_skill_id(sk.get("skill_id", ""))
    name = sk.get("name", tag)
    desc = sk.get("strategic_description", "")

    raw_steps = sk.get("protocol_raw", {}).get("steps", [])
    expected = report.get("expected_answer", "?")
    model_ans = report.get("model_answer", "?")

    parts = [f"--- Skill: {name} ({tag}) ---"]

    if desc:
        direction = desc if len(desc) <= 80 else desc[:77] + "..."
        parts.append(f"  Task: {direction}")

    if raw_steps:
        parts.append("")
        parts.append("  Example reasoning (from a representative case):")
        for i, step in enumerate(raw_steps[:4]):
            step_text = str(step)
            if len(step_text) > 120:
                step_text = step_text[:117] + "..."
            parts.append(f"    {i+1}. {step_text}")
        if len(raw_steps) > 4:
            parts.append(f"    ... ({len(raw_steps) - 4} more steps)")

        parts.append(f"  Answer: {expected} (gold) / {model_ans} (model)")
        correct = report.get("judge_correct")
        if correct is not None:
            parts.append(f"  Correct: {correct}")
    else:
        parts.append("  (no protocol_raw available for exemplar)")

    parts.append("--- end skill ---")
    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────

def run_comparison(task: str, skill_index: Optional[int] = None) -> None:
    records = load_skills(task)
    print(f"\n{'='*70}")
    print(f"  Task: {task} — {len(records)} skills")
    print(f"  Bank: {BANK_ROOT / task / 'skill_bank.jsonl'}")
    print(f"{'='*70}\n")

    if skill_index is not None:
        if skill_index >= len(records):
            print(f"ERROR: skill_index {skill_index} >= {len(records)}")
            return
        records = [records[skill_index]]

    total_a, total_b, total_c = 0, 0, 0

    for idx, rec in enumerate(records):
        sk = rec.get("skill", rec)
        sid = sk.get("skill_id", f"skill_{idx}")
        print(f"{'─'*60}")
        print(f"  [{idx}] {sid}")
        print(f"{'─'*60}")

        # Paradigm A
        prompt_a = render_paradigm_a(rec)
        tok_a = estimate_tokens(prompt_a)
        total_a += tok_a
        print(f"\n  [Paradigm A — Subgoal Tag] (~{tok_a} tokens)")
        print(f"    {prompt_a}")

        # Paradigm B
        prompt_b = render_paradigm_b(rec)
        tok_b = estimate_tokens(prompt_b)
        total_b += tok_b
        print(f"\n  [Paradigm B — Multi-step Protocol] (~{tok_b} tokens)")
        for line in prompt_b.split("\n"):
            print(f"    {line}")

        # Paradigm C
        prompt_c = render_paradigm_c(rec)
        tok_c = estimate_tokens(prompt_c)
        total_c += tok_c
        print(f"\n  [Paradigm C — Hybrid + Exemplar] (~{tok_c} tokens)")
        for line in prompt_c.split("\n"):
            print(f"    {line}")

        print()

    # Summary
    n = len(records)
    print(f"\n{'='*60}")
    print(f"  TOKEN BUDGET SUMMARY ({task}, {n} skills)")
    print(f"{'='*60}")
    print(f"  Paradigm A (subgoal tag):       {total_a:>5} tokens total, ~{total_a//max(n,1):>3} avg/skill")
    print(f"  Paradigm B (multi-step proto):  {total_b:>5} tokens total, ~{total_b//max(n,1):>3} avg/skill")
    print(f"  Paradigm C (hybrid+exemplar):   {total_c:>5} tokens total, ~{total_c//max(n,1):>3} avg/skill")
    print()
    print(f"  Note: only 1 skill is active at a time in the prompt.")
    print(f"  The per-skill avg is the actual prompt overhead.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="video_holmes",
                    help=f"Task to render. Available: {', '.join(AVAILABLE_TASKS)}")
    ap.add_argument("--skill-index", type=int, default=None,
                    help="Render only one skill by index (0-based)")
    ap.add_argument("--all-tasks", action="store_true",
                    help="Run for all available tasks")
    args = ap.parse_args()

    if args.all_tasks:
        for task in AVAILABLE_TASKS:
            run_comparison(task, args.skill_index)
    else:
        if args.task not in AVAILABLE_TASKS:
            print(f"ERROR: task '{args.task}' not found. Available: {AVAILABLE_TASKS}")
            return 1
        run_comparison(args.task, args.skill_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
