#!/usr/bin/env python
"""SMOKE TEST: ask GPT-5.4 to lift one skill from each of the 4 cohorts
into a normalised, modality-agnostic step template.

Goal: show what an "abstracted" step-template looks like and whether
GPT-5.4 can reliably produce them with the same controlled-vocabulary
operators across all cohorts.  If yes, the full pipeline runs the same
prompt over all 448 skills in the inventory.

Controlled operator vocabulary (modality-agnostic):
  PERCEIVE  — observe / scan / read inputs
  RECALL    — pull task or world state into context
  COMPARE   — set candidates against criteria
  FILTER    — drop options that fail constraints
  DECIDE    — pick one option / target / direction
  COMMIT    — execute the chosen action irreversibly
  VERIFY    — confirm the post-condition was achieved
  RECOVER   — restore safe state if execution fails / drifts

Each step also carries a 6-12 word predicate (modality-agnostic) so the
template stays grounded in the original skill semantics.

Usage::
    python scripts/lift_skill_templates_smoke.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO.parent
for p in [str(WORK), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key  # type: ignore
except Exception:
    pass

DEFAULT_MODEL = "gpt-5.4"
INVENTORY = REPO / "sft_data_inventory"

# One representative skill per cohort
SAMPLES = [
    ("gymv_game",  INVENTORY / "games/Temporal_Strider-v0/skill_bank.jsonl",       "COMMIT/EVADE"),
    ("env_wr_game",INVENTORY / "games/tetris/skill_bank.jsonl",                     "COMMIT/OPTIMIZE"),
    ("web",        INVENTORY / "non_game/webshop/skill_bank.jsonl",                  "COMMIT/OPTIMIZE"),
    ("vr_video",   INVENTORY / "non_game/video_holmes/skill_bank.jsonl",            "REASON/RULE_OUT"),
    ("vr_image",   INVENTORY / "non_game/tir_bench/skill_bank.jsonl",               "COMPARE/DEDUCE"),
]


SYSTEM_PROMPT = (
    "You are an expert at distilling reusable skills into modality-agnostic "
    "procedural templates.  You will be given ONE skill mined from agent "
    "trajectories — its strategic description, contract preconditions / "
    "postconditions, and any existing low-level protocol notes.  Your job "
    "is to produce a 2-5 step HIGH-LEVEL TEMPLATE that captures what the "
    "skill does *in a way that could plausibly transfer to a different "
    "task or modality*.  Output strict JSON only."
)


def _build_prompt(skill: dict, cohort: str, task: str) -> str:
    sid = skill.get("skill_id", "")
    name = skill.get("name", "")
    desc = (skill.get("strategic_description") or "").strip()
    contract = skill.get("contract") or {}
    pre = contract.get("preconditions") or []
    post = contract.get("postconditions") or []
    pred = contract.get("example_predicates") or []
    proto = skill.get("protocol")
    proto_lines = []
    if isinstance(proto, list):
        for s in proto:
            if isinstance(s, dict) and s.get("notes"):
                proto_lines.append(f"  - [op={s.get('op','?')}] {s.get('notes')}")
    elif isinstance(proto, dict):
        for s in proto.get("steps") or []:
            proto_lines.append(f"  - {s}")

    return "\n".join([
        f"COHORT : {cohort}",
        f"TASK   : {task}",
        f"SKILL  : {sid}  ({name})",
        "",
        "STRATEGIC_DESCRIPTION:",
        f"  {desc}",
        "",
        "CONTRACT — preconditions:",
        *(f"  - {x}" for x in pre[:6]),
        "CONTRACT — postconditions:",
        *(f"  - {x}" for x in post[:6]),
        "CONTRACT — example_predicates:",
        f"  {pred}",
        "",
        ("LOW-LEVEL PROTOCOL (only present for gym_v):" if proto_lines else "(no low-level protocol available)"),
        *proto_lines[:6],
        "",
        "TASK: Distil the skill into a 2-5 step modality-agnostic procedural",
        "      template.  Use ONLY operators from this controlled vocabulary:",
        "        PERCEIVE   — observe / scan / read inputs from the environment",
        "        RECALL     — pull task spec, prior state, or memory into context",
        "        COMPARE    — set candidates against criteria",
        "        FILTER     — drop options that fail constraints",
        "        DECIDE     — pick one option / target / direction",
        "        COMMIT     — execute the chosen action irreversibly",
        "        VERIFY     — confirm the post-condition was achieved",
        "        RECOVER    — restore safe state if execution fails or drifts",
        "",
        "Output STRICT JSON of the form:",
        "{",
        '  "template_steps": [',
        '    {"op": "<one of {PERCEIVE,RECALL,COMPARE,FILTER,DECIDE,COMMIT,VERIFY,RECOVER}>",',
        '     "predicate": "<6-12 word modality-agnostic description>"},',
        "    ... (2-5 steps total)",
        "  ],",
        '  "template_signature": "<OP1 → OP2 → OP3 ...>",   // joined ops, e.g. "PERCEIVE → COMPARE → COMMIT"',
        '  "transferable_to_cohorts": ["<cohort>", ...]      // subset of {gymv_game, env_wr_game, web, vr_image, vr_video} this template plausibly fits',
        "}",
        "",
        "Constraints:",
        "  - Predicates must be ABSTRACT — no game-pad buttons, no DOM xpaths,",
        "    no specific game/web vocabulary unique to this task.",
        "  - The predicate must paraphrase the original skill's semantics, not",
        "    invent new behaviour.",
        "  - 'transferable_to_cohorts' should err on the side of inclusiveness",
        "    only when the template genuinely fits — e.g. an answer-commit",
        "    template fits both vr_image and vr_video; an evade-jump template",
        "    fits only games.",
    ])


def _strip_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


def _extract_json(text: str):
    text = _strip_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    depth = 0; start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start: i + 1])
                except Exception:
                    start = -1
    return None


def _client():
    from openai import OpenAI  # type: ignore
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"])
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def main() -> int:
    client = _client()
    print(f"\n{'='*100}")
    print("SMOKE: lifting 5 skills (one per cohort) into modality-agnostic templates")
    print('='*100)

    for cohort, bank, target_sid in SAMPLES:
        if not bank.exists():
            print(f"\n[{cohort}] missing bank: {bank}")
            continue
        rec = None
        for line in bank.open():
            r = json.loads(line)
            sk = r.get("skill") or r
            if sk.get("skill_id") == target_sid:
                rec = sk
                break
        if rec is None:
            # Fallback: take first record
            rec = json.loads(bank.open().readline())["skill"]

        prompt = _build_prompt(rec, cohort, bank.parent.name)
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=600,
            )
        except Exception as exc:
            print(f"\n[{cohort}] LLM call failed: {exc}")
            continue

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        parsed = _extract_json(text)
        if parsed is None:
            print(f"\n[{cohort}] LLM returned unparsable JSON — raw:\n{text[:400]}")
            continue

        print(f"\n--- COHORT={cohort:<11}  TASK={bank.parent.name:<28}  SKILL={target_sid} ---")
        print(f"  name              : {rec.get('name','')}")
        print(f"  strategic_summary : {(rec.get('strategic_description') or '')[:140]}")
        print(f"  template_signature: {parsed.get('template_signature','')}")
        for i, st in enumerate(parsed.get("template_steps", []), 1):
            print(f"    [{i}] {st.get('op','?'):<9}  {st.get('predicate','')}")
        print(f"  transferable_to   : {parsed.get('transferable_to_cohorts', [])}")

    print(f"\n{'='*100}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
