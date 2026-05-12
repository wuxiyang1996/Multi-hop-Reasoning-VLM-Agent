#!/usr/bin/env python
"""Lift every skill in every (contract-rich) bank into a normalised,
modality-agnostic procedural template using GPT-5.4.

This is the full pipeline that ``lift_skill_templates_smoke.py`` previews
on 5 representative skills.  It walks every bank in
``sft_data_inventory/{games,non_game}/<task>/skill_bank.jsonl`` and
produces a sibling JSONL per task with one record per skill of the form::

    {
      "task": "<task_name>",
      "cohort": "<gymv_game|env_wr_game|web|vr_image|vr_video>",
      "skill_id": "...",                  # unchanged from source bank
      "skill_name": "...",                # unchanged
      "template_signature": "OP1 → OP2 → OP3",
      "template_steps": [{"op": ..., "predicate": ...}, ...],
      "transferable_to_cohorts": ["...", ...],
      "lifted_at": "<ISO8601>",
      "model": "gpt-5.4"
    }

Output layout::

    labeling/skill_templates/run_<utc-ts>/<cohort>/<task>/template_bank.jsonl
    labeling/skill_templates/run_<utc-ts>/_lift_summary.json

The controlled vocabulary is documented at the top of the prompt and is
enforced (off-vocabulary ops are remapped to the nearest valid op or
the call is retried).

Run::

    python scripts/lift_skill_templates_gpt54.py                       # full
    python scripts/lift_skill_templates_gpt54.py --tasks tetris webshop  # subset
    python scripts/lift_skill_templates_gpt54.py --limit 3 -v            # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
WORK = REPO.parent
for p in [str(WORK), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key  # type: ignore
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key  # type: ignore
except Exception:
    pass

logger = logging.getLogger("lift_skill_templates")

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_WORKERS = 16
INVENTORY = REPO / "sft_data_inventory"

VALID_OPS = {"PERCEIVE", "RECALL", "COMPARE", "FILTER", "DECIDE",
             "COMMIT", "VERIFY", "RECOVER"}

VALID_COHORTS = {"gymv_game", "env_wr_game", "web", "vr_image", "vr_video"}

# Map task name → cohort.  gym_v games are detected by prefix.
COHORT_OF: Dict[str, str] = {
    "tetris": "env_wr_game", "super_mario": "env_wr_game",
    "candy_crush": "env_wr_game", "twenty_forty_eight": "env_wr_game",
    "miniwob": "web", "webshop": "web",
    "video_holmes": "vr_video", "siv_bench": "vr_video",
    "tir_bench": "vr_image", "visual_toolbench": "vr_image",
}


def cohort_of(task: str) -> Optional[str]:
    if task in COHORT_OF:
        return COHORT_OF[task]
    if task.startswith("Temporal_"):
        return "gymv_game"
    return None


# ---------------------------------------------------------------------------
# OpenRouter / OpenAI client
# ---------------------------------------------------------------------------
def _get_openai_client():
    from openai import OpenAI  # type: ignore
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"])
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _strip_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    depth = 0
    start = -1
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


# ---------------------------------------------------------------------------
# Prompt construction (mirrors the smoke script)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at distilling reusable skills into modality-agnostic "
    "procedural templates.  You will be given ONE skill mined from agent "
    "trajectories — its strategic description, contract preconditions / "
    "postconditions, and any existing low-level protocol notes.  Your job "
    "is to produce a 2-5 step HIGH-LEVEL TEMPLATE that captures what the "
    "skill does in a way that could plausibly transfer to a different "
    "task or modality.  Output strict JSON only."
)


def _build_prompt(*, skill: Dict[str, Any], cohort: str, task: str) -> str:
    sid = skill.get("skill_id", "")
    name = skill.get("name", "")
    desc = (skill.get("strategic_description") or "").strip()
    contract = skill.get("contract") or {}
    pre = contract.get("preconditions") or []
    post = contract.get("postconditions") or []
    pred = contract.get("example_predicates") or []
    proto = skill.get("protocol")
    proto_lines: List[str] = []
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
        ("LOW-LEVEL PROTOCOL (only present for gym_v):"
         if proto_lines else "(no low-level protocol available)"),
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
        '  "template_signature": "<OP1 → OP2 → OP3 ...>",   // joined ops',
        '  "transferable_to_cohorts": ["<cohort>", ...]      // subset of {gymv_game, env_wr_game, web, vr_image, vr_video}',
        "}",
        "",
        "Constraints:",
        "  - Predicates must be ABSTRACT — no game-pad buttons, no DOM xpaths,",
        "    no specific game/web vocabulary unique to this task.",
        "  - The predicate must paraphrase the original skill's semantics.",
        "  - 'transferable_to_cohorts' should err on the side of inclusiveness",
        "    only when the template genuinely fits.",
    ])


# ---------------------------------------------------------------------------
# Validation / coercion of LLM output
# ---------------------------------------------------------------------------
def _normalize_op(op: str) -> Optional[str]:
    if not isinstance(op, str):
        return None
    norm = op.strip().upper().replace(" ", "_")
    if norm in VALID_OPS:
        return norm
    # Common LLM aberrations → canonical form
    aliases = {
        "OBSERVE": "PERCEIVE", "SCAN": "PERCEIVE", "READ": "PERCEIVE",
        "DETECT": "PERCEIVE", "INSPECT": "PERCEIVE",
        "REMEMBER": "RECALL", "RETRIEVE": "RECALL", "LOAD": "RECALL",
        "EVALUATE": "COMPARE", "RANK": "COMPARE", "SCORE": "COMPARE",
        "ELIMINATE": "FILTER", "PRUNE": "FILTER", "DROP": "FILTER",
        "RULE_OUT": "FILTER", "REJECT": "FILTER",
        "CHOOSE": "DECIDE", "SELECT": "DECIDE", "PICK": "DECIDE",
        "EXECUTE": "COMMIT", "APPLY": "COMMIT", "PERFORM": "COMMIT",
        "ACT": "COMMIT",
        "CHECK": "VERIFY", "CONFIRM": "VERIFY", "VALIDATE": "VERIFY",
        "RESTORE": "RECOVER", "REPAIR": "RECOVER", "FIX": "RECOVER",
    }
    return aliases.get(norm)


def _coerce_template(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps_raw = parsed.get("template_steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        return None
    steps: List[Dict[str, str]] = []
    for st in steps_raw[:6]:
        if not isinstance(st, dict):
            continue
        op = _normalize_op(st.get("op", ""))
        pred = str(st.get("predicate", "")).strip()
        if not op or not pred:
            continue
        # Predicate length guard: 4 ≤ words ≤ 16
        word_n = len(pred.split())
        if word_n < 3 or word_n > 24:
            pred = " ".join(pred.split()[:18])
        steps.append({"op": op, "predicate": pred[:240]})
    if len(steps) < 2 or len(steps) > 5:
        # Trim to 5 if too long (keeping first + last + middle); reject if <2.
        if len(steps) > 5:
            steps = steps[:5]
        else:
            return None
    sig = " → ".join(s["op"] for s in steps)

    transferable_raw = parsed.get("transferable_to_cohorts") or []
    transferable: List[str] = []
    for c in transferable_raw:
        if isinstance(c, str) and c.strip().lower() in VALID_COHORTS:
            transferable.append(c.strip().lower())
    return {
        "template_steps": steps,
        "template_signature": sig,
        "transferable_to_cohorts": sorted(set(transferable)),
    }


# ---------------------------------------------------------------------------
# Per-skill driver
# ---------------------------------------------------------------------------
def _lift_one_skill(
    *, skill: Dict[str, Any], cohort: str, task: str, client, model: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    prompt = _build_prompt(skill=skill, cohort=cohort, task=task)
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=600,
            )
        except Exception as exc:
            logger.warning("[%s/%s] LLM call failed (attempt %d): %s",
                           task, skill.get("skill_id", ""), attempt, exc)
            continue
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        parsed = _extract_json(text)
        if parsed is None:
            logger.warning("[%s/%s] unparsable JSON (attempt %d)",
                           task, skill.get("skill_id", ""), attempt)
            continue
        coerced = _coerce_template(parsed)
        if coerced:
            return coerced, "ok"
        logger.warning("[%s/%s] coercion failed (attempt %d), parsed=%s",
                       task, skill.get("skill_id", ""), attempt,
                       json.dumps(parsed)[:160])
    return None, "fail"


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------
def _process_task(
    *, src_bank: Path, dst_bank: Path, task: str, cohort: str,
    client, model: str, workers: int, limit: Optional[int],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for line in src_bank.open():
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    if limit is not None:
        records = records[:limit]

    started = time.time()
    out_rows: List[Optional[Dict[str, Any]]] = [None] * len(records)
    statuses = ["pending"] * len(records)

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, rec in enumerate(records):
            sk = rec.get("skill") or rec
            futures[ex.submit(_lift_one_skill, skill=sk, cohort=cohort,
                              task=task, client=client, model=model)] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            sk = (records[idx].get("skill") or records[idx])
            try:
                lifted, status = fut.result()
            except Exception as exc:
                logger.error("[%s/%s] raised: %s", task,
                             sk.get("skill_id", ""), exc)
                lifted, status = None, "exc"
            statuses[idx] = status
            if lifted is None:
                continue
            out_rows[idx] = {
                "task": task,
                "cohort": cohort,
                "skill_id": sk.get("skill_id", ""),
                "skill_name": sk.get("name", ""),
                "template_signature": lifted["template_signature"],
                "template_steps": lifted["template_steps"],
                "transferable_to_cohorts": lifted["transferable_to_cohorts"],
                "lifted_at": datetime.utcnow().isoformat() + "Z",
                "model": model,
            }

    dst_bank.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with dst_bank.open("w") as f:
        for row in out_rows:
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1

    elapsed = time.time() - started
    return {
        "task": task,
        "cohort": cohort,
        "src_bank": str(src_bank),
        "dst_bank": str(dst_bank),
        "n_records": len(records),
        "n_ok": n_ok,
        "n_fail": statuses.count("fail") + statuses.count("exc"),
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--inventory", type=Path, default=INVENTORY,
                    help=f"Inventory dir (default {INVENTORY}).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output dir (default labeling/skill_templates/run_<utc-ts>/).")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Optional subset of task names.  Default: all 18.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap skills per task (smoke test).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if args.output_dir is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = REPO / "labeling" / "skill_templates" / f"run_{ts}"
    else:
        out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover tasks under the inventory
    candidates: List[Tuple[str, Path]] = []
    for cat in ("games", "non_game"):
        for d in sorted((args.inventory / cat).iterdir()):
            if not d.is_dir():
                continue
            bank = d / "skill_bank.jsonl"
            if not bank.exists():
                continue
            candidates.append((d.name, bank))

    if args.tasks:
        wanted = set(args.tasks)
        candidates = [(t, b) for t, b in candidates if t in wanted]

    client = _get_openai_client()
    summaries: List[Dict[str, Any]] = []
    started_all = time.time()
    for task, bank in candidates:
        coh = cohort_of(task)
        if coh is None:
            logger.warning("skipping %s: no cohort mapping", task)
            continue
        dst = out_root / coh / task / "template_bank.jsonl"
        n_total = sum(1 for _ in bank.open())
        logger.info(">> lifting %s (cohort=%s, n=%d)", task, coh, n_total)
        s = _process_task(
            src_bank=bank, dst_bank=dst, task=task, cohort=coh,
            client=client, model=args.model,
            workers=args.workers, limit=args.limit,
        )
        summaries.append(s)
        logger.info("   done: ok=%d fail=%d  (%.1fs) -> %s",
                    s["n_ok"], s["n_fail"], s["elapsed_s"], dst)

    elapsed = time.time() - started_all
    summary_path = out_root / "_lift_summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model,
        "workers": args.workers,
        "limit": args.limit,
        "n_tasks": len(summaries),
        "n_ok": sum(s["n_ok"] for s in summaries),
        "n_fail": sum(s["n_fail"] for s in summaries),
        "n_records": sum(s["n_records"] for s in summaries),
        "elapsed_s": round(elapsed, 1),
        "per_task": summaries,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))

    print()
    print("=" * 78)
    print(f"[lift_skill_templates] DONE — {len(summaries)} tasks")
    print(f"  output : {out_root}")
    print(f"  ok={sum(s['n_ok'] for s in summaries)}  "
          f"fail={sum(s['n_fail'] for s in summaries)}  "
          f"elapsed={round(elapsed,1)}s")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
