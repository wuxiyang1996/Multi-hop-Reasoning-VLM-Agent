"""Backward DISCOVERY path: a new skill was mined / promoted /
crafter-proposed inside one task -> insert into PerTaskBank AND
upsert the lifted abstract into SharedAbstractBank.

This is the second half of the user-requested pipeline:

    "when new skill being discovered, insert in the per-game bank
     and update the shared bank via harness/crafter."

Discovery matching strategy (priority order):

  1. **Plan-level LLM judge** (default): lift the new skill's
     reasoning plan, compare against existing mega-skills' plans
     via LLM judge.  If score >= threshold (default 4), merge into
     the best-matching mega-skill.  This finds matches that name-
     based lookup would miss (same reasoning procedure, different
     name).

  2. **Name-based lookup** (fallback): match by normalised
     skill_id stem.  Used when ``--no-plan-judge`` is set or when
     the LLM judge finds no match.

  3. **Create new**: if neither method finds a match, create a
     brand-new SharedAbstractSkill.

Two upstream channels feed this script:

  1. **Mining / promotion**: the orchestrator's promotion hook
     accepts a new ``SkillRecord``, then invokes
     :func:`discover_one` here.  We compute a hash, write the
     concrete binding, lift the abstract template via GPT-5.4
     (reusing :mod:`scripts.lift_skill_templates_gpt54`), then
     upsert into the abstract bank.

  2. **Crafter v2 acceptance**: the crafter pipeline accepts a
     proposal (e.g. ``REASON/EVADE#v2:9a370dbc``) and persists it
     to ``runs/<run>/crafter_v2_offline/.../accepted.jsonl``.  We
     loop that file with ``--from-crafter-accepted``.

Idempotent: re-running on the same record either updates lineage
counts or produces ``binding=updated`` / ``abstract=merged`` and
no duplicates.

Single-record CLI usage::

    python scripts/discover_skill_to_shared_bank.py \\
        --bank-root shared_skill_bank/_latest \\
        --task candy_crush \\
        --from-skill-bank labeling/.../candy_crush/skill_bank.jsonl \\
        --skill-id COMMIT/CLEAR

Bulk crafter-accept ingestion::

    python scripts/discover_skill_to_shared_bank.py \\
        --bank-root shared_skill_bank/_latest \\
        --from-crafter-accepted runs/.../accepted.jsonl \\
        --task tf3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.lift_skill_templates_gpt54 import (                       # noqa: E402
    DEFAULT_MODEL, _coerce_template, _extract_json, _get_openai_client,
)
from scripts.build_shared_skill_bank import (                          # noqa: E402
    cohort_for, _peek_inner, _load_jsonl,
)
from skill_bank.shared_abstract_bank import (                          # noqa: E402
    BoundConcreteSkill, LineageEntry, ProtocolStep, SharedAbstractSkill,
    SubEpisodeRef, TemplateStep, TwoLayerSkillStore, hash_contract,
    normalise_skill_id, parse_skill_id_decorations,
)

logger = logging.getLogger("discover_skill_to_shared_bank")

# ---------------------------------------------------------------------------
# Plan-level judge: match new skill against existing mega-skills by reasoning
# ---------------------------------------------------------------------------
PLAN_JUDGE_SYSTEM = """You are an expert reasoning-plan analyst. You will receive one NEW skill's reasoning plan and a list of EXISTING mega-skill plans from the shared bank.

Your job: identify which existing mega-skill (if any) shares the SAME transferable cognitive procedure as the new skill — meaning a human expert seeing ONLY the step-by-step reasoning (no task names) would recognize them as the same general strategy.

For EACH candidate, rate:
  5 = IDENTICAL procedure — same cognitive strategy, different domain objects
  4 = HIGHLY SIMILAR — same core strategy with minor variations
  3 = MODERATELY SIMILAR — overlapping reasoning but meaningful differences
  2 = WEAKLY SIMILAR — same structure, different cognitive challenge
  1 = DIFFERENT — fundamentally different procedures

You MUST respond with ONLY a JSON object (no markdown):
{
  "best_match": "<candidate letter with highest score, or 'none'>",
  "best_score": <1-5 or 0 if none>,
  "matches": [
    {
      "candidate_id": "<A/B/C/...>",
      "score": <1-5>,
      "shared_reasoning": "<1-sentence shared strategy>"
    }
  ],
  "new_skill_summary": "<1-sentence summary of the new skill's reasoning>"
}

Only include candidates with score >= 3 in the matches list."""

PLAN_JUDGE_USER = """## NEW skill  (task: {task})
Name: {name}
Reasoning plan:
{plan_text}

---

## EXISTING mega-skills in shared bank

{candidates_text}

---

Which existing mega-skill (if any) shares the same reasoning procedure? Respond with JSON only."""


def _format_plan_text(skill_record: Dict[str, Any]) -> str:
    """Extract a human-readable reasoning plan from a skill record."""
    inner = _peek_inner(skill_record)
    protocol = inner.get("protocol") or []
    lines = []
    if isinstance(protocol, list):
        for i, s in enumerate(protocol[:8]):
            if isinstance(s, dict):
                op = s.get("op", "?")
                notes = s.get("notes", "")
                predicate = s.get("predicate", notes)
                lines.append(f"  {i+1}. [{op}] {predicate[:100]}")
    elif isinstance(protocol, dict):
        for i, s in enumerate((protocol.get("steps") or [])[:8]):
            if isinstance(s, str):
                lines.append(f"  {i+1}. {s[:100]}")
    return "\n".join(lines) if lines else "(no explicit steps)"


def _format_candidates(abstracts: List[SharedAbstractSkill]) -> str:
    """Format existing mega-skills as lettered candidates."""
    lines = []
    for i, a in enumerate(abstracts):
        letter = chr(65 + i)
        step_text = "\n".join(
            f"    {j+1}. [{s.op}] {s.predicate}"
            for j, s in enumerate(a.template_steps)
        )
        if not step_text:
            step_text = f"    (signature: {a.template_signature})"
        lines.append(f"### Candidate {letter}: {a.name[:80]}")
        lines.append(f"  ID: {a.abstract_skill_id}")
        lines.append(f"  Signature: {a.template_signature}")
        lines.append(f"  Plan:")
        lines.append(step_text)
        lines.append("")
    return "\n".join(lines)


def _plan_judge_match(
    skill_record: Dict[str, Any],
    *,
    task: str,
    bank: "TwoLayerSkillStore",
    model: str = DEFAULT_MODEL,
    threshold: int = 4,
    max_candidates: int = 20,
) -> Optional[Dict[str, Any]]:
    """Compare new skill's reasoning plan against existing mega-skills
    via LLM judge. Returns the best match info or None."""
    all_abstracts = bank.abstract.records
    if not all_abstracts:
        return None

    multi_task = [a for a in all_abstracts if a.n_bound_tasks >= 2]
    candidates = multi_task[:max_candidates] if multi_task else all_abstracts[:max_candidates]
    if not candidates:
        return None

    plan_text = _format_plan_text(skill_record)
    inner = _peek_inner(skill_record)
    name = inner.get("name", inner.get("skill_id", "unknown"))

    user_msg = PLAN_JUDGE_USER.format(
        task=task,
        name=name,
        plan_text=plan_text,
        candidates_text=_format_candidates(candidates),
    )

    client = _get_openai_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PLAN_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=800,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            best_id = result.get("best_match", "none")
            best_score = result.get("best_score", 0)
            if best_id == "none" or best_score < threshold:
                return None

            if len(best_id) == 1:
                idx = ord(best_id) - 65
                if 0 <= idx < len(candidates):
                    matched_abstract = candidates[idx]
                    return {
                        "abstract": matched_abstract,
                        "score": best_score,
                        "shared_reasoning": next(
                            (m.get("shared_reasoning", "")
                             for m in result.get("matches", [])
                             if m.get("candidate_id") == best_id),
                            "",
                        ),
                        "method": "plan_judge",
                    }
            return None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("  plan-judge attempt %d failed: %s", attempt + 1, e)
            import time
            time.sleep(1.5 * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
LIFT_SYSTEM_PROMPT = (
    "You are an expert at abstracting skills mined from one task "
    "into a modality-agnostic procedural template.  You will be "
    "given ONE concrete skill (its name, contract, and multi-hop "
    "protocol) and must return its 2-5 step Layer-C template using "
    "ops drawn from {PERCEIVE, RECALL, COMPARE, FILTER, DECIDE, "
    "COMMIT, VERIFY, RECOVER}.  Respond with strict JSON."
)


def _lift_template(
    skill_record: Dict[str, Any], *, task: str, model: str = DEFAULT_MODEL,
) -> Optional[Dict[str, Any]]:
    """Reproduces the ``lift_skill_templates_gpt54`` prompt for a
    single concrete record and returns the parsed template dict
    (with keys ``template_signature`` and ``template_steps``)."""
    inner = _peek_inner(skill_record)
    sid = inner.get("skill_id", "")
    name = inner.get("name", sid)
    contract = inner.get("contract") or {}
    protocol = inner.get("protocol") or []
    proto_lines: List[str] = []
    if isinstance(protocol, list):
        for s in protocol[:8]:
            if isinstance(s, dict):
                proto_lines.append(
                    f"  - op={s.get('op','?'):<10} notes={s.get('notes','')[:80]}"
                )
    elif isinstance(protocol, dict):
        for s in (protocol.get("steps") or [])[:8]:
            if isinstance(s, str):
                proto_lines.append(f"  - {s[:80]}")

    prompt = "\n".join([
        f"TASK             : {task}  (cohort={cohort_for(task)})",
        f"SKILL_ID         : {sid}",
        f"NAME             : {name}",
        "CONTRACT:",
        f"  preconditions      : {contract.get('preconditions') or []}",
        f"  postconditions     : {contract.get('postconditions') or []}",
        f"  example_predicates : {contract.get('example_predicates') or []}",
        f"  eff_add            : {contract.get('eff_add') or []}",
        f"  eff_del            : {contract.get('eff_del') or []}",
        "PROTOCOL:",
        *(proto_lines or ["  (none)"]),
        "",
        "Output strict JSON:",
        "{",
        '  "template_signature": "<arrow-joined op chain, 2-5 ops>",',
        '  "template_steps": [',
        '     {"op": "<one of the 8 ops>", "predicate": "<6-12 word abstract description>"},',
        "     ...",
        "  ],",
        '  "rationale": "<≤60 words>"',
        "}",
        "",
        "RULES: predicates must NOT contain task-specific tokens.",
    ])

    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LIFT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0, max_tokens=600,
    )
    text = resp.choices[0].message.content if resp.choices else ""
    return _extract_json(text or "")


# ---------------------------------------------------------------------------
def _record_to_concrete(
    skill_record: Dict[str, Any], *, task: str,
    binding_source: str = "mining",
) -> BoundConcreteSkill:
    inner = _peek_inner(skill_record)
    sid = inner.get("skill_id", "")
    stem = normalise_skill_id(sid)
    contract = inner.get("contract") or {}
    # Rich protocol (op / payload / slot_types / effects).
    proto_raw = inner.get("protocol") or []
    protocol_steps: List[ProtocolStep] = []
    if isinstance(proto_raw, list):
        for s in proto_raw:
            if isinstance(s, dict):
                protocol_steps.append(ProtocolStep.from_dict(s))
    elif isinstance(proto_raw, dict):
        for s in (proto_raw.get("steps") or []):
            if isinstance(s, str):
                protocol_steps.append(ProtocolStep(op="?", notes=s))
    # Receipts.
    sub_eps_raw = inner.get("sub_episodes") or []
    sub_episodes = [SubEpisodeRef.from_dict(s) for s in sub_eps_raw
                     if isinstance(s, dict)]
    for s in sub_episodes:
        if not s.task:
            s.task = task
    return BoundConcreteSkill(
        concrete_skill_id=stem,
        task=task,
        abstract_skill_id=stem,
        name=inner.get("name", "") or stem,
        protocol=protocol_steps,
        sub_episodes=sub_episodes,
        contract=contract,
        binding_status="VALIDATED" if binding_source == "mining" else "PENDING",
        binding_source=binding_source,
        raw_skill_id=sid,
        decorations=parse_skill_id_decorations(sid),
        n_episodes_verified=len(sub_episodes),
    )


def discover_one(
    *,
    bank: TwoLayerSkillStore,
    task: str,
    skill_record: Dict[str, Any],
    binding_source: str = "mining",
    do_llm_lift: bool = True,
    do_plan_judge: bool = True,
    plan_judge_threshold: int = 4,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Insert one newly-discovered skill into the per-task bank AND
    upsert the lifted abstract into the shared bank.

    Matching strategy (priority order):
      1. Plan-level LLM judge: compare reasoning plan against existing
         mega-skills.  If score >= threshold, merge into best match.
      2. Name-based lookup: match by normalised skill_id stem.
      3. Create new SharedAbstractSkill if neither matches.
    """
    inner = _peek_inner(skill_record)
    sid = inner.get("skill_id", "")
    if not sid:
        return {"ok": False, "reason": "no_skill_id"}
    stem = normalise_skill_id(sid)
    cohort = cohort_for(task)

    # ── 1. Build the concrete binding ────────────────────────────
    concrete = _record_to_concrete(skill_record, task=task,
                                    binding_source=binding_source)

    # ── 2. Lift abstract template first (needed for judge) ───────
    template_signature = "NO_TEMPLATE"
    template_steps: List[TemplateStep] = []
    lift_diag: Dict[str, Any] = {"lifted": False}
    if do_llm_lift:
        try:
            parsed = _lift_template(skill_record, task=task, model=model)
            if parsed:
                t = _coerce_template(parsed)
                if t:
                    template_signature = t["template_signature"]
                    template_steps = [TemplateStep.from_dict(s)
                                       for s in t["template_steps"]]
                    lift_diag = {"lifted": True,
                                 "template_signature": template_signature}
        except Exception as exc:                                       # noqa: BLE001
            lift_diag = {"lifted": False, "error": repr(exc)}

    # ── 3. Try to find an existing abstract match ────────────────
    abs_rec: Optional[SharedAbstractSkill] = None
    match_method = "none"
    match_diag: Dict[str, Any] = {}

    # 3a. Plan-level LLM judge (DEFAULT — matches by reasoning procedure)
    if do_plan_judge and do_llm_lift:
        judge_result = _plan_judge_match(
            skill_record,
            task=task,
            bank=bank,
            model=model,
            threshold=plan_judge_threshold,
        )
        if judge_result:
            abs_rec = judge_result["abstract"]
            template_signature = abs_rec.template_signature
            template_steps = list(abs_rec.template_steps)
            match_method = "plan_judge"
            match_diag = {
                "judge_score": judge_result["score"],
                "shared_reasoning": judge_result["shared_reasoning"],
                "matched_abstract_id": abs_rec.abstract_skill_id,
            }
            logger.info("  plan-judge matched %s → %s (score=%d: %s)",
                        stem, abs_rec.abstract_skill_id,
                        judge_result["score"],
                        judge_result["shared_reasoning"][:80])

    # 3b. Name-based lookup (FALLBACK)
    if abs_rec is None:
        cands = bank.abstract.by_abstract_id(stem)
        if cands:
            abs_rec = max(cands, key=lambda r: r.n_bound_tasks)
            template_signature = abs_rec.template_signature
            template_steps = list(abs_rec.template_steps)
            match_method = "name_stem"
            match_diag = {"matched_abstract_id": abs_rec.abstract_skill_id}

    # ── 4. Build / upsert the abstract record ────────────────────
    if abs_rec is None or abs_rec.template_signature != template_signature:
        abs_rec = SharedAbstractSkill(
            abstract_skill_id=stem,
            name=inner.get("name", "") or stem,
            template_signature=template_signature,
            template_steps=list(template_steps),
            protocol_steps=[
                ProtocolStep.from_dict(s)
                for s in (inner.get("protocol") or [])
                if isinstance(s, dict)
            ],
            discovered_via=binding_source,
        )
        match_method = "new_abstract"

    chash = hash_contract(concrete.contract)
    lineage = LineageEntry(
        task=task,
        concrete_skill_id=stem,
        raw_skill_id=sid,
        cohort=cohort,
        discovered_via=binding_source,
        is_native=True,
        contract_hash=chash,
        decorations=parse_skill_id_decorations(sid),
    )

    verdicts = bank.insert_discovered_skill(
        concrete=concrete, abstract=abs_rec, lineage=lineage,
    )
    return {
        "ok": True,
        "task": task,
        "stem": stem,
        "raw_skill_id": sid,
        "verdicts": verdicts,
        "match_method": match_method,
        "match_diag": match_diag,
        "lift_diag": lift_diag,
        "template_signature": template_signature,
        "n_template_steps": len(template_steps),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--from-skill-bank", default=None,
                    help="Path to a skill_bank.jsonl from a mining run.")
    ap.add_argument("--from-crafter-accepted", default=None,
                    help="Path to a crafter accepted.jsonl.")
    ap.add_argument("--skill-id", default=None,
                    help="If set, only ingest the matching record.")
    ap.add_argument("--binding-source", default="mining",
                    choices=["mining", "crafter", "promotion"])
    ap.add_argument("--no-llm-lift", action="store_true",
                    help="Skip the abstract-template lift (debug).")
    ap.add_argument("--no-plan-judge", action="store_true",
                    help="Disable plan-level judge matching (use name-only).")
    ap.add_argument("--plan-judge-threshold", type=int, default=4,
                    help="Minimum judge score to merge into existing mega-skill (default: 4).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-records", type=int, default=0,
                    help="Cap (for cheap demos).  0 = no cap.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    bank = TwoLayerSkillStore(Path(args.bank_root))
    bank.abstract.load()

    src_path: Optional[Path] = None
    if args.from_skill_bank:
        src_path = Path(args.from_skill_bank)
    elif args.from_crafter_accepted:
        src_path = Path(args.from_crafter_accepted)
        args.binding_source = "crafter"
    else:
        logger.error("either --from-skill-bank or --from-crafter-accepted required")
        return 2

    records = _load_jsonl(src_path)
    if args.skill_id:
        records = [r for r in records
                   if _peek_inner(r).get("skill_id") == args.skill_id]

    if args.max_records:
        records = records[:args.max_records]

    do_plan_judge = not args.no_plan_judge
    logger.info("ingesting %d record(s) from %s (plan_judge=%s, threshold=%d)",
                len(records), src_path, do_plan_judge, args.plan_judge_threshold)

    n_ok = 0
    n_new_abstract = 0
    n_merged_abstract = 0
    match_method_counts: Dict[str, int] = {}
    for r in records:
        try:
            rep = discover_one(
                bank=bank, task=args.task, skill_record=r,
                binding_source=args.binding_source,
                do_llm_lift=not args.no_llm_lift,
                do_plan_judge=do_plan_judge,
                plan_judge_threshold=args.plan_judge_threshold,
                model=args.model,
            )
        except Exception as exc:                                       # noqa: BLE001
            rep = {"ok": False, "reason": repr(exc)}
        logger.info("  %s", json.dumps(rep, default=str))
        if rep.get("ok"):
            n_ok += 1
            v = rep["verdicts"]
            if v.get("abstract") == "new":
                n_new_abstract += 1
            elif v.get("abstract") == "merged":
                n_merged_abstract += 1
            method = rep.get("match_method", "unknown")
            match_method_counts[method] = match_method_counts.get(method, 0) + 1

    logger.info("═" * 60)
    logger.info("done: %d ok / %d total", n_ok, len(records))
    logger.info("  abstract: new=%d  merged=%d", n_new_abstract, n_merged_abstract)
    logger.info("  match methods: %s", match_method_counts)
    if match_method_counts.get("plan_judge", 0):
        logger.info("  → %d skills matched via plan-level judge (same reasoning procedure)",
                    match_method_counts["plan_judge"])
    if match_method_counts.get("new_abstract", 0):
        logger.info("  → %d new mega-skills created", match_method_counts["new_abstract"])
    logger.info("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
