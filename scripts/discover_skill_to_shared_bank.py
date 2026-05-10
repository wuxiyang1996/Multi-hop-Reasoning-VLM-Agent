"""Backward DISCOVERY path: a new skill was mined / promoted /
crafter-proposed inside one task -> insert into PerTaskBank AND
upsert the lifted abstract into SharedAbstractBank.

This is the second half of the user-requested pipeline:

    "when new skill being discovered, insert in the per-game bank
     and update the shared bank via harness/crafter."

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
    TemplateStep, TwoLayerSkillStore, hash_contract, normalise_skill_id,
    parse_skill_id_decorations,
)

logger = logging.getLogger("discover_skill_to_shared_bank")

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
    proto_raw = inner.get("protocol") or []
    proto_list = proto_raw if isinstance(proto_raw, list) else []
    return BoundConcreteSkill(
        concrete_skill_id=stem,
        task=task,
        abstract_skill_id=stem,
        name=inner.get("name", "") or stem,
        contract=contract,
        protocol=proto_list,
        binding_status="VALIDATED" if binding_source == "mining" else "PENDING",
        binding_source=binding_source,
        raw_skill_id=sid,
        decorations=parse_skill_id_decorations(sid),
    )


def discover_one(
    *,
    bank: TwoLayerSkillStore,
    task: str,
    skill_record: Dict[str, Any],
    binding_source: str = "mining",
    do_llm_lift: bool = True,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Insert one newly-discovered skill into the per-task bank AND
    upsert the lifted abstract into the shared bank."""
    inner = _peek_inner(skill_record)
    sid = inner.get("skill_id", "")
    if not sid:
        return {"ok": False, "reason": "no_skill_id"}
    stem = normalise_skill_id(sid)
    cohort = cohort_for(task)

    # ── 1. Build the concrete binding ────────────────────────────
    concrete = _record_to_concrete(skill_record, task=task,
                                    binding_source=binding_source)

    # ── 2. Try to find an existing abstract match ────────────────
    cands = bank.abstract.by_abstract_id(stem)
    abs_rec: Optional[SharedAbstractSkill] = None
    template_signature = "NO_TEMPLATE"
    template_steps: List[TemplateStep] = []
    if cands:
        # Pick the candidate with the most lineage breadth.
        abs_rec = max(cands, key=lambda r: r.n_bound_tasks)
        template_signature = abs_rec.template_signature
        template_steps = list(abs_rec.template_steps)

    # ── 3. Lift abstract template if (a) no abstract found, (b)
    #       found abstract is NO_TEMPLATE, or (c) caller forces. ──
    lift_diag: Dict[str, Any] = {"lifted": False}
    if do_llm_lift and (abs_rec is None or template_signature == "NO_TEMPLATE"):
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

    # ── 4. Build / upsert the abstract record ────────────────────
    if abs_rec is None or abs_rec.template_signature != template_signature:
        # Either brand-new, or existing record had a NO_TEMPLATE
        # placeholder while we just lifted a real signature: emit a
        # fresh abstract record (the bank de-dups on
        # (skill_id, signature) pairs anyway).
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

    logger.info("ingesting %d record(s) from %s", len(records), src_path)

    n_ok = 0
    n_new_abstract = 0
    n_merged_abstract = 0
    for r in records:
        try:
            rep = discover_one(
                bank=bank, task=args.task, skill_record=r,
                binding_source=args.binding_source,
                do_llm_lift=not args.no_llm_lift,
                model=args.model,
            )
        except Exception as exc:                                       # noqa: BLE001
            rep = {"ok": False, "reason": repr(exc)}
        logger.info("  %s", json.dumps(rep))
        if rep.get("ok"):
            n_ok += 1
            v = rep["verdicts"]
            if v.get("abstract") == "new":
                n_new_abstract += 1
            elif v.get("abstract") == "merged":
                n_merged_abstract += 1

    logger.info("done: %d ok / %d total — abstract new=%d merged=%d",
                n_ok, len(records), n_new_abstract, n_merged_abstract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
