"""Re-lift the ``NO_TEMPLATE`` abstracts in the SharedAbstractBank.

When ``build_shared_skill_bank.py`` consolidates skills mined from
sources that don't carry a Layer-C ``template_signature`` (production
usage logs, crafter v2 proposals, QA-style mining banks), the
abstract is recorded with ``template_signature="NO_TEMPLATE"`` so it
ends up in the bank alongside its templated peers.  This script
walks every NO_TEMPLATE abstract, builds a synthetic skill dict from
its existing ``protocol_steps`` + lineage evidence, runs
``scripts.lift_skill_templates_gpt54._lift_one_skill`` on it (GPT-5.4
by default), and:

  * If the lift succeeds, the record's ``template_signature`` and
    ``template_steps`` are updated **in place** (the lineage,
    ``protocol_steps``, and ``cohorts_seen`` are preserved verbatim).
    Because :class:`SharedAbstractBank` keys records on
    ``(abstract_skill_id, template_signature)``, the lifted record
    lands under a NEW key — we explicitly drop the old NO_TEMPLATE
    entry so the bank doesn't carry a stale duplicate.

  * If the lift fails (LLM didn't return well-formed JSON, or the
    coercer couldn't validate the steps), the record is left as
    NO_TEMPLATE and reported in the failure summary.

The script writes the upgraded ``abstract.jsonl`` in place (the
PerTaskBank under ``by_task/`` is untouched).  A ``.bak`` of the
original is dropped beside it so the change is reversible.

Invocation::

    OPENROUTER_API_KEY=$KEY python -m scripts.relift_no_template_abstracts \\
        --bank-root shared_skill_bank/_latest \\
        --workers 8

Cost: ~157 LLM calls × 1 attempt = under 200 GPT-5.4 chat completions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from skill_bank.shared_abstract_bank import (                            # noqa: E402
    SharedAbstractBank, SharedAbstractSkill, TemplateStep,
)
from scripts.lift_skill_templates_gpt54 import (                          # noqa: E402
    DEFAULT_MODEL, _get_openai_client, _lift_one_skill, cohort_of,
)

logger = logging.getLogger("relift_no_template")


# ---------------------------------------------------------------------------
# Build a synthetic "skill" dict for the lifter.
# ---------------------------------------------------------------------------
def _abstract_to_skill_dict(
    rec: SharedAbstractSkill,
) -> Dict[str, Any]:
    """The lift prompt expects the schema produced by the legacy
    mining bank: ``{skill_id, name, strategic_description, contract,
    protocol}``.  Synthesise that from a SharedAbstractSkill's own
    fields plus its richest lineage entry."""
    notes_lines: List[str] = []
    for s in rec.protocol_steps:
        if s.notes:
            notes_lines.append(str(s.notes))

    # If protocol_steps is empty (often the case for production_usage-
    # only abstracts), fall back to the most-used lineage entry as
    # context so the LLM at least sees the source task name.
    most_used = max(rec.lineage, default=None,
                    key=lambda L: (L.n_uses, L.n_success))
    src_task = most_used.task if most_used else "(unknown)"

    n_tasks  = rec.n_native_tasks
    desc = (
        f"Cross-task skill {rec.abstract_skill_id!r} "
        f"(native in {n_tasks} task(s); most-used source = {src_task}). "
        f"Production aggregate: {rec.total_production_successes}"
        f"/{rec.total_production_uses} successes/uses across "
        f"{len(rec.cohorts_seen)} cohort(s)."
    )
    proto: List[Dict[str, Any]] = []
    for s in rec.protocol_steps:
        proto.append({
            "op":    s.op or "?",
            "notes": s.notes or "",
        })
    return {
        "skill_id": rec.abstract_skill_id,
        "name":     rec.name or rec.abstract_skill_id,
        "strategic_description": desc,
        "contract": {
            "preconditions":  [],
            "postconditions": [],
            "example_predicates": [],
        },
        "protocol": proto,
    }


def _cohort_for_abstract(rec: SharedAbstractSkill) -> str:
    """Pick a cohort label for the lift prompt — 'mixed' when the
    abstract spans multiple cohorts."""
    if not rec.cohorts_seen:
        # Fallback: derive from the most-used lineage entry.
        most_used = max(rec.lineage, default=None,
                        key=lambda L: (L.n_uses, L.n_success))
        if most_used:
            c = cohort_of(most_used.task)
            return c or "gymv_game"
        return "gymv_game"
    if len(rec.cohorts_seen) == 1:
        return rec.cohorts_seen[0]
    return "mixed"


# ---------------------------------------------------------------------------
# Per-record driver
# ---------------------------------------------------------------------------
def _relift_one(
    rec: SharedAbstractSkill, client, model: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if rec.template_signature != "NO_TEMPLATE":
        return None, "skipped_already_lifted"
    skill = _abstract_to_skill_dict(rec)
    cohort = _cohort_for_abstract(rec)
    most_used = max(rec.lineage, default=None,
                    key=lambda L: (L.n_uses, L.n_success))
    task = most_used.task if most_used else "(unknown)"
    return _lift_one_skill(
        skill=skill, cohort=cohort, task=task,
        client=client, model=model,
    )


# ---------------------------------------------------------------------------
# I/O — read/write abstract.jsonl in-place
# ---------------------------------------------------------------------------
def _read_records(jsonl_path: Path) -> List[SharedAbstractSkill]:
    """Read ``abstract.jsonl`` keeping only the latest record per
    ``stable_key`` (mirrors :meth:`SharedAbstractBank.load`)."""
    by_key: Dict[Tuple[str, str], SharedAbstractSkill] = {}
    if not jsonl_path.exists():
        return []
    for line in jsonl_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        rec = SharedAbstractSkill.from_dict(d)
        key = rec.stable_key()
        prev = by_key.get(key)
        if prev is None or rec.updated_at >= prev.updated_at:
            by_key[key] = rec
    return list(by_key.values())


def _write_records(
    jsonl_path: Path, records: List[SharedAbstractSkill],
) -> None:
    tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)


# ---------------------------------------------------------------------------
def relift_no_template(
    *,
    bank_root: Path,
    model: str = DEFAULT_MODEL,
    workers: int = 8,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sweep all NO_TEMPLATE abstracts in ``bank_root/abstract.jsonl``
    and re-lift them via GPT-5.4."""
    bank_root = Path(bank_root)
    jsonl_path = bank_root / "abstract.jsonl"

    all_records = _read_records(jsonl_path)
    no_template_all = [r for r in all_records if r.template_signature == "NO_TEMPLATE"]
    other           = [r for r in all_records if r.template_signature != "NO_TEMPLATE"]

    logger.info("loaded %d abstracts from %s (%d NO_TEMPLATE, %d already-lifted)",
                len(all_records), jsonl_path, len(no_template_all), len(other))

    # ``to_process`` is the subset we'll attempt to lift.  Any
    # remaining NO_TEMPLATE records (when ``--limit`` is in effect)
    # are kept verbatim in the rewritten bank — never dropped.
    to_process = list(no_template_all)
    if limit is not None:
        to_process = to_process[:limit]
        logger.info("limiting to first %d NO_TEMPLATE record(s); "
                    "the remaining %d stay as-is",
                    limit, len(no_template_all) - limit)
    unprocessed = no_template_all[len(to_process):]

    if dry_run:
        for r in to_process[:8]:
            logger.info("  would re-lift %-25s n_lineage=%d cohorts=%s",
                        r.abstract_skill_id, r.n_lineage, r.cohorts_seen)
        return {"n_no_template": len(to_process), "dry_run": True}

    if not to_process:
        return {"n_no_template": 0, "n_lifted": 0, "n_failed": 0}

    client = _get_openai_client()
    started = time.time()
    lifted: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_failed = 0
    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_relift_one, rec=r, client=client, model=model): r
            for r in to_process
        }
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                template, status = fut.result()
            except Exception as exc:                                       # noqa: BLE001
                logger.error("[%s] raised: %s", rec.abstract_skill_id, exc)
                template, status = None, "exc"
            n_done += 1
            if template is not None:
                lifted[rec.stable_key()] = template
                logger.info("[%4d/%d] %-25s sig=%s",
                            n_done, len(to_process),
                            rec.abstract_skill_id,
                            template.get("template_signature", "?"))
            else:
                n_failed += 1
                logger.warning("[%4d/%d] %-25s FAILED status=%s",
                               n_done, len(to_process),
                               rec.abstract_skill_id, status)

    elapsed = time.time() - started
    logger.info("re-lift done: %d lifted, %d failed in %.1fs",
                len(lifted), n_failed, elapsed)

    # ── 1. backup the original abstract.jsonl ─────────────────────
    backup = jsonl_path.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(jsonl_path, backup)
        logger.info("backup written to %s", backup)

    # ── 2. apply lifts in-memory ──────────────────────────────────
    upgraded: List[SharedAbstractSkill] = []
    keep_no_template: List[SharedAbstractSkill] = list(unprocessed)
    for r in to_process:
        if r.stable_key() not in lifted:
            keep_no_template.append(r)
            continue
        L = lifted[r.stable_key()]
        new_steps = [TemplateStep(op=s["op"], predicate=s["predicate"])
                      for s in L["template_steps"]]
        # Merge with an existing record under (stem, new_signature) if one
        # already exists — preserves any prior lineage from a parallel
        # mining-era extraction that happened to land under the new key.
        existing_lifted = next(
            (e for e in other
             if e.abstract_skill_id == r.abstract_skill_id
             and e.template_signature == L["template_signature"]),
            None,
        )
        if existing_lifted is not None:
            for ent in r.lineage:
                existing_lifted.upsert_lineage(ent)
            for c in r.cohorts_seen:
                if c not in existing_lifted.cohorts_seen:
                    existing_lifted.cohorts_seen.append(c)
            if not existing_lifted.protocol_steps and r.protocol_steps:
                existing_lifted.protocol_steps = list(r.protocol_steps)
            existing_lifted.updated_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            )
            logger.info("  merged %s lineage into existing lifted record %s/%s",
                        r.abstract_skill_id, existing_lifted.abstract_skill_id,
                        existing_lifted.template_signature)
        else:
            r.template_signature = L["template_signature"]
            r.template_steps = new_steps
            r.updated_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            )
            upgraded.append(r)

    # ── 3. write all records back (rewrite atomically) ────────────
    out_records = list(other) + upgraded + keep_no_template
    _write_records(jsonl_path, out_records)
    logger.info("wrote %d records back to %s "
                "(other=%d upgraded=%d still_no_template=%d)",
                len(out_records), jsonl_path,
                len(other), len(upgraded), len(keep_no_template))

    return {
        "n_no_template_total":  len(no_template_all),
        "n_processed":          len(to_process),
        "n_lifted":             len(lifted),
        "n_failed":             n_failed,
        "n_upgraded_keys":      len(upgraded),
        "n_merged_into_existing": len(lifted) - len(upgraded),
        "n_still_no_template":  len(keep_no_template),
        "elapsed_s":            round(elapsed, 1),
        "backup":               str(backup),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", required=True,
                    help="SharedAbstractBank root (e.g. shared_skill_bank/_latest)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap NO_TEMPLATE records processed (smoke test).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List records that would be re-lifted; don't call LLM.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    summary = relift_no_template(
        bank_root=Path(args.bank_root),
        model=args.model, workers=args.workers,
        limit=args.limit, dry_run=args.dry_run,
    )
    logger.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
