#!/usr/bin/env python3
"""Inject Layer-C procedural templates into per-task skill banks as
runtime-ready protocol dicts.

Reads every ``template_bank.jsonl`` produced by
``scripts/lift_skill_templates_gpt54.py`` and patches the corresponding
skill record in ``per_task_banks/<task>/skill_bank.jsonl`` so the
trainer's ``_format_skill_guidance_for_prompt`` renders a full
step-by-step reasoning plan for the 9B actor.

Before injection, many skills (especially tetris, VR, web) have empty
protocol.steps — the agent sees only the skill name with no plan.
After injection, every skill has 3-5 reasoning-level steps the agent
can follow with ``>>`` step tracking + intrinsic bonus.

Usage::

    python frontier_data/scripts/inject_layerc_protocols.py
    python frontier_data/scripts/inject_layerc_protocols.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("inject_layerc_protocols")

ROOT = Path(__file__).resolve().parent.parent.parent
LAYER_C_DIR = ROOT / "frontier_data" / "output" / "layer_c_templates"
BANK_DIR = ROOT / "frontier_data" / "output" / "per_task_banks"


def _template_steps_to_runtime(
    template_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert Layer-C template_steps → trainer-compatible protocol dict."""
    nl_steps: List[str] = []
    step_checks: List[str] = []
    pred_success: List[str] = []
    pred_abort: List[str] = []
    preconditions: List[str] = []
    success_criteria: List[str] = []
    abort_criteria: List[str] = []
    action_vocab: List[str] = []

    seen_ops: set = set()
    for i, s in enumerate(template_steps):
        op = (s.get("op") or "").upper()
        predicate = s.get("predicate", "")
        nl_steps.append(predicate)

        if op and op not in seen_ops:
            action_vocab.append(op)
            seen_ops.add(op)

        for e in s.get("effects_add", []):
            etype = e.get("type", "")
            if etype:
                step_checks.append(f"{etype}=true")
                break
        else:
            step_checks.append("")

        for p in s.get("preconditions", []):
            ptype = p.get("type", "")
            if ptype and i == 0:
                preconditions.append(ptype)

        for c in s.get("abort_criteria", []) or []:
            if c and str(c) not in abort_criteria:
                abort_criteria.append(str(c))

    last = template_steps[-1] if template_steps else {}
    for e in last.get("effects_add", []):
        etype = e.get("type", "")
        if etype:
            pred_success.append(f"{etype}=true")
            success_criteria.append(etype)

    if not abort_criteria:
        abort_criteria = ["No progress toward skill objective after several moves"]

    n = len(nl_steps) or 1
    return {
        "preconditions": preconditions[:6],
        "steps": nl_steps,
        "success_criteria": success_criteria[:6],
        "abort_criteria": abort_criteria[:4],
        "expected_duration": max(n * 3, 6),
        "step_checks": step_checks,
        "predicate_success": pred_success[:6],
        "predicate_abort": pred_abort[:6],
        "action_vocab": sorted(action_vocab),
        "source": "layer_c_lift",
    }


def load_templates(layer_c_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {task: {skill_id: template_record}}."""
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cohort in sorted(os.listdir(layer_c_dir)):
        cpath = layer_c_dir / cohort
        if not cpath.is_dir():
            continue
        for task in sorted(os.listdir(cpath)):
            tb = cpath / task / "template_bank.jsonl"
            if not tb.is_file():
                continue
            task_map: Dict[str, Dict[str, Any]] = {}
            with open(tb) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    task_map[r["skill_id"]] = r
            by_task[task] = task_map
    return by_task


def inject_into_bank(
    bank_path: Path,
    templates: Dict[str, Dict[str, Any]],
    *,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Patch skill_bank.jsonl in-place. Returns (total, patched, already_rich)."""
    if not bank_path.is_file():
        return 0, 0, 0

    records: List[str] = []
    total = 0
    patched = 0
    already_rich = 0

    with open(bank_path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)

            sk = r.get("skill", r)
            skill_id = sk.get("skill_id", "")
            tmpl = templates.get(skill_id)

            if tmpl is None:
                records.append(json.dumps(r, ensure_ascii=False))
                continue

            old_proto = sk.get("protocol", {})
            old_steps = (old_proto.get("steps", [])
                         if isinstance(old_proto, dict) else [])

            runtime_proto = _template_steps_to_runtime(tmpl["template_steps"])

            if old_steps and len(old_steps) >= len(runtime_proto["steps"]):
                already_rich += 1
                if isinstance(old_proto, dict) and not old_proto.get("step_checks"):
                    old_proto["step_checks"] = runtime_proto["step_checks"]
                    old_proto["action_vocab"] = runtime_proto["action_vocab"]
                    old_proto.setdefault("predicate_success",
                                        runtime_proto["predicate_success"])
                csig = tmpl.get("collapsed_signature", "")
                if csig and not sk.get("collapsed_signature"):
                    sk["collapsed_signature"] = csig
                    if "skill" in r:
                        r["skill"] = sk
                    else:
                        r = sk
                    patched += 1
                records.append(json.dumps(r, ensure_ascii=False))
                continue

            if isinstance(old_proto, dict):
                for key in ("preconditions", "success_criteria", "abort_criteria"):
                    existing = old_proto.get(key, [])
                    if existing and not runtime_proto[key]:
                        runtime_proto[key] = existing
            sk["protocol"] = runtime_proto

            sig = tmpl.get("template_signature", "")
            if sig:
                sk["template_signature"] = sig
            csig = tmpl.get("collapsed_signature", "")
            if csig:
                sk["collapsed_signature"] = csig
            xfer = tmpl.get("transferable_to_cohorts", [])
            if xfer:
                sk["transferable_to_cohorts"] = xfer

            if not sk.get("strategic_description"):
                desc = tmpl.get("strategic_description", "")
                if desc:
                    sk["strategic_description"] = desc

            if "skill" in r:
                r["skill"] = sk
            else:
                r = sk

            patched += 1
            records.append(json.dumps(r, ensure_ascii=False))

    if not dry_run and patched > 0:
        backup = bank_path.with_suffix(".jsonl.pre_layerc_bak")
        if not backup.exists():
            bank_path.rename(backup)
            bank_path.write_text("\n".join(records) + "\n")
        else:
            bank_path.write_text("\n".join(records) + "\n")

    return total, patched, already_rich


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer-c-dir", default=str(LAYER_C_DIR))
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    layer_c_dir = Path(args.layer_c_dir)
    bank_dir = Path(args.bank_dir)

    logger.info("Loading Layer-C templates from %s", layer_c_dir)
    templates = load_templates(layer_c_dir)
    n_templates = sum(len(v) for v in templates.values())
    logger.info("  %d templates across %d tasks", n_templates, len(templates))

    grand_total = 0
    grand_patched = 0
    grand_rich = 0

    for task in sorted(os.listdir(bank_dir)):
        bp = bank_dir / task / "skill_bank.jsonl"
        if not bp.is_file():
            continue
        tmpl = templates.get(task, {})
        if not tmpl:
            logger.debug("  skip %s (no templates)", task)
            continue

        total, patched, rich = inject_into_bank(bp, tmpl, dry_run=args.dry_run)
        grand_total += total
        grand_patched += patched
        grand_rich += rich

        status = "DRY-RUN" if args.dry_run else "PATCHED"
        logger.info("  %-30s total=%d  %s=%d  already_rich=%d",
                     task, total, status, patched, rich)

    logger.info("="*60)
    logger.info("DONE: %d skills total, %d patched with Layer-C protocols, "
                "%d already had rich protocols",
                grand_total, grand_patched, grand_rich)
    if args.dry_run:
        logger.info("  (dry-run mode — no files modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
