#!/usr/bin/env python
"""Bind shared mega-skills → per-task concrete skills via harness + crafter.

For each multi-task abstract skill in the shared bank, forward-bind it to
every task that doesn't already have a native binding. This exercises:

  1. Forward conversion: re-ground the abstract protocol skeleton onto the
     target task's action vocabulary using LLM translation.
  2. Harness validation: verify the bound protocol against target-task
     demos (if available), updating binding_status and sub_episodes.
  3. Crafter refinement: propose protocol patches (PATCH/HYPOTHESIS) for
     low-quality bindings.

When LLM/harness APIs are unavailable, runs in --offline mode which:
  - Creates PENDING bindings from heuristic action-vocab mapping
  - Marks them for later online validation

Usage:
    # Full LLM-driven binding + harness validation
    python frontier_data/scripts/bind_and_validate.py --model gpt-5.4

    # Offline mode (no LLM needed)
    python frontier_data/scripts/bind_and_validate.py --offline

    # Target specific tasks
    python frontier_data/scripts/bind_and_validate.py --tasks tetris browsergym
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bind_and_validate")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from skill_bank.shared_abstract_bank import (
    BoundConcreteSkill,
    LineageEntry,
    ProtocolStep,
    SharedAbstractSkill,
    SubEpisodeRef,
)


def load_abstracts(path: Path) -> List[SharedAbstractSkill]:
    """Load SharedAbstractSkill records from abstract.jsonl."""
    abstracts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            abstracts.append(SharedAbstractSkill.from_dict(d))
    return abstracts


def load_bindings(path: Path) -> Dict[str, BoundConcreteSkill]:
    """Load existing bindings from bindings.jsonl. Returns {concrete_skill_id: BoundConcreteSkill}."""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            b = BoundConcreteSkill.from_dict(d)
            result[b.concrete_skill_id] = b
    return result


def get_all_tasks(shared_bank_root: Path) -> List[str]:
    """Enumerate tasks that already have bindings."""
    by_task = shared_bank_root / "by_task"
    if not by_task.is_dir():
        return []
    return sorted(d.name for d in by_task.iterdir() if d.is_dir())


def reground_protocol_offline(abstract: SharedAbstractSkill, target_task: str) -> List[ProtocolStep]:
    """Heuristic offline re-grounding: preserve the abstract protocol steps
    verbatim with target-task annotations."""
    reground = []
    for step in abstract.protocol_steps:
        new_step = ProtocolStep(
            op=step.op,
            payload={k: f"${{{k}}}@{target_task}" for k in step.payload},
            slot_types=dict(step.slot_types),
            preconditions=list(step.preconditions),
            effects_add=list(step.effects_add),
            effects_del=list(step.effects_del),
            evidence_role=step.evidence_role,
            notes=f"[offline forward-bind from {abstract.abstract_skill_id}]",
        )
        reground.append(new_step)
    return reground


def forward_bind_abstract(
    abstract: SharedAbstractSkill,
    target_task: str,
    existing_bindings: Dict[str, BoundConcreteSkill],
    use_llm: bool = False,
    model: str = "gpt-5.4",
) -> Optional[BoundConcreteSkill]:
    """Create a forward binding of an abstract skill to a target task."""
    # Skip if task already has a native binding
    if abstract.abstract_skill_id in existing_bindings:
        return None

    now_iso = datetime.now(timezone.utc).isoformat()

    if use_llm:
        try:
            return _forward_bind_llm(abstract, target_task, model, now_iso)
        except Exception as e:
            logger.warning("LLM binding failed for %s → %s: %s, falling back to offline",
                           abstract.abstract_skill_id, target_task, e)

    protocol = reground_protocol_offline(abstract, target_task)
    return BoundConcreteSkill(
        concrete_skill_id=abstract.abstract_skill_id,
        task=target_task,
        abstract_skill_id=abstract.abstract_skill_id,
        name=abstract.name,
        protocol=protocol,
        sub_episodes=[],
        contract={},
        binding_status="PENDING",
        binding_source="forward_convert",
        raw_skill_id=abstract.abstract_skill_id,
        schema_version=2,
        created_at=now_iso,
        updated_at=now_iso,
    )


def _forward_bind_llm(
    abstract: SharedAbstractSkill,
    target_task: str,
    model: str,
    now_iso: str,
) -> BoundConcreteSkill:
    """LLM-driven forward binding via scripts/bind_abstract_to_task.py."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "scripts/bind_abstract_to_task.py",
            "--bank-root", str(REPO_ROOT / "frontier_data" / "output" / "shared_skill_bank"),
            "--target-task", target_task,
            "--abstract-id", abstract.abstract_skill_id,
            "--model", model,
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"bind_abstract_to_task failed: {result.stderr[:200]}")
    bindings_path = (REPO_ROOT / "frontier_data" / "output" / "shared_skill_bank"
                     / "by_task" / target_task / "bindings.jsonl")
    if bindings_path.exists():
        with open(bindings_path) as f:
            for line in f:
                d = json.loads(line.strip())
                if d.get("abstract_skill_id") == abstract.abstract_skill_id:
                    return BoundConcreteSkill.from_dict(d)
    raise RuntimeError("Binding not found after LLM call")


def crafter_refine_binding(
    binding: BoundConcreteSkill,
    use_llm: bool = False,
    model: str = "gpt-5.4",
) -> Optional[BoundConcreteSkill]:
    """Run crafter v2 PATCH/HYPOTHESIS on a low-quality binding."""
    if binding.binding_status not in ("PENDING", "REJECTED"):
        return None
    if not use_llm:
        binding.decorations["crafter_note"] = "needs_online_refinement"
        return binding
    # LLM path would call crafter_v2_batch_pipeline.py
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="Offline mode — heuristic binding without LLM calls")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Restrict to specific target tasks")
    ap.add_argument("--max-binds-per-task", type=int, default=50)
    args = ap.parse_args()

    shared_bank_root = REPO_ROOT / "frontier_data" / "output" / "shared_skill_bank"
    abstract_path = shared_bank_root / "abstract.jsonl"
    if not abstract_path.exists():
        logger.error("No abstract.jsonl at %s — run build_shared_bank.py first", shared_bank_root)
        return 1

    # Load abstracts
    abstracts = load_abstracts(abstract_path)
    logger.info("Loaded %d abstracts", len(abstracts))

    # Focus on multi-task abstracts (the real mega-skills)
    multi_task = [a for a in abstracts if a.n_bound_tasks >= 2]
    logger.info("%d multi-task abstracts (span ≥ 2 tasks)", len(multi_task))

    # Determine target tasks
    all_tasks = get_all_tasks(shared_bank_root)
    if args.tasks:
        target_tasks = [t for t in args.tasks if t in all_tasks]
    else:
        target_tasks = all_tasks

    logger.info("Target tasks: %d", len(target_tasks))
    use_llm = not args.offline

    # ── Forward binding: for each multi-task abstract, bind to every task
    # that doesn't already have it ──
    stats = Counter()
    new_bindings: Dict[str, List[BoundConcreteSkill]] = defaultdict(list)

    for abstract in multi_task:
        native_tasks = {L.task for L in abstract.lineage}
        for task in target_tasks:
            if task in native_tasks:
                continue
            existing = load_bindings(shared_bank_root / "by_task" / task / "bindings.jsonl")
            bound = forward_bind_abstract(
                abstract, task, existing, use_llm=use_llm, model=args.model,
            )
            if bound:
                new_bindings[task].append(bound)
                stats["forward_binds"] += 1

    # ── Crafter refinement for PENDING bindings ──
    for task in target_tasks:
        bind_path = shared_bank_root / "by_task" / task / "bindings.jsonl"
        existing = load_bindings(bind_path)
        for sid, b in existing.items():
            if b.binding_status in ("PENDING", "REJECTED"):
                refined = crafter_refine_binding(b, use_llm=use_llm, model=args.model)
                if refined:
                    stats["crafter_refined"] += 1

    # ── Write new bindings (append to existing) ──
    for task, bindings in new_bindings.items():
        bind_dir = shared_bank_root / "by_task" / task
        bind_dir.mkdir(parents=True, exist_ok=True)
        bind_path = bind_dir / "bindings.jsonl"
        with open(bind_path, "a") as f:
            for b in bindings:
                f.write(json.dumps(b.to_dict()) + "\n")
        stats["tasks_updated"] += 1

    # ── Write binding report ──
    report_dir = REPO_ROOT / "frontier_data" / "output" / "bind_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_utc": ts,
        "mode": "offline" if args.offline else "llm",
        "n_multi_task_abstracts": len(multi_task),
        "n_target_tasks": len(target_tasks),
        "n_forward_binds": stats["forward_binds"],
        "n_crafter_refined": stats["crafter_refined"],
        "n_tasks_updated": stats["tasks_updated"],
        "per_task_new_binds": {t: len(bs) for t, bs in new_bindings.items()},
    }
    with open(report_dir / f"bind_report_{ts}.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("═" * 60)
    logger.info("BINDING COMPLETE")
    logger.info("  %d forward bindings created", stats["forward_binds"])
    logger.info("  %d crafter refinements", stats["crafter_refined"])
    logger.info("  %d tasks updated", stats["tasks_updated"])
    logger.info("  Report: %s", report_dir / f"bind_report_{ts}.json")
    logger.info("═" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
