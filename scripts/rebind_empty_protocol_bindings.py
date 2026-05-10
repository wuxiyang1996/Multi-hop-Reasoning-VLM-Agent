"""Regenerate executable protocols for the empty-protocol bindings
that the QA / env-wrapper mining pipelines left behind.

Background.  ``build_shared_skill_bank.py`` consolidates skills from
five mining sources.  The gym_v cohort emits structured multi-hop
protocols, but the QA-style banks (``vr_image``, ``vr_video``,
``web``) and parts of the ``env_wr_game`` cohort mine the
``(name, strategic_description, contract)`` tuple WITHOUT a
protocol — by design, those pipelines are answer-once oriented
rather than multi-hop.  When the consolidator lands a binding from
those sources into PerTaskBank, ``BoundConcreteSkill.protocol``
ends up empty.

Empty-protocol bindings are inert from a Path-A harness perspective
(``GymvAdapter.can_handle`` returns False on ``not skill.protocol``,
and our pre-flight veto rejects them with
``protocol_too_short``).  They also can't be served as runtime
guidance to an agent — the trainer's prompt builder only emits
``Active skill: ...`` blocks for bindings with ≥ 2 hops.

This script walks every PerTaskBank, picks every binding with
``len(protocol) < 2``, looks up its parent abstract in the shared
bank, and runs the existing forward-bind path
(``bind_abstract_to_task.bind_one``) to synthesise a task-specific
protocol via GPT-5.4.  The :meth:`PerTaskBank.upsert_binding` is
already additive on ``sub_episodes`` — prior rollout receipts
survive the rebind.

Invocation::

    OPENROUTER_API_KEY=$KEY python -m scripts.rebind_empty_protocol_bindings \\
        --bank-root shared_skill_bank/_latest \\
        --workers 6

By default we run every empty-protocol binding across every task.
Use ``--tasks`` to restrict the sweep, ``--limit`` for smoke tests,
``--harness-validate`` to flip on the Path-A FewShotAdapter
verdict on each newly-minted binding.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from skill_bank.shared_abstract_bank import (                            # noqa: E402
    BoundConcreteSkill, SharedAbstractSkill, TwoLayerSkillStore,
)
from scripts.bind_abstract_to_task import (                               # noqa: E402
    DEFAULT_MODEL, bind_one,
)

logger = logging.getLogger("rebind_empty_protocol")


# ---------------------------------------------------------------------------
# Discovery: which bindings are missing an executable protocol?
# ---------------------------------------------------------------------------
def _is_empty_protocol(binding: BoundConcreteSkill, *, min_len: int = 2) -> bool:
    if len(binding.protocol) < min_len:
        return True
    # Defensive: a single-step protocol whose only step has op="?" is
    # still effectively empty (the legacy crafter-string carrier).
    if all((s.op or "") in ("", "?") for s in binding.protocol):
        return True
    return False


def _pick_richest_abstract(
    bank: TwoLayerSkillStore, abstract_skill_id: str,
) -> Optional[SharedAbstractSkill]:
    """When multiple abstracts share the same stem (different
    template_signatures), pick the one with the broadest lineage
    so the forward-bind LLM gets the richest evidence."""
    candidates = bank.abstract.by_abstract_id(abstract_skill_id)
    if not candidates:
        return None

    def score(a: SharedAbstractSkill) -> Tuple[int, int, int]:
        n_native = sum(1 for L in a.lineage if L.is_native)
        n_total  = len(a.lineage)
        has_template = 0 if a.template_signature == "NO_TEMPLATE" else 1
        return (has_template, n_native, n_total)

    return max(candidates, key=score)


def find_empty_protocol_bindings(
    bank: TwoLayerSkillStore,
    *,
    tasks: Optional[Iterable[str]] = None,
    skip_tasks: Iterable[str] = (),
) -> List[Tuple[str, BoundConcreteSkill]]:
    """Return ``(task, binding)`` pairs for every binding in the
    bank's PerTaskBanks that has an empty / placeholder protocol.

    ``skip_tasks`` is a denylist (defaults to empty); the caller can
    use it to skip tasks where forward-bind is known to be wasteful
    (e.g. tasks whose abstracts don't have cross-task lineage)."""
    target_tasks = list(tasks) if tasks else bank.list_tasks()
    out: List[Tuple[str, BoundConcreteSkill]] = []
    for t in target_tasks:
        if t in skip_tasks:
            continue
        for b in bank.per_task(t).records:
            if _is_empty_protocol(b):
                out.append((t, b))
    return out


# ---------------------------------------------------------------------------
# Per-binding driver
# ---------------------------------------------------------------------------
def _rebind_one(
    *, bank: TwoLayerSkillStore, task: str, binding: BoundConcreteSkill,
    model: str, do_harness_validate: bool,
) -> Dict[str, Any]:
    abstract = _pick_richest_abstract(bank, binding.abstract_skill_id)
    if abstract is None:
        return {
            "task": task, "skill_id": binding.concrete_skill_id,
            "ok": False, "reason": "no_abstract",
        }
    if abstract.template_signature == "NO_TEMPLATE":
        # Better to leave a NO_TEMPLATE-only abstract alone — the
        # forward-bind LLM has no skeleton to follow.  Surface as
        # skipped so the relift pass can be re-run first.
        return {
            "task": task, "skill_id": binding.concrete_skill_id,
            "ok": False, "reason": "abstract_no_template",
        }

    try:
        r = bind_one(
            abstract=abstract, target_task=task,
            bank=bank, model=model,
            do_harness_validate=do_harness_validate,
        )
    except Exception as exc:                                              # noqa: BLE001
        return {
            "task": task, "skill_id": binding.concrete_skill_id,
            "ok": False, "reason": "bind_one_raised",
            "error": repr(exc),
        }
    if not r.get("ok"):
        return {
            "task": task, "skill_id": binding.concrete_skill_id,
            "ok": False, "reason": r.get("reason", "bind_failed"),
            "raw_excerpt": r.get("raw_excerpt", "")[:120],
        }

    # bind_one() already wrote the binding back via
    # TwoLayerSkillStore.insert_validated_binding (which calls
    # PerTaskBank.upsert_binding — additive on sub_episodes,
    # protocol-replacing).  Pull the upgraded binding to confirm.
    new_b = bank.per_task(task).by_concrete_id(binding.concrete_skill_id)
    if new_b is None:
        return {
            "task": task, "skill_id": binding.concrete_skill_id,
            "ok": False, "reason": "binding_missing_post_bind",
        }
    return {
        "task": task,
        "skill_id": binding.concrete_skill_id,
        "ok": True,
        "binding_status": new_b.binding_status,
        "n_protocol_steps_before": 0,
        "n_protocol_steps_after":  len(new_b.protocol),
        "n_sub_episodes_preserved": len(new_b.sub_episodes),
        "validator_diag": r.get("validator_diag", {}),
    }


# ---------------------------------------------------------------------------
def rebind_empty_protocols(
    *, bank_root: Path, tasks: Optional[List[str]] = None,
    skip_tasks: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL, workers: int = 6,
    do_harness_validate: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    bank_root = Path(bank_root)
    bank = TwoLayerSkillStore(bank_root)
    bank.abstract.load()

    pairs = find_empty_protocol_bindings(
        bank, tasks=tasks, skip_tasks=skip_tasks or (),
    )
    logger.info("found %d empty-protocol bindings across %d task(s)",
                len(pairs), len({t for t, _ in pairs}))

    # Group by task for the dry-run preview.
    by_task: Dict[str, List[str]] = {}
    for t, b in pairs:
        by_task.setdefault(t, []).append(b.concrete_skill_id)
    for t, ids in sorted(by_task.items()):
        logger.info("  %-25s n=%3d  e.g. %s",
                    t, len(ids),
                    ", ".join(ids[:3]) + ("..." if len(ids) > 3 else ""))

    if limit is not None:
        pairs = pairs[:limit]
        logger.info("limiting to first %d binding(s)", limit)
    if dry_run:
        return {"n_empty": len(pairs), "by_task": {t: len(v)
                                                    for t, v in by_task.items()},
                "dry_run": True}

    if not pairs:
        return {"n_empty": 0}

    started = time.time()
    results: List[Dict[str, Any]] = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_rebind_one, bank=bank, task=t, binding=b,
                       model=model,
                       do_harness_validate=do_harness_validate): (t, b)
            for t, b in pairs
        }
        for fut in as_completed(futures):
            t, b = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:                                      # noqa: BLE001
                res = {"task": t, "skill_id": b.concrete_skill_id,
                       "ok": False, "reason": "exc", "error": repr(exc)}
            n_done += 1
            results.append(res)
            ok_marker = "✓" if res.get("ok") else "✗"
            logger.info("[%4d/%d] %s %-25s/%-25s  steps→%s  %s",
                        n_done, len(pairs), ok_marker,
                        res.get("task", ""),
                        res.get("skill_id", ""),
                        res.get("n_protocol_steps_after", "-"),
                        res.get("reason") or res.get("binding_status") or "")

    elapsed = time.time() - started

    # Roll up.
    n_ok      = sum(1 for r in results if r.get("ok"))
    n_failed  = len(results) - n_ok
    by_status: Dict[str, int] = {}
    for r in results:
        s = r.get("binding_status") if r.get("ok") else r.get("reason", "exc")
        by_status[s] = by_status.get(s, 0) + 1

    summary = {
        "n_empty":         len(pairs),
        "n_ok":            n_ok,
        "n_failed":        n_failed,
        "by_outcome":      by_status,
        "elapsed_s":       round(elapsed, 1),
        "harness_validate": bool(do_harness_validate),
    }
    return summary


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", required=True,
                    help="SharedAbstractBank root (e.g. shared_skill_bank/_latest)")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Restrict sweep to these tasks (default: all PerTaskBanks).")
    ap.add_argument("--skip-tasks", nargs="+", default=None,
                    help="Denylist of tasks to skip.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--harness-validate", action="store_true",
                    help="Run Path-A FewShotAdapter on every newly-minted binding.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap empty-protocol bindings processed (smoke test).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List bindings; don't call LLM.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    summary = rebind_empty_protocols(
        bank_root=Path(args.bank_root),
        tasks=args.tasks, skip_tasks=args.skip_tasks,
        model=args.model, workers=args.workers,
        do_harness_validate=args.harness_validate,
        limit=args.limit, dry_run=args.dry_run,
    )
    logger.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
