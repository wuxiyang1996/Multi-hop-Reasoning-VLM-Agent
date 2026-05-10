"""Reverse index over the lifted procedural-template banks (Layer C).

Each ``template_bank.jsonl`` line is the GPT-5.4-distilled, modality-
agnostic 2-5 step skeleton of one mined skill, produced by
``scripts/lift_skill_templates_gpt54.py``::

    {
      "task": "tetris",
      "cohort": "env_wr_game",
      "skill_id": "COMMIT/OPTIMIZE",
      "skill_name": "Flat Hole-Free Placement",
      "template_signature": "PERCEIVE → FILTER → COMPARE → DECIDE → COMMIT",
      "template_steps": [{"op": "PERCEIVE", "predicate": "..."},  ...],
      "transferable_to_cohorts": ["gymv_game", "vr_image", ...],
      ...
    }

This module is the small, dependency-free retrieval layer that the
Crafter / Transfer-matrix prefilter / Harness-validator reuse.  It is
**read-only**: it never writes to the bank and never calls an LLM.

Three retrieval surfaces:

* :meth:`TemplateIndex.lookup_by_signature` -- "give me K skills from
  *other* cohorts/tasks whose signature equals ``sig``" (used by the
  Crafter to enumerate cross-cohort transfer candidates).
* :meth:`TemplateIndex.signature_jaccard` -- cohort-vs-cohort or
  task-vs-task Jaccard over the *set* of distinct signatures (used by
  the Transfer-matrix prefilter).
* :meth:`TemplateIndex.task_has_signature` -- "has task ``T`` ever
  produced signature ``S`` in the wild?" (used by the harness
  validator's softening heuristic).

Default ``--template-run`` for every call site is the canonical
``labeling/skill_templates/run_20260510_053121/`` (where the 448-skill
lift output lives).  Override per-call with ``template_run=...`` so
unit tests can point at smaller fixtures.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("template_index")

# Canonical default — produced by scripts/lift_skill_templates_gpt54.py
DEFAULT_TEMPLATE_RUN = Path(__file__).resolve().parent.parent / \
    "labeling/skill_templates/run_20260510_053121"

# 5 cohorts the lift script emits.  Mirror of COHORT_OF in
# ``lift_skill_templates_gpt54.py``.
VALID_COHORTS: Tuple[str, ...] = (
    "gymv_game", "env_wr_game", "web", "vr_image", "vr_video",
)


@dataclass(frozen=True)
class TemplateRecord:
    """One lifted-template record, normalised."""
    task: str
    cohort: str
    skill_id: str
    skill_name: str
    template_signature: str
    template_steps: Tuple[Dict[str, str], ...]
    transferable_to_cohorts: Tuple[str, ...]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemplateRecord":
        steps = tuple(
            {"op": str(s.get("op", "")).upper(),
             "predicate": str(s.get("predicate", ""))}
            for s in (d.get("template_steps") or [])
            if isinstance(s, dict)
        )
        tt = tuple(
            c for c in (d.get("transferable_to_cohorts") or [])
            if isinstance(c, str) and c in VALID_COHORTS
        )
        return cls(
            task=str(d.get("task", "")),
            cohort=str(d.get("cohort", "")),
            skill_id=str(d.get("skill_id", "")),
            skill_name=str(d.get("skill_name", "")),
            template_signature=str(d.get("template_signature", "")),
            template_steps=steps,
            transferable_to_cohorts=tt,
        )


# ---------------------------------------------------------------------------
@dataclass
class TemplateIndex:
    """In-memory reverse index over a ``template_bank.jsonl`` corpus.

    Construct via :meth:`from_run` (default: scan the canonical lift run).
    All methods are pure / side-effect free; safe to share across threads.
    """

    records: List[TemplateRecord] = field(default_factory=list)

    # Precomputed indices (built in __post_init__)
    _by_sig: Dict[str, List[int]] = field(default_factory=dict, repr=False)
    _by_task: Dict[str, List[int]] = field(default_factory=dict, repr=False)
    _by_cohort: Dict[str, List[int]] = field(default_factory=dict, repr=False)
    _task_signatures: Dict[str, Set[str]] = field(default_factory=dict, repr=False)
    _cohort_signatures: Dict[str, Set[str]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def from_run(
        cls,
        template_run: Optional[Path] = None,
        *,
        cohorts: Optional[Sequence[str]] = None,
    ) -> "TemplateIndex":
        """Walk ``<template_run>/<cohort>/<task>/template_bank.jsonl`` files.

        ``template_run`` defaults to :data:`DEFAULT_TEMPLATE_RUN`.  Pass
        ``cohorts=[...]`` to restrict.  Returns an empty index (logged
        warning) when the run dir is missing — callers should treat the
        index as a soft enrichment, not a hard dependency.
        """
        run = Path(template_run or DEFAULT_TEMPLATE_RUN).resolve()
        records: List[TemplateRecord] = []
        if not run.is_dir():
            logger.warning("template_index: run dir missing: %s", run)
            idx = cls(records=records)
            idx._build()
            return idx

        wanted = set(cohorts) if cohorts else None
        for co_dir in sorted(run.iterdir()):
            if not co_dir.is_dir():
                continue
            if wanted is not None and co_dir.name not in wanted:
                continue
            if co_dir.name.startswith("_"):
                continue
            for task_dir in sorted(co_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                bank = task_dir / "template_bank.jsonl"
                if not bank.exists():
                    continue
                with bank.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        records.append(TemplateRecord.from_dict(d))

        idx = cls(records=records)
        idx._build()
        logger.info("template_index: loaded %d records from %s",
                    len(records), run)
        return idx

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "TemplateIndex":
        """Construct from in-memory dicts (test fixture path)."""
        recs = [TemplateRecord.from_dict(r) for r in records]
        idx = cls(records=recs)
        idx._build()
        return idx

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._by_sig.clear()
        self._by_task.clear()
        self._by_cohort.clear()
        self._task_signatures.clear()
        self._cohort_signatures.clear()
        for i, r in enumerate(self.records):
            self._by_sig.setdefault(r.template_signature, []).append(i)
            self._by_task.setdefault(r.task, []).append(i)
            self._by_cohort.setdefault(r.cohort, []).append(i)
            self._task_signatures.setdefault(r.task, set()).add(r.template_signature)
            self._cohort_signatures.setdefault(r.cohort, set()).add(r.template_signature)

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def n_unique_signatures(self) -> int:
        return len(self._by_sig)

    @property
    def n_tasks(self) -> int:
        return len(self._by_task)

    def all_signatures(self) -> List[str]:
        return list(self._by_sig.keys())

    def signatures_for_task(self, task: str) -> Set[str]:
        return self._task_signatures.get(task, set())

    def signatures_for_cohort(self, cohort: str) -> Set[str]:
        return self._cohort_signatures.get(cohort, set())

    def task_has_signature(self, task: str, signature: str) -> bool:
        return signature in self._task_signatures.get(task, set())

    # ------------------------------------------------------------------
    def lookup_by_signature(
        self,
        signature: str,
        *,
        exclude_task: Optional[str] = None,
        exclude_cohort: Optional[str] = None,
        prefer_cross_cohort: bool = True,
        k: int = 5,
    ) -> List[TemplateRecord]:
        """Retrieve up to ``k`` skills with the exact same signature.

        ``prefer_cross_cohort=True`` (default) sorts cross-cohort
        matches before same-cohort ones — useful for the Crafter,
        which is looking for *transfer* candidates, not native ones.
        """
        idxs = self._by_sig.get(signature) or []
        out: List[TemplateRecord] = []
        for i in idxs:
            r = self.records[i]
            if exclude_task and r.task == exclude_task:
                continue
            if exclude_cohort and r.cohort == exclude_cohort:
                continue
            out.append(r)
        if prefer_cross_cohort and exclude_cohort is None:
            # If we kept same-cohort matches, push them to the end so
            # the caller hits cross-cohort candidates first.
            out.sort(key=lambda r: (
                r.cohort == (exclude_cohort or ""),  # already filtered, no-op
                0,
            ))
        return out[:k]

    def lookup_for_target_task(
        self,
        target_task: str,
        signature: str,
        *,
        k: int = 5,
    ) -> List[TemplateRecord]:
        """Convenience: matches with the same signature, drawn from
        OTHER tasks, with the target task's own cohort de-prioritised.

        This is the canonical retrieval call from the Crafter's "find
        cross-cohort transfer candidates for failure-trace F on task
        T" path.
        """
        # Determine target cohort (if it appears in our index)
        target_cohort: Optional[str] = None
        if target_task in self._by_task:
            target_cohort = self.records[self._by_task[target_task][0]].cohort
        cands = self.lookup_by_signature(
            signature, exclude_task=target_task,
            prefer_cross_cohort=True, k=k * 4,  # over-fetch then filter
        )
        if target_cohort is None:
            return cands[:k]
        cross = [r for r in cands if r.cohort != target_cohort]
        same  = [r for r in cands if r.cohort == target_cohort]
        return (cross + same)[:k]

    # ------------------------------------------------------------------
    def signature_jaccard(self, a: str, b: str, *, by: str = "task") -> float:
        """Jaccard between the *signature sets* of two tasks (or cohorts).

        ``by="task"``  → use ``signatures_for_task``  (per-task sets).
        ``by="cohort"`` → use ``signatures_for_cohort`` (per-cohort sets).
        """
        if by == "task":
            sa = self._task_signatures.get(a, set())
            sb = self._task_signatures.get(b, set())
        elif by == "cohort":
            sa = self._cohort_signatures.get(a, set())
            sb = self._cohort_signatures.get(b, set())
        else:
            raise ValueError(f"by must be 'task' or 'cohort', got {by!r}")
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def cohort_pairs_above(self, *, by: str = "cohort", threshold: float = 0.10
                          ) -> List[Tuple[str, str, float]]:
        """All (a, b, J) triples with Jaccard above ``threshold``.

        Used by the transfer-matrix prefilter to enumerate which
        (source, target) cells are worth running.
        """
        if by == "task":
            keys = list(self._task_signatures.keys())
            sigs = self._task_signatures
        else:
            keys = list(self._cohort_signatures.keys())
            sigs = self._cohort_signatures
        out: List[Tuple[str, str, float]] = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                sa, sb = sigs[a], sigs[b]
                if not sa or not sb:
                    continue
                j = len(sa & sb) / max(1, len(sa | sb))
                if j >= threshold:
                    out.append((a, b, round(j, 4)))
        out.sort(key=lambda t: -t[2])
        return out


# ---------------------------------------------------------------------------
def _summary(idx: TemplateIndex) -> str:
    return (
        f"TemplateIndex(n_records={idx.size}, "
        f"n_unique_signatures={idx.n_unique_signatures}, "
        f"n_tasks={idx.n_tasks})"
    )


def _selftest() -> int:
    """Smoke check.  Runs against the canonical lift run if present."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    idx = TemplateIndex.from_run()
    print(_summary(idx))
    if idx.size == 0:
        return 1

    sig = "PERCEIVE → DECIDE → COMMIT → VERIFY"
    print(f"\nLookup_by_signature({sig!r}, k=4):")
    for r in idx.lookup_by_signature(sig, k=4):
        print(f"  {r.cohort:<12} {r.task:<28} {r.skill_id:<22} ({r.skill_name})")

    print(f"\nlookup_for_target_task('webshop', {sig!r}, k=4):")
    for r in idx.lookup_for_target_task("webshop", sig, k=4):
        print(f"  {r.cohort:<12} {r.task:<28} {r.skill_id:<22}")

    print("\nCohort-cohort signature Jaccard (top 5):")
    for a, b, j in idx.cohort_pairs_above(by="cohort", threshold=0.0)[:5]:
        print(f"  {a:<13} ↔ {b:<13}  J={j:.3f}")

    print(f"\ntask_has_signature('webshop', {sig!r}) = "
          f"{idx.task_has_signature('webshop', sig)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
