"""`SkillLifecycleManager` — sole authority for skill status transitions.

Spec: PLAN-UNIFIED-SKILL-GATE §6.

This is the only module in the codebase that may write to a `SkillStore`.
It enforces the canonical state-machine and physically migrates records
between stores on each transition (DRAFT → CANDIDATE moves the JSON file
from `draft_store/` to `candidate_store/`).

External callers go through this method exclusively; direct
`store.put(...)` calls raise `StoreLockedError`.
"""

from __future__ import annotations

import threading
from typing import Dict, FrozenSet, Iterable, Optional

from common.enums import SkillStatus
from data_structure.extensions.skill_record import SkillRecord
from skill_bank.repository import SkillRepository
from skill_bank.stores import SkillStore, StoreName, store_for_status


class LifecycleError(RuntimeError):
    """Raised on disallowed transitions or invariant violations."""


# Allowed transitions (PLAN-UNIFIED-SKILL-GATE §2.3).
_ALLOWED: Dict[SkillStatus, FrozenSet[SkillStatus]] = {
    SkillStatus.DRAFT: frozenset({SkillStatus.CANDIDATE, SkillStatus.REJECTED}),
    SkillStatus.CANDIDATE: frozenset(
        {SkillStatus.SHADOW, SkillStatus.PROVISIONAL, SkillStatus.ACTIVE, SkillStatus.REJECTED, SkillStatus.DEPRECATED}
    ),
    SkillStatus.SHADOW: frozenset(
        {SkillStatus.PROVISIONAL, SkillStatus.ACTIVE, SkillStatus.REJECTED, SkillStatus.DEPRECATED}
    ),
    SkillStatus.PROVISIONAL: frozenset(
        {SkillStatus.ACTIVE, SkillStatus.SHADOW, SkillStatus.DEPRECATED, SkillStatus.ROLLED_BACK}
    ),
    SkillStatus.ACTIVE: frozenset({SkillStatus.DEPRECATED, SkillStatus.ROLLED_BACK, SkillStatus.PROVISIONAL}),
    SkillStatus.DEPRECATED: frozenset({SkillStatus.ROLLED_BACK}),
    SkillStatus.REJECTED: frozenset(),
    SkillStatus.ROLLED_BACK: frozenset({SkillStatus.DEPRECATED}),
}


class SkillLifecycleManager:
    """The only module allowed to mutate skill stores.

    Construct with one repo; the manager owns the cross-store migration
    transactions. All transitions are serialized by a single mutex —
    bank state is always internally consistent at quiesce.
    """

    def __init__(self, repository: SkillRepository) -> None:
        self._repo = repository
        self._mutex = threading.RLock()
        self._token: object = object()

    # -- read-through helpers -------------------------------------------

    @property
    def repository(self) -> SkillRepository:
        return self._repo

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        return self._repo.get(skill_id)

    # -- ingest ---------------------------------------------------------

    def ingest_draft(self, record: SkillRecord) -> SkillRecord:
        """Insert a new skill at status=DRAFT.

        Used by the crafter and by mining flows. Always lands in the
        draft store regardless of any status the caller set on the record.
        """
        with self._mutex:
            if self._repo.get(record.skill_id) is not None:
                raise LifecycleError(f"Skill {record.skill_id!r} already exists.")
            record.status = SkillStatus.DRAFT
            self._write(self._repo.draft, record)
            return record

    # -- transition ------------------------------------------------------

    def transition(
        self,
        skill_id: str,
        *,
        to_status: SkillStatus,
        rationale: str,
    ) -> SkillRecord:
        with self._mutex:
            record = self._repo.get(skill_id)
            if record is None:
                raise LifecycleError(f"Unknown skill {skill_id!r}")
            if to_status not in _ALLOWED.get(record.status, frozenset()):
                raise LifecycleError(
                    f"Disallowed transition {record.status.value} -> {to_status.value} "
                    f"for skill {skill_id!r}."
                )
            self._validate_invariants(record, to_status, rationale)

            from_store = self._store_for(record.status)
            to_store = self._store_for(to_status)

            with from_store._unlocked(self._token):
                from_store.remove(record.skill_id, token=self._token)
            record.status = to_status
            with to_store._unlocked(self._token):
                to_store.put(record, token=self._token)
            return record

    # -- batch promotion (used by PromotionOrchestrator) -----------------

    def transition_many(
        self,
        transitions: Iterable[tuple[str, SkillStatus, str]],
    ) -> Dict[str, SkillRecord]:
        """Apply multiple transitions atomically. Either all succeed or all
        are rolled back."""
        with self._mutex:
            staged: list[tuple[SkillStatus, SkillRecord]] = []
            try:
                for skill_id, to_status, rationale in transitions:
                    record = self._repo.get(skill_id)
                    if record is None:
                        raise LifecycleError(f"Unknown skill {skill_id!r}")
                    staged.append((record.status, record))
                # Re-resolve and apply (transition() re-validates each).
                result: Dict[str, SkillRecord] = {}
                for skill_id, to_status, rationale in transitions:
                    result[skill_id] = self.transition(
                        skill_id, to_status=to_status, rationale=rationale
                    )
                return result
            except Exception:
                # Best-effort rollback: revert any record whose status changed.
                for original_status, record in staged:
                    if record.status == original_status:
                        continue
                    cur_store = self._store_for(record.status)
                    orig_store = self._store_for(original_status)
                    with cur_store._unlocked(self._token):
                        cur_store.remove(record.skill_id, token=self._token)
                    record.status = original_status
                    with orig_store._unlocked(self._token):
                        orig_store.put(record, token=self._token)
                raise

    # -- internals -------------------------------------------------------

    def _store_for(self, status: SkillStatus) -> SkillStore:
        target = store_for_status(status)
        if target == StoreName.DRAFT:
            return self._repo.draft
        if target == StoreName.CANDIDATE:
            return self._repo.candidate
        if target == StoreName.ACTIVE:
            return self._repo.active
        if target == StoreName.ARCHIVE:
            return self._repo.archive
        raise LifecycleError(f"No store for status {status.value!r}")

    def _write(self, store: SkillStore, record: SkillRecord) -> None:
        with store._unlocked(self._token):
            store.put(record, token=self._token)

    def _validate_invariants(
        self,
        record: SkillRecord,
        to_status: SkillStatus,
        rationale: str,
    ) -> None:
        if not rationale:
            raise LifecycleError("Transition requires a non-empty rationale.")
        # PLAN-SKILL-BANK §0.1: cannot become ACTIVE single-domain.
        if to_status == SkillStatus.ACTIVE and len(set(record.feasible_domains)) < 2:
            raise LifecycleError(
                f"Cannot promote {record.skill_id!r} to ACTIVE: "
                f"general-protocol invariant requires ≥2 feasible_domains "
                f"(got {sorted(set(record.feasible_domains))})."
            )
        # PLAN-UNIFIED-SKILL-GATE §3: ACTIVE requires non-empty
        # expected_evidence_roles.
        if to_status == SkillStatus.ACTIVE and not record.contract.expected_evidence_roles:
            raise LifecycleError(
                f"Cannot promote {record.skill_id!r} to ACTIVE: "
                f"contract.expected_evidence_roles is empty (G0 violation)."
            )


__all__ = ["LifecycleError", "SkillLifecycleManager"]
