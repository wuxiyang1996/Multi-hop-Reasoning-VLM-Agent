from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import Lifecycle, MotifCandidate, TransferReport


@dataclass(frozen=True)
class BankEntry:
    motif: MotifCandidate
    status: Lifecycle
    evidence_ids: tuple[str, ...] = ()


class MotifBank:
    """A version space, not a semantic ranking or voting system."""

    def __init__(self) -> None:
        self._entries: dict[str, BankEntry] = {}

    def add_candidate(self, motif: MotifCandidate) -> BankEntry:
        entry = BankEntry(motif, Lifecycle.CANDIDATE)
        self._entries[motif.motif_id] = entry
        return entry

    def mark_source_supported(self, motif_id: str, receipt_ids: tuple[str, ...]) -> BankEntry:
        if not receipt_ids:
            raise ValueError("source support requires receipts")
        current = self._entries[motif_id]
        entry = replace(current, status=Lifecycle.SOURCE_SUPPORTED, evidence_ids=receipt_ids)
        self._entries[motif_id] = entry
        return entry

    def apply_transfer_report(self, motif_id: str, report: TransferReport) -> BankEntry:
        allowed = {
            Lifecycle.POSITIVE_TRANSFER,
            Lifecycle.NEGATIVE_TRANSFER,
            Lifecycle.GENERIC_ONLY,
            Lifecycle.INCONCLUSIVE,
        }
        if report.status not in allowed or not report.outcomes:
            raise ValueError("promotion requires a matched outcome report")
        current = self._entries[motif_id]
        entry = replace(current, status=report.status)
        self._entries[motif_id] = entry
        return entry

    def get(self, motif_id: str) -> BankEntry:
        return self._entries[motif_id]
