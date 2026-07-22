from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class LegacyLineage:
    source_path: str
    record: dict[str, Any]
    authority: str = "LINEAGE_RETRIEVAL_ONLY"


def load_jsonl(path: str | Path) -> Iterable[LegacyLineage]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield LegacyLineage(str(source), json.loads(line))


def audit_legacy(records: Iterable[LegacyLineage]) -> dict[str, int]:
    rows = list(records)
    families = {str(row.record.get("family_id", "")) for row in rows}
    signatures = {str(row.record.get("signature", "")) for row in rows}
    executable = sum(bool(row.record.get("replay_receipt_ids")) for row in rows)
    return {
        "records": len(rows),
        "families": len(families - {""}),
        "signatures": len(signatures - {""}),
        "records_with_replay_receipts": executable,
    }
