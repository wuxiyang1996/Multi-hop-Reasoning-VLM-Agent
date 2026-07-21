"""Content-independent controls for source-conditioning ablations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConditioningControlReceipt:
    control_kind: str
    control_seed: int
    step: int
    original_contexts_sha256: str
    controlled_contexts_sha256: str
    payload_permutation: Sequence[Mapping[str, str]]
    receipt_sha256: str

    def unsigned_payload(self):
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("conditioning control receipt hash mismatch")

    def to_dict(self):
        self.validate_hash()
        return asdict(self)


def rotate_source_conditioning(
    contexts: Sequence[Mapping[str, Any]], *, seed: int, step: int,
) -> tuple[Sequence[Mapping[str, Any]], ConditioningControlReceipt]:
    """Rotate payloads across candidate identities without inspecting content.

    Candidate/hypothesis/node identities remain fixed.  Only the untrusted
    ``source_conditioning`` payload is reassigned.  Ordering is determined by
    a seeded hash of candidate identity, then rotated by one, so no semantic
    score, target reward, embedding or model judgement affects the control.
    """
    if len(contexts) < 2:
        raise ValueError("randomized conditioning requires at least two active contexts")
    rows = [dict(row) for row in contexts]
    indexed = sorted(
        range(len(rows)),
        key=lambda index: _hash({
            "seed": int(seed),
            "candidate_hash": str(rows[index]["candidate_hash"]),
        }),
    )
    source_for_target = {
        target_index: indexed[(position + 1) % len(indexed)]
        for position, target_index in enumerate(indexed)
    }
    controlled = []
    permutation = []
    for target_index, row in enumerate(rows):
        source_index = source_for_target[target_index]
        updated = dict(row)
        updated["source_conditioning"] = dict(
            rows[source_index].get("source_conditioning") or {}
        )
        controlled.append(updated)
        permutation.append({
            "target_candidate_hash": str(row["candidate_hash"]),
            "payload_source_candidate_hash": str(rows[source_index]["candidate_hash"]),
        })
    unsigned = {
        "control_kind": "ROTATE_SOURCE_CONDITIONING",
        "control_seed": int(seed),
        "step": int(step),
        "original_contexts_sha256": _hash(rows),
        "controlled_contexts_sha256": _hash(controlled),
        "payload_permutation": permutation,
    }
    receipt = ConditioningControlReceipt(
        control_kind=unsigned["control_kind"],
        control_seed=unsigned["control_seed"],
        step=unsigned["step"],
        original_contexts_sha256=unsigned["original_contexts_sha256"],
        controlled_contexts_sha256=unsigned["controlled_contexts_sha256"],
        payload_permutation=tuple(permutation),
        receipt_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return tuple(controlled), receipt


def conditioning_control_receipt_from_dict(
    payload: Mapping[str, Any],
) -> ConditioningControlReceipt:
    receipt = ConditioningControlReceipt(
        control_kind=str(payload["control_kind"]),
        control_seed=int(payload["control_seed"]),
        step=int(payload["step"]),
        original_contexts_sha256=str(payload["original_contexts_sha256"]),
        controlled_contexts_sha256=str(payload["controlled_contexts_sha256"]),
        payload_permutation=tuple(dict(row) for row in payload["payload_permutation"]),
        receipt_sha256=str(payload["receipt_sha256"]),
    )
    receipt.validate_hash()
    if receipt.control_kind != "ROTATE_SOURCE_CONDITIONING":
        raise ValueError("unsupported conditioning control kind")
    if receipt.original_contexts_sha256 == receipt.controlled_contexts_sha256:
        raise ValueError("conditioning control did not change contexts")
    return receipt


__all__ = [
    "ConditioningControlReceipt",
    "conditioning_control_receipt_from_dict",
    "rotate_source_conditioning",
]
