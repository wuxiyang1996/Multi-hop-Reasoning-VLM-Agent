"""Candidate-isolated typed visual claims with a conservative executor.

Each neural call sees one candidate claim but no slot and no competing
candidates.  Candidate/slot binding happens only after the receipt freezes.
The symbolic executor is deliberately conservative: an MCQ commitment changes
only when exactly one claim is fully supported and every competing claim is
refuted.  Unknown or internally incomplete evidence falls back to the direct
target-native answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CHECK_KINDS = (
    "ENTITY_BINDING",
    "PRECONDITION",
    "POSTCONDITION",
    "DIRECTIONAL_OR_CAUSAL_LINK",
    "CLAIM_ENTAILMENT",
)
CHECK_STATUSES = ("SUPPORTED", "REFUTED", "UNKNOWN", "NOT_APPLICABLE")
CLAIM_STATUSES = ("SUPPORTED", "REFUTED", "UNKNOWN")
EVIDENCE_BASES = ("OBSERVED", "INFERRED", "NOT_APPLICABLE")


def _index_list(value: Any, *, frame_count: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError("evidence_frames must be a list of at most three indices")
    indices: list[int] = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError("evidence frame must be an integer")
        normalized = raw
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized.upper().startswith("F"):
                normalized = normalized[1:]
        index = int(normalized)
        if isinstance(normalized, float) and index != normalized:
            raise ValueError("evidence frame must be an integer")
        if not 0 <= index < frame_count:
            raise ValueError("evidence frame is outside the proxy video")
        if index not in indices:
            indices.append(index)
    if indices != sorted(indices):
        raise ValueError("evidence frames must be chronological")
    return tuple(indices)


@dataclass(frozen=True)
class TypedClaimCheck:
    kind: str
    status: str
    confidence: float
    basis: str
    evidence_frames: tuple[int, ...]
    fact: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "confidence": self.confidence,
            "basis": self.basis,
            "evidence_frames": list(self.evidence_frames),
            "fact": self.fact,
        }


@dataclass(frozen=True)
class TypedClaimReceipt:
    claim_status: str
    confidence: float
    checks: tuple[TypedClaimCheck, ...]
    uncertainties: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_status": self.claim_status,
            "confidence": self.confidence,
            "checks": [check.as_dict() for check in self.checks],
            "uncertainties": list(self.uncertainties),
            "reason": self.reason,
        }


def parse_typed_claim_receipt(
    payload: Mapping[str, Any], *, frame_count: int,
) -> TypedClaimReceipt:
    forbidden = {
        "answer", "answer_slot", "choice", "choice_id", "correct_option", "slot",
    }
    leaked = sorted(forbidden & {str(key).casefold() for key in payload})
    if leaked:
        raise ValueError(f"candidate receipt leaked binding fields: {leaked}")
    claim_status = str(payload.get("claim_status") or "")
    if claim_status not in CLAIM_STATUSES:
        raise ValueError("invalid claim_status")
    confidence = float(payload.get("confidence", -1.0))
    if not 0 <= confidence <= 1:
        raise ValueError("claim confidence must be in [0,1]")
    rows = list(payload.get("checks") or ())
    if [str(row.get("kind")) for row in rows] != list(CHECK_KINDS):
        raise ValueError("typed checks must preserve the canonical order")
    checks: list[TypedClaimCheck] = []
    for row in rows:
        status = str(row.get("status") or "")
        if status not in CHECK_STATUSES:
            raise ValueError("invalid typed-check status")
        check_confidence = float(row.get("confidence", -1.0))
        if not 0 <= check_confidence <= 1:
            raise ValueError("typed-check confidence must be in [0,1]")
        basis = str(row.get("basis") or "")
        if basis not in EVIDENCE_BASES:
            raise ValueError("invalid typed-check evidence basis")
        if status == "NOT_APPLICABLE" and basis != "NOT_APPLICABLE":
            raise ValueError("NOT_APPLICABLE status requires matching basis")
        if status != "NOT_APPLICABLE" and basis == "NOT_APPLICABLE":
            raise ValueError("applicable check cannot use NOT_APPLICABLE basis")
        evidence = _index_list(row.get("evidence_frames"), frame_count=frame_count)
        if status in {"SUPPORTED", "REFUTED"} and not evidence:
            if basis != "INFERRED":
                raise ValueError("observed supported/refuted checks need evidence frames")
        checks.append(TypedClaimCheck(
            kind=str(row["kind"]),
            status=status,
            confidence=check_confidence,
            basis=basis,
            evidence_frames=evidence,
            fact=str(row.get("fact") or "").strip(),
        ))
    entailment = checks[-1]
    if entailment.status == "NOT_APPLICABLE":
        raise ValueError("CLAIM_ENTAILMENT cannot be NOT_APPLICABLE")
    expected = {
        "SUPPORTED": "SUPPORTED",
        "REFUTED": "REFUTED",
        "UNKNOWN": "UNKNOWN",
    }[entailment.status]
    if claim_status != expected:
        raise ValueError("claim_status must be determined by CLAIM_ENTAILMENT")
    if claim_status == "SUPPORTED":
        entity = checks[0]
        if entity.status != "SUPPORTED":
            raise ValueError("supported claim requires supported entity binding")
        for check in checks[1:-1]:
            if check.status not in {"SUPPORTED", "NOT_APPLICABLE"}:
                raise ValueError("supported claim has an unresolved required transition check")
    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list) or not all(
        isinstance(value, str) for value in uncertainties
    ):
        raise ValueError("uncertainties must be a string list")
    return TypedClaimReceipt(
        claim_status=claim_status,
        confidence=confidence,
        checks=tuple(checks),
        uncertainties=tuple(value.strip() for value in uncertainties),
        reason=str(payload.get("reason") or "").strip(),
    )


def _fully_supported(receipt: TypedClaimReceipt, *, required: Sequence[str]) -> bool:
    by_kind = {check.kind: check for check in receipt.checks}
    return bool(
        receipt.claim_status == "SUPPORTED"
        and all(by_kind[kind].status == "SUPPORTED" for kind in required)
    )


def execute_mcq_guard(
    baseline: str,
    bound: Sequence[Mapping[str, Any]],
    *,
    required_checks: Sequence[str],
) -> dict[str, Any]:
    slots = [str(row["slot"]) for row in bound]
    if baseline not in slots or len(slots) != len(set(slots)):
        raise ValueError("MCQ candidate bindings are invalid")
    supported = [
        str(row["slot"]) for row in bound
        if _fully_supported(row["receipt"], required=required_checks)
    ]
    refuted = [
        str(row["slot"]) for row in bound
        if row["receipt"].claim_status == "REFUTED"
    ]
    recover = bool(
        len(supported) == 1
        and all(slot == supported[0] or slot in refuted for slot in slots)
        and supported[0] != baseline
    )
    return {
        "answer": supported[0] if recover else baseline,
        "recover": recover,
        "supported_slots": supported,
        "refuted_slots": refuted,
    }


def execute_binary_vector_guard(
    baseline: str,
    bound: Sequence[Mapping[str, Any]],
    *,
    required_checks: Sequence[str],
) -> dict[str, Any]:
    if len(baseline) != len(bound) or set(baseline) - {"0", "1"}:
        raise ValueError("binary-vector candidate bindings are invalid")
    output = list(baseline)
    decided = []
    for index, row in enumerate(bound):
        receipt = row["receipt"]
        if _fully_supported(receipt, required=required_checks):
            output[index] = "1"
            decided.append(index)
        elif receipt.claim_status == "REFUTED":
            output[index] = "0"
            decided.append(index)
    answer = "".join(output)
    return {"answer": answer, "recover": answer != baseline, "decided_indices": decided}


def rotate_bindings(bound: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(bound)
    if len(rows) < 2:
        raise ValueError("binding rotation needs at least two candidates")
    receipts = [row["receipt"] for row in rows[1:] + rows[:1]]
    return [
        {"slot": str(row["slot"]), "receipt": receipt}
        for row, receipt in zip(rows, receipts)
    ]


__all__ = [
    "CHECK_KINDS",
    "CHECK_STATUSES",
    "CLAIM_STATUSES",
    "EVIDENCE_BASES",
    "TypedClaimCheck",
    "TypedClaimReceipt",
    "execute_binary_vector_guard",
    "execute_mcq_guard",
    "parse_typed_claim_receipt",
    "rotate_bindings",
]
