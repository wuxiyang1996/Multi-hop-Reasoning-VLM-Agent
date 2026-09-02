"""Three-valued visual evidence for AGQA Layer-B Harness decisions.

The raw event graph records positive hypotheses.  It cannot justify treating an
unobserved event as false.  This module adds a content-addressed, arm-shared
receipt for explicit support, refutation, or uncertainty and applies the
open-world guard semantics induced from source interventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .agqa_layer_b_contracts import AGQASemanticSlotReceipt
from .contracts import stable_hash


CLAIM_STATUSES = frozenset({"SUPPORTED", "REFUTED", "UNKNOWN"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset({
    "answer", "correct", "functional_program", "gold", "operator_sequence",
    "program", "selected_option", "source_controller", "source_game",
    "source_identity", "target_outcome",
})


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_KEYS or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(child) for child in value)
    return False


@dataclass(frozen=True)
class AtomicVisualClaim:
    claim_id: str
    semantic_root_slot_id: str
    proposition: str

    def validate(self, known_slots: set[str]) -> None:
        if re.fullmatch(r"C[0-9]+", self.claim_id) is None:
            raise ValueError("claim IDs must be C0,C1,...")
        if self.semantic_root_slot_id not in known_slots:
            raise ValueError("claim references an unknown semantic slot")
        if not self.proposition.strip():
            raise ValueError("claim proposition must be non-empty")


@dataclass(frozen=True)
class AtomicVisualClaimDecision:
    claim_id: str
    status: str
    confidence: float
    evidence_frame_indices: tuple[int, ...]
    evidence_frame_sha256s: tuple[str, ...]
    rationale: str

    def validate(self, known_claims: set[str], frame_count: int) -> None:
        if self.claim_id not in known_claims:
            raise ValueError("decision references an unknown claim")
        if self.status not in CLAIM_STATUSES:
            raise ValueError("invalid claim status")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("claim confidence must be in [0,1]")
        if tuple(sorted(set(self.evidence_frame_indices))) != self.evidence_frame_indices:
            raise ValueError("claim evidence frames must be unique and chronological")
        if any(index < 0 or index >= frame_count for index in self.evidence_frame_indices):
            raise ValueError("claim evidence frame is outside the frozen frame set")
        if len(self.evidence_frame_indices) != len(self.evidence_frame_sha256s):
            raise ValueError("claim evidence indices/hashes are misaligned")
        if any(_SHA256.fullmatch(value) is None for value in self.evidence_frame_sha256s):
            raise ValueError("claim evidence must use sha256 frame identities")
        # Both truth and falsity require cited pixels.  Absence is UNKNOWN.
        if self.status in {"SUPPORTED", "REFUTED"} and not self.evidence_frame_indices:
            raise ValueError("supported/refuted claims require pixel evidence")


@dataclass(frozen=True)
class AtomicVisualClaimReceipt:
    task_id: str
    semantic_receipt_sha256: str
    raw_event_graph_receipt_sha256: str
    claims: tuple[AtomicVisualClaim, ...]
    decisions: tuple[AtomicVisualClaimDecision, ...]
    verifier_backend_sha256: str
    frame_budget: int
    shared_across_all_harness_arms: bool
    answer_read: bool
    functional_program_read: bool
    source_controller_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, semantic_receipt_sha256: str,
        raw_event_graph_receipt_sha256: str, claims: Sequence[AtomicVisualClaim],
        decisions: Sequence[AtomicVisualClaimDecision], verifier_backend_sha256: str,
        frame_budget: int,
    ) -> "AtomicVisualClaimReceipt":
        claim_rows, decision_rows = tuple(claims), tuple(decisions)
        body = {
            "task_id": str(task_id),
            "semantic_receipt_sha256": str(semantic_receipt_sha256),
            "raw_event_graph_receipt_sha256": str(raw_event_graph_receipt_sha256),
            "claims": [asdict(row) for row in claim_rows],
            "decisions": [asdict(row) for row in decision_rows],
            "verifier_backend_sha256": str(verifier_backend_sha256),
            "frame_budget": int(frame_budget),
            "shared_across_all_harness_arms": True,
            "answer_read": False,
            "functional_program_read": False,
            "source_controller_read": False,
        }
        value = cls(
            **{**body, "claims": claim_rows, "decisions": decision_rows},
            receipt_sha256=stable_hash(body),
        )
        value.validate()
        return value

    def validate(self) -> None:
        for value in (
            self.semantic_receipt_sha256, self.raw_event_graph_receipt_sha256,
            self.verifier_backend_sha256, self.receipt_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("epistemic receipt hashes must be sha256")
        if not self.shared_across_all_harness_arms:
            raise ValueError("epistemic evidence must be shared across all arms")
        if self.answer_read or self.functional_program_read or self.source_controller_read:
            raise ValueError("epistemic verifier crossed an authority boundary")
        if self.frame_budget <= 0:
            raise ValueError("epistemic verifier needs a positive frozen frame budget")
        claim_ids = [row.claim_id for row in self.claims]
        decision_ids = [row.claim_id for row in self.decisions]
        if len(claim_ids) != len(set(claim_ids)) or set(decision_ids) != set(claim_ids):
            raise ValueError("epistemic receipt needs exactly one decision per claim")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("epistemic receipt has duplicate decisions")
        known_slots = {row.semantic_root_slot_id for row in self.claims}
        for row in self.claims:
            row.validate(known_slots | {row.semantic_root_slot_id})
        for row in self.decisions:
            row.validate(set(claim_ids), self.frame_budget)
        body = asdict(self); claimed = body.pop("receipt_sha256")
        if _contains_forbidden_key(body):
            raise ValueError("epistemic receipt contains forbidden outcome/program fields")
        if stable_hash(body) != claimed:
            raise ValueError("epistemic receipt hash mismatch")


def extract_atomic_claims(semantic: AGQASemanticSlotReceipt) -> tuple[AtomicVisualClaim, ...]:
    """Extract operator-free branch claims from the target-native slot graph."""

    semantic.validate()
    by_id = {row.slot_id: row for row in semantic.slots}

    def describe(slot_id: str) -> str:
        node = by_id[slot_id]
        leaves: list[str] = []

        def walk(value: str) -> None:
            child = by_id[value]
            if not child.children:
                if child.surface not in leaves:
                    leaves.append(child.surface)
                return
            for nested in child.children:
                walk(nested)

        walk(slot_id)
        suffix = "; ".join(leaves)
        return f"{node.surface}: {suffix}" if suffix else node.surface

    roots: list[str] = []
    for node in semantic.slots:
        if node.kind == "LOGICAL_CONSTRAINT" and (
            node.surface.startswith("require exactly one")
            or node.surface.startswith("require both")
        ):
            roots.extend(node.children)
    if not roots and semantic.answer_kind == "BOOLEAN":
        roots.append(semantic.root_slot_id)
    # Preserve graph order and prevent the verifier from receiving duplicate branches.
    roots = list(dict.fromkeys(roots))
    return tuple(
        AtomicVisualClaim(f"C{index}", slot_id, describe(slot_id))
        for index, slot_id in enumerate(roots)
    )


def source_open_world_commit(
    *, required_operators: Sequence[str], symbolic_status: str,
    symbolic_prediction: str | None, evidence: AtomicVisualClaimReceipt,
) -> tuple[bool, str]:
    """Apply only source-induced three-valued guard rules.

    The executor remains shared.  This function controls whether its result is
    safe to commit; UNKNOWN always falls back to the shared neural actor.
    """

    evidence.validate()
    if symbolic_status != "COMMITTED":
        return False, "SYMBOLIC_EXECUTOR_ABSTAINED"
    statuses = [row.status for row in evidence.decisions]
    operators = set(required_operators)
    if "XOR" in operators:
        if len(statuses) != 2:
            return False, "XOR_REQUIRES_TWO_EXPLICIT_BRANCH_CLAIMS"
        if symbolic_prediction not in {"yes", "no"}:
            # For a CHOOSE-style answer the shared strict executor determines
            # which grounded branch supplies the value.  The source guard only
            # certifies that the two alternatives are genuinely exclusive.
            safe = sorted(statuses) == ["REFUTED", "SUPPORTED"]
        elif symbolic_prediction == "yes":
            safe = sorted(statuses) == ["REFUTED", "SUPPORTED"]
        else:
            safe = len(set(statuses)) == 1 and statuses[0] in {"SUPPORTED", "REFUTED"}
        return safe, "EXCLUSIVE_GUARDS_CONFIRMED" if safe else "EXCLUSIVE_GUARDS_UNKNOWN"
    if "AND" in operators:
        safe = (
            all(value == "SUPPORTED" for value in statuses)
            if symbolic_prediction == "yes"
            else "REFUTED" in statuses
        )
        return safe, "CONJUNCTIVE_GUARD_CONFIRMED" if safe else "CONJUNCTIVE_GUARD_UNKNOWN"
    if "EXISTS" in operators and symbolic_prediction in {"yes", "no"}:
        expected = "SUPPORTED" if symbolic_prediction == "yes" else "REFUTED"
        safe = bool(statuses) and expected in statuses
        return safe, "CARDINALITY_GUARD_CONFIRMED" if safe else "CARDINALITY_GUARD_UNKNOWN"
    return True, "NO_OPEN_WORLD_BOOLEAN_GUARD_REQUIRED"


def source_root_open_world_commit(
    *, semantic: AGQASemanticSlotReceipt, symbolic_status: str,
    symbolic_prediction: str | None, evidence: AtomicVisualClaimReceipt,
) -> tuple[bool, str]:
    """Apply three-valued guards by semantic root, not any nested operator.

    Nested EXISTS calls are perceptual subroutines.  They must not turn an
    equality, choice, or entity query into a top-level presence decision.
    """
    semantic.validate(); by_id = {row.slot_id: row for row in semantic.slots}
    root = by_id[semantic.root_slot_id]
    if root.kind == "LOGICAL_CONSTRAINT" and root.surface.startswith("require exactly one"):
        required = ("XOR",)
    elif root.kind == "LOGICAL_CONSTRAINT" and root.surface.startswith("require both"):
        required = ("AND",)
    elif root.kind == "QUERY_GOAL" and root.surface.startswith("ask whether"):
        required = ("EXISTS",)
    else:
        required = ()
    return source_open_world_commit(
        required_operators=required, symbolic_status=symbolic_status,
        symbolic_prediction=symbolic_prediction, evidence=evidence,
    )


__all__ = [
    "AtomicVisualClaim", "AtomicVisualClaimDecision", "AtomicVisualClaimReceipt",
    "CLAIM_STATUSES", "extract_atomic_claims", "source_open_world_commit",
    "source_root_open_world_commit",
]
