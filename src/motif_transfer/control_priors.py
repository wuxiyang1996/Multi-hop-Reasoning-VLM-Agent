from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    Advisory,
    AdvisoryVerdict,
    BindingEvidence,
    BindingHypothesis,
    DecisionProposal,
    EvidenceVerdict,
    Lifecycle,
    Observation,
    TransitionReceipt,
    TransferObjectKind,
    stable_hash,
)


class ControlKnowledgeRole(str, Enum):
    """Interface roles, not a source-to-target semantic ontology."""

    CONTROL_REGULARITY = "CONTROL_REGULARITY"
    FAILURE_SIGNATURE = "FAILURE_SIGNATURE"
    VERIFICATION_ROUTINE = "VERIFICATION_ROUTINE"
    APPLICABILITY_BOUNDARY = "APPLICABILITY_BOUNDARY"


@dataclass(frozen=True)
class ReceiptGroundedClause:
    clause_id: str
    role: ControlKnowledgeRole
    untrusted_hypothesis: str
    source_receipt_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        role: ControlKnowledgeRole,
        untrusted_hypothesis: str,
        source_receipt_ids: Sequence[str],
    ) -> "ReceiptGroundedClause":
        body = {
            "role": role.value,
            "untrusted_hypothesis": str(untrusted_hypothesis).strip(),
            "source_receipt_ids": tuple(str(value) for value in source_receipt_ids),
        }
        return cls(stable_hash(body), role, body["untrusted_hypothesis"], body["source_receipt_ids"])

    def validate_hash(self) -> bool:
        return self.clause_id == stable_hash({
            "role": self.role.value,
            "untrusted_hypothesis": self.untrusted_hypothesis,
            "source_receipt_ids": self.source_receipt_ids,
        })


@dataclass(frozen=True)
class ReceiptGroundedKnowledge:
    knowledge_id: str
    source_lineage: tuple[str, ...]
    clauses: tuple[ReceiptGroundedClause, ...]
    status: Lifecycle = Lifecycle.CANDIDATE

    @classmethod
    def create(
        cls,
        source_lineage: Sequence[str],
        clauses: Sequence[ReceiptGroundedClause],
        *,
        status: Lifecycle = Lifecycle.CANDIDATE,
    ) -> "ReceiptGroundedKnowledge":
        rows = tuple(clauses)
        body = {
            "source_lineage": tuple(str(value) for value in source_lineage),
            "clauses": [
                {
                    **asdict(row),
                    "role": row.role.value,
                }
                for row in rows
            ],
            "status": status.value,
        }
        return cls(stable_hash(body), body["source_lineage"], rows, status)

    def validate_hash(self) -> bool:
        body = {
            "source_lineage": self.source_lineage,
            "clauses": [
                {
                    **asdict(row),
                    "role": row.role.value,
                }
                for row in self.clauses
            ],
            "status": self.status.value,
        }
        return self.knowledge_id == stable_hash(body)


@dataclass(frozen=True)
class ControlPriorAudit:
    accepted: bool
    failure_codes: tuple[str, ...]
    episode_support_by_clause: Mapping[str, int]


class KnowledgeJSONBackend(Protocol):
    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str: ...


def knowledge_to_mapping(knowledge: ReceiptGroundedKnowledge) -> dict[str, Any]:
    return {
        "knowledge_id": knowledge.knowledge_id,
        "source_lineage": list(knowledge.source_lineage),
        "clauses": [
            {
                "clause_id": row.clause_id,
                "role": row.role.value,
                "untrusted_hypothesis": row.untrusted_hypothesis,
                "source_receipt_ids": list(row.source_receipt_ids),
            }
            for row in knowledge.clauses
        ],
        "status": knowledge.status.value,
    }


def knowledge_from_mapping(value: Mapping[str, Any]) -> ReceiptGroundedKnowledge:
    clauses = tuple(
        ReceiptGroundedClause(
            clause_id=str(row["clause_id"]),
            role=ControlKnowledgeRole(str(row["role"])),
            untrusted_hypothesis=str(row["untrusted_hypothesis"]),
            source_receipt_ids=tuple(str(item) for item in row["source_receipt_ids"]),
        )
        for row in value.get("clauses") or ()
    )
    knowledge = ReceiptGroundedKnowledge(
        knowledge_id=str(value["knowledge_id"]),
        source_lineage=tuple(str(item) for item in value.get("source_lineage") or ()),
        clauses=clauses,
        status=Lifecycle(str(value.get("status") or Lifecycle.CANDIDATE.value)),
    )
    if not knowledge.validate_hash():
        raise ValueError("receipt-grounded knowledge hash mismatch")
    return knowledge


def audit_receipt_grounded_knowledge(
    knowledge: ReceiptGroundedKnowledge,
    *,
    receipt_to_episode: Mapping[str, str],
    require_source_supported: bool = True,
    minimum_episode_support: int = 2,
) -> ControlPriorAudit:
    """Mechanically check provenance and recurrence; never judge language truth."""
    failures: list[str] = []
    support: dict[str, int] = {}
    if not knowledge.validate_hash():
        failures.append("KNOWLEDGE_HASH_MISMATCH")
    if require_source_supported and knowledge.status != Lifecycle.SOURCE_SUPPORTED:
        failures.append("NOT_SOURCE_SUPPORTED")
    if not knowledge.source_lineage:
        failures.append("EMPTY_SOURCE_LINEAGE")
    if not knowledge.clauses:
        failures.append("EMPTY_KNOWLEDGE")
    if len({row.clause_id for row in knowledge.clauses}) != len(knowledge.clauses):
        failures.append("DUPLICATE_CLAUSE")
    for clause in knowledge.clauses:
        if not clause.validate_hash():
            failures.append(f"CLAUSE_HASH_MISMATCH:{clause.clause_id}")
        if not clause.untrusted_hypothesis:
            failures.append(f"EMPTY_HYPOTHESIS:{clause.clause_id}")
        if not clause.source_receipt_ids:
            failures.append(f"NO_SOURCE_RECEIPT:{clause.clause_id}")
        unknown = [
            receipt_id
            for receipt_id in clause.source_receipt_ids
            if receipt_id not in receipt_to_episode
        ]
        if unknown:
            failures.append(f"UNKNOWN_SOURCE_RECEIPT:{clause.clause_id}")
        episodes = {
            receipt_to_episode[receipt_id]
            for receipt_id in clause.source_receipt_ids
            if receipt_id in receipt_to_episode
        }
        support[clause.clause_id] = len(episodes)
        if len(episodes) < minimum_episode_support:
            failures.append(f"INSUFFICIENT_EPISODE_RECURRENCE:{clause.clause_id}")
    return ControlPriorAudit(not failures, tuple(failures), support)


def weak_prior_view(knowledge: ReceiptGroundedKnowledge) -> dict[str, object]:
    """Expose control knowledge without source actions, graph nodes, or graph edges."""
    return {
        "knowledge_id": knowledge.knowledge_id,
        "status": knowledge.status.value,
        "source_lineage_sha256": stable_hash(knowledge.source_lineage),
        "clauses": [
            {
                "clause_id": row.clause_id,
                "role": row.role.value,
                "untrusted_hypothesis": row.untrusted_hypothesis,
                "source_receipt_ids": list(row.source_receipt_ids),
                "source_receipt_count": len(row.source_receipt_ids),
            }
            for row in knowledge.clauses
        ],
    }


def compile_weak_prior_controls(
    authentic: ReceiptGroundedKnowledge,
    other_game: ReceiptGroundedKnowledge,
) -> dict[str, dict[str, Any]]:
    """Build matched payloads that separate knowledge content from context volume."""
    for name, knowledge in (("authentic", authentic), ("other_game", other_game)):
        if knowledge.status != Lifecycle.SOURCE_SUPPORTED or not knowledge.validate_hash():
            raise ValueError(f"{name} control prior is not a valid SOURCE_SUPPORTED artifact")
    if authentic.knowledge_id == other_game.knowledge_id:
        raise ValueError("other-game control must use a different source artifact")
    if len(authentic.clauses) < 2:
        raise ValueError("authentic prior needs at least two clauses for a shuffled-evidence control")

    authentic_view = weak_prior_view(authentic)
    other_view = weak_prior_view(other_game)
    clauses = list(authentic_view["clauses"])
    rotated_receipts = [
        clauses[(index + 1) % len(clauses)]["source_receipt_ids"]
        for index in range(len(clauses))
    ] if len(clauses) > 1 else [clauses[0]["source_receipt_ids"]]
    shuffled = {
        **authentic_view,
        "knowledge_id": "SHUFFLED_EVIDENCE_CONTROL",
        "clauses": [
            {
                **row,
                "source_receipt_ids": rotated_receipts[index],
                "source_receipt_count": len(rotated_receipts[index]),
            }
            for index, row in enumerate(clauses)
        ],
    }
    generic = {
        **authentic_view,
        "knowledge_id": "MATCHED_GENERIC_CONTROL",
        "source_lineage_sha256": "LINEAGE_SLOT",
        "clauses": [
            {
                **row,
                "clause_id": f"CLAUSE_SLOT_{index}",
                "untrusted_hypothesis": f"UNTRUSTED_CONTROL_STATEMENT_{index}",
                "source_receipt_ids": [
                    f"RECEIPT_SLOT_{index}_{receipt_index}"
                    for receipt_index, _ in enumerate(row["source_receipt_ids"])
                ],
            }
            for index, row in enumerate(clauses)
        ],
    }
    receipts_only = {
        "knowledge_id": "SOURCE_RECEIPTS_ONLY",
        "status": Lifecycle.SOURCE_SUPPORTED.value,
        "source_lineage_sha256": authentic_view["source_lineage_sha256"],
        "clauses": [
            {
                "clause_id": f"RECEIPT_GROUP_{index}",
                "role": "UNSPECIFIED",
                "untrusted_hypothesis": "",
                "source_receipt_ids": row["source_receipt_ids"],
                "source_receipt_count": row["source_receipt_count"],
            }
            for index, row in enumerate(clauses)
        ],
    }
    payloads = {
        "generic_reasoning": generic,
        "source_receipts_only": receipts_only,
        "authentic_weak_control_prior": authentic_view,
        "shuffled_evidence_prior": shuffled,
        "other_game_control_prior": other_view,
    }
    return {
        condition: {
            "schema_version": 1,
            "condition": condition,
            "transfer_object_kind": TransferObjectKind.WEAK_CONTROL_PRIOR.value,
            "payload": payload,
            "payload_sha256": stable_hash(payload),
        }
        for condition, payload in payloads.items()
    }


def initialize_weak_prior_hypothesis(
    knowledge: ReceiptGroundedKnowledge,
    *,
    adaptation_receipt_ids: Sequence[str],
    target_claim: str,
    testable_prediction: str,
    verifier_id: str,
) -> BindingHypothesis:
    """Create a target-time hypothesis with no structural or action alignment."""
    if knowledge.status != Lifecycle.SOURCE_SUPPORTED:
        raise ValueError("weak control prior requires SOURCE_SUPPORTED source knowledge")
    adaptation = tuple(str(value) for value in adaptation_receipt_ids)
    if not adaptation:
        raise ValueError("weak control prior requires live adaptation evidence")
    body = {
        "knowledge_id": knowledge.knowledge_id,
        "adaptation_receipt_ids": adaptation,
        "target_claim": str(target_claim).strip(),
        "testable_prediction": str(testable_prediction).strip(),
        "verifier_id": str(verifier_id),
        "transfer_object_kind": TransferObjectKind.WEAK_CONTROL_PRIOR.value,
    }
    if not body["target_claim"] or not body["testable_prediction"] or not body["verifier_id"]:
        raise ValueError("weak control prior hypothesis has an empty test field")
    return BindingHypothesis(
        binding_id=stable_hash(body),
        motif_id=knowledge.knowledge_id,
        target_claim=body["target_claim"],
        testable_prediction=body["testable_prediction"],
        adaptation_receipt_ids=adaptation,
        verifier_id=body["verifier_id"],
        status=Lifecycle.TARGET_PROVISIONAL,
        node_alignment=(),
        edge_alignment=(),
        invariance_signature="",
        transfer_object_kind=TransferObjectKind.WEAK_CONTROL_PRIOR,
    )


class ReceiptKnowledgeHarnessAgent:
    """No-action-authority Agent for weak source-knowledge use at target time."""

    def __init__(
        self,
        backend: KnowledgeJSONBackend,
        knowledge: Sequence[ReceiptGroundedKnowledge],
        *,
        allowed_verifier_ids: Sequence[str],
    ) -> None:
        self.backend = backend
        self.knowledge = {row.knowledge_id: row for row in knowledge}
        self.allowed_verifier_ids = frozenset(str(value) for value in allowed_verifier_ids)
        if not self.knowledge or not self.allowed_verifier_ids:
            raise ValueError("knowledge Harness requires source artifacts and registered verifiers")
        if any(
            row.status != Lifecycle.SOURCE_SUPPORTED or not row.validate_hash()
            for row in self.knowledge.values()
        ):
            raise ValueError("knowledge Harness accepts only valid SOURCE_SUPPORTED artifacts")

    @staticmethod
    def _json(raw: str) -> dict[str, Any]:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("knowledge Harness response must be one JSON object")
        return value

    @staticmethod
    def _validate_weak(binding: BindingHypothesis) -> None:
        if binding.transfer_object_kind != TransferObjectKind.WEAK_CONTROL_PRIOR:
            raise ValueError("knowledge Harness received a non-weak transfer object")
        if binding.node_alignment or binding.edge_alignment:
            raise ValueError("weak control prior may not carry target topology alignment")

    def initialize_from_example(
        self,
        knowledge_id: str,
        adaptation_example: Mapping[str, Any],
        *,
        adaptation_receipt_ids: Sequence[str],
        max_candidates: int = 4,
    ) -> tuple[BindingHypothesis, ...]:
        knowledge = self.knowledge[knowledge_id]
        payload = {
            "receipt_grounded_knowledge": weak_prior_view(knowledge),
            "one_target_adaptation_example": dict(adaptation_example),
            "adaptation_receipt_ids": list(adaptation_receipt_ids),
            "registered_verifier_ids": sorted(self.allowed_verifier_ids),
            "max_candidates": max_candidates,
        }
        system = (
            "Use source-derived knowledge only as a weak, falsifiable test-time hypothesis. "
            "Do not map source graph nodes or edges, and do not output, select, rank, or rewrite a target action. "
            "Return exact JSON with abstain and candidates. Each candidate has target_claim, "
            "testable_prediction, verifier_id, and cited_clause_ids. Evidence must be testable by a future "
            "live target receipt. If the example does not support a use, abstain."
        )
        raw = self._json(self.backend.complete("knowledge_initialize", system, payload))
        if set(raw) - {"abstain", "candidates"}:
            raise ValueError("knowledge initialization returned forbidden fields")
        if raw.get("abstain") is True:
            return ()
        candidates = raw.get("candidates") or []
        if not isinstance(candidates, list) or len(candidates) > max_candidates:
            raise ValueError("knowledge initialization returned an invalid candidate count")
        known_clauses = {row.clause_id for row in knowledge.clauses}
        result = []
        for row in candidates:
            if set(row) - {
                "target_claim", "testable_prediction", "verifier_id", "cited_clause_ids"
            }:
                raise ValueError("knowledge candidate contains forbidden fields")
            citations = {str(value) for value in row.get("cited_clause_ids") or ()}
            if not citations or not citations <= known_clauses:
                raise ValueError("knowledge candidate fabricated or omitted a source clause")
            verifier_id = str(row.get("verifier_id") or "")
            if verifier_id not in self.allowed_verifier_ids:
                raise ValueError("knowledge candidate selected an unregistered verifier")
            result.append(initialize_weak_prior_hypothesis(
                knowledge,
                adaptation_receipt_ids=adaptation_receipt_ids,
                target_claim=str(row.get("target_claim") or ""),
                testable_prediction=str(row.get("testable_prediction") or ""),
                verifier_id=verifier_id,
            ))
        if len({row.binding_id for row in result}) != len(result):
            raise ValueError("knowledge initialization returned duplicate hypotheses")
        return tuple(result)

    def review_bindings(
        self,
        proposal: DecisionProposal,
        observation: Observation,
        bindings: Sequence[BindingHypothesis],
        history: Sequence[TransitionReceipt],
    ) -> Advisory:
        rows = tuple(bindings)
        if not rows:
            raise ValueError("knowledge review requires at least one hypothesis")
        for row in rows:
            self._validate_weak(row)
            if row.motif_id not in self.knowledge:
                raise ValueError("knowledge review references an unregistered source artifact")
        payload = {
            "already_selected_target_native_proposal": asdict(proposal),
            "observation": dict(observation.state),
            "native_action_count": len(observation.native_actions),
            "hypotheses": [asdict(row) for row in rows],
            "source_knowledge": [
                weak_prior_view(self.knowledge[knowledge_id])
                for knowledge_id in sorted({row.motif_id for row in rows})
            ],
            "recent_live_receipts": [asdict(row) for row in history[-6:]],
        }
        system = (
            "Review an action already selected by the Decision Agent. Never output, replace, rank, or select an "
            "action. Return exact JSON with candidate_verdicts, one per binding_id. Each verdict contains "
            "binding_id, verdict (ADMIT, REPLAN, ABSTAIN), reason, cited_clause_ids, current_role, "
            "open_hypotheses, information_need, expected_transition, failure_route, and termination_test. "
            "Use only live-testable evidence. Disagreement must remain visible."
        )
        raw = self._json(self.backend.complete("knowledge_review", system, payload))
        if set(raw) != {"candidate_verdicts"}:
            raise ValueError("knowledge review returned forbidden or missing fields")
        verdict_rows = raw["candidate_verdicts"]
        by_id = {str(row.get("binding_id")): row for row in verdict_rows}
        if set(by_id) != {row.binding_id for row in rows} or len(by_id) != len(verdict_rows):
            raise ValueError("knowledge review did not cover the exact hypothesis version space")
        verdicts = []
        scalar_fields = (
            "current_role", "information_need", "expected_transition",
            "failure_route", "termination_test",
        )
        scalar_values: dict[str, list[str]] = {field: [] for field in scalar_fields}
        hypothesis_sets = []
        for binding in rows:
            row = by_id[binding.binding_id]
            if set(row) - {
                "binding_id", "verdict", "reason", "cited_clause_ids", "current_role",
                "open_hypotheses", "information_need", "expected_transition",
                "failure_route", "termination_test",
            }:
                raise ValueError("knowledge review verdict contains forbidden fields")
            known_clauses = {
                clause.clause_id for clause in self.knowledge[binding.motif_id].clauses
            }
            citations = {str(value) for value in row.get("cited_clause_ids") or ()}
            if not citations or not citations <= known_clauses:
                raise ValueError("knowledge review fabricated or omitted a source clause")
            verdicts.append(AdvisoryVerdict(str(row["verdict"])))
            for field in scalar_fields:
                scalar_values[field].append(str(row.get(field) or ""))
            hypothesis_sets.append({
                str(value) for value in row.get("open_hypotheses") or ()
            })
        unanimous = len(set(verdicts)) == 1
        common = {
            field: values[0] if len(set(values)) == 1 else ""
            for field, values in scalar_values.items()
        }
        common_hypotheses = (
            tuple(sorted(set.intersection(*hypothesis_sets))) if hypothesis_sets else ()
        )
        return Advisory(
            verdicts[0] if unanimous else AdvisoryVerdict.ABSTAIN,
            "unanimous weak-prior verdict" if unanimous else "weak-prior hypotheses disagreed",
            (),
            common["current_role"],
            common_hypotheses,
            common["information_need"],
            common["expected_transition"],
            common["failure_route"],
            common["termination_test"],
        )

    def verify_bindings(
        self,
        bindings: Sequence[BindingHypothesis],
        before: Observation,
        proposal: DecisionProposal,
        after: Observation,
        transition: TransitionReceipt,
        history: Sequence[TransitionReceipt],
    ) -> tuple[BindingEvidence, ...]:
        rows = tuple(bindings)
        for row in rows:
            self._validate_weak(row)
        payload = {
            "hypotheses": [asdict(row) for row in rows],
            "before_observation": dict(before.state),
            "already_executed_target_native_proposal": asdict(proposal),
            "after_observation": dict(after.state),
            "live_transition_receipt": asdict(transition),
            "recent_live_receipt_ids": [row.receipt_id for row in history[-6:]],
        }
        system = (
            "Verify each weak source-knowledge hypothesis against the supplied live target receipt. "
            "Do not output actions. Return exact JSON with candidate_evidence, one per binding_id, containing "
            "binding_id, verdict (SUPPORTED, REFUTED, INCONCLUSIVE), and reason."
        )
        raw = self._json(self.backend.complete("knowledge_verify", system, payload))
        if set(raw) != {"candidate_evidence"}:
            raise ValueError("knowledge verification returned forbidden or missing fields")
        evidence_rows = raw["candidate_evidence"]
        by_id = {str(row.get("binding_id")): row for row in evidence_rows}
        if set(by_id) != {row.binding_id for row in rows} or len(by_id) != len(evidence_rows):
            raise ValueError("knowledge verification did not cover the exact hypothesis version space")
        result = []
        for binding in rows:
            row = by_id[binding.binding_id]
            if set(row) != {"binding_id", "verdict", "reason"}:
                raise ValueError("knowledge verification evidence has an invalid schema")
            result.append(BindingEvidence(
                binding.binding_id,
                transition.receipt_id,
                binding.verifier_id,
                EvidenceVerdict(str(row["verdict"])),
            ))
        return tuple(result)
