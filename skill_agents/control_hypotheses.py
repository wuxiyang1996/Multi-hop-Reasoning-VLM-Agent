"""Evidence-qualified, Agent-proposed multi-step control hypotheses.

Validation proves reference integrity and observed ordering only.  It never
turns Agent prose or a single observed path into a causal/control fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from skill_agents.evidence_query import EvidenceResponse
from skill_bank.trace_program_ir import ControlClaimKind, TraceProgram


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HypothesisNode:
    node_id: str
    transition_ids: Sequence[str]


@dataclass(frozen=True)
class HypothesisEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    kind: ControlClaimKind
    # Agent-authored and untrusted.  It is retained for later intervention
    # design, never evaluated as a predicate by this validator.
    agent_claim: Mapping[str, Any]
    # Exact content hashes of observed intervention receipts. The validator
    # checks identity and source-program compatibility, never prose semantics.
    intervention_receipt_sha256s: Sequence[str] = ()


@dataclass(frozen=True)
class AgentControlHypothesis:
    hypothesis_id: str
    program_id: str
    program_hash: str
    proposal_source: str
    evidence_response_hashes: Sequence[str]
    nodes: Sequence[HypothesisNode]
    edges: Sequence[HypothesisEdge]
    abstain: bool = False

    def content_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class QualifiedHypothesis:
    hypothesis: AgentControlHypothesis
    hypothesis_hash: str
    status: str
    checks: Mapping[str, bool]
    failure_codes: Sequence[str]


class ControlHypothesisValidator:
    """Fail-closed structural validator with no semantic scorer."""

    def validate(
        self,
        hypothesis: AgentControlHypothesis,
        *,
        program: TraceProgram,
        evidence_responses: Sequence[EvidenceResponse],
        intervention_receipts: Sequence[Mapping[str, Any]] = (),
        require_full_partition: bool = False,
        require_multinode: bool = False,
    ) -> QualifiedHypothesis:
        known_transitions = {item.transition_id for item in program.transitions}
        known_responses = {item.response_sha256 for item in evidence_responses}
        shown_transitions = {
            str(row.get("transition_id") or "")
            for response in evidence_responses
            for row in response.transitions
        }
        response_identities_match = all(
            response.program_id == program.program_id
            and response.program_hash == program.content_hash()
            for response in evidence_responses
        )
        for response in evidence_responses:
            response.validate_hash()
        node_ids = [item.node_id for item in hypothesis.nodes]
        cited = [tid for node in hypothesis.nodes for tid in node.transition_ids]
        cited_receipts = [
            receipt_hash
            for edge in hypothesis.edges
            for receipt_hash in edge.intervention_receipt_sha256s
        ]
        receipts_by_hash = {
            str(item.get("receipt_sha256") or ""): item
            for item in intervention_receipts
        }
        positions = {
            item.transition_id: index for index, item in enumerate(program.transitions)
        }
        contiguous_nodes = all(
            bool(node.transition_ids)
            and all(
                positions[right] == positions[left] + 1
                for left, right in zip(node.transition_ids, node.transition_ids[1:])
            )
            for node in hypothesis.nodes
            if all(item in positions for item in node.transition_ids)
        )
        ordered_full_partition = cited == [
            item.transition_id for item in program.transitions
        ]
        native_by_transition = {
            str(row.get("transition_id") or ""): row.get("native_evidence") or {}
            for response in evidence_responses
            for row in response.transitions
        }

        def _receipt_compatible(receipt: Mapping[str, Any]) -> bool:
            prefix = [str(item) for item in receipt.get("prefix_actions") or ()]
            fork_index = len(prefix)
            if fork_index >= len(program.transitions):
                return False
            if prefix != [item.action for item in program.transitions[:fork_index]]:
                return False
            if str(receipt.get("expected_fork_state_sha256") or "") != (
                program.transitions[fork_index].state_sha256
            ):
                return False
            native = list(native_by_transition.get(
                program.transitions[fork_index].transition_id, {}
            ).get("available_actions") or ())
            return str(receipt.get("alternative_action") or "") in native

        nodes_by_id = {item.node_id: set(item.transition_ids) for item in hypothesis.nodes}

        def _edge_receipts_attached(edge: HypothesisEdge) -> bool:
            anchors = nodes_by_id.get(edge.source_node_id, set()) | nodes_by_id.get(
                edge.target_node_id, set()
            )
            for receipt_hash in edge.intervention_receipt_sha256s:
                receipt = receipts_by_hash.get(receipt_hash)
                if receipt is None:
                    return False
                fork_index = len(receipt.get("prefix_actions") or ())
                if fork_index >= len(program.transitions):
                    return False
                if program.transitions[fork_index].transition_id not in anchors:
                    return False
            return True

        cited_rows = [receipts_by_hash[item] for item in cited_receipts if item in receipts_by_hash]

        def _receipt_hash_valid(receipt: Mapping[str, Any]) -> bool:
            payload = {
                key: receipt.get(key) for key in (
                    "intervention_id", "seed", "prefix_actions",
                    "expected_fork_state_sha256", "replayed_fork_state_sha256",
                    "alternative_action", "admissible_actions_sha256",
                    "alternative_next_state_sha256", "status", "failure_codes",
                )
            }
            return _hash(payload) == str(receipt.get("receipt_sha256") or "")

        checks = {
            "not_abstain": not hypothesis.abstain,
            "program_id": hypothesis.program_id == program.program_id,
            "program_hash": hypothesis.program_hash == program.content_hash(),
            "evidence_responses_known": bool(hypothesis.evidence_response_hashes)
            and set(hypothesis.evidence_response_hashes).issubset(known_responses),
            "evidence_response_identities_match": response_identities_match,
            "node_ids_unique": bool(node_ids) and len(node_ids) == len(set(node_ids)),
            "transition_references_known": bool(cited) and set(cited).issubset(known_transitions),
            "transition_references_were_shown": bool(cited)
            and set(cited).issubset(shown_transitions),
            "nodes_contiguous_on_observed_path": contiguous_nodes,
            "nodes_partition_full_observed_path": (
                ordered_full_partition if require_full_partition else True
            ),
            "multiple_nodes": len(hypothesis.nodes) >= 2 if require_multinode else True,
            "edges_reference_nodes": all(
                edge.source_node_id in node_ids and edge.target_node_id in node_ids
                for edge in hypothesis.edges
            ),
            "edge_ids_unique": len({edge.edge_id for edge in hypothesis.edges})
            == len(hypothesis.edges),
            "intervention_receipts_unique": len(cited_receipts) == len(set(cited_receipts)),
            "intervention_receipts_known": set(cited_receipts).issubset(receipts_by_hash),
            "intervention_receipt_hashes_valid": all(
                _receipt_hash_valid(item) for item in cited_rows
            ) and len(cited_rows) == len(cited_receipts),
            "intervention_receipts_observed": all(
                str(item.get("status") or "") == "INTERVENTION_OBSERVED"
                for item in cited_rows
            ) and len(cited_rows) == len(cited_receipts),
            "intervention_receipts_program_compatible": all(
                _receipt_compatible(item) for item in cited_rows
            ) and len(cited_rows) == len(cited_receipts),
            "intervention_receipts_edge_attached": all(
                _edge_receipts_attached(edge) for edge in hypothesis.edges
            ),
        }
        failures = [name.upper() for name, passed in checks.items() if not passed]
        return QualifiedHypothesis(
            hypothesis=hypothesis,
            hypothesis_hash=hypothesis.content_hash(),
            status="AGENT_HYPOTHESIS" if not failures else "REJECTED",
            checks=checks,
            failure_codes=failures,
        )


def union_qualified_hypotheses(
    candidates: Sequence[QualifiedHypothesis],
) -> Sequence[QualifiedHypothesis]:
    """Content-hash deduplication + set union; deliberately no vote/rank."""
    accepted = {
        item.hypothesis_hash: item
        for item in candidates
        if item.status == "AGENT_HYPOTHESIS"
    }
    return tuple(accepted[key] for key in sorted(accepted))


__all__ = [
    "AgentControlHypothesis",
    "ControlHypothesisValidator",
    "HypothesisEdge",
    "HypothesisNode",
    "QualifiedHypothesis",
    "union_qualified_hypotheses",
]
