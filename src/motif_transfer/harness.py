from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import (
    ConditionOutcome,
    DecisionProposal,
    Lifecycle,
    MotifCandidate,
    Observation,
    TransferReport,
    TransitionReceipt,
    stable_hash,
)


class HarnessReject(ValueError):
    pass


@dataclass(frozen=True)
class MotifAudit:
    accepted: bool
    structural_fingerprint: str | None
    reason: str


class DeterministicHarness:
    REQUIRED_CONDITIONS = frozenset(
        {"authentic", "target_only", "generic_protocol", "shuffled_topology", "other_source"}
    )

    def validate_proposal(self, observation: Observation, proposal: DecisionProposal) -> None:
        if proposal.agent_id != "decision-agent":
            raise HarnessReject("only decision-agent may choose an action")
        if proposal.action not in observation.native_actions:
            raise HarnessReject("proposal is not an exact member of the native action set")

    def validate_receipt(self, receipt: TransitionReceipt) -> None:
        if not receipt.validate():
            raise HarnessReject("receipt hash mismatch")

    def audit_motif(
        self, candidate: MotifCandidate, receipts: Mapping[str, TransitionReceipt]
    ) -> MotifAudit:
        if len(candidate.nodes) < 2:
            return MotifAudit(False, None, "motif must contain at least two nodes")
        node_ids = [node.node_id for node in candidate.nodes]
        if len(node_ids) != len(set(node_ids)):
            return MotifAudit(False, None, "duplicate node id")
        referenced: list[str] = []
        for node in candidate.nodes:
            referenced.extend(node.transition_receipt_ids)
        if len(referenced) != len(set(referenced)):
            return MotifAudit(False, None, "a transition may not be assigned twice")
        if not referenced or any(receipt_id not in receipts for receipt_id in referenced):
            return MotifAudit(False, None, "unknown transition receipt")
        if any(not receipts[receipt_id].validate() for receipt_id in referenced):
            return MotifAudit(False, None, "invalid transition receipt")
        known_nodes = set(node_ids)
        for edge in candidate.edges:
            if edge.source not in known_nodes or edge.target not in known_nodes:
                return MotifAudit(False, None, "edge references unknown node")
            if any(receipt_id not in receipts for receipt_id in edge.replay_receipt_ids):
                return MotifAudit(False, None, "edge references unknown replay receipt")

        # Natural-language descriptions and edge claims are excluded on purpose.
        verified_shape = {
            "nodes": [
                {
                    "receipts": node.transition_receipt_ids,
                    "decision_signatures": [signature.__dict__ for signature in node.decision_signatures],
                }
                for node in candidate.nodes
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "receipts": edge.replay_receipt_ids}
                for edge in candidate.edges
            ],
        }
        return MotifAudit(True, stable_hash(verified_shape), "STRUCTURALLY_NONUNIFORM_CANDIDATE")

    def evaluate_matched(self, outcomes: Iterable[ConditionOutcome]) -> TransferReport:
        rows = tuple(outcomes)
        by_name = {row.condition: row for row in rows}
        missing = self.REQUIRED_CONDITIONS - by_name.keys()
        if missing:
            return TransferReport(Lifecycle.INCONCLUSIVE, f"missing conditions: {sorted(missing)}", rows)
        identity_fields = ("initial_state_hash", "prefix_hash", "policy_hash", "budget_hash")
        if any(len({getattr(row, field) for row in rows}) != 1 for field in identity_fields):
            return TransferReport(Lifecycle.INCONCLUSIVE, "matched-run identity mismatch", rows)

        authentic = by_name["authentic"]
        target_only = by_name["target_only"]
        generic = by_name["generic_protocol"]
        controls = (by_name["shuffled_topology"], by_name["other_source"])
        if authentic.official_success and not target_only.official_success:
            if generic.official_success:
                return TransferReport(Lifecycle.GENERIC_ONLY, "generic protocol explains the gain", rows)
            if not any(control.official_success for control in controls):
                return TransferReport(Lifecycle.POSITIVE_TRANSFER, "authentic motif uniquely improves official outcome", rows)
        if target_only.official_success and not authentic.official_success:
            return TransferReport(Lifecycle.NEGATIVE_TRANSFER, "authentic motif harms official outcome", rows)
        return TransferReport(Lifecycle.INCONCLUSIVE, "no attributable outcome separation", rows)
