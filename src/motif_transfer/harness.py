from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import (
    ConditionOutcome,
    DecisionProposal,
    DecisionProposalSet,
    DecisionCycleReceipt,
    Lifecycle,
    MotifCandidate,
    Observation,
    ReplayForkReceipt,
    SourceTransitionReceipt,
    SourceStepSignature,
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

    def validate_proposal_set(self, observation: Observation, proposal_set: DecisionProposalSet) -> None:
        if not proposal_set.proposals:
            raise HarnessReject("decision agent supplied an empty proposal set")
        ids = [row.proposal_id for row in proposal_set.proposals]
        if len(ids) != len(set(ids)):
            raise HarnessReject("proposal ids must be unique")
        try:
            proposal_set.selected
        except ValueError as exc:
            raise HarnessReject(str(exc)) from exc
        for proposal in proposal_set.proposals:
            self.validate_proposal(observation, proposal)

    def validate_receipt(self, receipt: TransitionReceipt | SourceTransitionReceipt) -> None:
        if not receipt.validate():
            raise HarnessReject("receipt hash mismatch")

    def validate_cycle(self, cycle: DecisionCycleReceipt) -> None:
        if not cycle.validate():
            raise HarnessReject("decision-cycle hash mismatch")

    def audit_motif(
        self,
        candidate: MotifCandidate,
        receipts: Mapping[str, TransitionReceipt | SourceTransitionReceipt | ReplayForkReceipt],
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
        if not referenced or any(
            receipt_id not in receipts
            or not isinstance(receipts[receipt_id], (TransitionReceipt, SourceTransitionReceipt))
            for receipt_id in referenced
        ):
            return MotifAudit(False, None, "unknown transition receipt")
        if any(not receipts[receipt_id].validate() for receipt_id in referenced):
            return MotifAudit(False, None, "invalid transition receipt")
        known_nodes = set(node_ids)
        if not candidate.edges:
            return MotifAudit(False, None, "motif must contain at least one control edge")
        for edge in candidate.edges:
            if edge.source not in known_nodes or edge.target not in known_nodes:
                return MotifAudit(False, None, "edge references unknown node")
            if any(
                receipt_id not in receipts or not isinstance(receipts[receipt_id], ReplayForkReceipt)
                for receipt_id in edge.replay_receipt_ids
            ):
                return MotifAudit(False, None, "edge references unknown replay receipt")
            if not edge.replay_receipt_ids:
                return MotifAudit(False, None, "control edge lacks a replay-fork receipt")
            if any(not receipts[receipt_id].validate() for receipt_id in edge.replay_receipt_ids):
                return MotifAudit(False, None, "invalid replay-fork receipt")
            source_transitions = set(
                candidate.nodes[node_ids.index(edge.source)].transition_receipt_ids
            )
            if any(
                receipts[receipt_id].source_transition_id not in source_transitions
                for receipt_id in edge.replay_receipt_ids
            ):
                return MotifAudit(
                    False, None, "replay fork is not grounded in edge source node"
                )
            exhaustive_forks = {
                receipt_id
                for receipt_id, receipt in receipts.items()
                if isinstance(receipt, ReplayForkReceipt)
                and receipt.source_transition_id in source_transitions
            }
            if set(edge.replay_receipt_ids) != exhaustive_forks:
                return MotifAudit(
                    False, None, "edge must carry every observed fork from its source node"
                )

        skill_class_map: dict[int, int] = {}
        for node in candidate.nodes:
            for signature in node.decision_signatures:
                if (
                    isinstance(signature, SourceStepSignature)
                    and signature.skill_class_ordinal is not None
                    and signature.skill_class_ordinal not in skill_class_map
                ):
                    skill_class_map[signature.skill_class_ordinal] = len(skill_class_map)
        signatures = {
            stable_hash(self._behavioral_signature(signature, skill_class_map))
            for node in candidate.nodes
            for signature in node.decision_signatures
        }
        outdegree = {node_id: 0 for node_id in node_ids}
        for edge in candidate.edges:
            outdegree[edge.source] += 1
        if len(signatures) < 2 and max(outdegree.values(), default=0) < 2:
            return MotifAudit(False, None, "no observable control variation")

        # Natural-language descriptions and edge claims are excluded on purpose.
        node_ordinals = {node_id: index for index, node_id in enumerate(node_ids)}
        def signature_rle(node):
            result = []
            for signature in node.decision_signatures:
                normalized = self._behavioral_signature(signature, skill_class_map)
                if not result or result[-1] != normalized:
                    result.append(normalized)
            return result

        verified_shape = {
            "nodes": [
                {
                    # Exact receipts remain mandatory evidence, but repeated time spent
                    # in one control state is domain timing, not motif identity.
                    "decision_signature_rle": signature_rle(node),
                }
                for node in candidate.nodes
            ],
            "edges": [
                {
                    "source_ordinal": node_ordinals[edge.source],
                    "target_ordinal": node_ordinals[edge.target],
                    "replay_count": len(edge.replay_receipt_ids),
                }
                for edge in candidate.edges
            ],
        }
        return MotifAudit(True, stable_hash(verified_shape), "STRUCTURALLY_NONUNIFORM_CANDIDATE")

    @staticmethod
    def _behavioral_signature(signature, skill_class_map=None):
        values = dict(signature.__dict__)
        if isinstance(signature, SourceStepSignature):
            # Treatment membership is provenance, not behavioral structure.
            values.pop("skill_conditioned", None)
            ordinal = values.get("skill_class_ordinal")
            if ordinal is not None and skill_class_map is not None:
                values["skill_class_ordinal"] = skill_class_map[ordinal]
        return values

    def evaluate_matched(self, outcomes: Iterable[ConditionOutcome]) -> TransferReport:
        rows = tuple(outcomes)
        pairs: dict[str, list[ConditionOutcome]] = {}
        for row in rows:
            pairs.setdefault(row.pair_id, []).append(row)
        identity_fields = ("initial_state_hash", "prefix_hash", "policy_hash", "budget_hash")
        for pair_id, pair_rows in pairs.items():
            names = {row.condition for row in pair_rows}
            missing = self.REQUIRED_CONDITIONS - names
            if missing:
                return TransferReport(
                    Lifecycle.INCONCLUSIVE,
                    f"pair {pair_id} missing conditions: {sorted(missing)}",
                    rows,
                )
            if len(pair_rows) != len(names):
                return TransferReport(Lifecycle.INCONCLUSIVE, f"pair {pair_id} has duplicate conditions", rows)
            if any(len({getattr(row, field) for row in pair_rows}) != 1 for field in identity_fields):
                return TransferReport(Lifecycle.INCONCLUSIVE, f"pair {pair_id} identity mismatch", rows)

        rates = {
            condition: sum(row.official_success for row in rows if row.condition == condition) / len(pairs)
            for condition in self.REQUIRED_CONDITIONS
        }
        authentic = rates["authentic"]
        target_only = rates["target_only"]
        generic = rates["generic_protocol"]
        other_controls = max(rates["shuffled_topology"], rates["other_source"])
        if authentic > target_only:
            if generic >= authentic:
                return TransferReport(Lifecycle.GENERIC_ONLY, "generic protocol explains the gain", rows, rates)
            if authentic > other_controls:
                return TransferReport(
                    Lifecycle.POSITIVE_TRANSFER,
                    "authentic motif improves paired official outcomes in this pilot",
                    rows,
                    rates,
                )
        if target_only > authentic:
            return TransferReport(Lifecycle.NEGATIVE_TRANSFER, "authentic motif harms paired official outcomes", rows, rates)
        return TransferReport(Lifecycle.INCONCLUSIVE, "no attributable paired outcome separation", rows, rates)
