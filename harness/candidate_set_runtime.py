"""Episode-local exact action-set intersection over a frozen candidate set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from harness.alfworld_grammar import parse_alfworld_action
from harness.multistep_binding import MultiStepAdmissionArtifact, QualifiedBindingCandidate


@dataclass(frozen=True)
class CandidateActionProposal:
    proposal_scope_hash: str
    command: str | None
    abstain: bool = False
    endpoint_error: str | None = None


@dataclass(frozen=True)
class ConsensusDecision:
    command: str | None
    status: str
    reason: str | None
    proposals: Sequence[CandidateActionProposal]


class CandidateRuntimeStatus(str, Enum):
    """Episode-local status; the frozen admission artifact never changes."""

    ACTIVE = "ACTIVE"
    DEFERRED = "DEFERRED"
    REFUTED = "REFUTED"
    FINISHED = "FINISHED"


class FrozenCandidateSetRuntime:
    """Artifact is immutable; only per-episode cursors change."""

    def __init__(self, artifact: MultiStepAdmissionArtifact) -> None:
        if artifact.schema_version != 3:
            raise ValueError("candidate-set runtime requires v3 artifact")
        if artifact.node_binding_version != 3:
            raise ValueError("candidate-set runtime requires source-conditioning binding v3")
        self.artifact = artifact
        self._cursor = {item.candidate_hash: 0 for item in artifact.candidates}
        self._status = {
            item.candidate_hash: CandidateRuntimeStatus.ACTIVE
            for item in artifact.candidates
        }
        self._status_receipt: dict[str, str | None] = {
            item.candidate_hash: None for item in artifact.candidates
        }

    @property
    def cursors(self) -> Mapping[str, int]:
        return dict(self._cursor)

    @property
    def statuses(self) -> Mapping[str, str]:
        return {key: value.value for key, value in self._status.items()}

    @property
    def status_receipts(self) -> Mapping[str, str | None]:
        return dict(self._status_receipt)

    def _active_candidates(self) -> tuple[QualifiedBindingCandidate, ...]:
        return tuple(
            item for item in self.artifact.candidates
            if self._status[item.candidate_hash] == CandidateRuntimeStatus.ACTIVE
        )

    def align_to_same_demo_prefix(self, *, demo_hash: str, completed_steps: int) -> None:
        """Align a fresh runtime after a verified same-demo treatment switch.

        Admission requires every candidate to partition the complete demo in
        exact target-transition order.  The caller must additionally provide
        the immutable demo identity; this method never guesses alignment from
        action text or semantic similarity.
        """
        if demo_hash != self.artifact.demo_hash:
            raise ValueError("cannot align runtimes admitted from different demos")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        for qualified in self.artifact.candidates:
            total = sum(len(node.target_steps) for node in qualified.candidate.nodes)
            if completed_steps > total:
                raise ValueError("completed prefix exceeds candidate program length")
        for key in self._cursor:
            self._cursor[key] = int(completed_steps)
            self._status[key] = CandidateRuntimeStatus.ACTIVE
            self._status_receipt[key] = None

    def active_source_conditioning(self) -> Sequence[Mapping[str, object]]:
        """Return every active candidate context; never select or rank one."""
        demo_receipts = {
            item.target_transition_index: item
            for item in self.artifact.demo_transition_contract_receipts
        }
        contexts = []
        for qualified in self._active_candidates():
            cursor = self._cursor[qualified.candidate_hash]
            offset = 0
            for node in qualified.candidate.nodes:
                upper = offset + len(node.target_steps)
                if offset <= cursor < upper:
                    if node.source_conditioning:
                        target_step = node.target_steps[cursor - offset]
                        demo_receipt = demo_receipts.get(
                            target_step.target_transition_index
                        )
                        contexts.append({
                            "candidate_hash": qualified.candidate_hash,
                            "source_hypothesis_hash": qualified.candidate.source_hypothesis_hash,
                            "node_id": node.node_id,
                            "target_transition_index": (
                                target_step.target_transition_index
                            ),
                            "demo_transition_receipt_sha256": (
                                demo_receipt.receipt_sha256
                                if demo_receipt is not None else None
                            ),
                            "demo_supported_evidence": (
                                list(demo_receipt.supported_evidence)
                                if demo_receipt is not None else []
                            ),
                            "source_conditioning": dict(node.source_conditioning),
                        })
                    break
                offset = upper
        return tuple(contexts)

    def choose(
        self,
        *,
        admissible: Sequence[str],
        actor: Callable[[Sequence[QualifiedBindingCandidate], Sequence[str], str], CandidateActionProposal],
    ) -> ConsensusDecision:
        active = self._active_candidates()
        if not active:
            reason = (
                "ALL_CANDIDATE_PROGRAMS_FINISHED"
                if self._status and all(
                    value == CandidateRuntimeStatus.FINISHED
                    for value in self._status.values()
                )
                else "NO_ACTIVE_CANDIDATES"
            )
            return ConsensusDecision(None, "ABSTAIN", reason, ())
        allowed_by_candidate: list[set[str]] = []
        for qualified in active:
            index = self._cursor[qualified.candidate_hash]
            flattened = [
                step for node in qualified.candidate.nodes for step in node.target_steps
            ]
            if index >= len(flattened):
                raise RuntimeError("ACTIVE candidate cursor is past its program")
            node = flattened[index]
            allowed = []
            for command in admissible:
                try:
                    parsed = parse_alfworld_action(command, admissible=admissible)
                except ValueError:
                    continue
                if (
                    parsed.operator == node.target_operator
                    and dict(parsed.argument_types) == dict(node.argument_types)
                ):
                    allowed.append(command)
            allowed_by_candidate.append(set(allowed))
        common = set.intersection(*allowed_by_candidate) if allowed_by_candidate else set()
        common_ordered = tuple(command for command in admissible if command in common)
        if not common_ordered:
            return ConsensusDecision(None, "ABSTAIN", "NO_COMMON_EXACT_COMMAND", ())
        scope_payload = {
            "artifact_hash": self.artifact.artifact_hash,
            "candidate_hashes": [item.candidate_hash for item in active],
            "cursors": self._cursor,
            "common_actions": common_ordered,
        }
        scope_hash = hashlib.sha256(json.dumps(
            scope_payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        proposal = actor(active, common_ordered, scope_hash)
        proposals = (proposal,)
        if proposal.proposal_scope_hash != scope_hash:
            return ConsensusDecision(None, "ABSTAIN", "PROPOSAL_SCOPE_MISMATCH", proposals)
        if proposal.endpoint_error:
            return ConsensusDecision(None, "ERROR", "ACTOR_ENDPOINT_ERROR", proposals)
        if proposal.abstain or proposal.command is None:
            return ConsensusDecision(None, "ABSTAIN", "ACTOR_ABSTAINED", proposals)
        if proposal.command not in common_ordered:
            return ConsensusDecision(None, "ABSTAIN", "INVALID_COMMON_COMMAND", proposals)
        return ConsensusDecision(proposal.command, "EXECUTE", None, proposals)

    def observe_executed(self, decision: ConsensusDecision, *, executed_command: str) -> None:
        """Unconditionally advance active candidates.

        This method is for target-native or Harness-off treatments.  Source
        Harness treatments must call :meth:`observe_evidence_contract` after
        a predeclared contract has been mechanically checked.
        """
        if decision.status != "EXECUTE" or decision.command != executed_command:
            raise ValueError("cursor can advance only after the exact consensus command executes")
        self._advance(self._active_candidates())

    def _advance(self, candidates: Sequence[QualifiedBindingCandidate]) -> None:
        for qualified in candidates:
            key = qualified.candidate_hash
            self._cursor[key] += 1
            total = sum(len(node.target_steps) for node in qualified.candidate.nodes)
            if self._cursor[key] >= total:
                self._status[key] = CandidateRuntimeStatus.FINISHED

    def observe_evidence_contract(
        self,
        decision: ConsensusDecision,
        *,
        executed_command: str,
        candidate_results: Mapping[str, bool],
        verification_receipt_sha256: str,
    ) -> None:
        """Independently commit/refute active candidates after frozen contracts."""
        if decision.status != "EXECUTE" or decision.command != executed_command:
            raise ValueError("contract result must match the exact executed consensus command")
        self._apply_candidate_results(
            candidate_results=candidate_results,
            verification_receipt_sha256=verification_receipt_sha256,
        )

    def _apply_candidate_results(
        self,
        *,
        candidate_results: Mapping[str, bool],
        verification_receipt_sha256: str,
    ) -> None:
        if len(verification_receipt_sha256) != 64:
            raise ValueError("verification receipt must be a sha256 identity")
        active = self._active_candidates()
        active_by_hash = {item.candidate_hash: item for item in active}
        if set(candidate_results) != set(active_by_hash):
            raise ValueError("contract results must cover every active candidate exactly")
        self._advance(tuple(
            active_by_hash[key] for key, satisfied in candidate_results.items()
            if bool(satisfied)
        ))
        for key, satisfied in candidate_results.items():
            if not satisfied:
                self._status[key] = CandidateRuntimeStatus.REFUTED
                self._status_receipt[key] = verification_receipt_sha256

    def observe_admitted_rebind_executed(
        self,
        *,
        binding_receipt_sha256: str,
        known_binding_receipt_sha256s: Sequence[str],
        covered_candidate_hashes: Sequence[str],
        common_actions: Sequence[str],
        executed_command: str,
        candidate_results: Mapping[str, bool],
        verification_receipt_sha256: str,
    ) -> None:
        """Advance after an externally admitted one-step online binding.

        This verifies only registered identity, complete candidate coverage and
        exact native action membership.  It does not turn the binding into a
        semantic or positive-transfer claim.
        """
        if binding_receipt_sha256 not in set(known_binding_receipt_sha256s):
            raise ValueError("online binding receipt is not registered")
        active = self._active_candidates()
        active_hashes = {item.candidate_hash for item in active}
        if set(covered_candidate_hashes) != active_hashes:
            raise ValueError("online binding does not cover the active candidate set")
        if len(covered_candidate_hashes) != len(set(covered_candidate_hashes)):
            raise ValueError("online binding candidate coverage contains duplicates")
        if executed_command not in common_actions:
            raise ValueError("executed command is outside admitted rebind common actions")
        self._apply_candidate_results(
            candidate_results=candidate_results,
            verification_receipt_sha256=verification_receipt_sha256,
        )

    def observe_external_command_if_compatible(
        self, *, admissible: Sequence[str], executed_command: str,
    ) -> bool:
        """Shadow-advance only when every candidate allows the exact command.

        This supports a fail-closed treatment switch.  It never infers cursor
        alignment from step count after an action outside the fallback
        artifact's current native signature.
        """
        if executed_command not in admissible:
            return False
        try:
            parsed = parse_alfworld_action(executed_command, admissible=admissible)
        except ValueError:
            return False
        active = self._active_candidates()
        for qualified in active:
            index = self._cursor[qualified.candidate_hash]
            flattened = [
                step for node in qualified.candidate.nodes for step in node.target_steps
            ]
            if index >= len(flattened):
                return False
            expected = flattened[index]
            if (
                parsed.operator != expected.target_operator
                or dict(parsed.argument_types) != dict(expected.argument_types)
            ):
                return False
        self._advance(active)
        return True


__all__ = [
    "CandidateActionProposal",
    "CandidateRuntimeStatus",
    "ConsensusDecision",
    "FrozenCandidateSetRuntime",
]
