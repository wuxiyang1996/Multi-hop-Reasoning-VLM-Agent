"""Target-native multi-step admission without cross-domain semantic predicates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Sequence
from pathlib import Path

from harness.skill_admission import TargetActionEvidence, TargetDemoReceipt


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ProgramOrigin(str, Enum):
    SOURCE_HYPOTHESIS = "SOURCE_HYPOTHESIS"
    TARGET_NATIVE_SAME_DEMO = "TARGET_NATIVE_SAME_DEMO"


_EVIDENCE_QUERY_ORDER = (
    "COMMAND_WAS_ADMISSIBLE",
    "OBSERVATION_CHANGED",
    "ADMISSIBLE_SET_CHANGED",
    "EXECUTED_ACTION_DISAPPEARED",
    "POSITIVE_NATIVE_REWARD",
    "OFFICIAL_SUCCESS",
)


@dataclass(frozen=True)
class DemoTransitionContractReceipt:
    target_transition_index: int
    target_operator: str
    argument_types: Mapping[str, str]
    supported_evidence: Sequence[str]
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("demo transition contract receipt hash mismatch")


def build_demo_transition_contract_receipt(
    action: TargetActionEvidence,
) -> DemoTransitionContractReceipt:
    supported = {
        "COMMAND_WAS_ADMISSIBLE": action.action in action.before_admissible_actions,
        "OBSERVATION_CHANGED": action.state_sha256 != action.next_state_sha256,
        "ADMISSIBLE_SET_CHANGED": (
            action.admissible_actions_sha256 != action.next_admissible_actions_sha256
        ),
        "EXECUTED_ACTION_DISAPPEARED": action.action not in action.after_admissible_actions,
        "POSITIVE_NATIVE_REWARD": action.reward > 0.0,
        "OFFICIAL_SUCCESS": action.official_success_after,
    }
    unsigned = {
        "target_transition_index": int(action.transition_index),
        "target_operator": str(action.operator),
        "argument_types": dict(action.argument_types),
        "supported_evidence": [
            kind for kind in _EVIDENCE_QUERY_ORDER if supported[kind]
        ],
    }
    receipt = DemoTransitionContractReceipt(
        target_transition_index=unsigned["target_transition_index"],
        target_operator=unsigned["target_operator"],
        argument_types=unsigned["argument_types"],
        supported_evidence=tuple(unsigned["supported_evidence"]),
        receipt_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return receipt


@dataclass(frozen=True)
class TargetStepBinding:
    target_transition_index: int
    target_operator: str
    argument_types: Mapping[str, str]


@dataclass(frozen=True)
class TargetNodeBinding:
    node_id: str
    target_steps: Sequence[TargetStepBinding]
    # Exact source-side receipts attached mechanically by node identity. The
    # runtime may show these to the Actor, but Harness admission never assigns
    # them target semantics.
    source_conditioning: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiStepBindingCandidate:
    candidate_id: str
    origin: ProgramOrigin
    proposal_source: str
    proposal_receipt_sha256: str
    nodes: Sequence[TargetNodeBinding]
    source_hypothesis_hash: str | None = None

    def content_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class QualifiedBindingCandidate:
    candidate: MultiStepBindingCandidate
    candidate_hash: str
    checks: Mapping[str, bool]


@dataclass(frozen=True)
class MultiStepAdmissionArtifact:
    target_domain: str
    task_family: str
    demo_id: str
    demo_hash: str
    candidates: Sequence[QualifiedBindingCandidate]
    rejected_candidates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    demo_transition_contract_receipts: Sequence[DemoTransitionContractReceipt] = field(
        default_factory=tuple
    )
    source_treatment: str | None = None
    source_control_receipt_sha256: str | None = None
    schema_version: int = 3
    node_binding_version: int = 3
    semantic_alignment_claimed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload["source_treatment"] is None:
            payload.pop("source_treatment")
        if payload["source_control_receipt_sha256"] is None:
            payload.pop("source_control_receipt_sha256")
        if not payload["demo_transition_contract_receipts"]:
            # Preserve exact hashes for legacy v3 artifacts. They remain
            # loadable but cannot enter receipt-grounded source control.
            payload.pop("demo_transition_contract_receipts")
        for row in payload["candidates"]:
            row["candidate"]["origin"] = row["candidate"]["origin"].value
        payload["artifact_hash"] = _hash(payload)
        return payload

    @property
    def artifact_hash(self) -> str:
        return self.to_dict()["artifact_hash"]


class MultiStepTargetAdmission:
    """Qualify exact node→demo-transition proposals and retain their set."""

    def admit(
        self,
        *,
        candidates: Sequence[MultiStepBindingCandidate],
        demo: TargetDemoReceipt,
        known_proposal_receipt_hashes: Sequence[str],
        known_source_hypothesis_nodes: Mapping[str, Sequence[str]] | None = None,
        known_source_node_conditioning: (
            Mapping[str, Mapping[str, Mapping[str, Any]]] | None
        ) = None,
        source_treatment: str | None = None,
        source_control_receipt_sha256: str | None = None,
    ) -> MultiStepAdmissionArtifact:
        demo.validate_for_admission()
        known_proposals = set(known_proposal_receipt_hashes)
        known_source_hypotheses = {
            str(key): tuple(str(item) for item in value)
            for key, value in (known_source_hypothesis_nodes or {}).items()
        }
        known_conditioning = {
            str(hypothesis_hash): {
                str(node_id): dict(payload)
                for node_id, payload in nodes.items()
            }
            for hypothesis_hash, nodes in (known_source_node_conditioning or {}).items()
        }
        qualified: list[QualifiedBindingCandidate] = []
        rejected: list[Mapping[str, Any]] = []
        for candidate in candidates:
            node_ids = [node.node_id for node in candidate.nodes]
            target_steps = [step for node in candidate.nodes for step in node.target_steps]
            indices = [step.target_transition_index for step in target_steps]
            expected_source_nodes = known_source_hypotheses.get(
                candidate.source_hypothesis_hash or ""
            )
            checks = {
                "candidate_id": bool(candidate.candidate_id),
                "proposal_receipt_known": (
                    _valid_digest(candidate.proposal_receipt_sha256)
                    and candidate.proposal_receipt_sha256 in known_proposals
                ),
                "source_identity_consistent": (
                    candidate.origin == ProgramOrigin.TARGET_NATIVE_SAME_DEMO
                    and candidate.source_hypothesis_hash is None
                ) or (
                    candidate.origin == ProgramOrigin.SOURCE_HYPOTHESIS
                    and _valid_digest(candidate.source_hypothesis_hash or "")
                    and candidate.source_hypothesis_hash in known_source_hypotheses
                ),
                "source_nodes_exact": (
                    candidate.origin == ProgramOrigin.TARGET_NATIVE_SAME_DEMO
                    or tuple(node_ids) == expected_source_nodes
                ),
                "source_conditioning_exact": (
                    candidate.origin == ProgramOrigin.TARGET_NATIVE_SAME_DEMO
                    and all(not node.source_conditioning for node in candidate.nodes)
                ) or (
                    candidate.origin == ProgramOrigin.SOURCE_HYPOTHESIS
                    and candidate.source_hypothesis_hash in known_conditioning
                    and all(
                        dict(node.source_conditioning)
                        == known_conditioning[candidate.source_hypothesis_hash].get(node.node_id)
                        for node in candidate.nodes
                    )
                ),
                "at_least_two_nodes": len(candidate.nodes) >= 2,
                "node_ids_unique": len(node_ids) == len(set(node_ids)),
                "each_node_has_target_span": all(node.target_steps for node in candidate.nodes),
                "target_indices_observed": all(0 <= index < len(demo.actions) for index in indices),
                # One-shot observation proves only this linear order.  It does
                # not prove a branch, loop, retry, or guard.
                "target_indices_strictly_increasing": all(
                    right > left for left, right in zip(indices, indices[1:])
                ),
                "target_indices_partition_full_demo": indices == list(range(len(demo.actions))),
                "native_signatures_exact": all(
                    0 <= step.target_transition_index < len(demo.actions)
                    and step.target_operator
                    == demo.actions[step.target_transition_index].operator
                    and dict(step.argument_types)
                    == dict(demo.actions[step.target_transition_index].argument_types)
                    for step in target_steps
                ),
            }
            if all(checks.values()):
                qualified.append(QualifiedBindingCandidate(
                    candidate=candidate,
                    candidate_hash=candidate.content_hash(),
                    checks=checks,
                ))
            else:
                rejected.append({
                    "candidate_id": candidate.candidate_id,
                    "failure_codes": [key.upper() for key, value in checks.items() if not value],
                })
        # Set union + content-hash dedup only.  Never select a winner.
        by_hash = {item.candidate_hash: item for item in qualified}
        return MultiStepAdmissionArtifact(
            target_domain=demo.target_domain,
            task_family=demo.task_family,
            demo_id=demo.demo_id,
            demo_hash=demo.content_hash(),
            candidates=tuple(by_hash[key] for key in sorted(by_hash)),
            rejected_candidates=tuple(rejected),
            demo_transition_contract_receipts=tuple(
                build_demo_transition_contract_receipt(action)
                for action in demo.actions
            ),
            source_treatment=source_treatment,
            source_control_receipt_sha256=source_control_receipt_sha256,
        )


class FrozenMultiStepArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def freeze(self, artifact: MultiStepAdmissionArtifact) -> Path:
        payload = artifact.to_dict()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{artifact.artifact_hash}.json"
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise RuntimeError("immutable v3 artifact collision")
            return path
        fd, temporary = tempfile.mkstemp(prefix=".multistep.", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return path


def multistep_candidate_from_dict(payload: Mapping[str, Any]) -> MultiStepBindingCandidate:
    return MultiStepBindingCandidate(
        candidate_id=str(payload["candidate_id"]),
        origin=ProgramOrigin(str(payload["origin"])),
        proposal_source=str(payload["proposal_source"]),
        proposal_receipt_sha256=str(payload["proposal_receipt_sha256"]),
        source_hypothesis_hash=(
            str(payload["source_hypothesis_hash"])
            if payload.get("source_hypothesis_hash") is not None else None
        ),
        nodes=tuple(TargetNodeBinding(
            node_id=str(item["node_id"]),
            target_steps=tuple(TargetStepBinding(
                target_transition_index=int(step["target_transition_index"]),
                target_operator=str(step["target_operator"]),
                argument_types=dict(step.get("argument_types") or {}),
            ) for step in item.get("target_steps") or ()),
            source_conditioning=dict(item.get("source_conditioning") or {}),
        ) for item in payload.get("nodes") or ()),
    )


def multistep_artifact_from_dict(payload: Mapping[str, Any]) -> MultiStepAdmissionArtifact:
    claimed_hash = str(payload.get("artifact_hash") or "")
    candidates = []
    for row in payload.get("candidates") or ():
        candidate = multistep_candidate_from_dict(row["candidate"])
        qualified = QualifiedBindingCandidate(
            candidate=candidate,
            candidate_hash=str(row["candidate_hash"]),
            checks=dict(row.get("checks") or {}),
        )
        if qualified.candidate_hash != candidate.content_hash() or not all(qualified.checks.values()):
            raise ValueError("invalid qualified candidate in frozen artifact")
        candidates.append(qualified)
    artifact = MultiStepAdmissionArtifact(
        target_domain=str(payload["target_domain"]),
        task_family=str(payload["task_family"]),
        demo_id=str(payload["demo_id"]),
        demo_hash=str(payload["demo_hash"]),
        candidates=tuple(candidates),
        rejected_candidates=tuple(payload.get("rejected_candidates") or ()),
        demo_transition_contract_receipts=tuple(
            DemoTransitionContractReceipt(
                target_transition_index=int(row["target_transition_index"]),
                target_operator=str(row["target_operator"]),
                argument_types=dict(row.get("argument_types") or {}),
                supported_evidence=tuple(str(item) for item in row.get("supported_evidence") or ()),
                receipt_sha256=str(row["receipt_sha256"]),
            )
            for row in payload.get("demo_transition_contract_receipts") or ()
        ),
        source_treatment=(
            str(payload["source_treatment"])
            if payload.get("source_treatment") is not None else None
        ),
        source_control_receipt_sha256=(
            str(payload["source_control_receipt_sha256"])
            if payload.get("source_control_receipt_sha256") is not None else None
        ),
        schema_version=int(payload.get("schema_version", 0)),
        node_binding_version=int(payload.get("node_binding_version", 0)),
        semantic_alignment_claimed=bool(payload.get("semantic_alignment_claimed", False)),
    )
    if (
        artifact.schema_version != 3
        or artifact.node_binding_version != 3
        or artifact.semantic_alignment_claimed
    ):
        raise ValueError("unsupported or semantically widened v3 artifact")
    if artifact.artifact_hash != claimed_hash:
        raise ValueError("v3 artifact hash mismatch")
    for receipt in artifact.demo_transition_contract_receipts:
        receipt.validate_hash()
    indices = [
        receipt.target_transition_index
        for receipt in artifact.demo_transition_contract_receipts
    ]
    if indices and indices != list(range(len(indices))):
        raise ValueError("demo transition contract receipts are not contiguous")
    receipts_by_index = {
        receipt.target_transition_index: receipt
        for receipt in artifact.demo_transition_contract_receipts
    }
    for receipt in artifact.demo_transition_contract_receipts:
        if (
            list(receipt.supported_evidence)
            != [
                kind for kind in _EVIDENCE_QUERY_ORDER
                if kind in set(receipt.supported_evidence)
            ]
            or len(receipt.supported_evidence) != len(set(receipt.supported_evidence))
        ):
            raise ValueError("demo transition supported evidence is not canonical")
    if receipts_by_index and any(
        (
            step.target_transition_index not in receipts_by_index
            or receipts_by_index[step.target_transition_index].target_operator
            != step.target_operator
            or dict(receipts_by_index[step.target_transition_index].argument_types)
            != dict(step.argument_types)
        )
        for candidate in artifact.candidates
        for node in candidate.candidate.nodes
        for step in node.target_steps
    ):
        raise ValueError("candidate target step is not anchored to its demo receipt")
    return artifact


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


__all__ = [
    "MultiStepAdmissionArtifact",
    "DemoTransitionContractReceipt",
    "MultiStepBindingCandidate",
    "MultiStepTargetAdmission",
    "FrozenMultiStepArtifactStore",
    "ProgramOrigin",
    "QualifiedBindingCandidate",
    "TargetNodeBinding",
    "TargetStepBinding",
    "build_demo_transition_contract_receipt",
    "multistep_artifact_from_dict",
    "multistep_candidate_from_dict",
]
