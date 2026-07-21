"""Content-addressed multi-example version spaces for target bindings.

No semantic similarity, clustering, voting, or reward-based source selection is
performed here.  A version is an exact source-node -> target-native schema that
an untrusted Agent proposed and the admission Harness qualified independently
on one or more frozen adaptation examples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence

from harness.multistep_binding import (
    MultiStepAdmissionArtifact,
    QualifiedBindingCandidate,
)
from harness.online_transfer_runtime import NativeTransitionEvidence


_EVIDENCE_QUERY_ORDER = (
    "COMMAND_WAS_ADMISSIBLE",
    "OBSERVATION_CHANGED",
    "ADMISSIBLE_SET_CHANGED",
    "EXECUTED_ACTION_DISAPPEARED",
    "POSITIVE_NATIVE_REWARD",
    "OFFICIAL_SUCCESS",
)


def _hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class VersionSpaceStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    READY = "READY"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET = (
        "NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET"
    )


class VersionTransitionVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    NOT_APPLICABLE_NOW = "NOT_APPLICABLE_NOW"
    NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET = (
        "NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET"
    )


@dataclass(frozen=True)
class AdaptationExampleReceipt:
    example_index: int
    demo_id: str
    demo_hash: str
    admission_artifact_hash: str
    proposed_schema_hashes: Sequence[str]
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("adaptation example receipt hash mismatch")


@dataclass(frozen=True)
class VersionStepEvidence:
    step_index: int
    demo_hash: str
    demo_transition_receipt_sha256: str
    supported_evidence: Sequence[str]


@dataclass(frozen=True)
class BindingSchemaVersion:
    schema_hash: str
    schema: Mapping[str, Any]
    supporting_example_indices: Sequence[int]
    witness_candidate_hashes: Sequence[str]
    step_evidence: Sequence[VersionStepEvidence]


@dataclass(frozen=True)
class ReceiptVersionSpaceArtifact:
    adaptation_set_id: str
    target_domain: str
    task_family: str
    expected_example_count: int
    examples: Sequence[AdaptationExampleReceipt]
    versions: Sequence[BindingSchemaVersion]
    viable_schema_hashes: Sequence[str]
    status: VersionSpaceStatus
    source_treatment: str | None
    schema_version: int = 1

    def unsigned_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    def to_dict(self) -> Dict[str, Any]:
        payload = self.unsigned_payload()
        payload["artifact_hash"] = _hash(payload)
        return payload

    @property
    def artifact_hash(self) -> str:
        return self.to_dict()["artifact_hash"]


def candidate_binding_schema(
    candidate: QualifiedBindingCandidate,
) -> Mapping[str, Any]:
    value = candidate.candidate
    return {
        "origin": value.origin.value,
        "source_hypothesis_hash": value.source_hypothesis_hash,
        "nodes": [{
            "node_id": node.node_id,
            "target_steps": [{
                "target_operator": step.target_operator,
                "argument_types": dict(step.argument_types),
            } for step in node.target_steps],
        } for node in value.nodes],
    }


def candidate_binding_schema_hash(
    candidate: QualifiedBindingCandidate,
) -> str:
    return _hash(candidate_binding_schema(candidate))


def _build_example_receipt(
    *, index: int, artifact: MultiStepAdmissionArtifact,
    schema_hashes: Sequence[str],
) -> AdaptationExampleReceipt:
    unsigned = {
        "example_index": int(index),
        "demo_id": artifact.demo_id,
        "demo_hash": artifact.demo_hash,
        "admission_artifact_hash": artifact.artifact_hash,
        "proposed_schema_hashes": sorted(set(schema_hashes)),
    }
    receipt = AdaptationExampleReceipt(
        **unsigned, receipt_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return receipt


def build_receipt_version_space(
    *,
    adaptation_set_id: str,
    artifacts: Sequence[MultiStepAdmissionArtifact],
    expected_example_count: int,
) -> ReceiptVersionSpaceArtifact:
    """Build a bounded exact-schema version space from frozen examples.

    The viable set is the literal intersection of Agent-proposed, Harness-
    qualified schema identities across the examples seen so far.  Therefore a
    final empty set means only "not recovered within this registered adaptation
    protocol", never that the source skill is globally impossible.
    """
    if not adaptation_set_id:
        raise ValueError("adaptation_set_id is required")
    if expected_example_count < 1:
        raise ValueError("expected_example_count must be positive")
    if not artifacts or len(artifacts) > expected_example_count:
        raise ValueError("artifact count is outside the registered adaptation set")
    domains = {artifact.target_domain for artifact in artifacts}
    families = {artifact.task_family for artifact in artifacts}
    treatments = {artifact.source_treatment for artifact in artifacts}
    demo_hashes = [artifact.demo_hash for artifact in artifacts]
    if len(domains) != 1 or len(families) != 1 or len(treatments) != 1:
        raise ValueError("adaptation artifacts do not share protocol identity")
    if len(demo_hashes) != len(set(demo_hashes)):
        raise ValueError("adaptation examples must have distinct demo hashes")
    if any(
        artifact.schema_version != 3
        or artifact.node_binding_version != 3
        or not artifact.demo_transition_contract_receipts
        for artifact in artifacts
    ):
        raise ValueError("version space requires receipt-grounded v3 artifacts")

    schemas: dict[str, Mapping[str, Any]] = {}
    supporting_indices: dict[str, set[int]] = {}
    candidate_hashes: dict[str, set[str]] = {}
    evidence: dict[str, list[VersionStepEvidence]] = {}
    examples = []
    per_example_schema_sets = []
    for example_index, artifact in enumerate(artifacts):
        receipts = {
            item.target_transition_index: item
            for item in artifact.demo_transition_contract_receipts
        }
        example_schema_hashes = []
        for candidate in artifact.candidates:
            schema = candidate_binding_schema(candidate)
            schema_hash = candidate_binding_schema_hash(candidate)
            schemas.setdefault(schema_hash, schema)
            if schemas[schema_hash] != schema:
                raise RuntimeError("binding schema hash collision")
            supporting_indices.setdefault(schema_hash, set()).add(example_index)
            candidate_hashes.setdefault(schema_hash, set()).add(
                candidate.candidate_hash
            )
            example_schema_hashes.append(schema_hash)
            flattened = [
                step
                for node in candidate.candidate.nodes
                for step in node.target_steps
            ]
            for step_index, step in enumerate(flattened):
                receipt = receipts[step.target_transition_index]
                row = VersionStepEvidence(
                    step_index=step_index,
                    demo_hash=artifact.demo_hash,
                    demo_transition_receipt_sha256=receipt.receipt_sha256,
                    supported_evidence=tuple(receipt.supported_evidence),
                )
                rows = evidence.setdefault(schema_hash, [])
                if row not in rows:
                    rows.append(row)
        example_set = set(example_schema_hashes)
        per_example_schema_sets.append(example_set)
        examples.append(_build_example_receipt(
            index=example_index,
            artifact=artifact,
            schema_hashes=tuple(example_set),
        ))

    viable = set.intersection(*per_example_schema_sets)
    complete = len(artifacts) == expected_example_count
    if viable:
        status = VersionSpaceStatus.READY if complete else VersionSpaceStatus.PROVISIONAL
    else:
        status = (
            VersionSpaceStatus.NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET
            if complete else VersionSpaceStatus.NEED_MORE_EVIDENCE
        )
    versions = tuple(BindingSchemaVersion(
        schema_hash=schema_hash,
        schema=schemas[schema_hash],
        supporting_example_indices=tuple(sorted(supporting_indices[schema_hash])),
        witness_candidate_hashes=tuple(sorted(candidate_hashes[schema_hash])),
        step_evidence=tuple(sorted(
            evidence[schema_hash],
            key=lambda row: (
                row.step_index, row.demo_hash,
                row.demo_transition_receipt_sha256,
            ),
        )),
    ) for schema_hash in sorted(schemas))
    artifact = ReceiptVersionSpaceArtifact(
        adaptation_set_id=adaptation_set_id,
        target_domain=next(iter(domains)),
        task_family=next(iter(families)),
        expected_example_count=int(expected_example_count),
        examples=tuple(examples),
        versions=versions,
        viable_schema_hashes=tuple(sorted(viable)),
        status=status,
        source_treatment=next(iter(treatments)),
    )
    receipt_version_space_from_dict(artifact.to_dict())
    return artifact


def _canonical_supported_evidence(values: Sequence[str]) -> bool:
    return (
        list(values)
        == [kind for kind in _EVIDENCE_QUERY_ORDER if kind in set(values)]
        and len(values) == len(set(values))
    )


def receipt_version_space_from_dict(
    payload: Mapping[str, Any],
) -> ReceiptVersionSpaceArtifact:
    examples = tuple(AdaptationExampleReceipt(
        example_index=int(row["example_index"]),
        demo_id=str(row["demo_id"]),
        demo_hash=str(row["demo_hash"]),
        admission_artifact_hash=str(row["admission_artifact_hash"]),
        proposed_schema_hashes=tuple(str(item) for item in row["proposed_schema_hashes"]),
        receipt_sha256=str(row["receipt_sha256"]),
    ) for row in payload.get("examples") or ())
    versions = tuple(BindingSchemaVersion(
        schema_hash=str(row["schema_hash"]),
        schema=dict(row["schema"]),
        supporting_example_indices=tuple(
            int(item) for item in row["supporting_example_indices"]
        ),
        witness_candidate_hashes=tuple(
            str(item) for item in row["witness_candidate_hashes"]
        ),
        step_evidence=tuple(VersionStepEvidence(
            step_index=int(item["step_index"]),
            demo_hash=str(item["demo_hash"]),
            demo_transition_receipt_sha256=str(
                item["demo_transition_receipt_sha256"]
            ),
            supported_evidence=tuple(
                str(value) for value in item["supported_evidence"]
            ),
        ) for item in row["step_evidence"]),
    ) for row in payload.get("versions") or ())
    artifact = ReceiptVersionSpaceArtifact(
        adaptation_set_id=str(payload["adaptation_set_id"]),
        target_domain=str(payload["target_domain"]),
        task_family=str(payload["task_family"]),
        expected_example_count=int(payload["expected_example_count"]),
        examples=examples,
        versions=versions,
        viable_schema_hashes=tuple(
            str(item) for item in payload.get("viable_schema_hashes") or ()
        ),
        status=VersionSpaceStatus(str(payload["status"])),
        source_treatment=(
            str(payload["source_treatment"])
            if payload.get("source_treatment") is not None else None
        ),
        schema_version=int(payload.get("schema_version", 0)),
    )
    claimed_hash = str(payload.get("artifact_hash") or "")
    if artifact.schema_version != 1 or artifact.artifact_hash != claimed_hash:
        raise ValueError("receipt version-space artifact hash mismatch")
    if (
        not artifact.adaptation_set_id
        or artifact.expected_example_count < 1
        or not artifact.examples
        or len(artifact.examples) > artifact.expected_example_count
    ):
        raise ValueError("invalid registered adaptation set")
    if [row.example_index for row in artifact.examples] != list(range(len(examples))):
        raise ValueError("adaptation example indices are not contiguous")
    if len({row.demo_hash for row in examples}) != len(examples):
        raise ValueError("adaptation demo hashes are not unique")
    for row in examples:
        row.validate_hash()
        if (
            not _valid_sha256(row.demo_hash)
            or not _valid_sha256(row.admission_artifact_hash)
            or list(row.proposed_schema_hashes) != sorted(set(row.proposed_schema_hashes))
        ):
            raise ValueError("invalid adaptation example receipt")
    by_hash = {row.schema_hash: row for row in versions}
    registered_schema_hashes = set().union(*(
        set(row.proposed_schema_hashes) for row in examples
    ))
    if (
        len(by_hash) != len(versions)
        or set(artifact.viable_schema_hashes) - set(by_hash)
        or set(by_hash) != registered_schema_hashes
        or list(artifact.viable_schema_hashes)
        != sorted(set(artifact.viable_schema_hashes))
    ):
        raise ValueError("invalid version schema registry")
    for version in versions:
        if version.schema_hash != _hash(version.schema):
            raise ValueError("binding schema hash mismatch")
        expected_support = tuple(
            row.example_index for row in examples
            if version.schema_hash in row.proposed_schema_hashes
        )
        if version.supporting_example_indices != expected_support:
            raise ValueError("version supporting examples are inconsistent")
        if (
            not version.witness_candidate_hashes
            or list(version.witness_candidate_hashes)
            != sorted(set(version.witness_candidate_hashes))
            or any(not _valid_sha256(item) for item in version.witness_candidate_hashes)
        ):
            raise ValueError("invalid version witness registry")
        if set(version.schema) != {
            "origin", "source_hypothesis_hash", "nodes",
        }:
            raise ValueError("invalid binding schema shape")
        flattened_steps = [
            step for node in version.schema["nodes"]
            for step in node["target_steps"]
        ]
        supporting_demo_hashes = {
            examples[index].demo_hash for index in expected_support
        }
        if any(
            not _valid_sha256(item.demo_transition_receipt_sha256)
            or not _canonical_supported_evidence(item.supported_evidence)
            or item.step_index < 0
            or item.step_index >= len(flattened_steps)
            or item.demo_hash not in supporting_demo_hashes
            for item in version.step_evidence
        ):
            raise ValueError("invalid version step evidence")
        if any(
            {
                item.demo_hash for item in version.step_evidence
                if item.step_index == step_index
            } != supporting_demo_hashes
            for step_index in range(len(flattened_steps))
        ):
            raise ValueError("version step evidence does not cover its witnesses")
    expected_viable = set.intersection(*(
        set(row.proposed_schema_hashes) for row in examples
    ))
    if set(artifact.viable_schema_hashes) != expected_viable:
        raise ValueError("viable schema set is not the exact example intersection")
    complete = len(examples) == artifact.expected_example_count
    expected_status = (
        VersionSpaceStatus.READY if expected_viable and complete
        else VersionSpaceStatus.PROVISIONAL if expected_viable
        else VersionSpaceStatus.NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET
        if complete else VersionSpaceStatus.NEED_MORE_EVIDENCE
    )
    if artifact.status != expected_status:
        raise ValueError("version-space status is inconsistent")
    return artifact


def transition_supported_evidence(
    transition: NativeTransitionEvidence,
) -> tuple[str, ...]:
    transition.validate_hash()
    supported = {
        "COMMAND_WAS_ADMISSIBLE": transition.command_was_admissible,
        "OBSERVATION_CHANGED": (
            transition.before_observation_sha256
            != transition.after_observation_sha256
        ),
        "ADMISSIBLE_SET_CHANGED": (
            transition.before_actions_sha256 != transition.after_actions_sha256
        ),
        "EXECUTED_ACTION_DISAPPEARED": (
            not transition.executed_action_admissible_after
        ),
        "POSITIVE_NATIVE_REWARD": transition.reward > 0.0,
        "OFFICIAL_SUCCESS": transition.official_success,
    }
    return tuple(kind for kind in _EVIDENCE_QUERY_ORDER if supported[kind])


@dataclass(frozen=True)
class VersionTransitionResult:
    schema_hash: str
    schema_matches: bool
    signature_observed: bool


@dataclass(frozen=True)
class VersionSpaceTransitionReceipt:
    version_space_artifact_hash: str
    transition_receipt_sha256: str
    cursor_before: int
    observed_operator: str
    observed_argument_types: Mapping[str, str]
    observed_supported_evidence: Sequence[str]
    version_results: Sequence[VersionTransitionResult]
    verdict: VersionTransitionVerdict
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("version transition receipt hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_hash()
        payload = dict(self.unsigned_payload())
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


class ReceiptVersionSpaceRuntime:
    """Episode-local selective execution over a frozen version space."""

    def __init__(self, artifact: ReceiptVersionSpaceArtifact) -> None:
        receipt_version_space_from_dict(artifact.to_dict())
        self.artifact = artifact
        self.cursor = 0
        self.paused_for_evidence = artifact.status in {
            VersionSpaceStatus.NEED_MORE_EVIDENCE,
            VersionSpaceStatus.NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET,
        }

    def current_schemas(self) -> Mapping[str, Mapping[str, Any]]:
        by_hash = {row.schema_hash: row for row in self.artifact.versions}
        out = {}
        for schema_hash in self.artifact.viable_schema_hashes:
            steps = [
                step
                for node in by_hash[schema_hash].schema["nodes"]
                for step in node["target_steps"]
            ]
            if self.cursor < len(steps):
                out[schema_hash] = dict(steps[self.cursor])
        return out

    def observe_transition(
        self,
        *,
        transition: NativeTransitionEvidence,
        observed_operator: str,
        observed_argument_types: Mapping[str, str],
    ) -> VersionSpaceTransitionReceipt:
        transition.validate_hash()
        if self.artifact.status == (
            VersionSpaceStatus.NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET
        ):
            verdict = (
                VersionTransitionVerdict
                .NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET
            )
            results: tuple[VersionTransitionResult, ...] = ()
        else:
            observed_signature = transition_supported_evidence(transition)
            by_hash = {row.schema_hash: row for row in self.artifact.versions}
            results_list = []
            for schema_hash, schema_step in self.current_schemas().items():
                schema_matches = (
                    str(schema_step["target_operator"]) == str(observed_operator)
                    and dict(schema_step["argument_types"])
                    == dict(observed_argument_types)
                )
                known_signatures = {
                    tuple(item.supported_evidence)
                    for item in by_hash[schema_hash].step_evidence
                    if item.step_index == self.cursor
                }
                results_list.append(VersionTransitionResult(
                    schema_hash=schema_hash,
                    schema_matches=schema_matches,
                    signature_observed=(
                        schema_matches and observed_signature in known_signatures
                    ),
                ))
            results = tuple(results_list)
            if not results or not all(row.schema_matches for row in results):
                verdict = VersionTransitionVerdict.NOT_APPLICABLE_NOW
            elif all(row.signature_observed for row in results):
                verdict = VersionTransitionVerdict.SUPPORTED
                self.cursor += 1
            else:
                # An unseen effect is new evidence, not semantic refutation.
                verdict = VersionTransitionVerdict.NEED_MORE_EVIDENCE
                self.paused_for_evidence = True
        observed_signature = transition_supported_evidence(transition)
        unsigned = {
            "version_space_artifact_hash": self.artifact.artifact_hash,
            "transition_receipt_sha256": transition.receipt_sha256,
            "cursor_before": self.cursor - 1 if verdict == VersionTransitionVerdict.SUPPORTED else self.cursor,
            "observed_operator": str(observed_operator),
            "observed_argument_types": dict(observed_argument_types),
            "observed_supported_evidence": list(observed_signature),
            "version_results": [asdict(row) for row in results],
            "verdict": verdict.value,
        }
        receipt = VersionSpaceTransitionReceipt(
            version_space_artifact_hash=unsigned["version_space_artifact_hash"],
            transition_receipt_sha256=unsigned["transition_receipt_sha256"],
            cursor_before=unsigned["cursor_before"],
            observed_operator=unsigned["observed_operator"],
            observed_argument_types=unsigned["observed_argument_types"],
            observed_supported_evidence=tuple(unsigned["observed_supported_evidence"]),
            version_results=results,
            verdict=verdict,
            receipt_sha256=_hash(unsigned),
        )
        receipt.validate_hash()
        return receipt


__all__ = [
    "AdaptationExampleReceipt",
    "BindingSchemaVersion",
    "ReceiptVersionSpaceArtifact",
    "ReceiptVersionSpaceRuntime",
    "VersionSpaceStatus",
    "VersionSpaceTransitionReceipt",
    "VersionTransitionResult",
    "VersionTransitionVerdict",
    "build_receipt_version_space",
    "candidate_binding_schema",
    "candidate_binding_schema_hash",
    "receipt_version_space_from_dict",
    "transition_supported_evidence",
]
