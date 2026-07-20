"""Strict one-shot binding admission for evidence-qualified programs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from skill_bank.program_ir import CanonicalSkillProgram, ProgramStatus
from skill_bank.source_effects import (
    AGENT_LOCATION_CHANGED,
    MOVABLE_LOCATION_CHANGED,
    POSSESSION_ACQUIRED,
    RECEPTACLE_OPENED,
)


TARGET_OPERATOR_SCHEMAS: Dict[str, Dict[str, Mapping[str, str]]] = {
    "alfworld": {
        "LOOK": {},
        "INVENTORY": {},
        "GOTO": {"location": "location"},
        "OPEN": {"receptacle": "receptacle"},
        "CLOSE": {"receptacle": "receptacle"},
        "TAKE": {"object": "object", "receptacle": "receptacle"},
        "PUT": {"object": "object", "relation": "relation", "receptacle": "receptacle"},
        "MOVE_TO": {"object": "object", "receptacle": "receptacle"},
        "HEAT": {"object": "object", "tool": "tool"},
        "COOL": {"object": "object", "tool": "tool"},
        "CLEAN": {"object": "object", "tool": "tool"},
        "TOGGLE": {"object": "object"},
        "USE": {"object": "object"},
        "EXAMINE": {"object": "object"},
    }
}

TARGET_OPERATOR_EFFECTS: Dict[str, Dict[str, str]] = {
    "alfworld": {
        "GOTO": AGENT_LOCATION_CHANGED,
        "OPEN": RECEPTACLE_OPENED,
        "TAKE": POSSESSION_ACQUIRED,
        "MOVE_TO": MOVABLE_LOCATION_CHANGED,
    }
}


class AdmissionStatus(str, Enum):
    ADMITTED = "ADMITTED"
    CONDITIONAL = "CONDITIONAL"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class TargetActionEvidence:
    transition_index: int
    action: str
    operator: str
    arguments: Mapping[str, str]
    argument_types: Mapping[str, str]
    admissible_actions_sha256: str
    state_sha256: str
    next_state_sha256: str


@dataclass(frozen=True)
class TargetDemoReceipt:
    demo_id: str
    target_domain: str
    task_family: str
    split: str
    episode_id: str
    source_file_sha256: str
    executor_kind: str
    evaluator: str
    official_success: bool
    official_score: float
    actions: Sequence[TargetActionEvidence]
    held_out: bool = False

    def validate_for_admission(self) -> None:
        if self.held_out or self.split.lower() in {
            "test", "valid_seen", "valid_unseen", "eval_in_distribution",
            "eval_out_of_distribution",
        }:
            raise ValueError("held-out/test evidence cannot be used for admission")
        if self.executor_kind != "real":
            raise ValueError("admission requires a real target executor")
        if not self.evaluator or self.evaluator in {"llm", "model_self_report"}:
            raise ValueError("admission requires an official deterministic evaluator")
        if not self.actions:
            raise ValueError("target demo must contain executed action evidence")
        if not self.official_success:
            raise ValueError("the fixed admission demonstration did not succeed")
        digests = [self.source_file_sha256]
        for action in self.actions:
            digests.extend([
                action.admissible_actions_sha256,
                action.state_sha256,
                action.next_state_sha256,
            ])
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in digests
        ):
            raise ValueError("demo evidence contains an invalid sha256 digest")

    def content_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class BindingCandidate:
    candidate_id: str
    source_program_id: str
    source_program_hash: str
    source_step_id: str
    target_domain: str
    task_family: str
    target_operator: str
    argument_types: Mapping[str, str]
    source_effect: str
    proposal_source: str

    @property
    def executable_signature(self) -> str:
        slots = ",".join(f"{key}:{self.argument_types[key]}" for key in sorted(self.argument_types))
        return f"{self.source_step_id}[{self.source_effect}]->{self.target_operator}({slots})"


@dataclass(frozen=True)
class VerifiedScope:
    target_domain: str
    task_family: str
    operators: Sequence[str]
    argument_types: Mapping[str, str]
    source_steps: Sequence[str]
    effects: Sequence[str] = field(default_factory=tuple)
    required_conditions: Sequence[str] = field(default_factory=tuple)
    source_effect_evidence_count: int = 0
    source_evidence_count: int = 0


@dataclass
class AdmissionArtifact:
    status: AdmissionStatus
    source_program_id: str
    source_program_hash: str
    target_domain: str
    task_family: str
    demo_id: str
    demo_hash: str
    admitted_candidate_id: str | None
    candidate_ids: List[str]
    verified_scope: VerifiedScope | None
    proof_trace: List[Dict[str, Any]]
    failure_codes: List[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["artifact_hash"] = _stable_hash(payload)
        return payload

    @property
    def artifact_hash(self) -> str:
        return self.to_dict()["artifact_hash"]


class StrictOneShotAdmission:
    """Fail-closed verifier; model confidence and prose are ignored."""

    def admit(
        self,
        *,
        program: CanonicalSkillProgram,
        candidates: Sequence[BindingCandidate],
        demo: TargetDemoReceipt,
    ) -> AdmissionArtifact:
        program.validate()
        if program.status != ProgramStatus.SOURCE_VERIFIED:
            return self._failure(program, candidates, demo, AdmissionStatus.REJECTED, "SOURCE_NOT_VERIFIED")
        try:
            demo.validate_for_admission()
        except ValueError as exc:
            return self._failure(
                program, candidates, demo, AdmissionStatus.REJECTED,
                f"INVALID_DEMO:{str(exc).replace(' ', '_').upper()}",
            )

        proof: List[Dict[str, Any]] = [
            {"check": "source_program_verified", "pass": True, "program_hash": program.content_hash()},
            {"check": "one_fixed_real_demo", "pass": True, "demo_hash": demo.content_hash()},
            {"check": "official_target_success", "pass": True, "score": demo.official_score},
        ]
        action_by_signature: Dict[tuple[str, tuple[tuple[str, str], ...]], List[TargetActionEvidence]] = {}
        for action in demo.actions:
            key = (action.operator, tuple(sorted(action.argument_types.items())))
            action_by_signature.setdefault(key, []).append(action)

        source_steps = {step.step_id for step in program.steps}
        source_effects = {
            effect.predicate
            for step in program.steps
            for effect in step.effects
            if effect.value_type == "typed_source_state_delta"
        }
        source_effect_support = {
            effect.predicate: set(effect.evidence_step_ids)
            for step in program.steps
            for effect in step.effects
            if effect.value_type == "typed_source_state_delta"
        }
        passing: List[BindingCandidate] = []
        failures: List[str] = []
        uncovered: List[str] = []
        for candidate in candidates:
            canonical_schema = TARGET_OPERATOR_SCHEMAS.get(candidate.target_domain, {}).get(
                candidate.target_operator
            )
            checks = {
                "program_id": candidate.source_program_id == program.program_id,
                "program_hash": candidate.source_program_hash == program.content_hash(),
                "source_step": candidate.source_step_id in source_steps,
                "source_effect_verified": candidate.source_effect in source_effects,
                "target_domain": candidate.target_domain == demo.target_domain,
                "task_family": candidate.task_family == demo.task_family,
                "target_schema_supported": (
                    canonical_schema is not None
                    and dict(candidate.argument_types) == dict(canonical_schema)
                ),
                "source_target_effect_exact": (
                    TARGET_OPERATOR_EFFECTS.get(candidate.target_domain, {}).get(
                        candidate.target_operator
                    ) == candidate.source_effect
                ),
                "operator_and_types": (
                    candidate.target_operator,
                    tuple(sorted(candidate.argument_types.items())),
                ) in action_by_signature,
            }
            passed = all(checks.values())
            proof.append({
                "check": "candidate_binding",
                "candidate_id": candidate.candidate_id,
                "pass": passed,
                "details": checks,
                "proposal_source_untrusted": candidate.proposal_source,
            })
            if passed:
                passing.append(candidate)
            else:
                structural = all(
                    checks[key]
                    for key in (
                        "program_id", "program_hash", "source_step",
                        "source_effect_verified", "target_domain", "task_family",
                        "target_schema_supported", "source_target_effect_exact",
                    )
                )
                if structural and not checks["operator_and_types"]:
                    uncovered.append(f"OPERATOR_NOT_COVERED:{candidate.candidate_id}")
                else:
                    failures.append(f"CANDIDATE_FAILED:{candidate.candidate_id}")

        if not passing:
            return AdmissionArtifact(
                status=(
                    AdmissionStatus.INCONCLUSIVE
                    if uncovered and not failures
                    else AdmissionStatus.REJECTED
                ),
                source_program_id=program.program_id,
                source_program_hash=program.content_hash(),
                target_domain=demo.target_domain,
                task_family=demo.task_family,
                demo_id=demo.demo_id,
                demo_hash=demo.content_hash(),
                admitted_candidate_id=None,
                candidate_ids=[item.candidate_id for item in candidates],
                verified_scope=None,
                proof_trace=proof,
                failure_codes=failures + uncovered or ["NO_BINDING_CANDIDATE"],
            )

        signatures = {item.executable_signature for item in passing}
        if len(signatures) > 1:
            return AdmissionArtifact(
                status=AdmissionStatus.INCONCLUSIVE,
                source_program_id=program.program_id,
                source_program_hash=program.content_hash(),
                target_domain=demo.target_domain,
                task_family=demo.task_family,
                demo_id=demo.demo_id,
                demo_hash=demo.content_hash(),
                admitted_candidate_id=None,
                candidate_ids=[item.candidate_id for item in passing],
                verified_scope=None,
                proof_trace=proof,
                failure_codes=["NON_EQUIVALENT_BINDING_AMBIGUITY"],
            )

        chosen = sorted(passing, key=lambda item: item.candidate_id)[0]
        covered = {item.source_step_id for item in passing if item.executable_signature == chosen.executable_signature}
        support_count = len(source_effect_support.get(chosen.source_effect, set()))
        source_evidence_count = len(program.evidence)
        effect_is_universal = support_count == source_evidence_count
        status = (
            AdmissionStatus.ADMITTED
            if covered == source_steps and effect_is_universal
            else AdmissionStatus.CONDITIONAL
        )
        scope = VerifiedScope(
            target_domain=demo.target_domain,
            task_family=demo.task_family,
            operators=[chosen.target_operator],
            argument_types=dict(chosen.argument_types),
            source_steps=sorted(covered),
            effects=[chosen.source_effect],
            required_conditions=["exact_current_environment_admissibility"],
            source_effect_evidence_count=support_count,
            source_evidence_count=source_evidence_count,
        )
        proof.append({
            "check": "scope_bounded_admission",
            "pass": True,
            "covered_source_steps": sorted(covered),
            "all_source_steps": sorted(source_steps),
            "source_effect": chosen.source_effect,
            "source_effect_evidence_count": support_count,
            "source_evidence_count": source_evidence_count,
            "source_effect_universal": effect_is_universal,
            "status": status.value,
        })
        return AdmissionArtifact(
            status=status,
            source_program_id=program.program_id,
            source_program_hash=program.content_hash(),
            target_domain=demo.target_domain,
            task_family=demo.task_family,
            demo_id=demo.demo_id,
            demo_hash=demo.content_hash(),
            admitted_candidate_id=chosen.candidate_id,
            candidate_ids=[item.candidate_id for item in passing],
            verified_scope=scope,
            proof_trace=proof,
            failure_codes=[],
        )

    def _failure(
        self,
        program: CanonicalSkillProgram,
        candidates: Sequence[BindingCandidate],
        demo: TargetDemoReceipt,
        status: AdmissionStatus,
        code: str,
    ) -> AdmissionArtifact:
        return AdmissionArtifact(
            status=status,
            source_program_id=program.program_id,
            source_program_hash=program.content_hash(),
            target_domain=demo.target_domain,
            task_family=demo.task_family,
            demo_id=demo.demo_id,
            demo_hash=demo.content_hash(),
            admitted_candidate_id=None,
            candidate_ids=[item.candidate_id for item in candidates],
            verified_scope=None,
            proof_trace=[{"check": code, "pass": False}],
            failure_codes=[code],
        )


class FrozenAdmissionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def freeze(self, artifact: AdmissionArtifact) -> Path:
        payload = artifact.to_dict()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{artifact.artifact_hash}.json"
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise RuntimeError(f"immutable admission artifact collision: {path}")
            return path
        fd, tmp = tempfile.mkstemp(prefix=".admission.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        return path


def runtime_scope_allows(
    artifact: AdmissionArtifact,
    *,
    target_domain: str,
    task_family: str,
    operator: str,
    argument_types: Mapping[str, str],
) -> bool:
    if artifact.status not in {AdmissionStatus.ADMITTED, AdmissionStatus.CONDITIONAL}:
        return False
    scope = artifact.verified_scope
    return bool(
        scope
        and scope.target_domain == target_domain
        and scope.task_family == task_family
        and operator in scope.operators
        and dict(argument_types) == dict(scope.argument_types)
    )


def target_demo_receipt_from_dict(payload: Mapping[str, Any]) -> TargetDemoReceipt:
    receipt = TargetDemoReceipt(
        demo_id=str(payload["demo_id"]),
        target_domain=str(payload["target_domain"]),
        task_family=str(payload["task_family"]),
        split=str(payload["split"]),
        episode_id=str(payload["episode_id"]),
        source_file_sha256=str(payload["source_file_sha256"]),
        executor_kind=str(payload["executor_kind"]),
        evaluator=str(payload["evaluator"]),
        official_success=bool(payload["official_success"]),
        official_score=float(payload.get("official_score") or 0.0),
        actions=[
            TargetActionEvidence(
                transition_index=int(item["transition_index"]),
                action=str(item["action"]),
                operator=str(item["operator"]),
                arguments=dict(item.get("arguments") or {}),
                argument_types=dict(item.get("argument_types") or {}),
                admissible_actions_sha256=str(item["admissible_actions_sha256"]),
                state_sha256=str(item["state_sha256"]),
                next_state_sha256=str(item["next_state_sha256"]),
            )
            for item in payload.get("actions", [])
        ],
        held_out=bool(payload.get("held_out", False)),
    )
    expected_hash = payload.get("demo_hash")
    if expected_hash and expected_hash != receipt.content_hash():
        raise ValueError(f"demo hash mismatch for {receipt.demo_id}")
    return receipt


def admission_artifact_from_dict(payload: Mapping[str, Any]) -> AdmissionArtifact:
    """Load a frozen artifact and reject any content/hash mismatch.

    This is deliberately a deserializer, not an admission entry point.  A
    held-out evaluator can consume a preregistered decision without importing
    a verifier or seeing target outcomes.
    """
    scope_payload = payload.get("verified_scope")
    scope = None
    if scope_payload is not None:
        scope = VerifiedScope(
            target_domain=str(scope_payload["target_domain"]),
            task_family=str(scope_payload["task_family"]),
            operators=[str(item) for item in scope_payload.get("operators", [])],
            argument_types=dict(scope_payload.get("argument_types") or {}),
            source_steps=[str(item) for item in scope_payload.get("source_steps", [])],
            effects=[str(item) for item in scope_payload.get("effects", [])],
            required_conditions=[
                str(item) for item in scope_payload.get("required_conditions", [])
            ],
            source_effect_evidence_count=int(
                scope_payload.get("source_effect_evidence_count", 0)
            ),
            source_evidence_count=int(scope_payload.get("source_evidence_count", 0)),
        )
    artifact = AdmissionArtifact(
        status=AdmissionStatus(str(payload["status"])),
        source_program_id=str(payload["source_program_id"]),
        source_program_hash=str(payload["source_program_hash"]),
        target_domain=str(payload["target_domain"]),
        task_family=str(payload["task_family"]),
        demo_id=str(payload["demo_id"]),
        demo_hash=str(payload["demo_hash"]),
        admitted_candidate_id=(
            str(payload["admitted_candidate_id"])
            if payload.get("admitted_candidate_id") is not None
            else None
        ),
        candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
        verified_scope=scope,
        proof_trace=[dict(item) for item in payload.get("proof_trace", [])],
        failure_codes=[str(item) for item in payload.get("failure_codes", [])],
        schema_version=int(payload.get("schema_version", 1)),
    )
    expected_hash = payload.get("artifact_hash")
    if not expected_hash:
        raise ValueError("frozen admission artifact is missing artifact_hash")
    if str(expected_hash) != artifact.artifact_hash:
        raise ValueError(
            f"admission artifact hash mismatch: expected {expected_hash}, "
            f"computed {artifact.artifact_hash}"
        )
    return artifact


def load_frozen_admission_manifest(path: str | Path) -> List[AdmissionArtifact]:
    """Load only the immutable artifacts referenced by a one-shot manifest."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("one_shot") is not True:
        raise ValueError("admission manifest is not marked one_shot=true")
    if int(manifest.get("target_gradient_updates", -1)) != 0:
        raise ValueError("admission manifest contains target gradient updates")
    artifacts: List[AdmissionArtifact] = []
    for item in manifest.get("bindings", []):
        artifact_path = Path(str(item["artifact_path"]))
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact = admission_artifact_from_dict(payload)
        if artifact.artifact_hash != str(item["artifact_hash"]):
            raise ValueError(f"manifest/artifact hash mismatch: {artifact_path}")
        if artifact.status.value != str(item["status"]):
            raise ValueError(f"manifest/artifact status mismatch: {artifact_path}")
        artifacts.append(artifact)
    return artifacts


def _stable_hash(value: Any) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "AdmissionArtifact",
    "AdmissionStatus",
    "BindingCandidate",
    "FrozenAdmissionStore",
    "StrictOneShotAdmission",
    "TargetActionEvidence",
    "TargetDemoReceipt",
    "VerifiedScope",
    "admission_artifact_from_dict",
    "load_frozen_admission_manifest",
    "runtime_scope_allows",
    "target_demo_receipt_from_dict",
    "TARGET_OPERATOR_SCHEMAS",
    "TARGET_OPERATOR_EFFECTS",
]
