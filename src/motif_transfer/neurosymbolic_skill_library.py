"""Evidence-bound dispatch for heterogeneous neural-symbolic transfer skills.

The library deliberately does not translate source actions into target actions.
It can only select one source-qualified symbolic program (or abstain).  The
selected route names a target-native grounder and executor that retain all
authority over target symbols and actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from motif_transfer.contracts import stable_hash


class SkillLibraryReject(ValueError):
    """Raised when a supposedly frozen registry or receipt is inconsistent."""


class EvidenceTier(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    MECHANISM = "MECHANISM"
    FRESH_FORMAL = "FRESH_FORMAL"


_TIER_ORDER = {
    EvidenceTier.DEVELOPMENT: 0,
    EvidenceTier.MECHANISM: 1,
    EvidenceTier.FRESH_FORMAL: 2,
}


class DispatchVerdict(str, Enum):
    SELECT_SKILL = "SELECT_SKILL"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class TargetRequest:
    domain: str
    interface: str
    capabilities: tuple[str, ...]

    @classmethod
    def create(
        cls, domain: str, interface: str, capabilities: Sequence[str],
    ) -> "TargetRequest":
        return cls(
            domain=str(domain),
            interface=str(interface),
            capabilities=tuple(sorted({str(value) for value in capabilities})),
        )


@dataclass(frozen=True)
class FrozenRoute:
    target_domain: str
    target_interface: str
    required_capabilities: tuple[str, ...]
    target_adapter: str
    target_grounder: str
    target_executor: str
    action_authority: str


@dataclass(frozen=True)
class FrozenSkill:
    skill_id: str
    source_domains: tuple[str, ...]
    symbolic_payload: str
    source_artifact_sha256: str
    evidence_tier: EvidenceTier
    evidence_status: str
    routes: tuple[FrozenRoute, ...]


@dataclass(frozen=True)
class DispatchReceipt:
    verdict: DispatchVerdict
    reason: str
    request: TargetRequest
    skill_id: str | None
    source_artifact_sha256: str | None
    evidence_tier: EvidenceTier | None
    target_adapter: str | None
    target_grounder: str | None
    target_executor: str | None
    action_authority: str | None
    source_permission: str
    receipt_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within_repo(repo: Path, relative: str) -> Path:
    candidate = (repo / relative).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise SkillLibraryReject(f"registry path escapes repository: {relative}") from exc
    if not candidate.is_file():
        raise SkillLibraryReject(f"registry file is absent: {relative}")
    return candidate


def _lookup(payload: Any, path: str) -> Any:
    value = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise SkillLibraryReject(f"receipt claim path is absent: {path}")
        value = value[component]
    return value


def _validate_file_hash(repo: Path, spec: Mapping[str, Any]) -> Path:
    path = _within_repo(repo, str(spec["path"]))
    observed = _file_sha256(path)
    expected = str(spec["file_sha256"])
    if observed != expected:
        raise SkillLibraryReject(
            f"frozen file hash mismatch for {spec['path']}: {observed} != {expected}"
        )
    return path


def _validate_bound_file(repo: Path, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    path = _validate_file_hash(repo, spec)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillLibraryReject(f"frozen file is not JSON: {spec['path']}") from exc
    if not isinstance(payload, Mapping):
        raise SkillLibraryReject(f"frozen file must contain an object: {spec['path']}")
    for claim in spec.get("claims") or ():
        actual = _lookup(payload, str(claim["path"]))
        if actual != claim["equals"]:
            raise SkillLibraryReject(
                f"frozen receipt claim failed: {spec['path']}::{claim['path']}"
            )
    return payload


class FrozenNeurosymbolicSkillLibrary:
    """Validated skill contracts with exact, non-semantic route dispatch."""

    def __init__(self, skills: Sequence[FrozenSkill], registry_sha256: str):
        identifiers = [skill.skill_id for skill in skills]
        if len(identifiers) != len(set(identifiers)):
            raise SkillLibraryReject("duplicate skill_id in registry")
        self.skills = tuple(skills)
        self.registry_sha256 = registry_sha256

    @classmethod
    def load(cls, registry_path: str | Path, *, repo: str | Path) -> "FrozenNeurosymbolicSkillLibrary":
        repo_path = Path(repo).resolve()
        path = Path(registry_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "neurosymbolic-skill-library-v1":
            raise SkillLibraryReject("unsupported skill-library schema")
        if payload.get("dispatch_authority") != "SELECT_SKILL_OR_ABSTAIN_ONLY":
            raise SkillLibraryReject("registry gives the dispatcher excess authority")
        skills = []
        for row in payload.get("skills") or ():
            source_payload = _validate_bound_file(repo_path, row["source_receipt"])
            evidence_payload = _validate_bound_file(repo_path, row["evidence_receipt"])
            expected_source_hash = str(row["source_artifact_sha256"])
            source_hash_path = row.get("source_artifact_hash_path")
            source_hash = (
                str(_lookup(source_payload, str(source_hash_path)))
                if source_hash_path
                else str(row["source_receipt"]["file_sha256"])
            )
            if source_hash != expected_source_hash:
                raise SkillLibraryReject(f"source content hash mismatch: {row['skill_id']}")
            evidence_status = str(_lookup(evidence_payload, row["evidence_status_path"]))
            if evidence_status != str(row["evidence_status"]):
                raise SkillLibraryReject(f"evidence status mismatch: {row['skill_id']}")
            adapter_files = row.get("adapter_files") or ()
            for adapter_file in adapter_files:
                _validate_file_hash(repo_path, adapter_file)
            routes = []
            for route in row.get("routes") or ():
                authority = str(route["action_authority"])
                if authority != "TARGET_NATIVE_GROUNDER_AND_EXECUTOR":
                    raise SkillLibraryReject(
                        f"route gives source direct target-action authority: {row['skill_id']}"
                    )
                routes.append(FrozenRoute(
                    target_domain=str(route["target_domain"]),
                    target_interface=str(route["target_interface"]),
                    required_capabilities=tuple(sorted({
                        str(value) for value in route["required_capabilities"]
                    })),
                    target_adapter=str(route["target_adapter"]),
                    target_grounder=str(route["target_grounder"]),
                    target_executor=str(route["target_executor"]),
                    action_authority=authority,
                ))
            if not routes:
                raise SkillLibraryReject(f"skill has no target routes: {row['skill_id']}")
            skills.append(FrozenSkill(
                skill_id=str(row["skill_id"]),
                source_domains=tuple(str(value) for value in row["source_domains"]),
                symbolic_payload=str(row["symbolic_payload"]),
                source_artifact_sha256=expected_source_hash,
                evidence_tier=EvidenceTier(str(row["evidence_tier"])),
                evidence_status=evidence_status,
                routes=tuple(routes),
            ))
        if not skills:
            raise SkillLibraryReject("registry has no skills")
        return cls(skills, _file_sha256(path))

    def dispatch(
        self,
        request: TargetRequest,
        *,
        minimum_evidence: EvidenceTier = EvidenceTier.MECHANISM,
    ) -> DispatchReceipt:
        capabilities = set(request.capabilities)
        candidates: list[tuple[FrozenSkill, FrozenRoute]] = []
        underqualified = False
        for skill in self.skills:
            for route in skill.routes:
                if (
                    route.target_domain == request.domain
                    and route.target_interface == request.interface
                    and set(route.required_capabilities) <= capabilities
                ):
                    if _TIER_ORDER[skill.evidence_tier] >= _TIER_ORDER[minimum_evidence]:
                        candidates.append((skill, route))
                    else:
                        underqualified = True
        if len(candidates) > 1:
            raise SkillLibraryReject(
                "ambiguous exact dispatch; registry must make target routes disjoint"
            )
        if not candidates:
            reason = "EVIDENCE_TIER_BELOW_REQUIREMENT" if underqualified else "NO_EXACT_ROUTE"
            return self._receipt(request=request, verdict=DispatchVerdict.ABSTAIN, reason=reason)
        skill, route = candidates[0]
        return self._receipt(
            request=request,
            verdict=DispatchVerdict.SELECT_SKILL,
            reason="EXACT_INTERFACE_AND_CAPABILITY_MATCH",
            skill=skill,
            route=route,
        )

    @staticmethod
    def _receipt(
        *,
        request: TargetRequest,
        verdict: DispatchVerdict,
        reason: str,
        skill: FrozenSkill | None = None,
        route: FrozenRoute | None = None,
    ) -> DispatchReceipt:
        body = {
            "verdict": verdict.value,
            "reason": reason,
            "request": asdict(request),
            "skill_id": skill.skill_id if skill else None,
            "source_artifact_sha256": skill.source_artifact_sha256 if skill else None,
            "evidence_tier": skill.evidence_tier.value if skill else None,
            "target_adapter": route.target_adapter if route else None,
            "target_grounder": route.target_grounder if route else None,
            "target_executor": route.target_executor if route else None,
            "action_authority": route.action_authority if route else None,
            "source_permission": "SELECT_SYMBOLIC_PROGRAM_OR_ABSTAIN; NEVER_EMIT_TARGET_ACTION",
        }
        return DispatchReceipt(
            verdict=verdict,
            reason=reason,
            request=request,
            skill_id=body["skill_id"],
            source_artifact_sha256=body["source_artifact_sha256"],
            evidence_tier=skill.evidence_tier if skill else None,
            target_adapter=body["target_adapter"],
            target_grounder=body["target_grounder"],
            target_executor=body["target_executor"],
            action_authority=body["action_authority"],
            source_permission=body["source_permission"],
            receipt_sha256=stable_hash(body),
        )


def validate_dispatch_receipt(receipt: DispatchReceipt) -> None:
    body = asdict(receipt)
    claimed = body.pop("receipt_sha256")
    body["verdict"] = receipt.verdict.value
    body["evidence_tier"] = (
        receipt.evidence_tier.value if receipt.evidence_tier else None
    )
    if stable_hash(body) != claimed:
        raise SkillLibraryReject("dispatch receipt self-hash mismatch")
    if receipt.verdict == DispatchVerdict.SELECT_SKILL:
        if receipt.action_authority != "TARGET_NATIVE_GROUNDER_AND_EXECUTOR":
            raise SkillLibraryReject("selected skill violates target-native authority")
        if not all((receipt.skill_id, receipt.target_grounder, receipt.target_executor)):
            raise SkillLibraryReject("selected dispatch receipt is incomplete")


__all__ = [
    "DispatchReceipt",
    "DispatchVerdict",
    "EvidenceTier",
    "FrozenNeurosymbolicSkillLibrary",
    "FrozenRoute",
    "FrozenSkill",
    "SkillLibraryReject",
    "TargetRequest",
    "validate_dispatch_receipt",
]
