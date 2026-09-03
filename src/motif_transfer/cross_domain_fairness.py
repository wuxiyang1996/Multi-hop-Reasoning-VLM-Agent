"""Fail-closed fairness gates for cross-domain memory comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .cross_domain_memory_baselines import MemoryBaseline, validate_memory_artifact


class FairnessProtocolError(ValueError):
    """The requested comparison would violate the frozen fairness contract."""


@dataclass(frozen=True)
class SuiteFairnessAudit:
    target_domain: str
    methods: tuple[str, ...]
    source_superset_sha256: str
    source_episode_count: int
    adaptation_payload_sha256: str
    admitted_items: Mapping[str, int]
    artifact_sha256: Mapping[str, str]
    implementation_fidelity: str
    formal_ready: bool
    exact_baseline_ready: bool
    blockers: tuple[str, ...]


def require_nonpilot_embedding(identity: Mapping[str, Any], *, run_mode: str) -> None:
    """Prevent a dependency-light smoke retriever entering a formal result."""
    if run_mode not in {"pilot", "formal"}:
        raise FairnessProtocolError(f"unknown run mode: {run_mode!r}")
    if run_mode == "formal" and bool(identity.get("pilot_only")):
        raise FairnessProtocolError("formal comparison forbids a pilot-only embedding backend")


def audit_target_bound_suite(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    target_domain: str,
    expected_source_episodes: int = 96,
    implementation_fidelity: str = "clean_room_style",
) -> SuiteFairnessAudit:
    """Check the invariants required before comparing the three methods.

    Different method-specific source projections are allowed (AWM legitimately
    reads successful demonstrations only), but their frozen *superset* and the
    target adaptation examples must be identical.
    """
    expected = {row.value for row in MemoryBaseline}
    supplied = set(map(str, artifacts))
    if supplied != expected:
        raise FairnessProtocolError(
            f"suite must contain exactly {sorted(expected)}; got {sorted(supplied)}"
        )
    supersets: set[str] = set()
    superset_ids: set[tuple[str, ...]] = set()
    adaptations: set[str] = set()
    counts: set[int] = set()
    admitted: dict[str, int] = {}
    artifact_hashes: dict[str, str] = {}
    for method in sorted(expected):
        artifact = artifacts[method]
        validate_memory_artifact(artifact)
        if str(artifact["method"]) != method:
            raise FairnessProtocolError(f"artifact key/method mismatch for {method}")
        binding = artifact.get("target_binding") or {}
        if str(binding.get("target_domain")) != target_domain:
            raise FairnessProtocolError(f"{method} is not bound to {target_domain}")
        supersets.add(str(artifact.get("source_superset_sha256") or ""))
        ids = tuple(map(str, artifact.get("source_superset_episode_ids") or ()))
        superset_ids.add(ids)
        counts.add(len(ids))
        adaptations.add(str(binding.get("adaptation_payload_sha256") or ""))
        admitted[method] = len(artifact.get("items") or ())
        artifact_hashes[method] = str(artifact["artifact_sha256"])
    if "" in supersets or len(supersets) != 1 or len(superset_ids) != 1:
        raise FairnessProtocolError("methods do not share one identical source superset")
    if "" in adaptations or len(adaptations) != 1:
        raise FairnessProtocolError("methods do not share one target adaptation payload")
    if len(counts) != 1:
        raise FairnessProtocolError("source superset episode counts differ")
    count = next(iter(counts))
    blockers = []
    if count != int(expected_source_episodes):
        blockers.append(f"source_episode_count={count}, expected={expected_source_episodes}")
    exact_ready = implementation_fidelity == "upstream_pinned" and not blockers
    return SuiteFairnessAudit(
        target_domain=target_domain,
        methods=tuple(sorted(expected)),
        source_superset_sha256=next(iter(supersets)),
        source_episode_count=count,
        adaptation_payload_sha256=next(iter(adaptations)),
        admitted_items=admitted,
        artifact_sha256=artifact_hashes,
        implementation_fidelity=implementation_fidelity,
        formal_ready=not blockers,
        exact_baseline_ready=exact_ready,
        blockers=tuple(blockers),
    )


def require_formal_suite_audit(
    audit_path: str | Path | None,
    *,
    run_mode: str,
    target_domain: str,
    method: str | None = None,
    artifact_sha256: str | None = None,
) -> None:
    """Require a hash-valid, passing suite audit before any formal target call."""
    if run_mode != "formal":
        return
    if audit_path is None:
        raise FairnessProtocolError("formal run requires --fairness-audit")
    report = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    from .contracts import stable_hash
    claimed = str(report.get("report_sha256") or "")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if not claimed or stable_hash(body) != claimed:
        raise FairnessProtocolError("fairness audit hash mismatch")
    if not report.get("all_domains_formal_ready"):
        raise FairnessProtocolError("fairness audit has unresolved formal blockers")
    domain = (report.get("domains") or {}).get(target_domain)
    if not isinstance(domain, Mapping) or not domain.get("formal_ready"):
        raise FairnessProtocolError(f"target domain is not formal-ready: {target_domain}")
    if method is not None:
        expected = str((domain.get("artifact_sha256") or {}).get(method) or "")
        if not artifact_sha256 or expected != artifact_sha256:
            raise FairnessProtocolError("runtime artifact does not match the fairness audit")


def assert_paired_target_receipts(
    target_only: Mapping[str, Any], memory: Mapping[str, Any]
) -> None:
    """Reject a comparison unless task, initial state, model and budgets match."""
    fields = (
        "target_domain", "task_id", "seed", "decision_model",
        "maximum_steps", "max_steps", "candidate_count",
        "maximum_output_tokens", "decision_max_tokens",
    )
    for field in fields:
        left, right = target_only.get(field), memory.get(field)
        if left is not None or right is not None:
            if left != right:
                raise FairnessProtocolError(f"paired receipt mismatch at {field}: {left!r} != {right!r}")
    for field in ("initial_state_hash", "resolved_game_file", "image_sha256"):
        left, right = target_only.get(field), memory.get(field)
        if left is not None or right is not None:
            if not left or left != right:
                raise FairnessProtocolError(f"paired receipt mismatch at {field}")


__all__ = [
    "FairnessProtocolError", "SuiteFairnessAudit", "assert_paired_target_receipts",
    "audit_target_bound_suite", "require_nonpilot_embedding",
    "require_formal_suite_audit",
]
