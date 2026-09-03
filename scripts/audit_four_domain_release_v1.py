#!/usr/bin/env python3
"""Fail-closed audit of the portable four-domain evidence release."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.neurosymbolic_skill_library import (  # noqa: E402
    DispatchVerdict,
    EvidenceTier,
    FrozenNeurosymbolicSkillLibrary,
    TargetRequest,
    validate_dispatch_receipt,
)


class ReleaseAuditError(ValueError):
    """Raised when a release input differs from the frozen manifest."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_file(relative: str, *, repo: Path) -> Path:
    path = (repo / relative).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError as exc:
        raise ReleaseAuditError(f"release path escapes repository: {relative}") from exc
    if not path.is_file():
        raise ReleaseAuditError(f"release file is absent: {relative}")
    return path


def _require_hash(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ReleaseAuditError(
            f"release file hash mismatch for {path}: {observed} != {expected}"
        )


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def audit_release(manifest_path: Path, *, repo: Path = REPO) -> dict[str, Any]:
    repo = repo.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "four-domain-neurosymbolic-release-v1":
        raise ReleaseAuditError("unsupported release manifest")
    if manifest.get("status") != (
        "PORTABLE_AUDIT_AND_ALFWORLD_ARTIFACT_BUNDLE_FROZEN"
    ):
        raise ReleaseAuditError("release manifest is not frozen")

    minimum_python = str(manifest["audit"]["minimum_python"])
    current_python = sys.version_info[:3]
    if current_python < _version_tuple(minimum_python):
        raise ReleaseAuditError(
            f"Python {minimum_python}+ is required for the portable audit"
        )

    registry_spec = manifest["audit"]["registry"]
    registry_path = _repo_file(str(registry_spec["path"]), repo=repo)
    _require_hash(registry_path, str(registry_spec["file_sha256"]))
    summary_spec = manifest["audit"]["frozen_summary"]
    summary_path = _repo_file(str(summary_spec["path"]), repo=repo)
    _require_hash(summary_path, str(summary_spec["file_sha256"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != summary_spec["required_status"]:
        raise ReleaseAuditError("frozen unified summary has the wrong status")
    summary_body = dict(summary)
    claimed_summary_hash = summary_body.pop("report_sha256")
    if stable_hash(summary_body) != claimed_summary_hash:
        raise ReleaseAuditError("frozen unified summary self-hash mismatch")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    library = FrozenNeurosymbolicSkillLibrary.load(registry_path, repo=repo)
    positive = []
    for row in registry.get("verification_requests") or ():
        receipt = library.dispatch(
            TargetRequest.create(
                row["domain"], row["interface"], row["capabilities"]
            ),
            minimum_evidence=EvidenceTier(str(row["minimum_evidence"])),
        )
        validate_dispatch_receipt(receipt)
        if (
            receipt.verdict != DispatchVerdict.SELECT_SKILL
            or receipt.skill_id != row["expected_skill_id"]
        ):
            raise ReleaseAuditError(f"positive dispatch failed: {row['domain']}")
        positive.append(asdict(receipt))

    negative = []
    for row in registry.get("negative_verification_requests") or ():
        receipt = library.dispatch(
            TargetRequest.create(
                row["domain"], row["interface"], row.get("capabilities") or ()
            ),
            minimum_evidence=EvidenceTier(
                str(row.get("minimum_evidence") or "MECHANISM")
            ),
        )
        validate_dispatch_receipt(receipt)
        if receipt.verdict != DispatchVerdict.ABSTAIN:
            raise ReleaseAuditError(f"negative dispatch did not abstain: {row}")
        negative.append(asdict(receipt))

    expected_domains = list(map(str, manifest["audit"]["expected_domains"]))
    observed_domains = [
        str(row["request"]["domain"])
        for row in positive
    ]
    if observed_domains != expected_domains:
        raise ReleaseAuditError(
            f"dispatch domain order changed: {observed_domains} != {expected_domains}"
        )
    if len(positive) != int(manifest["audit"]["expected_positive_dispatches"]):
        raise ReleaseAuditError("positive dispatch count changed")
    if len(negative) != int(manifest["audit"]["expected_negative_dispatches"]):
        raise ReleaseAuditError("negative dispatch count changed")

    artifact_rows = []
    artifact_payloads: dict[str, Mapping[str, Any]] = {}
    for spec in manifest.get("bundled_artifacts") or ():
        if spec.get("compression") != "gzip-no-name":
            raise ReleaseAuditError("unsupported release artifact compression")
        path = _repo_file(str(spec["path"]), repo=repo)
        _require_hash(path, str(spec["file_sha256"]))
        try:
            with gzip.open(path, "rb") as handle:
                raw = handle.read()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseAuditError(f"invalid bundled JSON artifact: {path}") from exc
        observed_content_hash = _sha256_bytes(raw)
        if observed_content_hash != str(spec["uncompressed_sha256"]):
            raise ReleaseAuditError(f"uncompressed artifact hash mismatch: {path}")
        role = str(spec["role"])
        artifact_payloads[role] = payload
        artifact_rows.append({
            "role": role,
            "compressed_bytes": path.stat().st_size,
            "file_sha256": str(spec["file_sha256"]),
            "uncompressed_sha256": observed_content_hash,
        })

    expected_roles = {
        "alfworld_frozen_candidate",
        "alfworld_development_report",
        "alfworld_final_report",
    }
    if set(artifact_payloads) != expected_roles:
        raise ReleaseAuditError("ALFWorld artifact bundle is incomplete")
    expected_statuses = {
        "alfworld_frozen_candidate": "QUALIFICATION_AUTHORIZED",
        "alfworld_development_report": "QUALIFICATION_CANDIDATE_PASSED",
        "alfworld_final_report": "FINAL_HELDOUT_PASSED",
    }
    for role, expected in expected_statuses.items():
        if artifact_payloads[role].get("status") != expected:
            raise ReleaseAuditError(f"bundled artifact status failed for {role}")

    body = {
        "schema_version": "four-domain-neurosymbolic-release-audit-v1",
        "status": "PORTABLE_FOUR_DOMAIN_AUDIT_AND_ALFWORLD_BUNDLE_VALIDATED",
        "release_manifest_file_sha256": _sha256(manifest_path),
        "registry_file_sha256": _sha256(registry_path),
        "frozen_summary_file_sha256": _sha256(summary_path),
        "domains": observed_domains,
        "positive_dispatches": len(positive),
        "negative_abstentions": len(negative),
        "target_native_action_authority": all(
            row["action_authority"] == "TARGET_NATIVE_GROUNDER_AND_EXECUTOR"
            for row in positive
        ),
        "bundled_artifacts": artifact_rows,
        "alfworld_full_artifact_bundle_present": True,
        "third_party_dependencies_used_by_audit": [],
        "python": ".".join(map(str, current_python)),
        "claim_boundary": manifest["claim_boundary"],
    }
    body["audit_sha256"] = stable_hash(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs/four_domain_neurosymbolic_release_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_release(args.manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
