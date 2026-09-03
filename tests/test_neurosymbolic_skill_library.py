from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from motif_transfer.neurosymbolic_skill_library import (
    DispatchVerdict,
    EvidenceTier,
    FrozenNeurosymbolicSkillLibrary,
    SkillLibraryReject,
    TargetRequest,
    validate_dispatch_receipt,
)


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(tmp_path: Path, *, authority: str = "TARGET_NATIVE_GROUNDER_AND_EXECUTOR") -> Path:
    source_hash = "source-content-hash"
    source_file = _write(tmp_path / "source.json", {"artifact": {"sha": source_hash}})
    evidence_file = _write(tmp_path / "evidence.json", {"status": "PASSED"})
    adapter_file = _write(tmp_path / "adapter.json", {"adapter": True})
    registry = {
        "schema_version": "neurosymbolic-skill-library-v1",
        "dispatch_authority": "SELECT_SKILL_OR_ABSTAIN_ONLY",
        "skills": [{
            "skill_id": "topology-v1",
            "source_domains": ["sokoban"],
            "symbolic_payload": "anonymous edge execution",
            "source_artifact_sha256": source_hash,
            "source_artifact_hash_path": "artifact.sha",
            "source_receipt": {
                "path": "source.json", "file_sha256": source_file,
                "claims": [{"path": "artifact.sha", "equals": source_hash}],
            },
            "evidence_tier": "FRESH_FORMAL",
            "evidence_status": "PASSED",
            "evidence_status_path": "status",
            "evidence_receipt": {
                "path": "evidence.json", "file_sha256": evidence_file,
                "claims": [{"path": "status", "equals": "PASSED"}],
            },
            "adapter_files": [{
                "path": "adapter.json", "file_sha256": adapter_file,
            }],
            "routes": [{
                "target_domain": "tir",
                "target_interface": "single_image_maze",
                "required_capabilities": ["pixel_graph", "direction_binding"],
                "target_adapter": "tir-maze-v2",
                "target_grounder": "target.bind",
                "target_executor": "target.execute",
                "action_authority": authority,
            }],
        }],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    return path


def test_exact_dispatch_selects_program_but_never_an_action(tmp_path: Path) -> None:
    library = FrozenNeurosymbolicSkillLibrary.load(
        _registry(tmp_path), repo=tmp_path,
    )
    receipt = library.dispatch(TargetRequest.create(
        "tir", "single_image_maze", ["direction_binding", "pixel_graph", "extra"],
    ), minimum_evidence=EvidenceTier.FRESH_FORMAL)
    assert receipt.verdict == DispatchVerdict.SELECT_SKILL
    assert receipt.skill_id == "topology-v1"
    assert receipt.action_authority == "TARGET_NATIVE_GROUNDER_AND_EXECUTOR"
    assert not hasattr(receipt, "action")
    validate_dispatch_receipt(receipt)


def test_unknown_interface_and_underqualified_route_abstain(tmp_path: Path) -> None:
    library = FrozenNeurosymbolicSkillLibrary.load(
        _registry(tmp_path), repo=tmp_path,
    )
    missing = library.dispatch(TargetRequest.create("tir", "rotation", []))
    assert missing.verdict == DispatchVerdict.ABSTAIN
    assert missing.reason == "NO_EXACT_ROUTE"
    assert library.dispatch(
        TargetRequest.create("tir", "single_image_maze", ["pixel_graph"]),
    ).verdict == DispatchVerdict.ABSTAIN


def test_registry_rejects_hash_drift_and_source_action_authority(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    (tmp_path / "evidence.json").write_text('{"status":"CHANGED"}')
    with pytest.raises(SkillLibraryReject, match="hash mismatch"):
        FrozenNeurosymbolicSkillLibrary.load(registry, repo=tmp_path)

    clean = tmp_path / "clean"
    clean.mkdir()
    with pytest.raises(SkillLibraryReject, match="direct target-action authority"):
        FrozenNeurosymbolicSkillLibrary.load(
            _registry(clean, authority="SOURCE_EMITS_TARGET_ACTION"), repo=clean,
        )


def test_source_artifact_may_be_the_bound_receipt_file(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    registry = json.loads(registry_path.read_text())
    row = registry["skills"][0]
    row.pop("source_artifact_hash_path")
    row["source_artifact_sha256"] = row["source_receipt"]["file_sha256"]
    registry_path.write_text(json.dumps(registry))
    library = FrozenNeurosymbolicSkillLibrary.load(registry_path, repo=tmp_path)
    assert library.skills[0].source_artifact_sha256 == row["source_artifact_sha256"]
