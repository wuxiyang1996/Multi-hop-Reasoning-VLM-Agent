import json
from pathlib import Path

import pytest

from scripts.freeze_agqa_track_verified_formal_protocol import (
    _sha256,
    build_protocol,
)


def _inputs(tmp_path: Path, *, passed: bool = True):
    qualification_protocol = {
        "frozen_grounder": {
            "candidate_verifier": {"formula": "fixed"},
            "candidate_support_threshold": 0.76,
        },
        "source_harness": {
            "source_capability_file": "source.json",
            "source_capability_file_sha256": "source-file",
            "source_capability_sha256": "source",
            "anonymous_controller_file": "controller.json",
            "anonymous_controller_file_sha256": "controller-file",
            "anonymous_controller_sha256": "controller",
        },
    }
    protocol_path = tmp_path / "qualification_protocol.json"
    protocol_path.write_text(json.dumps(qualification_protocol))
    qualification = {
        "status": (
            "QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED"
            if passed else "QUERY_GROUNDER_V2_STRICT_BOUNDARY_NOT_QUALIFIED"
        ),
        "gates": {"all": passed},
        "protocol_file_sha256": _sha256(protocol_path),
        "report_sha256": "qualified-report",
    }
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(json.dumps(qualification))
    return qualification, qualification_protocol, qualification_path, protocol_path


def test_builds_five_arm_protocol_only_after_pass(tmp_path: Path) -> None:
    qualification, qualification_protocol, qualification_path, protocol_path = _inputs(tmp_path)
    result = build_protocol(
        qualification, qualification_protocol,
        qualification_file=qualification_path,
        qualification_protocol_file=protocol_path,
        videos=512, tasks_per_video=2, selection_salt="fresh",
    )
    assert result["formal_cohort"]["query_object_tasks"] == 1024
    assert result["qualified_grounder"]["candidate_support_threshold"] == 0.76
    assert result["arms"] == [
        "neural_only", "generic_scaffold", "source_permuted",
        "source_induced", "target_written_isomorphic",
    ]
    assert result["formal_gates"]["maximum_source_permuted_commit_fraction"] == 0.05


def test_rejects_failed_qualification(tmp_path: Path) -> None:
    qualification, qualification_protocol, qualification_path, protocol_path = _inputs(
        tmp_path, passed=False,
    )
    with pytest.raises(ValueError, match="passed grounder qualification"):
        build_protocol(
            qualification, qualification_protocol,
            qualification_file=qualification_path,
            qualification_protocol_file=protocol_path,
            videos=512, tasks_per_video=2, selection_salt="fresh",
        )
