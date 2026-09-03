from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit_source_provenance_identifiability_v15.py"


def _module():
    spec = importlib.util.spec_from_file_location("provenance_v15", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_disentangles_execution_from_source_acquisition() -> None:
    report = _module().run()
    assert report["status"] == "SOURCE_CONTENT_AND_ACQUISITION_VALUE_DISENTANGLED"
    assert all(report["gates"].values())
    assert report["alfworld_execution_equivalence"] == {
        "tasks": 45,
        "raw_target_only_successes": 24,
        "source_induced_successes": 38,
        "target_written_successes": 38,
        "exact_action_trace_matches": 45,
        "source_artifact_reads": 0,
    }
    assert report["matched_acquisition_information"][
        "complete_target_trajectories_replaced"
    ] == 1


def test_content_specificity_is_not_source_identity_routing() -> None:
    report = _module().run()
    assert report["content_specificity"] == {
        "source_program_catalog": 11,
        "distinct_selected_program_bodies": 3,
        "wrong_family_abstentions": 4,
        "target_routes": 4,
    }
    assert report["answer"]["source_provenance_after_program_specification"] == (
        "Neither necessary nor behaviorally identifiable."
    )
