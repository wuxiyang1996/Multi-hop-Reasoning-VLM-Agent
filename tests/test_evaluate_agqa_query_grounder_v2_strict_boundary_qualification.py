from scripts.evaluate_agqa_query_grounder_v2_strict_boundary_qualification import (
    _preoutcome_gates,
)


def test_preoutcome_gate_requires_candidate_support_for_every_commit():
    protocol = {
        "qualification_cohort": {"query_tasks": 2},
        "qualification_gates": {
            "outcome_blind_source_symbolic_commit_fraction_minimum": .2,
            "outcome_blind_source_permuted_commit_fraction_maximum": .05,
            "candidate_support_required_for_every_symbolic_commit": True,
        },
        "source_harness": {
            "source_capability_sha256": "source",
            "anonymous_controller_sha256": "controller",
        },
    }
    audit = {
        "tasks": 2, "source_commit_fraction": .5,
        "permuted_commit_fraction": 0.0,
        "query_grounding_report_sha256": "grounding",
        "source_capability_sha256": "source",
        "anonymous_controller_sha256": "controller",
        "rows": [
            {"source_commit": True, "permuted_commit": False, "candidate_supported": True},
            {"source_commit": False, "permuted_commit": False, "candidate_supported": False},
        ],
    }
    assert all(_preoutcome_gates(protocol, audit, {"report_sha256": "grounding"}).values())
    audit["rows"][0]["candidate_supported"] = False
    assert not _preoutcome_gates(
        protocol, audit, {"report_sha256": "grounding"}
    )["candidate_support_required_for_every_symbolic_commit"]
