#!/usr/bin/env python3
"""Audit program-content efficacy separately from source acquisition value."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_V15 = REPO / "runs/alfworld_target_written_provenance_v15/report.json"
DEFAULT_SOURCE_VALUE = (
    REPO / "docs/results/discoveryworld_normal_source_value_v26_qualification.json"
)
DEFAULT_HETEROGENEITY = (
    REPO / "docs/results/phase9_source_program_heterogeneity_v1.json"
)
DEFAULT_SOURCE_ARTIFACT = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"
DEFAULT_SOURCE_CONFIRMATION = (
    REPO / "runs/sokoban_goal_acquisition_v1/fresh_confirmation_report.json"
)
DEFAULT_OUTPUT = REPO / "docs/results/source_provenance_identifiability_v15.json"


def _read(path: Path) -> dict[str, Any]:
    actual = path
    if not actual.exists() and Path(f"{path}.gz").exists():
        actual = Path(f"{path}.gz")
    if actual.suffix == ".gz":
        raw = gzip.open(actual, "rt", encoding="utf-8").read()
    else:
        raw = actual.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {actual}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _verify_v15_nested(report: Mapping[str, Any]) -> None:
    for population in report["populations"]:
        body = dict(population)
        claimed = str(body.pop("population_sha256", ""))
        if stable_hash(body) != claimed:
            raise ValueError("invalid V15 population hash")
        for row in population["rows"]:
            row_body = dict(row)
            row_claimed = str(row_body.pop("row_sha256", ""))
            if stable_hash(row_body) != row_claimed:
                raise ValueError("invalid V15 task-row hash")


def run(
    *, v15_path: Path = DEFAULT_V15,
    source_value_path: Path = DEFAULT_SOURCE_VALUE,
    heterogeneity_path: Path = DEFAULT_HETEROGENEITY,
    source_artifact_path: Path = DEFAULT_SOURCE_ARTIFACT,
    source_confirmation_path: Path = DEFAULT_SOURCE_CONFIRMATION,
) -> dict[str, Any]:
    v15 = _read(v15_path)
    source_value = _read(source_value_path)
    heterogeneity = _read(heterogeneity_path)
    source_artifact = _read(source_artifact_path)
    source_confirmation = _read(source_confirmation_path)
    for value in (v15, source_value, heterogeneity):
        _verify_self_hash(value, "report_sha256")
    _verify_self_hash(source_artifact, "artifact_sha256")
    _verify_self_hash(source_confirmation, "report_sha256")
    _verify_v15_nested(v15)

    curve = {
        int(row["complete_ordered_target_trajectory_budget"]): row
        for row in source_value["target_only_induction_curve"]
    }
    route_audits = list(heterogeneity["route_audits"])
    wrong_families_abstain = all(
        row["wrong_family_selection"]["status"]
        == "SOURCE_CONTRACT_SELECTION_ABSTAINED"
        for row in route_audits
    )
    gates = {
        "target_written_reads_no_source_artifact": bool(
            v15["gates"]["zero_source_artifact_reads"]
        ),
        "target_written_exactly_matches_all_45_action_traces": (
            int(v15["combined"]["exact_action_trace_matches"]) == 45
            and bool(v15["gates"]["all_state_effect_traces_exactly_match"])
        ),
        "program_content_has_nonzero_success_utility": (
            int(v15["combined"][
                "source_induced_and_target_written_gain_over_raw"
            ]) == 14
        ),
        "source_artifact_is_source_only_induced": (
            source_artifact.get("induction_authority")
            == "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY"
            and source_artifact.get("target_data_read") is False
            and source_artifact.get("named_controller_template_used") is False
        ),
        "source_artifact_passes_heldout_and_shuffled_effect_controls": (
            bool(source_confirmation.get("source_gate_passed"))
            and bool(source_confirmation["gates"][
                "heldout_transition_conformance"
            ])
            and bool(source_confirmation["gates"][
                "authentic_beats_shuffled_effect_conformance"
            ])
            and bool(source_confirmation["gates"][
                "shuffled_effect_binding_rejected"
            ])
        ),
        "matched_target_zero_demo_induction_abstains": (
            curve[0]["status"] == "ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY"
            and curve[0]["matches_source_phase_program"] is False
        ),
        "matched_target_one_demo_recovers_program": (
            curve[1]["status"] == "TARGET_ONLY_PROGRAM_INDUCED"
            and curve[1]["matches_source_phase_program"] is True
        ),
        "source_replaces_one_complete_target_trajectory": (
            int(source_value["metrics"][
                "complete_target_trajectories_replaced"
            ]) == 1
        ),
        "program_selection_is_content_specific": (
            int(heterogeneity["source_catalog_size"]) == 11
            and int(heterogeneity["selected_distinct_programs"]) == 3
            and wrong_families_abstain
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "source-provenance-identifiability-audit-v15",
        "status": (
            "SOURCE_CONTENT_AND_ACQUISITION_VALUE_DISENTANGLED"
            if passed else "SOURCE_PROVENANCE_IDENTIFIABILITY_AUDIT_FAILED"
        ),
        "question": (
            "Does efficacy come from structure learned by source intervention, "
            "or can an isomorphic target-written controller work?"
        ),
        "answer": {
            "execution_effect": (
                "The program structure is effective; an extensionally "
                "isomorphic target-written controller works identically."
            ),
            "source_provenance_after_program_specification": (
                "Neither necessary nor behaviorally identifiable."
            ),
            "demonstrated_source_intervention_value": (
                "Automatic source-only acquisition of the structure; in the "
                "matched DiscoveryWorld induction curve it replaces one "
                "complete ordered successful target trajectory."
            ),
            "not_measured": (
                "Human target-controller authoring cost and whether source "
                "beats every possible target-side prior or program synthesizer."
            ),
        },
        "alfworld_execution_equivalence": {
            "tasks": int(v15["combined"]["tasks"]),
            "raw_target_only_successes": int(
                v15["combined"]["raw_target_only_successes"]
            ),
            "source_induced_successes": int(
                v15["combined"]["authentic_source_induced_successes"]
            ),
            "target_written_successes": int(
                v15["combined"]["target_written_successes"]
            ),
            "exact_action_trace_matches": int(
                v15["combined"]["exact_action_trace_matches"]
            ),
            "source_artifact_reads": sum(
                int(row["source_artifact_read_attempts"])
                for row in v15["populations"]
            ),
        },
        "matched_acquisition_information": {
            "source_complete_target_trajectory_budget": 0,
            "target_only_k0_status": str(curve[0]["status"]),
            "target_only_k1_status": str(curve[1]["status"]),
            "complete_target_trajectories_replaced": int(
                source_value["metrics"][
                    "complete_target_trajectories_replaced"
                ]
            ),
        },
        "content_specificity": {
            "source_program_catalog": int(
                heterogeneity["source_catalog_size"]
            ),
            "distinct_selected_program_bodies": int(
                heterogeneity["selected_distinct_programs"]
            ),
            "wrong_family_abstentions": sum(
                row["wrong_family_selection"]["status"]
                == "SOURCE_CONTRACT_SELECTION_ABSTAINED"
                for row in route_audits
            ),
            "target_routes": len(route_audits),
        },
        "claim_boundary": (
            "The evidence supports source-induced program acquisition plus "
            "target-native grounding. It does not support a provenance magic "
            "claim: once identical program content is supplied, origin cannot "
            "change or identify extensional behavior. V15 is a posthoc "
            "consumed-population diagnostic; prospective success evidence "
            "remains the original V13/V14 runs."
        ),
        "lineage": {
            "alfworld_v15_report_sha256": str(v15["report_sha256"]),
            "discoveryworld_source_value_report_sha256": str(
                source_value["report_sha256"]
            ),
            "phase9_heterogeneity_report_sha256": str(
                heterogeneity["report_sha256"]
            ),
            "source_artifact_sha256": str(
                source_artifact["artifact_sha256"]
            ),
            "source_confirmation_sha256": str(
                source_confirmation["report_sha256"]
            ),
            "auditor_file_sha256": _sha(Path(__file__)),
        },
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run()
    output = args.output if args.output.is_absolute() else REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
