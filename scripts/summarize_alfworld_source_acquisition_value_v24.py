#!/usr/bin/env python3
"""Build a compact audit of ALFWorld program-content acquisition value."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_goal_relation_macro import (  # noqa: E402
    AUTHENTIC,
    CARDINALITY_CONTROL,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_V16 = (
    REPO / "docs/results/alfworld_target_acquisition_value_v16_qualification.json"
)
DEFAULT_V17 = (
    REPO / "runs/alfworld_target_induced_policy_v17_consumed/report.json.gz"
)
DEFAULT_V24 = (
    REPO / "runs/alfworld_target_acquisition_py311_v23/"
    "report_deterministic_reconstruction.json.gz"
)
DEFAULT_OUTPUT = (
    REPO / "docs/results/alfworld_source_acquisition_value_v24_summary.json"
)
TARGET_K1 = "target_only_k1_induced_program"
TARGET_PERMUTED = "target_only_k1_binding_relation_permuted"


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        raw = gzip.open(path, "rt", encoding="utf-8").read()
    else:
        raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def summarize(v16_path: Path, v17_path: Path, v24_path: Path) -> dict[str, Any]:
    v16, v17, v24 = map(_read, (v16_path, v17_path, v24_path))
    for value in (v16, v17, v24):
        _self_hash(value, "report_sha256")
        if not all(value["gates"].values()):
            raise ValueError(f"failed input report: {value['status']}")

    summaries = v24["summaries"]
    trace_matches = v24["target_k1_source_comparisons"]
    curve = v16["target_only_induction_curve"]
    if [row["complete_target_trajectory_budget"] for row in curve[:2]] != [0, 1]:
        raise ValueError("V16 does not expose the registered K=0/K=1 contrast")
    body = {
        "schema_version": "alfworld-source-acquisition-value-v24-summary",
        "status": "ALFWORLD_SOURCE_ACQUISITION_VALUE_MECHANISM_VALIDATED",
        "answer": (
            "Correct symbolic program content, not source provenance, causes "
            "the execution gain. Source interventions contribute acquisition "
            "value by obtaining that content with zero complete target "
            "trajectories; the source-blind target learner abstains at K=0 "
            "and needs one complete target trajectory to recover it."
        ),
        "retrospective_acquisition_curve": {
            "development_complete_target_trajectories": int(
                v16["development"]["eligible_trajectories"]
            ),
            "heldout_complete_target_trajectories": int(
                v16["qualification"]["eligible_trajectories"]
            ),
            "target_k0_status": str(curve[0]["status"]),
            "target_k0_heldout_support": int(
                curve[0]["qualification_support"]
            ),
            "target_k1_status": str(curve[1]["status"]),
            "target_k1_heldout_support": int(
                curve[1]["qualification_support"]
            ),
            "target_k1_heldout_total": int(
                v16["qualification"]["eligible_trajectories"]
            ),
            "target_k1_shuffled_effect_support": int(
                curve[1]["qualification_shuffled_effect_support"]
            ),
            "target_k1_binding_relation_permuted_support": int(
                curve[1][
                    "qualification_binding_relation_permuted_support"
                ]
            ),
            "source_complete_target_trajectory_budget": 0,
            "target_complete_target_trajectory_budget": 1,
        },
        "consumed_execution_equivalence": {
            "tasks": int(v17["tasks"]),
            "raw_successes": int(v17["raw_target_only_reference_successes"]),
            "source_successes": int(
                v17["source_induced_reference_successes"]
            ),
            "target_k1_successes": int(v17["target_induced_successes"]),
            "exact_action_and_state_effect_traces": sum(
                row["actions_exactly_match_source"]
                and row["state_effect_trace_exactly_matches_source"]
                for row in v17["comparisons"]
            ),
        },
        "fresh_compiler_valid_replication": {
            "tasks": len(v24["episodes"][RAW]),
            "raw_successes": int(summaries[RAW]["successes"]),
            "source_successes": int(summaries[AUTHENTIC]["successes"]),
            "source_cardinality_control_successes": int(
                summaries[CARDINALITY_CONTROL]["successes"]
            ),
            "source_effect_control_successes": int(
                summaries[EFFECT_CONTROL]["successes"]
            ),
            "generic_scaffold_successes": int(
                summaries[GENERIC]["successes"]
            ),
            "target_k1_successes": int(summaries[TARGET_K1]["successes"]),
            "target_k1_permuted_successes": int(
                summaries[TARGET_PERMUTED]["successes"]
            ),
            "source_vs_raw": dict(v24["paired"]["source_vs_raw"]),
            "target_k1_vs_raw": dict(v24["paired"]["target_k1_vs_raw"]),
            "target_k1_vs_permuted": dict(
                v24["paired"]["target_k1_vs_target_permuted"]
            ),
            "source_target_k1_exact_action_traces": sum(
                row["actions_exactly_match"] for row in trace_matches
            ),
            "source_target_k1_exact_state_effect_traces": sum(
                row["state_effect_trace_exactly_matches"]
                for row in trace_matches
            ),
            "source_target_k1_trace_total": len(trace_matches),
            "mechanism_not_powered_population_claim": True,
        },
        "recovery_boundary": dict(v24["recovery_audit"]),
        "claim_limits": [
            "A correctly hand-written isomorphic target controller can match the source program.",
            "The measured source advantage is one complete ordered target trajectory, not human authoring time or all target-side priors.",
            "The fresh n=14 compiler-valid slice is a mechanism replication, not a powered ALFWorld population estimate.",
            "This validates the ALFWorld multiplicity interface, not arbitrary ALFWorld task families.",
        ],
        "lineage": {
            "v16_report_sha256": str(v16["report_sha256"]),
            "v17_report_sha256": str(v17["report_sha256"]),
            "v24_report_sha256": str(v24["report_sha256"]),
            "target_k1_program_sha256": str(
                v16["lineage"]["target_k1_program_sha256"]
            ),
        },
    }
    return body | {"summary_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v17", type=Path, default=DEFAULT_V17)
    parser.add_argument("--v24", type=Path, default=DEFAULT_V24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [
        path if path.is_absolute() else REPO / path
        for path in (args.v16, args.v17, args.v24)
    ]
    output = args.output if args.output.is_absolute() else REPO / args.output
    summary = summarize(*paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
