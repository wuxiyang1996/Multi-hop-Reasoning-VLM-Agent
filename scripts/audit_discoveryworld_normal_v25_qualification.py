#!/usr/bin/env python3
"""Decide whether another DiscoveryWorld Normal formal reserve may open.

The audit uses consumed development evidence only.  It records the successful
target-acquisition repair separately from the transfer/headroom gates so that
coverage improvement cannot silently authorize a formal transfer experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


V22 = REPO / "docs/results/discoveryworld_v22_normal_formal_early_stop.json"
ACQUISITION = (
    REPO / "runs/discoveryworld_proteomics_normal_v24_development/"
    "acquisition_summary.json"
)
MATCHED = (
    REPO / "runs/discoveryworld_proteomics_normal_v24_matched_development/"
    "proteomics.normal.seed2.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}: {claimed}")


def build_report() -> dict[str, Any]:
    v22 = _read(V22)
    acquisition = _read(ACQUISITION)
    matched = _read(MATCHED)
    _self_hash(v22, "summary_sha256")
    _self_hash(acquisition, "summary_sha256")
    _self_hash(matched, "result_sha256")
    if not matched["all_matched_forks"] or not matched[
        "all_selection_receipts_valid"
    ]:
        raise ValueError("V24 matched development receipt is invalid")

    conditions = matched["conditions"]
    target_success = bool(conditions["target_native_myopic"]["official_success"])
    authentic_success = bool(
        conditions["authentic_sokoban_effect_plus_target"]["official_success"]
    )
    availability_success = bool(
        conditions["commit_availability_control_plus_target"]["official_success"]
    )
    binding = matched["target_binding"]
    development_tasks = int(acquisition["tasks"])
    matched_forks = 1
    acquisition_successes = sum(
        bool(row["official_success"]) for row in acquisition["episodes"]
    )
    gates = {
        "v22_incompatible_space_sick_normal_route_removed": (
            v22["interpretation"]["formal_design_error"].startswith(
                "Difficulty did not preserve the intervention interface"
            )
        ),
        "proteomics_normal_target_acquisition_reaches_commit": (
            acquisition["status"] == "TARGET_ACQUISITION_GATE_PASSED"
            and int(acquisition["commit_coverage"]) == development_tasks
        ),
        "minimum_eight_development_tasks": development_tasks >= 8,
        "minimum_six_valid_matched_forks": matched_forks >= 6,
        "target_relation_vocabulary_represents_required_adjacency": (
            str(binding["commit_subject_relation_to_target"]) == "adjacent"
        ),
        "target_comparator_has_positive_success_headroom_at_fork": (
            not target_success
        ),
        "authentic_is_nonnegative_vs_target_on_development": (
            authentic_success or not target_success
        ),
        "authentic_strictly_beats_availability_control_on_development": (
            authentic_success and not availability_success
        ),
        "zero_oracle_use_in_acquisition_and_selection": (
            acquisition["zero_source_input"] is True
            and acquisition["zero_policy_oracle_scorecard_use"] is True
            and matched["policy_runtime_saw_oracle_scorecard"] is False
        ),
    }
    qualifying_gates = {
        key: value for key, value in gates.items()
        if key != "v22_incompatible_space_sick_normal_route_removed"
    }
    allowed = all(qualifying_gates.values())
    body = {
        "schema_version": "discoveryworld-normal-v25-qualification-audit-v1",
        "status": (
            "DISCOVERYWORLD_NORMAL_FORMAL_AUTHORIZED"
            if allowed else
            "DISCOVERYWORLD_NORMAL_FORMAL_REMAINS_BLOCKED"
        ),
        "formal_reserve_authorized": allowed,
        "claim_boundary": (
            "Consumed-development qualification only. The target-native "
            "Proteomics Normal acquisition repair reached and solved the final "
            "commit on both development tasks, but the single matched commit "
            "fork had no target headroom and authentic source transfer was "
            "negative. No fresh Normal task may be opened from this audit."
        ),
        "development_evidence": {
            "tasks": development_tasks,
            "commit_coverage": int(acquisition["commit_coverage"]),
            "official_successes": acquisition_successes,
            "matched_forks": matched_forks,
            "target_native_myopic_successes": int(target_success),
            "authentic_source_successes": int(authentic_success),
            "availability_control_successes": int(availability_success),
            "binding_relation": binding[
                "commit_subject_relation_to_target"
            ],
            "binding_distance": binding["target_distance"],
        },
        "gates": gates,
        "failed_qualification_gates": sorted(
            key for key, value in qualifying_gates.items() if not value
        ),
        "next_legal_experiment": {
            "target_interface": "Proteomics Normal only",
            "intervention_point": (
                "an earlier measurement/tool-acquisition decision with "
                "demonstrated target-policy headroom, not the final DROP"
            ),
            "required_repairs": [
                "learn an adjacent relation in the target-native vocabulary",
                "collect at least eight source-blind development tasks",
                "obtain at least six matched intervention forks",
                "freeze a neural target grounder before fresh target reset",
                "require nonnegative authentic utility and strict source-control separation",
            ],
            "unchanged_components": [
                "source-only programs",
                "anonymous structural IR",
                "source-permuted and generic controls",
                "unified fail-closed authority chain",
            ],
        },
        "integrity": {
            "v22_file_sha256": _sha(V22),
            "v22_summary_sha256": v22["summary_sha256"],
            "acquisition_file_sha256": _sha(ACQUISITION),
            "acquisition_summary_sha256": acquisition["summary_sha256"],
            "matched_file_sha256": _sha(MATCHED),
            "matched_result_sha256": matched["result_sha256"],
        },
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / (
            "docs/results/discoveryworld_normal_v25_qualification_stop.json"
        ),
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "formal_reserve_authorized": report["formal_reserve_authorized"],
        "development_evidence": report["development_evidence"],
        "failed_qualification_gates": report["failed_qualification_gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
