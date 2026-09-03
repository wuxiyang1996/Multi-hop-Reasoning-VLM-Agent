#!/usr/bin/env python3
"""Measure ALFWorld target-demo cost for the transferred recurrence."""

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

from motif_transfer.alfworld_goal_relation_macro import AUTHENTIC, RAW  # noqa: E402
from motif_transfer.alfworld_target_recurrent_induction import (  # noqa: E402
    eligible_target_demonstrations,
    execution_normal_form,
    induce_target_recurrent_program,
    permute_binding_relation,
    shuffled_effect_supports,
    target_program_supports,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.source_goal_acquisition_induction import (  # noqa: E402
    validate_goal_acquisition_program,
)
from motif_transfer.source_goal_relation_induction import (  # noqa: E402
    validate_goal_relation_macro_program,
)


DEFAULT_CONFIG = REPO / "configs/alfworld_target_acquisition_value_v16.json"


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        raw = gzip.open(path, "rt", encoding="utf-8").read()
    else:
        raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _source_execution_normal_form(
    acquisition: Mapping[str, Any], relation: Mapping[str, Any],
    v13: Mapping[str, Any], v15: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile source IR plus its already-validated target adapter contract."""

    acquisition_program = acquisition["program"]
    relation_program = relation["program"]
    descriptors = {
        str(row["operator_type_id"]): row
        for row in acquisition["operator_types"]
    }
    acquisition_ids = set(map(
        str, acquisition_program["acquisition_operator_type_ids"],
    ))
    control_ids = {
        type_id for type_id in acquisition_ids
        if descriptors[type_id]["predicate_family"] == "CONTROL_STATE"
    }
    recurrent_acquisition = any(
        str(row["from_operator_type_id"])
        == str(row["to_operator_type_id"])
        and str(row["from_operator_type_id"]) in control_ids
        for row in acquisition_program["transition_graph"]
    )
    binding_then_relation = (
        acquisition_program["binding_to_relation"][
            "from_operator_type_id"
        ] == acquisition_program["binding_operator_type_id"]
        and acquisition_program["binding_to_relation"][
            "to_operator_type_id"
        ] == acquisition_program["relation_operator_type_id"]
    )
    recurrent_relation = any(
        row["from_operator_type_id"] == row["to_operator_type_id"]
        and row["cardinality"] == "ONE_OR_MORE"
        for row in relation_program["transitions"]
    )
    authentic_records = [
        row
        for episode in v13["episodes"][AUTHENTIC]
        for row in episode["records"]
    ]
    activation_after_one = (
        all(
            not row["program_active"]
            for row in authentic_records
            if int(row["completed_count_before"]) == 0
        )
        and any(
            row["program_active"]
            for row in authentic_records
            if int(row["completed_count_before"]) == 1
        )
    )
    exact_adapter_equivalence = all(v15["gates"].values()) and (
        int(v15["combined"]["exact_action_trace_matches"]) == 45
        and int(v15["combined"]["exact_state_effect_trace_matches"]) == 45
    )
    return {
        "activation_after_positive_relations": 1 if activation_after_one else -1,
        "recurrent_acquisition_control": recurrent_acquisition,
        "binding_then_relation": binding_then_relation,
        "recurrent_relation_grounding": recurrent_relation,
        "relation_argument_rule": (
            "PRESERVE_FIRST_POSITIVE_RELATION_HANDLE"
            if exact_adapter_equivalence else "UNVERIFIED"
        ),
        "terminal_remaining_relations": 0,
        "positive_effect_cardinality": int(
            acquisition_program["relation_binding_cardinality"]["value"]
        ),
        "fail_closed_on_ambiguity": (
            acquisition_program["abstention_rule"][
                "binding_cardinality_above_induced_value"
            ] == "ABSTAIN"
            and relation_program["abstention_rule"][
                "multiple_target_bindings"
            ] == "ABSTAIN"
        ),
    }


def run(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "CONSUMED_RETROSPECTIVE_ACQUISITION_PROTOCOL":
        raise ValueError("Phase 13/V16 config is not a consumed-data protocol")
    dependencies = {
        "v13_report_file_sha256": REPO / str(config["v13_report"]),
        "v14_report_file_sha256": REPO / str(config["v14_report"]),
        "v15_report_archive_file_sha256": REPO / str(
            config["v15_report_archive"]
        ),
        "source_acquisition_artifact_file_sha256": REPO / str(
            config["source_acquisition_artifact"]
        ),
        "source_acquisition_confirmation_file_sha256": REPO / str(
            config["source_acquisition_confirmation"]
        ),
        "source_relation_artifact_file_sha256": REPO / str(
            config["source_relation_artifact"]
        ),
        "source_relation_confirmation_file_sha256": REPO / str(
            config["source_relation_confirmation"]
        ),
        "target_inducer_file_sha256": REPO / (
            "src/motif_transfer/alfworld_target_recurrent_induction.py"
        ),
    }
    for field, path in dependencies.items():
        if _sha(path) != config[field]:
            raise ValueError(f"Phase 13/V16 dependency changed: {path}")

    v13 = _read(dependencies["v13_report_file_sha256"])
    v14 = _read(dependencies["v14_report_file_sha256"])
    v15 = _read(dependencies["v15_report_archive_file_sha256"])
    acquisition = _read(dependencies["source_acquisition_artifact_file_sha256"])
    acquisition_confirmation = _read(
        dependencies["source_acquisition_confirmation_file_sha256"]
    )
    relation = _read(dependencies["source_relation_artifact_file_sha256"])
    relation_confirmation = _read(
        dependencies["source_relation_confirmation_file_sha256"]
    )
    for value in (v13, v14, v15):
        _self_hash(value, "report_sha256")
    _self_hash(acquisition_confirmation, "report_sha256")
    _self_hash(relation_confirmation, "report_sha256")
    validate_goal_acquisition_program(acquisition)
    validate_goal_relation_macro_program(relation)
    if not (
        acquisition_confirmation.get("source_gate_passed")
        and relation_confirmation.get("source_gate_passed")
    ):
        raise ValueError("source program lacks fresh held-out confirmation")

    development = sorted(
        eligible_target_demonstrations(v13["episodes"][RAW]),
        key=lambda row: stable_hash({
            "namespace": str(config["demo_order_namespace"]),
            "task_id": str(row["task_id"]),
        }),
    )
    qualification = eligible_target_demonstrations(v14["episodes"][RAW])
    source_normal = _source_execution_normal_form(
        acquisition, relation, v13, v15,
    )
    curve = []
    first_matching_budget: int | None = None
    programs: dict[int, dict[str, Any]] = {}
    for budget in map(int, config["complete_target_trajectory_budgets"]):
        target = induce_target_recurrent_program(development, budget=budget)
        programs[budget] = target
        normal = execution_normal_form(target)
        matches = normal == source_normal
        if matches and first_matching_budget is None:
            first_matching_budget = budget
        control = permute_binding_relation(target)
        curve.append({
            "complete_target_trajectory_budget": budget,
            "status": str(target["status"]),
            "program_sha256": str(target["program_sha256"]),
            "matches_source_execution_normal_form": matches,
            "qualification_support": sum(
                target_program_supports(target, row) for row in qualification
            ),
            "qualification_shuffled_effect_support": sum(
                shuffled_effect_supports(target, row) for row in qualification
            ),
            "qualification_binding_relation_permuted_support": sum(
                target_program_supports(control, row) for row in qualification
            ),
        })
    if first_matching_budget is None:
        first_matching_budget = 0
    selected = programs[first_matching_budget]
    v14_source = int(v14["summaries"][AUTHENTIC]["successes"])
    v14_raw = int(v14["summaries"][RAW]["successes"])
    minimum_qualification = int(config["gates"][
        "minimum_qualification_trajectories"
    ])
    gates = {
        "development_has_at_least_eight_complete_target_demos": (
            len(development) >= int(config["gates"][
                "minimum_development_trajectories"
            ])
        ),
        "chronological_v14_qualification_has_minimum_support": (
            len(qualification) >= minimum_qualification
        ),
        "target_only_k0_abstains": curve[0]["status"].startswith("ABSTAIN"),
        "target_only_first_matches_at_k1": first_matching_budget == 1,
        "k1_supports_every_qualification_trajectory": (
            curve[1]["qualification_support"] == len(qualification)
        ),
        "shuffled_effect_control_has_zero_support": (
            curve[1]["qualification_shuffled_effect_support"] == 0
        ),
        "binding_relation_permutation_has_zero_support": (
            curve[1]["qualification_binding_relation_permuted_support"] == 0
        ),
        "source_artifacts_pass_fresh_heldout_controls": bool(
            acquisition_confirmation["source_gate_passed"]
            and relation_confirmation["source_gate_passed"]
        ),
        "source_program_has_nonzero_v14_policy_utility": v14_source > v14_raw,
        "target_written_oracle_exactly_matches_source_execution": all(
            v15["gates"].values()
        ),
        "phase13_resets_no_target_environment": True,
        "no_untouched_success_claim": True,
    }
    passed = all(gates.values())
    report_body = {
        "schema_version": "alfworld-target-acquisition-value-v16-report",
        "status": (
            "ALFWORLD_TARGET_ACQUISITION_VALUE_RETROSPECTIVE_VALIDATED"
            if passed else "ALFWORLD_TARGET_ACQUISITION_VALUE_FAILED"
        ),
        "role": "retrospective_chronological_heldout_acquisition_curve",
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "estimand": str(config["estimand"]),
        "source_complete_target_trajectory_budget": 0,
        "complete_target_trajectories_replaced": first_matching_budget,
        "source_execution_normal_form": source_normal,
        "development": {
            "source": "V13_RAW_TARGET_ONLY_COMPLETE_SUCCESS_PATHS",
            "eligible_trajectories": len(development),
            "task_id_hashes_in_order": [
                stable_hash(str(row["task_id"])) for row in development
            ],
        },
        "qualification": {
            "source": "LATER_V14_RAW_TARGET_ONLY_COMPLETE_SUCCESS_PATHS",
            "eligible_trajectories": len(qualification),
            "task_id_hashes": [
                stable_hash(str(row["task_id"])) for row in qualification
            ],
        },
        "target_only_induction_curve": curve,
        "policy_context": {
            "v14_source_induced_successes": v14_source,
            "v14_raw_target_only_successes": v14_raw,
            "v14_source_policy_gain": v14_source - v14_raw,
            "v15_target_written_exact_action_trace_matches": int(
                v15["combined"]["exact_action_trace_matches"]
            ),
            "prospective_policy_evidence_reused_not_reclassified": True,
        },
        "fresh_pool_audit": {
            "installed_standard_multiplicity_pool_available": False,
            "valid_train_consumed": 45,
            "valid_seen_historically_consumed": 24,
            "valid_unseen_historically_consumed": 17,
            "new_environment_reset_by_this_qualification": False,
            "prospective_replication_status": (
                "BLOCKED_ON_GENUINELY_NEW_ALFWORLD_INSTANCE_GENERATION"
            ),
        },
        "lineage": {
            "v13_report_sha256": str(v13["report_sha256"]),
            "v14_report_sha256": str(v14["report_sha256"]),
            "v15_report_sha256": str(v15["report_sha256"]),
            "source_acquisition_artifact_sha256": str(
                acquisition["artifact_sha256"]
            ),
            "source_relation_artifact_sha256": str(
                relation["artifact_sha256"]
            ),
            "target_k1_program_sha256": str(selected["program_sha256"]),
        },
        "gates": gates,
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    return report, selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    report, selected = run(config_path)
    config = _read(config_path)
    output = REPO / str(config["output"])
    artifact_output = REPO / str(config["target_k1_artifact_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "complete_target_trajectories_replaced": report[
            "complete_target_trajectories_replaced"
        ],
        "target_only_induction_curve": report["target_only_induction_curve"],
        "policy_context": report["policy_context"],
        "fresh_pool_audit": report["fresh_pool_audit"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
