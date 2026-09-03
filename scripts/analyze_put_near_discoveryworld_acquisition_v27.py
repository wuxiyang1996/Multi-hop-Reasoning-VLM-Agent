#!/usr/bin/env python3
"""Audit a second, finite program family for source acquisition value."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    StructuralPath,
    induce_structural_program,
    sequence_contains,
    validate_structural_program,
)
from motif_transfer.target_structural_induction import (  # noqa: E402
    induce_target_partial_order_program,
    source_sequence_support,
    target_program_supports,
    validate_target_program,
)


DEFAULT_CONFIG = (
    REPO / "configs/put_near_discoveryworld_acquisition_v27.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def _validate_portable_receipt(receipt: Mapping[str, Any]) -> None:
    _self_hash(receipt, "receipt_sha256")
    if receipt.get("target_data_read") is not False:
        raise ValueError("target data leaked into PutNear source receipt")
    if receipt.get("raw_source_action_tokens_exported") is not False:
        raise ValueError("source action token leaked into PutNear receipt")
    for collections in receipt.get("roles", {}).values():
        for collection in collections:
            for path in collection.get("paths") or ():
                _self_hash(path, "path_sha256")
                for step in path.get("steps") or ():
                    _self_hash(step, "transition_sha256")


def _paths(
    collections: Sequence[Mapping[str, Any]], *, split: str,
) -> tuple[StructuralPath, ...]:
    return tuple(
        StructuralPath(
            split=split,
            success=bool(path["success"]),
            steps=tuple(path["steps"]),
        )
        for collection in collections
        for path in collection["paths"]
    )


def source_finite_normal_form(program: Mapping[str, Any]) -> dict[str, Any]:
    validate_structural_program(program)
    descriptors = {
        str(row["operator_type_id"]): row["operator_type_descriptor"]
        for row in program["operators"]
    }
    return {
        "ir_kind": "FINITE_STRUCTURAL_DELTA_SEQUENCE",
        "recurrent": False,
        "operator_sequence": [
            {
                key: descriptors[str(type_id)][key]
                for key in (
                    "operation", "predicate_family", "arity", "value_kind"
                )
            }
            for type_id in program["induced_sequence"]
        ],
        "terminal_rule": "TERMINAL_OUTCOME_CHECK_AFTER_SEQUENCE",
        "fail_closed": all(
            value in {"ABSTAIN", "ABSTAIN_TO_TARGET_POLICY"}
            for value in program["abstention_rule"].values()
        ),
    }


def target_partial_order_normal_form(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    validate_target_program(program)
    return {
        "operator_minimum_counts": {
            str(row["operator_type_id"]): int(row["minimum_count"])
            for row in program["operator_requirements"]
        },
        "precedence_edges": sorted(
            (str(row["before"]), str(row["after"]))
            for row in program["precedence_edges"]
        ),
        "terminal_rule": str(program["terminal_rule"]),
        "fail_closed": all(
            value == "ABSTAIN"
            for value in program["abstention_rule"].values()
        ),
    }


def target_contains_source_subprogram(
    target: Mapping[str, Any], source: Mapping[str, Any],
) -> bool:
    validate_target_program(target)
    validate_structural_program(source)
    sequence = list(map(str, source["induced_sequence"]))
    if len(sequence) != 2:
        return False
    requirements = {
        str(row["operator_type_id"]): int(row["minimum_count"])
        for row in target["operator_requirements"]
    }
    edges = {
        (str(row["before"]), str(row["after"]))
        for row in target["precedence_edges"]
    }
    return (
        all(requirements.get(type_id, 0) >= 1 for type_id in sequence)
        and tuple(sequence) in edges
    )


def _permuted_precedence(
    program: Mapping[str, Any], source_sequence: Sequence[str],
) -> dict[str, Any]:
    validate_target_program(program)
    left, right = map(str, source_sequence)
    value = deepcopy(dict(program))
    value.pop("program_sha256", None)
    changed = False
    for edge in value["precedence_edges"]:
        if (str(edge["before"]), str(edge["after"])) == (left, right):
            edge["before"], edge["after"] = right, left
            changed = True
    if not changed:
        raise ValueError("target program omitted the finite source precedence")
    value["control_kind"] = "SOURCE_SUBPROGRAM_PRECEDENCE_REVERSED"
    return value | {"program_sha256": stable_hash(value)}


def source_analysis(
    receipt: Mapping[str, Any], frozen_source: Mapping[str, Any],
    fresh_report: Mapping[str, Any], *, order_namespace: str,
) -> dict[str, Any]:
    _validate_portable_receipt(receipt)
    validate_structural_program(frozen_source)
    discovery = sorted(
        receipt["roles"]["discovery"],
        key=lambda row: stable_hash({
            "namespace": order_namespace,
            "seed": int(row["seed"]),
        }),
    )
    qualification = list(receipt["roles"]["source_qualification"])
    fresh = list(receipt["roles"]["fresh_confirmation"])
    curve = [{
        "complete_source_intervention_collections": 0,
        "status": "ABSTAIN_NO_SOURCE_SUCCESS_PATH",
        "matches_frozen_source_normal_form": False,
    }]
    first = None
    for budget in range(1, len(discovery) + 1):
        selected = discovery[:budget]
        induction_paths = (
            *_paths(selected, split="development"),
            *_paths(qualification, split="qualification"),
        )
        hashes = [
            str(row["collection_sha256"])
            for row in (*selected, *qualification)
        ]
        program = induce_structural_program(
            induction_paths,
            source_receipts_sha256=stable_hash(hashes),
        )
        validate_structural_program(program)
        structural_match = (
            program["status"] == "SOURCE_STRUCTURAL_PROGRAM_QUALIFIED"
            and source_finite_normal_form(program)
            == source_finite_normal_form(frozen_source)
        )
        receipt_row = {
            "complete_source_intervention_collections": budget,
            "status": str(program["status"]),
            "program_sha256": str(program["program_sha256"]),
            "matches_frozen_source_normal_form": structural_match,
            "exact_artifact_hash_match": (
                program["program_sha256"] == frozen_source["program_sha256"]
            ),
            "induced_sequence": list(program["induced_sequence"]),
            "qualification_gates": dict(program["qualification_gates"]),
            "retained_discovery_success_path_transitions": sum(
                len(path["steps"])
                for collection in selected
                for path in collection["paths"]
                if path["success"]
            ),
            "fixed_source_qualification_collections": len(qualification),
        }
        curve.append(receipt_row)
        if structural_match and first is None:
            first = receipt_row

    fresh_success = [
        StructuralPath(
            split="qualification", success=True, steps=tuple(path["steps"])
        )
        for collection in fresh
        for path in collection["paths"]
        if path["success"]
    ]
    sequence = list(map(str, frozen_source["induced_sequence"]))
    reversed_sequence = list(reversed(sequence))
    fresh_lineage = next(
        row for row in fresh_report["lineages"]
        if row["task_id"] == "put_near"
    )
    return {
        "curve": curve,
        "first_structurally_complete_budget": (
            first["complete_source_intervention_collections"]
            if first else None
        ),
        "first_structurally_complete_receipt": first,
        "frozen_normal_form": source_finite_normal_form(frozen_source),
        "source_qualification_collections_always_visible": len(qualification),
        "fresh_confirmation": {
            "collections": len(fresh),
            "success_paths": len(fresh_success),
            "authentic_sequence_support": sum(
                sequence_contains(row.effects, sequence)
                for row in fresh_success
            ),
            "reversed_sequence_support": sum(
                sequence_contains(row.effects, reversed_sequence)
                for row in fresh_success
            ),
            "authentic_correct_bindings": int(
                fresh_lineage["grounding"]["authentic_correct_bindings"]
            ),
            "shuffled_correct_bindings": int(
                fresh_lineage["grounding"]["shuffled_correct_bindings"]
            ),
            "wrong_family_sequence_support": int(
                fresh_lineage["permuted_sequence_supported"]
            ),
        },
        "source_target_data_read": bool(frozen_source["target_data_read"]),
        "source_action_identity_exported": bool(
            frozen_source["source_action_identity_exported"]
        ),
        "source_task_identity_used_as_feature": bool(
            frozen_source["source_task_identity_used_as_feature"]
        ),
    }


def target_analysis(
    target_report: Mapping[str, Any], source: Mapping[str, Any],
    wrong_family: Mapping[str, Any],
) -> dict[str, Any]:
    _self_hash(target_report, "report_sha256")
    validate_structural_program(source)
    validate_structural_program(wrong_family)
    development = list(target_report["development_target_sequences"])
    qualification = list(target_report["qualification_target_sequences"])
    source_sequence = list(map(str, source["induced_sequence"]))
    wrong_sequence = list(map(str, wrong_family["induced_sequence"]))
    receipts = []
    normals = []
    for index, sequence in enumerate(development):
        program = induce_target_partial_order_program(
            (sequence,),
            development_receipts_sha256=stable_hash({
                "target_only_complete_path_index": index,
                "sequence": sequence,
            }),
        )
        normal = target_partial_order_normal_form(program)
        normals.append(normal)
        control = _permuted_precedence(program, source_sequence)
        receipts.append({
            "complete_target_trajectory_budget": 1,
            "development_path_index": index,
            "substantive_structural_events": len(sequence),
            "program_sha256": str(program["program_sha256"]),
            "source_program_copied_as_target_body": bool(
                program["source_program_copied_as_target_body"]
            ),
            "contains_source_finite_subprogram": (
                target_contains_source_subprogram(program, source)
            ),
            "heldout_target_program_support": sum(
                target_program_supports(program, row)
                for row in qualification
            ),
            "heldout_precedence_permuted_support": sum(
                target_program_supports(control, row)
                for row in qualification
            ),
            "heldout_source_sequence_support": sum(
                source_sequence_support(source_sequence, row)
                for row in qualification
            ),
            "heldout_reversed_source_sequence_support": sum(
                source_sequence_support(
                    list(reversed(source_sequence)), row,
                )
                for row in qualification
            ),
            "heldout_wrong_family_sequence_support": sum(
                source_sequence_support(wrong_sequence, row)
                for row in qualification
            ),
            "semantic_normal_form": normal,
        })
    return {
        "target_k0": {
            "complete_target_trajectory_budget": 0,
            "status": "ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY",
            "heldout_support": 0,
        },
        "development_complete_structural_paths": len(development),
        "heldout_complete_structural_paths": len(qualification),
        "single_demo_robustness": {
            "independent_single_demo_programs": len(receipts),
            "all_semantic_normal_forms_equal": len({
                stable_hash(row) for row in normals
            }) == 1,
            "all_contain_source_finite_subprogram": all(
                row["contains_source_finite_subprogram"] for row in receipts
            ),
            "all_support_every_heldout_path": all(
                row["heldout_target_program_support"] == len(qualification)
                for row in receipts
            ),
            "all_precedence_permuted_support_zero": all(
                row["heldout_precedence_permuted_support"] == 0
                for row in receipts
            ),
            "all_reversed_and_wrong_family_support_zero": all(
                row["heldout_reversed_source_sequence_support"] == 0
                and row["heldout_wrong_family_sequence_support"] == 0
                for row in receipts
            ),
            "receipts": receipts,
        },
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_RETROSPECTIVE_SECOND_FAMILY_PROTOCOL":
        raise ValueError("second-family acquisition protocol is not frozen")
    dependencies = {
        field: REPO / str(config[path_field])
        for field, path_field in config["dependency_fields"].items()
    }
    for hash_field, path in dependencies.items():
        if _sha(path) != config[hash_field]:
            raise ValueError(f"dependency changed: {path}")

    receipt = _read(REPO / str(config["source_portable_receipt"]))
    source_program = _read(REPO / str(config["source_program"]))
    wrong_program = _read(REPO / str(config["wrong_family_program"]))
    fresh_report = _read(REPO / str(config["source_fresh_report"]))
    target_report = _read(REPO / str(config["target_development_report"]))
    formal = _read(REPO / str(config["target_formal_report"]))
    heterogeneity = _read(REPO / str(config["phase9_heterogeneity_report"]))
    phase14 = _read(REPO / str(config["phase14_acquisition_report"]))
    for report in (fresh_report, formal, heterogeneity, phase14):
        field = (
            "report_sha256" if "report_sha256" in report
            else "summary_sha256"
        )
        _self_hash(report, field)

    source = source_analysis(
        receipt, source_program, fresh_report,
        order_namespace=str(config["source_order_namespace"]),
    )
    target = target_analysis(target_report, source_program, wrong_program)
    single = target["single_demo_robustness"]
    route = next(
        row for row in heterogeneity["route_audits"]
        if row["route_id"]
        == "minigrid-put-near-to-discoveryworld-easy-v1"
    )
    selection = route["selection"]
    formal_source = int(formal["condition_successes"]["source_induced"])
    formal_neural = int(formal["condition_successes"]["neural_only"])
    formal_permuted = int(formal["condition_successes"]["source_permuted"])
    phase14_normal = phase14["source"][
        "reference_full_source_normal_form"
    ]
    family_distinction = {
        "put_near_ir_kind": source["frozen_normal_form"]["ir_kind"],
        "put_near_recurrent": source["frozen_normal_form"]["recurrent"],
        "put_near_operator_sequence": source["frozen_normal_form"][
            "operator_sequence"
        ],
        "alfworld_recurrent_acquisition_control": bool(
            phase14_normal["recurrent_acquisition_control"]
        ),
        "alfworld_recurrent_relation_update": bool(
            phase14_normal["recurrent_relation_update"]
        ),
        "same_program_family": False,
    }
    gates = {
        "source_k0_abstains": source["curve"][0]["status"].startswith(
            "ABSTAIN"
        ),
        "source_k1_fails_frozen_minimum_support": (
            source["curve"][1]["status"]
            == "SOURCE_STRUCTURAL_PROGRAM_ABSTAINING"
            and source["curve"][1]["qualification_gates"][
                "minimum_discovery_success_paths"
            ] is False
        ),
        "source_k2_recovers_frozen_program_normal_form": (
            source["first_structurally_complete_budget"] == 2
        ),
        "source_induction_uses_no_target_identity_or_action_tokens": not any((
            source["source_target_data_read"],
            source["source_action_identity_exported"],
            source["source_task_identity_used_as_feature"],
        )),
        "source_fresh_success_paths_all_support_program": (
            source["fresh_confirmation"]["authentic_sequence_support"]
            == source["fresh_confirmation"]["success_paths"] == 4
        ),
        "source_fresh_controls_have_zero_support": (
            source["fresh_confirmation"]["reversed_sequence_support"] == 0
            and source["fresh_confirmation"][
                "wrong_family_sequence_support"
            ] == 0
            and source["fresh_confirmation"][
                "shuffled_correct_bindings"
            ] == 0
        ),
        "target_k0_abstains": target["target_k0"]["status"].startswith(
            "ABSTAIN"
        ),
        "every_target_k1_recovers_same_semantic_program": bool(
            single["all_semantic_normal_forms_equal"]
        ),
        "every_target_k1_contains_source_subprogram": bool(
            single["all_contain_source_finite_subprogram"]
        ),
        "every_target_k1_supports_all_heldout_paths": bool(
            single["all_support_every_heldout_path"]
        ),
        "target_precedence_and_wrong_family_controls_zero": bool(
            single["all_precedence_permuted_support_zero"]
            and single["all_reversed_and_wrong_family_support_zero"]
        ),
        "anonymous_selector_selects_put_near_and_rejects_wrong_family": (
            selection["selected_program_sha256"]
            == source_program["program_sha256"]
            and selection["source_identity_used_as_feature"] is False
            and route["wrong_family_selection"]["selected_program_sha256"]
            is None
        ),
        "existing_formal_utility_is_positive_and_control_specific": (
            formal_source == 12
            and formal_source > formal_neural == formal_permuted == 3
            and all(formal["gates"].values())
        ),
        "finite_family_is_distinct_from_phase14_recurrence": (
            family_distinction["put_near_recurrent"] is False
            and family_distinction["alfworld_recurrent_acquisition_control"]
            and family_distinction["alfworld_recurrent_relation_update"]
        ),
        "no_new_target_execution_or_success_claim": True,
    }
    passed = all(gates.values())
    body = {
        "schema_version": (
            "put-near-discoveryworld-acquisition-v27-report"
        ),
        "status": (
            "SECOND_DISTINCT_PROGRAM_FAMILY_ACQUISITION_VALIDATED"
            if passed else "SECOND_PROGRAM_FAMILY_ACQUISITION_FAILED"
        ),
        "role": "retrospective_second_program_family_acquisition_audit",
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "answer": (
            "The acquisition result is not confined to the recurrent ALFWorld "
            "controller. A distinct finite ADD-slot then REMOVE-slot program "
            "is source-induced at K=2 source discovery collections, while a "
            "source-blind target learner abstains at K=0 and every one of "
            "three independent complete target structural paths recovers a "
            "target program containing the same ordered subprogram at K=1."
        ),
        "source": source,
        "target": target,
        "family_distinction": family_distinction,
        "existing_prospective_formal_context": {
            "tasks": int(formal["tasks"]),
            "source_induced_successes": formal_source,
            "neural_only_successes": formal_neural,
            "source_permuted_successes": formal_permuted,
            "source_vs_neural": dict(formal["source_vs_neural"]),
            "prospective_evidence_reused_not_reclassified": True,
        },
        "cost_boundary": {
            "source_first_structurally_complete_discovery_collections": source[
                "first_structurally_complete_budget"
            ],
            "fixed_source_qualification_collections": source[
                "source_qualification_collections_always_visible"
            ],
            "target_complete_structural_path_budget": 1,
            "target_substantive_events_per_path": sorted({
                row["substantive_structural_events"]
                for row in single["receipts"]
            }),
            "common_cost_unit_available": False,
            "reason": (
                "Source collections include exhaustive intervention forks, "
                "whereas the portable receipt retains the selected structural "
                "paths; target entries are complete domain trajectories "
                "represented by substantive event sequences."
            ),
        },
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    report = run(config_path)
    config = _read(config_path)
    output = REPO / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "source_first_budget": report["source"][
            "first_structurally_complete_budget"
        ],
        "target_single_demo_programs": report["target"][
            "single_demo_robustness"
        ]["independent_single_demo_programs"],
        "existing_formal": report["existing_prospective_formal_context"],
        "gates_passed": sum(report["gates"].values()),
        "gates_total": len(report["gates"]),
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
