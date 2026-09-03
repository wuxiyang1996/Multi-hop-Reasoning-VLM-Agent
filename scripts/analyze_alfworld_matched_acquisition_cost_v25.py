#!/usr/bin/env python3
"""Compare source and target acquisition evidence for the ALFWorld program.

This is a retrospective mechanism analysis.  It does not execute ALFWorld or
claim that a source simulator episode and a target environment trajectory have
the same cost.  It asks two narrower questions:

1. how much source-only intervention evidence is needed to recover the already
   validated structural program; and
2. whether *every* eligible single target demonstration can independently
   recover the same execution normal form and reject equal-size controls.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_goal_relation_macro import RAW  # noqa: E402
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
    confirm_goal_acquisition_program,
    induce_goal_acquisition_program,
)
from motif_transfer.source_goal_relation_causal_budget import (  # noqa: E402
    induce_causal_goal_relation_program,
)
from motif_transfer.source_goal_relation_induction import (  # noqa: E402
    confirm_goal_relation_macro_program,
)


DEFAULT_CONFIG = REPO / "configs/alfworld_matched_acquisition_cost_v25.json"


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


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["snapshot_id"]), str(row["episode_id"])


def _subset_dataset(
    dataset: Mapping[str, Any], *, collection: str,
    selected: set[tuple[str, str]], relation_sha256: str | None = None,
) -> dict[str, Any]:
    value = deepcopy(dict(dataset))
    value.pop("dataset_sha256", None)
    value[collection] = [
        row for row in value[collection] if _key(row) in selected
    ]
    if relation_sha256 is not None:
        value["relation_artifact_sha256"] = relation_sha256
    value["dataset_sha256"] = stable_hash(value)
    return value


def _ordered_keys(
    keys: Sequence[tuple[str, str]], namespace: str,
) -> list[tuple[str, str]]:
    return sorted(
        keys,
        key=lambda key: stable_hash({
            "namespace": namespace,
            "snapshot_id": key[0],
            "episode_id": key[1],
        }),
    )


def source_structural_normal_form(
    acquisition: Mapping[str, Any], relation: Mapping[str, Any],
) -> dict[str, Any]:
    """Project source artifacts onto target-independent execution semantics."""

    acquisition_program = acquisition["program"]
    relation_program = relation["program"]
    descriptors = {
        str(row["operator_type_id"]): row
        for row in acquisition["operator_types"]
    }
    control_ids = {
        str(type_id)
        for type_id in acquisition_program["acquisition_operator_type_ids"]
        if descriptors[str(type_id)]["predicate_family"] == "CONTROL_STATE"
    }
    terminal = list(relation_program["terminal_predicates"])
    return {
        "recurrent_acquisition_control": any(
            str(row["from_operator_type_id"])
            == str(row["to_operator_type_id"])
            and str(row["from_operator_type_id"]) in control_ids
            for row in acquisition_program["transition_graph"]
        ),
        "binding_then_relation": (
            acquisition_program["binding_to_relation"][
                "from_operator_type_id"
            ] == acquisition_program["binding_operator_type_id"]
            and acquisition_program["binding_to_relation"][
                "to_operator_type_id"
            ] == acquisition_program["relation_operator_type_id"]
        ),
        "recurrent_relation_update": any(
            row["from_operator_type_id"] == row["to_operator_type_id"]
            and row["cardinality"] == "ONE_OR_MORE"
            for row in relation_program["transitions"]
        ),
        "terminal_relation_coverage": (
            len(terminal) == 1
            and terminal[0].get("predicate_family")
            == "ENTITY_GOAL_RELATION"
            and terminal[0].get("operator") == "EQ"
            and float(terminal[0].get("value", -1)) == 1.0
        ),
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


def _source_cost(
    acquisition: Mapping[str, Any], relation: Mapping[str, Any],
    selected: set[tuple[str, str]],
) -> dict[str, int]:
    acquisition_rows = [
        row for row in acquisition["trajectories"] if _key(row) in selected
    ]
    relation_rows = [
        row for row in relation["episodes"] if _key(row) in selected
    ]
    successful_macros = [
        candidate
        for episode in relation_rows
        for candidate in episode["candidates"]
        if candidate["success_from_state_only"]
    ]
    return {
        "complete_source_intervention_episodes": len(selected),
        "observed_success_path_primitive_transitions": sum(
            len(row["transitions"]) for row in acquisition_rows
        ),
        "observed_success_path_relation_macro_transitions": sum(
            len(row["macro_tuples"]) for row in successful_macros
        ),
    }


def _source_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _read(REPO / str(config[path_field]))
        for name, path_field in {
            "discovery_acquisition": "source_discovery_acquisition",
            "fresh_acquisition": "source_fresh_acquisition",
            "discovery_relation": "source_discovery_relation",
            "fresh_relation": "source_fresh_relation",
        }.items()
    }


def _induce_source_budget(
    inputs: Mapping[str, Mapping[str, Any]],
    order: Sequence[tuple[str, str]], budget: int,
    *, confirm: bool,
) -> dict[str, Any]:
    selected = set(order[:budget])
    relation_dataset = _subset_dataset(
        inputs["discovery_relation"], collection="episodes",
        selected=selected,
    )
    relation = induce_causal_goal_relation_program(relation_dataset)
    acquisition_dataset = _subset_dataset(
        inputs["discovery_acquisition"], collection="trajectories",
        selected=selected, relation_sha256=str(relation["artifact_sha256"]),
    )
    acquisition = induce_goal_acquisition_program(acquisition_dataset)
    result: dict[str, Any] = {
        "normal_form": source_structural_normal_form(acquisition, relation),
        "cost": _source_cost(
            inputs["discovery_acquisition"],
            inputs["discovery_relation"], selected,
        ),
        "relation_artifact_sha256": str(relation["artifact_sha256"]),
        "acquisition_artifact_sha256": str(acquisition["artifact_sha256"]),
        "source_target_data_read": bool(
            relation.get("target_data_read")
            or acquisition.get("target_data_read")
        ),
        "terminal_candidate_authority": dict(
            relation["terminal_candidate_authority"]
        ),
    }
    if confirm:
        relation_report = confirm_goal_relation_macro_program(
            relation, inputs["fresh_relation"],
        )
        fresh_acquisition = _subset_dataset(
            inputs["fresh_acquisition"], collection="trajectories",
            selected={_key(row) for row in inputs["fresh_acquisition"][
                "trajectories"
            ]},
            relation_sha256=str(relation["artifact_sha256"]),
        )
        acquisition_report = confirm_goal_acquisition_program(
            acquisition, fresh_acquisition,
        )
        result["heldout"] = {
            "relation_episodes": int(
                relation_report["metrics"]["heldout_episodes"]
            ),
            "relation_gate_passed": bool(
                relation_report["source_gate_passed"]
            ),
            "relation_shuffled_effect_bindings": int(
                relation_report["metrics"]["shuffled_effect_bindings"]
            ),
            "acquisition_trajectories": int(
                acquisition_report["metrics"]["heldout_trajectories"]
            ),
            "acquisition_gate_passed": bool(
                acquisition_report["source_gate_passed"]
            ),
            "acquisition_shuffled_effect_bindings": int(
                acquisition_report["metrics"]["shuffled_effect_bindings"]
            ),
        }
    return result


def source_analysis(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _source_inputs(config)
    relation_keys = {_key(row) for row in inputs["discovery_relation"][
        "episodes"
    ]}
    acquisition_keys = {_key(row) for row in inputs[
        "discovery_acquisition"
    ]["trajectories"]}
    common = sorted(relation_keys & acquisition_keys)
    maximum = int(config["source_budget"]["maximum_episodes"])

    full_relation = induce_causal_goal_relation_program(
        inputs["discovery_relation"]
    )
    full_acquisition_dataset = _subset_dataset(
        inputs["discovery_acquisition"], collection="trajectories",
        selected=acquisition_keys,
        relation_sha256=str(full_relation["artifact_sha256"]),
    )
    full_acquisition = induce_goal_acquisition_program(
        full_acquisition_dataset
    )
    reference_normal = source_structural_normal_form(
        full_acquisition, full_relation,
    )

    primary_order = _ordered_keys(
        common, str(config["source_budget"]["primary_order_namespace"]),
    )
    curve: list[dict[str, Any]] = [{
        "complete_source_intervention_episodes": 0,
        "status": "ABSTAIN_NO_SOURCE_INTERVENTION_TRAJECTORY",
        "matches_full_source_normal_form": False,
    }]
    primary_first: dict[str, Any] | None = None
    for budget in range(1, maximum + 1):
        try:
            row = _induce_source_budget(
                inputs, primary_order, budget, confirm=True,
            )
            matches = row["normal_form"] == reference_normal
            heldout = row["heldout"]
            qualified = bool(
                matches
                and heldout["relation_gate_passed"]
                and heldout["acquisition_gate_passed"]
            )
            receipt = {
                "complete_source_intervention_episodes": budget,
                "status": (
                    "SOURCE_PROGRAM_HELDOUT_QUALIFIED" if qualified
                    else "SOURCE_PROGRAM_NOT_YET_QUALIFIED"
                ),
                "matches_full_source_normal_form": matches,
                **row,
            }
        except ValueError as error:
            receipt = {
                "complete_source_intervention_episodes": budget,
                "status": "ABSTAIN_INSUFFICIENT_SOURCE_STRUCTURE",
                "matches_full_source_normal_form": False,
                "diagnostic": str(error),
            }
            qualified = False
        curve.append(receipt)
        if qualified and primary_first is None:
            primary_first = receipt

    repeat_count = int(config["source_budget"]["robustness_repetitions"])
    prefix = str(config["source_budget"]["robustness_namespace_prefix"])
    robustness = []
    for repeat in range(repeat_count):
        namespace = f"{prefix}{repeat}"
        order = _ordered_keys(common, namespace)
        first: dict[str, Any] | None = None
        for budget in range(1, maximum + 1):
            try:
                row = _induce_source_budget(
                    inputs, order, budget, confirm=False,
                )
            except ValueError:
                continue
            if row["normal_form"] == reference_normal:
                first = {
                    "repeat": repeat,
                    "order_namespace": namespace,
                    "first_structurally_complete_budget": budget,
                    "cost": row["cost"],
                    "source_target_data_read": row[
                        "source_target_data_read"
                    ],
                }
                break
        robustness.append(first or {
            "repeat": repeat,
            "order_namespace": namespace,
            "first_structurally_complete_budget": None,
        })
    recovered = [
        int(row["first_structurally_complete_budget"])
        for row in robustness
        if row["first_structurally_complete_budget"] is not None
    ]
    summary = {
        "shared_discovery_episodes": len(common),
        "reference_full_source_normal_form": reference_normal,
        "primary_curve": curve,
        "primary_first_qualified": primary_first,
        "robustness": {
            "repetitions": repeat_count,
            "recovered_within_maximum": len(recovered),
            "first_budget_min": min(recovered) if recovered else None,
            "first_budget_median": median(recovered) if recovered else None,
            "first_budget_max": max(recovered) if recovered else None,
            "receipts": robustness,
            "qualification_note": (
                "Robustness orders recover the same semantic normal form. "
                "Held-out confirmation is executed on every primary budget; "
                "confirmation depends on this semantic form, not support "
                "counts or the order namespace."
            ),
        },
    }
    return summary, inputs


def target_analysis(config: Mapping[str, Any]) -> dict[str, Any]:
    v13 = _read(REPO / str(config["target_v13_report"]))
    v14 = _read(REPO / str(config["target_v14_report"]))
    v16 = _read(REPO / str(config["phase13_acquisition_report"]))
    _self_hash(v13, "report_sha256")
    _self_hash(v14, "report_sha256")
    _self_hash(v16, "report_sha256")
    expected = dict(v16["source_execution_normal_form"])
    development = sorted(
        eligible_target_demonstrations(v13["episodes"][RAW]),
        key=lambda row: stable_hash(str(row["task_id"])),
    )
    qualification = eligible_target_demonstrations(v14["episodes"][RAW])
    zero = induce_target_recurrent_program((), budget=0)
    receipts = []
    for episode in development:
        program = induce_target_recurrent_program((episode,), budget=1)
        control = permute_binding_relation(program)
        diagnostics = program["induction_diagnostics"]
        receipts.append({
            "task_id_sha256": stable_hash(str(episode["task_id"])),
            "full_episode_primitive_transitions": len(episode["records"]),
            "post_intervention_induction_transitions": (
                int(diagnostics["acquisition_control_steps"])
                + int(diagnostics["relation_grounding_steps"]) + 2
            ),
            "program_sha256": str(program["program_sha256"]),
            "matches_source_execution_normal_form": (
                execution_normal_form(program) == expected
            ),
            "heldout_support": sum(
                target_program_supports(program, row)
                for row in qualification
            ),
            "heldout_shuffled_effect_support": sum(
                shuffled_effect_supports(program, row)
                for row in qualification
            ),
            "heldout_binding_relation_permuted_support": sum(
                target_program_supports(control, row)
                for row in qualification
            ),
        })
    full_steps = [
        int(row["full_episode_primitive_transitions"]) for row in receipts
    ]
    induction_steps = [
        int(row["post_intervention_induction_transitions"])
        for row in receipts
    ]
    return {
        "development_eligible_complete_trajectories": len(development),
        "heldout_eligible_complete_trajectories": len(qualification),
        "target_k0": {
            "status": str(zero["status"]),
            "heldout_support": sum(
                target_program_supports(zero, row) for row in qualification
            ),
        },
        "single_demo_robustness": {
            "independent_single_demo_programs": len(receipts),
            "all_match_source_execution_normal_form": all(
                row["matches_source_execution_normal_form"]
                for row in receipts
            ),
            "all_support_every_heldout_trajectory": all(
                row["heldout_support"] == len(qualification)
                for row in receipts
            ),
            "all_shuffled_effect_support_zero": all(
                row["heldout_shuffled_effect_support"] == 0
                for row in receipts
            ),
            "all_binding_relation_permuted_support_zero": all(
                row["heldout_binding_relation_permuted_support"] == 0
                for row in receipts
            ),
            "full_episode_transition_min": min(full_steps),
            "full_episode_transition_median": median(full_steps),
            "full_episode_transition_max": max(full_steps),
            "post_intervention_transition_min": min(induction_steps),
            "post_intervention_transition_median": median(induction_steps),
            "post_intervention_transition_max": max(induction_steps),
            "receipts": receipts,
        },
        "source_execution_normal_form": expected,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_RETROSPECTIVE_MECHANISM_PROTOCOL":
        raise ValueError("matched acquisition protocol is not frozen")
    dependencies = {
        field: REPO / str(config[path_field])
        for field, path_field in config["dependency_fields"].items()
    }
    for hash_field, path in dependencies.items():
        if _sha(path) != config[hash_field]:
            raise ValueError(f"dependency changed: {path}")

    source, _ = source_analysis(config)
    target = target_analysis(config)
    primary = source["primary_first_qualified"]
    robustness = source["robustness"]
    single = target["single_demo_robustness"]
    gates = {
        "source_k0_abstains": (
            source["primary_curve"][0]["status"]
            == "ABSTAIN_NO_SOURCE_INTERVENTION_TRAJECTORY"
        ),
        "source_primary_qualifies_within_frozen_maximum": primary is not None,
        "source_primary_uses_no_target_data": bool(
            primary and not primary["source_target_data_read"]
        ),
        "source_terminal_feature_not_provided_by_name": bool(
            primary
            and primary["terminal_candidate_authority"][
                "named_terminal_feature_provided"
            ] is False
        ),
        "source_primary_passes_both_heldout_gates": bool(
            primary
            and primary["heldout"]["relation_gate_passed"]
            and primary["heldout"]["acquisition_gate_passed"]
        ),
        "source_primary_rejects_shuffled_effects": bool(
            primary
            and primary["heldout"]["relation_shuffled_effect_bindings"] == 0
            and primary["heldout"][
                "acquisition_shuffled_effect_bindings"
            ] == 0
        ),
        "all_source_orders_recover_within_frozen_maximum": (
            robustness["recovered_within_maximum"]
            == robustness["repetitions"]
        ),
        "target_k0_abstains": str(target["target_k0"]["status"]).startswith(
            "ABSTAIN"
        ) and target["target_k0"]["heldout_support"] == 0,
        "every_eligible_target_k1_matches_source": bool(
            single["all_match_source_execution_normal_form"]
        ),
        "every_target_k1_supports_all_heldout": bool(
            single["all_support_every_heldout_trajectory"]
        ),
        "all_target_controls_have_zero_support": bool(
            single["all_shuffled_effect_support_zero"]
            and single["all_binding_relation_permuted_support_zero"]
        ),
        "no_new_target_execution_or_success_claim": True,
        "cost_units_reported_without_false_equivalence": True,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "alfworld-matched-acquisition-cost-v25-report",
        "status": (
            "ALFWORLD_MATCHED_ACQUISITION_MECHANISM_VALIDATED"
            if passed else "ALFWORLD_MATCHED_ACQUISITION_MECHANISM_FAILED"
        ),
        "role": "retrospective_source_vs_target_acquisition_audit",
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "answer": (
            "An isomorphic target controller is sufficient once its content "
            "is supplied. Source interventions add acquisition value: they "
            "recover the same structural content without a complete target "
            "trajectory. This audit does not claim that source episodes and "
            "target trajectories have equal cost or that source is more "
            "sample-efficient under a common unit."
        ),
        "source": source,
        "target": target,
        "cost_comparison": {
            "source_primary": primary["cost"] if primary else None,
            "target_complete_trajectory_budget": 1,
            "target_full_episode_transition_range": [
                single["full_episode_transition_min"],
                single["full_episode_transition_max"],
            ],
            "target_post_intervention_transition_range": [
                single["post_intervention_transition_min"],
                single["post_intervention_transition_max"],
            ],
            "common_cost_unit_available": False,
            "reason": (
                "Source receipts retain successful intervention paths but do "
                "not retain all simulator candidate-fork primitive steps; "
                "source simulator and target environment interactions also "
                "have different costs."
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
        "source_first_budget": (
            report["source"]["primary_first_qualified"] or {}
        ).get("complete_source_intervention_episodes"),
        "source_robustness": {
            key: report["source"]["robustness"][key]
            for key in (
                "recovered_within_maximum", "first_budget_min",
                "first_budget_median", "first_budget_max",
            )
        },
        "target_single_demo_programs": report["target"][
            "single_demo_robustness"
        ]["independent_single_demo_programs"],
        "gates_passed": sum(report["gates"].values()),
        "gates_total": len(report["gates"]),
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
