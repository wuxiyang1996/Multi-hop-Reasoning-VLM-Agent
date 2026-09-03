from __future__ import annotations

import json

from motif_transfer.agqa_layer_b_harness import (
    plan_harness_arm, semantic_operator_plan, source_permuted_compositions,
)
from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target, serialize_compact_semantic_target
from motif_transfer.contracts import stable_hash


PROGRAM="Query(class, OnlyItem(Iterate(Localize(before, holding a broom), Filter(frame, [relations, carrying, objects]))))"


def receipt():
    return parse_compact_semantic_target(
        serialize_compact_semantic_target(PROGRAM),task_id="A0",question_sha256=stable_hash("q"),
        parser_sha256=stable_hash("parser"),parser_training_authority="DEV_ONLY",
    )


def capabilities():
    return json.load(open("runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json"))


def test_semantic_planner_requires_transferable_structure_without_reading_program_at_runtime():
    operators,edges=semantic_operator_plan(receipt())
    assert {"PROJECT","UNIQUE","INTERVAL_OF","TEMPORAL_SELECT","FILTER_EQ"} <= set(operators)
    assert ("TEMPORAL_SELECT","UNIQUE") in set(edges)


def test_five_arms_are_matched_and_generic_is_not_crippled():
    semantic=receipt(); source=capabilities(); all_ops=sorted(source["capabilities"])
    plans={arm:plan_harness_arm(semantic,arm=arm,source_capabilities=source,all_vm_operators=all_ops)
           for arm in ("neural_only","generic_scaffold","source_permuted","source_induced","target_written_isomorphic")}
    assert plans["neural_only"].status=="ABSTAINED"
    assert plans["source_permuted"].status=="ABSTAINED"
    assert plans["generic_scaffold"].status=="PLANNED"
    assert plans["source_induced"].status=="PLANNED"
    assert plans["target_written_isomorphic"].status=="PLANNED"
    assert plans["generic_scaffold"].commit_policy.startswith("EAGER")
    assert plans["source_induced"].required_operators==plans["target_written_isomorphic"].required_operators
    assert plans["source_induced"].source_capability_sha256==source["artifact_sha256"]
    assert plans["source_permuted"].source_capability_sha256==source["artifact_sha256"]
    assert plans["source_permuted"].commit_policy.startswith("FIXED_DERANGEMENT")
    assert plans["target_written_isomorphic"].source_capability_sha256 is None


def test_endpoint_direction_selects_one_operator_not_first_then_last():
    semantic=parse_compact_semantic_target(
        "goal(class, single_reference(ordered_endpoint(forward, semantic_tuple(cup, table))))",
        task_id="A1",question_sha256=stable_hash("q1"),parser_sha256=stable_hash("parser"),
        parser_training_authority="DEV_ONLY",
    )
    operators,edges=semantic_operator_plan(semantic)
    assert "FIRST" in operators and "LAST" not in operators
    assert ("FIRST","LAST") not in edges


def test_source_permutation_preserves_inventory_and_edge_count_but_breaks_lineage():
    source = capabilities(); operators = source["authorized_operators"]
    authentic = {tuple(edge) for edge in source["authorized_compositions"]}
    permuted = set(source_permuted_compositions(operators, authentic))
    assert len(permuted) == len(authentic)
    assert permuted != authentic
    assert {value for edge in permuted for value in edge} <= set(operators)


def test_semantic_equality_does_not_alias_source_effect_comparison():
    semantic=parse_compact_semantic_target(
        "equality_condition(semantic_tuple(cup), semantic_tuple(table))",
        task_id="A2",question_sha256=stable_hash("q2"),parser_sha256=stable_hash("parser"),
        parser_training_authority="DEV_ONLY",
    )
    operators,_=semantic_operator_plan(semantic)
    assert "SEMANTIC_EQUALS" in operators
    assert "COMPARE" not in operators
    source=capabilities()
    source_plan=plan_harness_arm(
        semantic,arm="source_induced",source_capabilities=source,
        all_vm_operators=tuple(source["authorized_operators"])+("SEMANTIC_EQUALS",),
    )
    generic_plan=plan_harness_arm(
        semantic,arm="generic_scaffold",source_capabilities=source,
        all_vm_operators=tuple(source["authorized_operators"])+("SEMANTIC_EQUALS",),
    )
    assert source_plan.status=="ABSTAINED"
    assert source_plan.missing_operators==("SEMANTIC_EQUALS",)
    assert generic_plan.status=="PLANNED"
