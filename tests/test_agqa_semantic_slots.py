from __future__ import annotations

from motif_transfer.agqa_semantic_slots import (
    parse_compact_semantic_target, parse_semantic_target, relation_grounding_obligations, semantic_supervision_target,
    serialize_compact_semantic_target, serialize_semantic_target,
)
from motif_transfer.contracts import stable_hash


PROGRAM = (
    "Query(class, OnlyItem(Iterate(Localize(before, holding a broom), "
    "Filter(frame, [relations, carrying, objects]))))"
)


def test_program_supervision_lowers_to_operator_free_semantic_graph() -> None:
    target = semantic_supervision_target(PROGRAM)
    rendered = serialize_semantic_target(PROGRAM)
    assert target["answer_kind"] == "ENTITY"
    assert target["functional_program_in_target"] is False
    assert target["operator_sequence_in_target"] is False
    for forbidden in ("PROJECT", "TEMPORAL_SELECT", "UNIQUE", "FILTER_EQ"):
        assert forbidden not in rendered
    kinds = {row["kind"] for row in target["slots"]}
    assert {"QUERY_GOAL", "TEMPORAL_CONSTRAINT", "RELATION"} <= kinds


def test_serialized_semantic_target_round_trips_to_frozen_receipt() -> None:
    receipt = parse_semantic_target(
        serialize_semantic_target(PROGRAM), task_id="A0",
        question_sha256=stable_hash("question"), parser_sha256=stable_hash("parser"),
        parser_training_authority="AGQA_TRAIN_DEV_PROGRAMS_LOWERED_TO_OPERATOR_FREE_SLOTS",
    )
    receipt.validate()
    assert receipt.root_slot_id in {row.slot_id for row in receipt.slots}
    assert not receipt.operator_sequence_emitted


def test_compact_target_is_short_operator_free_and_round_trips() -> None:
    rendered = serialize_compact_semantic_target(PROGRAM)
    assert len(rendered) < 300
    assert "Query" not in rendered and "Localize" not in rendered
    for forbidden in ("PROJECT", "TEMPORAL_SELECT", "UNIQUE", "FILTER_EQ"):
        assert forbidden not in rendered
    receipt = parse_compact_semantic_target(
        rendered, task_id="A0", question_sha256=stable_hash("question"),
        parser_sha256=stable_hash("compact-parser"),
        parser_training_authority="AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS",
    )
    receipt.validate()
    assert receipt.answer_kind == "ENTITY"


def test_all_official_functions_have_semantic_lowering() -> None:
    programs = (
        "Exists(Iterate(video, Filter(frame, [relations, carrying, objects])))",
        "Choose([cup, table], Exists(Iterate(video, Filter(frame, [relations, touching, objects]))))",
        "Equals(OnlyItem([cup]), OnlyItem([cup]))",
        "Compare(more, Subtract(Superlative(max, [walking, running]), walking), running)",
        "XOR(Exists([a]), Exists([b]))",
        "AND(Exists([a]), HasItem([b], b))",
        "Query(class, OnlyItem(IterateUntil(forward, [cup, table])))",
        "Query(class, ToAction(holding, OnlyItem([cup])))",
    )
    for program in programs:
        target = semantic_supervision_target(program)
        assert target["slots"]


def test_relation_obligations_extract_explicit_predicates_not_categories() -> None:
    receipt = parse_compact_semantic_target(
        serialize_compact_semantic_target(
            "Query(class, OnlyItem(Iterate(Localize(while, smiling at something), "
            "Filter(frame, [relations, in, objects]))))"
        ),
        task_id="R0", question_sha256=stable_hash("rq"),
        parser_sha256=stable_hash("rp"), parser_training_authority="OPERATOR_FREE_TEST",
    )
    assert relation_grounding_obligations(receipt) == (("in", "S6"),)
