from __future__ import annotations

from motif_transfer.agqa_semantic_slots import (
    action_anchor_obligations,
    parse_compact_semantic_target,
    serialize_compact_semantic_target,
)
from motif_transfer.contracts import stable_hash


def _semantic(program: str):
    return parse_compact_semantic_target(
        serialize_compact_semantic_target(program),
        task_id="T0",
        question_sha256=stable_hash("question"),
        parser_sha256=stable_hash("parser"),
        parser_training_authority="SOURCE_FREE_OPERATOR_FREE_SEMANTICS",
    )


def test_between_temporal_tuple_yields_both_exact_action_obligations() -> None:
    semantic = _semantic(
        "Query(class, OnlyItem(Iterate(Localize(between, "
        "[holding some food, putting some food somewhere]), "
        "Filter(frame, [relations, beside, objects]))))"
    )

    assert action_anchor_obligations(semantic) == (
        ("holding some food", "S2"),
        ("putting some food somewhere", "S3"),
    )


def test_single_temporal_anchor_still_yields_one_obligation() -> None:
    semantic = _semantic(
        "Query(class, OnlyItem(Iterate(Localize(before, holding a broom), "
        "Filter(frame, [relations, carrying, objects]))))"
    )

    assert action_anchor_obligations(semantic) == (("holding a broom", "S2"),)
