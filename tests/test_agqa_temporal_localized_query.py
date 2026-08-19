import pytest

from motif_transfer.agqa_program_transfer import (
    RELATION_IR,
    RELATION_OPERATOR,
    TEMPORAL_IR,
    TEMPORAL_PAIR_OPERATOR,
)
from motif_transfer.agqa_temporal_localized_query import (
    calibrate_window_object_consensus,
    consensus_anchor_interval,
    execute_temporal_window,
    parse_temporal_localized_object_question,
    select_composite_source_programs,
)
from motif_transfer.structural_ir_applicability import SourceIRContract


def _source(name, ir_kind, operator, terminal=()):
    return SourceIRContract.create(
        program_sha256=name,
        ir_kind=ir_kind,
        operator_sequence=(operator,),
        recurrent=True,
        terminal_predicate_families=terminal,
        source_intervention_qualified=True,
        source_confirmation_sha256="fresh-" + name,
    )


TEMPORAL = _source("candy", TEMPORAL_IR, TEMPORAL_PAIR_OPERATOR)
RELATION = _source(
    "sokoban", RELATION_IR, RELATION_OPERATOR, ("ENTITY_GOAL_RELATION",),
)


@pytest.mark.parametrize(
    ("question", "operator", "relation", "anchor_a", "anchor_b"),
    [
        (
            "Before watching outside of a window, what was the person holding?",
            "BEFORE", "holding", "watching outside of a window", "",
        ),
        (
            "What did they touch after making some food?",
            "AFTER", "touching", "making some food", "",
        ),
        (
            "Which object were they holding while taking a broom from somewhere?",
            "WHILE", "holding", "taking a broom from somewhere", "",
        ),
        (
            "Which object did the person hold before starting to sneeze somewhere?",
            "BEFORE", "holding", "sneezing somewhere", "",
        ),
        (
            "Which object were they standing on between sneezing somewhere and "
            "putting a broom somewhere?",
            "BETWEEN", "standing on", "sneezing somewhere",
            "putting a broom somewhere",
        ),
        (
            "Before sitting in a bed but after putting clothes somewhere, "
            "which object did the person go behind?",
            "BETWEEN", "behind", "putting clothes somewhere",
            "sitting in a bed",
        ),
        (
            "Which object were they opening before putting some food somewhere "
            "but after putting clothes somewhere?",
            "BETWEEN", "opening", "putting clothes somewhere",
            "putting some food somewhere",
        ),
    ],
)
def test_public_question_compiler(question, operator, relation, anchor_a, anchor_b):
    plan = parse_temporal_localized_object_question(question)
    assert plan is not None
    assert plan.temporal_operator == operator
    assert plan.relation == relation
    assert plan.anchor_a == anchor_a
    assert plan.anchor_b == anchor_b
    assert plan.answer_read is False
    assert plan.functional_program_read is False


def test_compiler_abstains_when_object_relation_is_not_explicit():
    assert parse_temporal_localized_object_question(
        "Which object did the person hold after making some food?"
    ) is not None
    assert parse_temporal_localized_object_question(
        "Before watching outside, what was the person doing?"
    ) is None
    assert parse_temporal_localized_object_question(
        "What happened after making food?"
    ) is None


@pytest.mark.parametrize(
    ("question", "relation"),
    [
        ("After holding a bag, what was the person above?", "above"),
        ("After making food, what did they go behind?", "behind"),
        (
            "Before opening a door, which object were they on the side of?",
            "on the side of",
        ),
        ("What did they hold after making food?", "holding"),
    ],
)
def test_relation_surface_normalizes_to_closed_agqa_predicate(question, relation):
    plan = parse_temporal_localized_object_question(question)
    assert plan is not None
    assert plan.relation == relation


@pytest.mark.parametrize(
    ("operator", "a", "b", "expected"),
    [
        ("BEFORE", (10, 14), None, (0, 9)),
        ("AFTER", (10, 14), None, (15, 47)),
        ("WHILE", (10, 14), None, (10, 14)),
        ("BETWEEN", (4, 8), (20, 24), (9, 19)),
        ("BETWEEN", (20, 24), (4, 8), (9, 19)),
    ],
)
def test_symbolic_window_execution(operator, a, b, expected):
    receipt = execute_temporal_window(
        temporal_operator=operator, frame_count=48,
        anchor_a_interval=a, anchor_b_interval=b,
    )
    assert receipt.authorized
    assert (receipt.window_start_frame, receipt.window_end_frame) == expected
    assert receipt.answer_read is False


def test_too_small_window_abstains():
    receipt = execute_temporal_window(
        temporal_operator="BEFORE", frame_count=48,
        anchor_a_interval=(2, 4), minimum_window_frames=3,
    )
    assert not receipt.authorized
    assert receipt.window_start_frame is None


def _anchor(start, end, confidence=0.9):
    return {"observations": [{
        "observability": "OBSERVED", "confidence": confidence,
        "start_frame": start, "end_frame": end,
        "evidence_frames": [start, end],
    }]}


def test_recurrent_anchor_consensus_uses_independent_views():
    receipt = consensus_anchor_interval((_anchor(10, 15), _anchor(12, 17)))
    assert receipt.authorized
    assert receipt.consensus_interval == (10, 17)
    assert receipt.maximum_endpoint_spread == 2


def test_recurrent_anchor_consensus_fails_closed():
    missing = {"observations": [{
        "observability": "UNOBSERVED", "confidence": 0.9,
        "start_frame": None, "end_frame": None, "evidence_frames": [],
    }]}
    assert not consensus_anchor_interval((_anchor(10, 15), missing)).authorized
    assert not consensus_anchor_interval(
        (_anchor(2, 5), _anchor(20, 25)), maximum_endpoint_spread=8,
    ).authorized


def test_three_view_anchor_tiebreak_selects_unique_two_view_cluster():
    receipt = consensus_anchor_interval((
        _anchor(4, 8), _anchor(30, 36), _anchor(5, 10),
    ))
    assert receipt.authorized
    assert receipt.consensus_interval == (4, 10)
    assert receipt.maximum_endpoint_spread == 2


def test_composition_requires_both_exact_qualified_source_types():
    selected = select_composite_source_programs(
        (RELATION, TEMPORAL), grounder_qualified=True,
    )
    assert selected["status"] == "AUTHORIZED"
    assert selected["temporal_program_sha256"] == "candy"
    assert selected["relation_program_sha256"] == "sokoban"
    assert select_composite_source_programs(
        (RELATION,), grounder_qualified=True,
    )["status"] == "ABSTAINED"
    assert select_composite_source_programs(
        (RELATION, TEMPORAL), grounder_qualified=False,
    )["reason"] == "TARGET_GROUNDER_NOT_QUALIFIED"
    assert select_composite_source_programs(
        (RELATION, TEMPORAL), grounder_qualified=True,
        formal_outcome_read=True,
    )["reason"] == "CURRENT_TARGET_OUTCOME_EXPOSED"


def test_window_consensus_counts_model_families_not_prompts():
    result = calibrate_window_object_consensus(
        model_family_responses={"qwen": "the bed", "gemini3": "bed"},
        ontology_family_receipts={}, ontology_minimum_confidence=0.8,
    )
    assert result["decision"] == "bed"
    assert result["full_video_direct_response_read"] is False
    disagreement = calibrate_window_object_consensus(
        model_family_responses={"qwen": "bed", "gemini3": "chair"},
        ontology_family_receipts={"gemini3": {
            "decision": "bed", "relation_observed": True, "confidence": 0.9,
            "evidence_frames": [2],
        }},
        ontology_minimum_confidence=0.8,
    )
    # Gemini3's own prompts conflict, so it cannot join Qwen's vote.
    assert disagreement["decision"] is None
