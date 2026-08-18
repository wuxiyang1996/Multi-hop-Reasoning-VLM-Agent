import pytest

from motif_transfer.agqa_active_frame_grounder import (
    choose_operand_receipt,
    merge_operand_receipts,
    operand_needs_rescan,
    parse_operand_receipt,
    parse_public_question_plan,
    parse_query_plan,
    reconcile_recurrent_receipts,
    remap_operand_receipt,
    source_controller_for_plan,
)
from motif_transfer.agqa_frame_grounder import execute_grounding_receipt
from motif_transfer.agqa_program_transfer import (
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
)
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
)


def _plan(kind=TEMPORAL_PAIR_ROUTE, comparison="BEFORE_AFTER"):
    return parse_query_plan({
        "obligation_kind": kind,
        "comparison": comparison,
        "operand_a": "opening a laptop" if kind != RELATION_ROUTE else "chair",
        "operand_b": "sitting down" if kind != RELATION_ROUTE else "table",
        "visual_query_a": "person opening a laptop",
        "visual_query_b": "person going from standing to sitting",
        "parser_uncertainties": [],
    })


def _operand(role, requested, start, end, *, confidence=0.9, coverage="SUFFICIENT"):
    observed = start is not None
    return parse_operand_receipt({
        "operand_role": role,
        "requested_operand": requested,
        "observations": [{
            "occurrence_id": "O0",
            "label": requested,
            "subject": "person",
            "predicate": requested,
            "object": "laptop" if "laptop" in requested else "",
            "observability": "OBSERVED" if observed else "UNOBSERVED",
            "start_frame": start,
            "end_frame": end,
            "evidence_frames": [start, end] if observed else [],
            "confidence": confidence,
            "uncertainties": [],
        }],
        "coverage": coverage,
        "uncertainties": [],
    }, expected_role=role, expected_operand=requested, frame_count=48)


def _source(kind):
    if kind == RELATION_ROUTE:
        ir = "RECURRENT_GOAL_RELATION_PROGRAM"
        operator = OperatorSignature(
            "UPDATE", "ENTITY_GOAL_RELATION", 2, "RELATION_COVERAGE",
        )
        recurrent = True
        terminal = ("ENTITY_GOAL_RELATION",)
    elif kind == TEMPORAL_PAIR_ROUTE:
        ir = "SPARSE_TEMPORAL_EFFECT_FUNCTION"
        operator = OperatorSignature(
            "SCORE", "TEMPORAL_EFFECT_VECTOR", 2, "NORMALIZED_PROBABILITY",
        )
        recurrent = True
        terminal = ()
    else:
        ir = "SPARSE_TEMPORAL_EFFECT_FUNCTION"
        operator = OperatorSignature(
            "SCORE", "TEMPORAL_EFFECT_VECTOR", 1, "NORMALIZED_PROBABILITY",
        )
        recurrent = False
        terminal = ()
    return SourceIRContract.create(
        program_sha256=f"program-{kind}", ir_kind=ir,
        operator_sequence=(operator,), recurrent=recurrent,
        terminal_predicate_families=terminal,
        source_intervention_qualified=True,
        source_confirmation_sha256="held-out-source",
    )


def test_question_parser_rejects_answer_and_program_leakage():
    payload = _plan().as_dict()
    payload["answer"] = "after"
    with pytest.raises(ValueError, match="forbidden"):
        parse_query_plan(payload)


def test_query_object_parser_rejects_a_generic_object_without_relation():
    with pytest.raises(ValueError, match="non-generic"):
        parse_query_plan({
            "obligation_kind": RELATION_ROUTE,
            "comparison": "QUERY_OBJECT",
            "operand_a": "an unknown object",
            "operand_b": "",
            "visual_query_a": "an unknown object",
            "visual_query_b": "",
            "parser_uncertainties": [],
        })


def test_query_object_parser_accepts_equivalent_relation_inflection():
    plan = parse_query_plan({
        "obligation_kind": RELATION_ROUTE,
        "comparison": "QUERY_OBJECT",
        "operand_a": "put down",
        "operand_b": "",
        "visual_query_a": "a person putting down an unknown object",
        "visual_query_b": "",
        "parser_uncertainties": [],
    })
    assert plan.operand_a == "put down"


@pytest.mark.parametrize(("question", "comparison", "operand_a", "operand_b"), [
    (
        "Which object were they sitting on?", "QUERY_OBJECT",
        "sitting on", "",
    ),
    (
        "What was the person putting down?", "QUERY_OBJECT",
        "putting down", "",
    ),
    (
        "Was a paper or a table the thing they went on the side of?",
        "CHOOSE_OBJECT", "a paper", "a table",
    ),
    (
        "Was the person opening a laptop before or after going from standing to sitting?",
        "BEFORE_AFTER", "opening a laptop", "going from standing to sitting",
    ),
    (
        "Compared to consuming some medicine, did they hold a cup of something for longer?",
        "VERIFY_A_LONGER", "holding a cup of something", "consuming some medicine",
    ),
    (
        "Did the person spend a shorter amount of time holding a paper than they spent holding a book?",
        "VERIFY_A_SHORTER", "holding a paper", "holding a book",
    ),
    (
        "Which did they do for longer, holding a vacuum or smiling at something?",
        "SELECT_LONGER", "holding a vacuum", "smiling at something",
    ),
    (
        "Was the person leaning on a chair?", "EXISTS",
        "leaning on a chair", "",
    ),
    (
        "In the video, did they sit on a window?", "EXISTS",
        "sitting on a window", "",
    ),
])
def test_public_question_grammar_keeps_explicit_operands(
    question, comparison, operand_a, operand_b,
):
    plan = parse_public_question_plan(question)
    assert plan is not None
    assert (plan.comparison, plan.operand_a, plan.operand_b) == (
        comparison, operand_a, operand_b,
    )


def test_public_question_grammar_abstains_without_explicit_operands():
    assert parse_public_question_plan(
        "What were they doing for the most time?"
    ) is None


def test_choose_object_uses_one_candidate_blind_relation_scan():
    plan = parse_query_plan({
        "obligation_kind": RELATION_ROUTE, "comparison": "CHOOSE_OBJECT",
        "operand_a": "paper", "operand_b": "table",
        "visual_query_a": "person going on the side of an unknown object",
        "visual_query_b": "", "parser_uncertainties": [],
    })
    grounded = parse_operand_receipt({
        "operand_role": "A",
        "requested_operand": plan.visual_query_a,
        "observations": [{
            "occurrence_id": "O0", "label": "beside paper",
            "subject": "person", "predicate": "on the side of", "object": "paper",
            "observability": "OBSERVED", "start_frame": 2, "end_frame": 6,
            "evidence_frames": [2, 6], "confidence": 0.9, "uncertainties": [],
        }],
        "coverage": "SUFFICIENT", "uncertainties": [],
    }, expected_role="A", expected_operand=plan.visual_query_a, frame_count=12)
    merged = merge_operand_receipts(
        plan, operand_a=grounded, operand_b=None, frame_count=12,
    )
    assert execute_grounding_receipt(merged)["decision"] == "paper"


def test_operand_receipt_rejects_competing_operand_or_changed_request():
    payload = _operand("A", "opening a laptop", 2, 5).as_dict()
    payload["competing_operand"] = "sitting down"
    with pytest.raises(ValueError):
        parse_operand_receipt(
            payload, expected_role="A", expected_operand="opening a laptop",
            frame_count=48,
        )
    clean = _operand("A", "opening a laptop", 2, 5).as_dict()
    clean["requested_operand"] = "holding a book"
    with pytest.raises(ValueError, match="changed"):
        parse_operand_receipt(
            clean, expected_role="A", expected_operand="opening a laptop",
            frame_count=48,
        )


def test_source_ir_controls_arity_and_recurrence_without_source_identity():
    sources = tuple(_source(kind) for kind in (
        RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE,
    ))
    pair = source_controller_for_plan(_plan(), sources)
    assert pair.required_operands == 2
    assert pair.recurrent is True
    assert pair.maximum_rescans_per_operand == 1
    assert "candy" not in str(pair.as_dict()).lower()

    duration = source_controller_for_plan(
        _plan(TEMPORAL_SINGLE_ROUTE, "VERIFY_A_SHORTER"), sources,
    )
    assert duration.required_operands == 2
    assert duration.recurrent is False
    assert duration.maximum_rescans_per_operand == 0


def test_only_recurrent_source_requests_rescan_for_weak_operand():
    sources = tuple(_source(kind) for kind in (
        RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE,
    ))
    weak = _operand(
        "A", "opening a laptop", None, None,
        confidence=0.2, coverage="INSUFFICIENT",
    )
    recurrent = source_controller_for_plan(_plan(), sources)
    assert operand_needs_rescan(weak, controller=recurrent, confidence_threshold=0.7)
    nonrecurrent = source_controller_for_plan(
        _plan(TEMPORAL_SINGLE_ROUTE, "VERIFY_A_SHORTER"), sources,
    )
    assert not operand_needs_rescan(
        weak, controller=nonrecurrent, confidence_threshold=0.7,
    )


def test_recurrent_observed_unobserved_conflict_downgrades_to_partial():
    observed = _operand("A", "opening a laptop", 2, 5)
    unobserved = _operand(
        "A", "opening a laptop", None, None,
        confidence=0.9, coverage="SUFFICIENT",
    )
    reconciled = reconcile_recurrent_receipts(observed, unobserved)
    assert reconciled.coverage == "PARTIAL"
    assert all(row.observability == "PARTIAL" for row in reconciled.observations)
    assert "RECURRENT_OBSERVABILITY_CONFLICT_DOWNGRADED_TO_PARTIAL" in (
        reconciled.canonicalizations
    )


def test_recurrent_double_unobserved_can_prove_exists_no():
    primary = _operand(
        "A", "leaning on a chair", None, None,
        confidence=0.9, coverage="SUFFICIENT",
    )
    rescan = _operand(
        "A", "leaning on a chair", None, None,
        confidence=0.8, coverage="SUFFICIENT",
    )
    reconciled = reconcile_recurrent_receipts(primary, rescan)
    plan = parse_query_plan({
        "obligation_kind": RELATION_ROUTE,
        "comparison": "EXISTS",
        "operand_a": "leaning on a chair",
        "operand_b": "",
        "visual_query_a": "a person leaning on a chair",
        "visual_query_b": "",
        "parser_uncertainties": [],
    })
    merged = merge_operand_receipts(
        plan, operand_a=reconciled, operand_b=None, frame_count=48,
    )
    execution = execute_grounding_receipt(merged)
    assert execution["decision"] == "no"
    assert execution["reason"] == "QUERY_RELATION_DOUBLE_SCAN_UNOBSERVED"


def test_rescan_is_mapped_to_global_timeline_and_selected_without_outcome():
    primary = _operand(
        "A", "opening a laptop", None, None,
        confidence=0.2, coverage="INSUFFICIENT",
    )
    local = parse_operand_receipt({
        "operand_role": "A",
        "requested_operand": "opening a laptop",
        "observations": [{
            "occurrence_id": "O0", "label": "opening a laptop",
            "subject": "person", "predicate": "opening", "object": "laptop",
            "observability": "OBSERVED", "start_frame": 3, "end_frame": 6,
            "evidence_frames": [3, 6], "confidence": 0.92,
            "uncertainties": [],
        }],
        "coverage": "SUFFICIENT", "uncertainties": [],
    }, expected_role="A", expected_operand="opening a laptop", frame_count=8)
    mapped = remap_operand_receipt(
        local,
        local_seconds=[4, 5, 6, 7, 8, 9, 10, 11],
        global_seconds=list(range(48)),
    )
    assert mapped.observations[0].start_frame == 7
    assert mapped.observations[0].end_frame == 10
    assert choose_operand_receipt(primary, mapped) == mapped


def test_contradictory_unobserved_pixel_claim_is_downgraded_not_promoted():
    receipt = parse_operand_receipt({
        "operand_role": "A", "requested_operand": "person sitting on object",
        "observations": [{
            "occurrence_id": "O0", "label": "sitting", "subject": "person",
            "predicate": "sitting on", "object": "chair",
            "observability": "UNOBSERVED", "start_frame": 2, "end_frame": 5,
            "evidence_frames": [2, 5], "confidence": 0.9,
            "uncertainties": [],
        }],
        "coverage": "PARTIAL", "uncertainties": [],
    }, expected_role="A", expected_operand="person sitting on object", frame_count=8)
    assert receipt.observations[0].observability == "PARTIAL"
    assert receipt.canonicalizations == (
        "O0:DOWNGRADED_CONTRADICTORY_UNOBSERVED_TO_PARTIAL",
    )


def test_isolated_receipts_merge_into_existing_unified_executor():
    plan = _plan()
    a = _operand("A", plan.visual_query_a, 3, 6)
    b = _operand("B", plan.visual_query_b, 20, 24)
    receipt = merge_operand_receipts(
        plan, operand_a=a, operand_b=b, frame_count=48,
    )
    assert execute_grounding_receipt(receipt)["decision"] == "before"
    assert receipt.answer_read is False
    assert receipt.functional_program_read is False


def test_wrong_source_type_cannot_instantiate_controller():
    with pytest.raises(ValueError, match="found 0"):
        source_controller_for_plan(_plan(), (_source(RELATION_ROUTE),))
