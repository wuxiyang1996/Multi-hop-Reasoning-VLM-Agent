import pytest

from motif_transfer.agqa_frame_grounder import (
    execute_grounding_receipt,
    parse_frame_grounding_receipt,
    select_source_for_grounding,
)
from motif_transfer.agqa_program_transfer import (
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
)
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
)


def _event(
    event_id, role, label, start, end, evidence, *, status="OBSERVED",
):
    return {
        "event_id": event_id,
        "operand_role": role,
        "label": label,
        "subject": "person",
        "predicate": label,
        "object": "",
        "observability": status,
        "start_frame": start,
        "end_frame": end,
        "evidence_frames": evidence,
        "confidence": 0.8,
        "uncertainties": [],
    }


def _payload(kind, comparison, a, b, events, coverage="SUFFICIENT"):
    return {
        "obligation_kind": kind,
        "comparison": comparison,
        "operand_a": a,
        "operand_b": b,
        "events": events,
        "coverage": coverage,
        "uncertainties": [],
    }


def _source(kind, arity, recurrent):
    if kind == RELATION_ROUTE:
        ir = "RECURRENT_GOAL_RELATION_PROGRAM"
        signature = OperatorSignature(
            "UPDATE", "ENTITY_GOAL_RELATION", 2, "RELATION_COVERAGE",
        )
        terminal = ("ENTITY_GOAL_RELATION",)
    else:
        ir = "SPARSE_TEMPORAL_EFFECT_FUNCTION"
        signature = OperatorSignature(
            "SCORE", "TEMPORAL_EFFECT_VECTOR", arity, "NORMALIZED_PROBABILITY",
        )
        terminal = ()
    return SourceIRContract.create(
        program_sha256=f"source-{kind}",
        ir_kind=ir,
        operator_sequence=(signature,),
        recurrent=recurrent,
        terminal_predicate_families=terminal,
        source_intervention_qualified=True,
        source_confirmation_sha256="heldout-source-confirmation",
    )


def test_temporal_pair_receipt_executes_order_without_answer_field():
    receipt = parse_frame_grounding_receipt(_payload(
        TEMPORAL_PAIR_ROUTE,
        "BEFORE_AFTER",
        "interacting with a shoe",
        "grasping a doorknob",
        [
            _event("E0", "A", "interacting with a shoe", 2, 4, [2, 4]),
            _event("E1", "B", "grasping a doorknob", 9, 11, [9, 11]),
        ],
    ), frame_count=16)
    execution = execute_grounding_receipt(receipt)
    assert execution["decision"] == "before"
    assert execution["official_answer_read"] is False
    assert receipt.functional_program_read is False


def test_duration_receipt_merges_intervals_and_returns_operand_text():
    receipt = parse_frame_grounding_receipt(_payload(
        TEMPORAL_SINGLE_ROUTE,
        "SELECT_SHORTER",
        "holding a phone",
        "sitting on the floor",
        [
            _event("E0", "A", "holding a phone", 1, 3, [1, 3]),
            _event("E1", "B", "sitting on the floor", 8, 13, [8, 13]),
        ],
    ), frame_count=16)
    assert execute_grounding_receipt(receipt)["decision"] == "holding a phone"


def test_boolean_duration_receipt_returns_yes_or_no():
    receipt = parse_frame_grounding_receipt(_payload(
        TEMPORAL_SINGLE_ROUTE,
        "VERIFY_A_SHORTER",
        "holding a paper",
        "holding a book",
        [
            _event("E0", "A", "holding a paper", 1, 3, [1, 3]),
            _event("E1", "B", "holding a book", 7, 12, [7, 12]),
        ],
    ), frame_count=16)
    assert execute_grounding_receipt(receipt)["decision"] == "yes"


def test_relation_receipt_can_commit_only_from_observed_evidence():
    observed = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE,
        "EXISTS",
        "person undressing a shoe",
        "",
        [_event("E0", "A", "person undressing a shoe", 3, 5, [3, 5])],
    ), frame_count=16)
    assert execute_grounding_receipt(observed)["decision"] == "yes"
    unobserved_row = _event(
        "E0", "A", "person undressing a shoe", None, None, [],
        status="UNOBSERVED",
    )
    unobserved = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE, "EXISTS", "person undressing a shoe", "",
        [unobserved_row], coverage="PARTIAL",
    ), frame_count=16)
    assert execute_grounding_receipt(unobserved)["status"] == "ABSTAIN"


def test_relation_object_query_and_choice_are_target_native_executions():
    query = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE,
        "QUERY_OBJECT",
        "object the person is sitting on",
        "",
        [{
            **_event("E0", "A", "person sitting on a chair", 3, 7, [3, 7]),
            "object": "chair",
        }],
    ), frame_count=12)
    assert execute_grounding_receipt(query)["decision"] == "chair"

    choice = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE,
        "CHOOSE_OBJECT",
        "paper",
        "table",
        [
            _event("E0", "A", "person beside paper", 2, 4, [2, 4]),
            _event(
                "E1", "B", "person beside table", None, None, [],
                status="UNOBSERVED",
            ),
        ],
    ), frame_count=12)
    assert execute_grounding_receipt(choice)["decision"] == "paper"


def test_reversed_interval_endpoints_are_explicitly_canonicalized():
    receipt = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE,
        "QUERY_OBJECT",
        "object the person sits on",
        "",
        [{
            **_event("E0", "A", "person sitting on chair", 8, 3, [3, 8]),
            "object": "chair",
        }],
    ), frame_count=12)
    assert receipt.events[0].start_frame == 3
    assert receipt.events[0].end_frame == 8
    assert receipt.canonicalizations == (
        "E0:SWAPPED_REVERSED_INTERVAL_ENDPOINTS",
    )
    reparsed = parse_frame_grounding_receipt(receipt.as_dict(), frame_count=12)
    assert reparsed.receipt_sha256 == receipt.receipt_sha256


def test_interval_expands_to_model_cited_evidence_with_audit_marker():
    receipt = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE,
        "EXISTS",
        "person holding book",
        "",
        [_event("E0", "A", "person holding book", 5, 8, [3, 9])],
    ), frame_count=12)
    assert receipt.events[0].start_frame == 3
    assert receipt.events[0].end_frame == 9
    assert receipt.canonicalizations == (
        "E0:EXPANDED_INTERVAL_TO_COVER_EVIDENCE",
    )


@pytest.mark.parametrize("leak", [
    {"answer": "yes"},
    {"nested": {"functional_program": "Exists(...)"}},
    {"events": [{"sg_grounding": {"0-2": []}}]},
    {"source_identity": "sokoban"},
])
def test_receipt_rejects_answer_annotation_and_source_leakage(leak):
    payload = _payload(
        RELATION_ROUTE, "EXISTS", "relation", "",
        [_event("E0", "A", "relation", 1, 2, [1, 2])],
    )
    payload.update(leak)
    with pytest.raises(ValueError, match="forbidden"):
        parse_frame_grounding_receipt(payload, frame_count=8)


def test_predicted_grounding_type_selects_only_exact_anonymous_source():
    receipt = parse_frame_grounding_receipt(_payload(
        TEMPORAL_PAIR_ROUTE,
        "BEFORE_AFTER",
        "event a",
        "event b",
        [
            _event("E0", "A", "event a", 1, 2, [1, 2]),
            _event("E1", "B", "event b", 5, 6, [5, 6]),
        ],
    ), frame_count=8)
    relation = _source(RELATION_ROUTE, 2, True)
    pair = _source(TEMPORAL_PAIR_ROUTE, 2, True)
    single = _source(TEMPORAL_SINGLE_ROUTE, 1, False)
    selected = select_source_for_grounding(
        (relation, pair, single), task_id="q1", receipt=receipt,
        target_grounder_sha256="frame-grounder", grounder_qualified=True,
    )
    assert selected["selected_program_sha256"] == pair.program_sha256
    wrong = select_source_for_grounding(
        (single,), task_id="q1", receipt=receipt,
        target_grounder_sha256="frame-grounder", grounder_qualified=True,
    )
    assert wrong["selected_program_sha256"] is None


def test_unqualified_grounder_cannot_authorize_source_program():
    receipt = parse_frame_grounding_receipt(_payload(
        RELATION_ROUTE,
        "EXISTS",
        "relation",
        "",
        [_event("E0", "A", "relation", 1, 2, [1, 2])],
    ), frame_count=8)
    source = _source(RELATION_ROUTE, 2, True)
    result = select_source_for_grounding(
        (source,), task_id="q2", receipt=receipt,
        target_grounder_sha256="frame-grounder", grounder_qualified=False,
    )
    assert result["selected_program_sha256"] is None
    assert result["source_contracts"][0]["reason"] == "TARGET_GROUNDER_NOT_QUALIFIED"
