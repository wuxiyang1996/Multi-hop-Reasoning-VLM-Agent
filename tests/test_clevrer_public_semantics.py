from motif_transfer.clevrer_public_semantics import parse_public_semantics
from scripts.audit_clevrer_public_parser_v1 import _runtime_normalize


def test_operator_free_descriptive_receipt() -> None:
    receipt = parse_public_semantics(
        task_id="video_1.Q0", question="How many objects enter the scene?",
        question_family="descriptive", public_subtype="count",
    )
    assert receipt.answer_kind == "COUNT"
    assert receipt.operator_sequence_emitted is False
    assert receipt.functional_program_read is False
    assert receipt.answer_read is False


def test_operator_free_causal_receipt_validates_choices() -> None:
    receipt = parse_public_semantics(
        task_id="video_1.Q1", question="Which event will happen next?",
        question_family="predictive",
        choices=("The red sphere collides with the blue cube",),
    )
    assert receipt.answer_kind == "CHOICE_VECTOR"
    assert receipt.choice_sha256s
    assert receipt.target_outcome_read is False


def test_annotation_aliases_normalize_to_runtime_vocabulary() -> None:
    assert _runtime_normalize(["objects", "end", "get_frame", "get_object", "get_col_partner"]) == [
        "objects", "events", "filter_end", "query_frame", "query_object",
        "query_collision_partner",
    ]
