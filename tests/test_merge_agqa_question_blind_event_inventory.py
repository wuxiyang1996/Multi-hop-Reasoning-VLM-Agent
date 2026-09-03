from scripts.merge_agqa_question_blind_typed_event_inventory_v1 import _merged_status


def test_consumed_development_merge_cannot_claim_frozen_transfer_evidence() -> None:
    assert _merged_status(True) == (
        "CONSUMED_DEVELOPMENT_EVENT_INVENTORY_NOT_TRANSFER_EVIDENCE"
    )
    assert _merged_status(False).endswith("FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME")
