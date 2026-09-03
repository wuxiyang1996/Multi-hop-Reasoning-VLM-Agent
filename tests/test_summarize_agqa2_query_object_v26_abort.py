from scripts.summarize_agqa2_query_object_v26_abort import RETRY_HISTORY


def test_v26_retry_ledger_preserves_terminal_failure():
    assert RETRY_HISTORY["GL2JW-1948"]["eventually_completed"] is True
    assert RETRY_HISTORY["HLB3J-14232"]["eventually_completed"] is True
    assert RETRY_HISTORY["QMIKJ-29239"] == {
        "initial_error": "ValueError: provider response omitted a JSON object",
        "same_protocol_resume_attempts": 2,
        "eventually_completed": False,
    }
