from motif_transfer.real_transfer_gate import finalize_transfer_gate


def test_grounder_failure_blocks_target_execution() -> None:
    report = finalize_transfer_gate(
        {"status": "SOURCE_GATE_PASSED"},
        {"status": "SOURCE_GROUNDER_GATE_FAILED"},
    )
    assert report["status"] == "TRANSFER_BLOCKED_AT_SOURCE_GROUNDER"
    assert report["target_execution_authorized"] is False
    assert report["conditions_executed"] == []


def test_both_gates_only_authorize_target_test() -> None:
    report = finalize_transfer_gate(
        {"status": "SOURCE_GATE_PASSED"},
        {"status": "SOURCE_GROUNDER_GATE_PASSED"},
    )
    assert report["status"] == "TARGET_FOUR_CONDITION_RUN_AUTHORIZED"
    assert report["target_execution_authorized"] is True
    assert report["cross_domain_transfer_supported"] is False
