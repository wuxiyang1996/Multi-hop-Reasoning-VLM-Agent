from pathlib import Path

from motif_transfer.phase1_six_game_transfer_audit import (
    EXPECTED_GAMES,
    TARGETS,
    build_phase1_six_game_four_target_audit,
)


REPO = Path(__file__).resolve().parents[1]


def test_current_phase1_six_by_four_audit_fails_closed() -> None:
    report = build_phase1_six_game_four_target_audit(REPO)

    assert tuple(report["source_games"]) == EXPECTED_GAMES
    assert tuple(report["target_domains"]) == TARGETS
    assert report["aggregate"]["total_cells"] == 24
    assert report["aggregate"]["phase1_source_qualified_games"] == 0
    assert report["aggregate"]["validated_phase1_transfer_cells"] == 0
    assert (
        report["aggregate"]["target_mechanism_ready_cells_conditional_on_same_ir"]
        == 24
    )
    assert not report["compositional_validation_contract"]["authorized_6x4_claim"]
    assert all(
        not cell["validated"]
        for targets in report["cells"].values()
        for cell in targets.values()
    )


def test_candy_receipts_do_not_override_failed_grounder_gate() -> None:
    report = build_phase1_six_game_four_target_audit(REPO)
    candy = report["source_games"]["candy_crush"]

    assert candy["source_stage"]["matched_intervention_receipts"]
    assert not candy["source_stage"]["independent_source_ir_value_qualified"]
    diagnostic = candy["best_available_diagnostics"][0]
    assert diagnostic["receipt_gate"] == "SOURCE_GATE_PASSED"
    assert diagnostic["neural_grounder_gate"] == "SOURCE_GROUNDER_GATE_FAILED"
    assert diagnostic["target_conditions_executed"] == 0

