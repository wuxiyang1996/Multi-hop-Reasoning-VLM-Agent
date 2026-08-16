from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_source_induction import (
    ACTIVATE_DELTA,
    RELEASE_DELTA,
    TERMINATE_DELTA,
    build_lineage_report,
    decision_examples_from_ledgers,
    execute_program_on_ledgers,
    evaluate_program_on_split,
    induce_program,
    validate_program,
)


def _receipt(snapshot, split, verified, count=3):
    body = {
        "schema_version": "sokoban-search-attempt-ledger-v16",
        "snapshot_id": snapshot,
        "source_split": {
            "discovery": "development",
            "calibration": "qualification",
            "heldout": "heldout",
        }[split],
        "automaton_split": {"heldout": "fresh"}.get(split, split),
        "candidate_count": count,
        "verified_candidate_rank": verified,
        "verification_authority": "TEST_SOURCE_RETURN",
        "return_gap": 2.0 + verified,
        "attempts": [
            {
                "candidate_receipt_id": stable_hash(f"{snapshot}-{rank}"),
                "verified": rank == verified,
                "refuted": rank != verified,
                "observed_actions": 1,
                "transition_hashes": [stable_hash(f"transition-{snapshot}-{rank}")],
                "official_cumulative_return": float(rank == verified),
            }
            for rank in range(count)
        ],
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _ledgers():
    rows = []
    for split in ("discovery", "calibration", "heldout"):
        for index, verified in enumerate((0, 1, 2, 1)):
            rows.append(_receipt(f"{split}-{index}", split, verified))
    return rows


def _thresholds():
    return {
        "maximum_literals": 3,
        "minimum_discovery_support": 2,
        "minimum_qualification_support": 2,
        "minimum_qualification_precision": 1.0,
        "minimum_qualification_coverage": 1.0,
        "minimum_heldout_coverage": 1.0,
        "minimum_heldout_selective_accuracy": 1.0,
        "maximum_shuffled_heldout_accuracy": 0.75,
    }


def test_source_only_induction_discovers_anonymous_state_delta_rules():
    examples = decision_examples_from_ledgers(
        _ledgers(), source_game="synthetic_source",
    )
    program = induce_program(
        examples,
        source_game="synthetic_source",
        source_induction_receipts_sha256=stable_hash("rows"),
        **{key: value for key, value in _thresholds().items() if key in {
            "maximum_literals",
            "minimum_discovery_support",
            "minimum_qualification_support",
            "minimum_qualification_precision",
            "minimum_qualification_coverage",
        }},
    )
    validate_program(program)
    assert program["status"] == "SOURCE_INDUCED_PROGRAM_QUALIFIED"
    deltas = {
        tuple(tuple(value) for value in row["state_delta"])
        for row in program["operators"]
    }
    assert ACTIVATE_DELTA in deltas
    assert RELEASE_DELTA in deltas
    assert TERMINATE_DELTA in deltas
    serialized = str(program)
    assert "EXPLORE_UNTRIED" not in serialized
    assert "BACKTRACK_REPLAN" not in serialized
    assert "COMMIT_VERIFY" not in serialized
    heldout = evaluate_program_on_split(
        program, examples, source_split="heldout",
    )
    assert heldout["coverage"] == 1.0
    assert heldout["selective_accuracy"] == 1.0
    execution = execute_program_on_ledgers(
        program, _ledgers(), source_split="heldout",
    )
    assert execution["success_rate"] == 1.0


def test_unknown_state_abstains_by_having_no_unique_operator():
    examples = decision_examples_from_ledgers(
        _ledgers(), source_game="synthetic_source",
    )
    program = induce_program(
        examples,
        source_game="synthetic_source",
        source_induction_receipts_sha256=stable_hash("rows"),
    )
    from motif_transfer.phase3_source_induction import operators_from_program, route_state

    assert route_state(operators_from_program(program), {
        "active_presence": "PRESENT",
        "active_effect": "UNKNOWN",
        "has_untried": True,
        "terminal": False,
        "suspended": False,
    }) is None


def test_shuffled_effect_binding_cannot_relabel_true_continuation_value():
    authentic = decision_examples_from_ledgers(
        _ledgers(), source_game="synthetic_source",
    )
    shuffled = decision_examples_from_ledgers(
        _ledgers(), source_game="synthetic_source", shuffled_effect_binding=True,
    )
    shuffled_program = induce_program(
        shuffled,
        source_game="synthetic_source",
        source_induction_receipts_sha256=stable_hash("rows"),
    )
    heldout = evaluate_program_on_split(
        shuffled_program, authentic, source_split="heldout",
    )
    assert heldout["overall_accuracy"] < 1.0
    execution = execute_program_on_ledgers(
        shuffled_program, _ledgers(), source_split="heldout",
    )
    assert execution["success_rate"] == 0.0


def test_raw_row_hashes_are_fail_closed(tmp_path):
    # Exercise the real loader format with a minimal matched candidate set.
    from motif_transfer.phase3_source_induction import load_source_ledgers

    rows = []
    for rank, value in enumerate((2.0, 0.0)):
        for repeat in range(2):
            body = {
                "schema_version": "PHASE1_MATCHED_OPTION_FORK_V1",
                "snapshot_id": "snapshot",
                "source_split": "development",
                "candidate_rank": rank,
                "candidate_id": stable_hash(f"candidate-{rank}"),
                "status": "INTERVENTION_OBSERVED",
                "cumulative_returns": {"h8": value},
                "final_observable_sha256": stable_hash(f"final-{rank}"),
                "transition_hashes": [stable_hash(f"transition-{rank}")],
                "observed_actions": 1,
                "repeat_index": repeat,
            }
            rows.append(body | {"row_sha256": stable_hash(body)})
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))
    ledgers, audit = load_source_ledgers(path, primary_horizon=8)
    assert len(ledgers) == 1
    assert audit["eligible_ledgers"] == 1

    rows[0]["observed_actions"] = 9
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))
    import pytest

    with pytest.raises(ValueError, match="invalid source row hash"):
        load_source_ledgers(path, primary_horizon=8)
