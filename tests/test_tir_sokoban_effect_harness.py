from __future__ import annotations

import copy
from pathlib import Path
import json

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.tir_sokoban_effect_harness import (
    AUTHENTIC,
    evaluate_tir_effect_transfer,
    execute_condition,
    validate_source_receipt,
)


REPO = Path(__file__).resolve().parents[1]


def _source() -> dict:
    return json.loads((
        REPO / "docs/results/sokoban_effect_program_v2_compact_receipt.json"
    ).read_text())


def _candidate(candidate_id: str, score: float, answer: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "planner_score": score,
        "answer": {"answer": answer},
        "wrapper_receipt": {"tool": "zoom_region"},
    }


def _receipt(sample_id: str, baseline: str, answers: list[str], gold: str) -> dict:
    return {
        "sample_id": sample_id,
        "family": "maze",
        "gold_answer": gold,
        "baseline": {"answer": {"answer": baseline}},
        "candidates": [
            _candidate(f"C{i}", 1.0 - i / 10.0, value)
            for i, value in enumerate(answers)
        ],
    }


def test_source_receipt_is_content_addressed() -> None:
    source = _source()
    validate_source_receipt(source)
    bad = copy.deepcopy(source)
    bad["program"]["rules"][0]["select"] = "POSITION"
    with pytest.raises(ValueError, match="self hash"):
        validate_source_receipt(bad)


def test_authentic_requires_corroboration_before_changed_commit() -> None:
    receipt = _receipt("x", "A", ["B", "C", "B", "D"], "B")
    result = execute_condition(receipt, condition=AUTHENTIC, shuffle_seed="s")
    assert result["committed_answer"] == "B"
    assert result["tests"] == 3
    assert [row["source_option"] for row in result["source_decisions"]] == [
        "POSITION", "POSITION", "POSITION", "COMMIT",
    ]


def test_authentic_abstains_when_changed_answer_is_not_verified() -> None:
    receipt = _receipt("x", "A", ["B", "C", "D", "E"], "A")
    result = execute_condition(receipt, condition=AUTHENTIC, shuffle_seed="s")
    assert result["committed_answer"] == "A"
    assert result["tests"] == 4
    assert result["source_decisions"][-2]["source_option"] == "REPLAN_OR_ABSTAIN"


def test_runtime_policy_does_not_read_gold() -> None:
    receipt = _receipt("x", "A", ["B", "B", "C", "D"], "A")
    first = execute_condition(receipt, condition=AUTHENTIC, shuffle_seed="s")
    receipt["gold_answer"] = "F"
    second = execute_condition(receipt, condition=AUTHENTIC, shuffle_seed="s")
    assert first == second


def test_evaluator_checks_frozen_coverage_and_hashes_report() -> None:
    rows = [
        _receipt("a", "A", ["B", "B", "C", "D"], "B"),
        _receipt("b", "C", ["C", "D", "E", "F"], "C"),
    ]
    report = evaluate_tir_effect_transfer(
        rows,
        source_receipt=_source(),
        expected_ids=["a", "b"],
        claim_boundary="unit test",
        evidence_tier="CONSUMED_DEVELOPMENT",
    )
    body = dict(report)
    claimed = body.pop("report_sha256")
    assert stable_hash(body) == claimed
    with pytest.raises(ValueError, match="order/coverage"):
        evaluate_tir_effect_transfer(
            rows,
            source_receipt=_source(),
            expected_ids=["b", "a"],
            claim_boundary="unit test",
            evidence_tier="CONSUMED_DEVELOPMENT",
        )
