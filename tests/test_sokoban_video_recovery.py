from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.sokoban_video_recovery import (
    authentic_recovery_decision,
    exact_binomial_two_sided,
    parse_executor_effect,
    validate_source_receipt,
)


REPO = Path(__file__).resolve().parents[1]


def test_confirmed_source_receipt_has_recovery_contract() -> None:
    receipt = json.loads(
        (REPO / "docs/results/sokoban_effect_program_v2_compact_receipt.json")
        .read_text(encoding="utf-8")
    )
    validate_source_receipt(receipt)


def test_typed_error_triggers_recovery_but_negative_answer_does_not() -> None:
    observed = parse_executor_effect(["yes", "no", "no"])
    refuted = parse_executor_effect(["yes", "error", "no"])
    assert not authentic_recovery_decision(observed)
    assert authentic_recovery_decision(refuted)


def test_exact_paired_probability() -> None:
    assert exact_binomial_two_sided(7, 0) == 0.015625
