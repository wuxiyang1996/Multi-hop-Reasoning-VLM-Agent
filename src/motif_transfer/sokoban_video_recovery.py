"""Retarget the confirmed Sokoban VERIFY/REFUTED/REPLAN contract to video.

The source program never sees CLEVRER tokens, programs, answers, or dynamics
representations.  The target reports only whether its native symbolic executor
returned a well-typed effect (yes/no) or refuted the expected effect (error).
The authentic source contract keeps the primary representation after an
observed effect and replans through the target-native alternate representation
after a refutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExecutorEffectReceipt:
    expected_effect_observed: bool
    expected_effect_refuted: bool
    raw_result_count: int
    error_count: int


def validate_source_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("artifact_version") != "SOKOBAN_EFFECT_PROGRAM_V2":
        raise ValueError("unexpected Sokoban effect-program version")
    if receipt.get("source_domain") != "Sokoban":
        raise ValueError("video recovery requires the confirmed Sokoban source")
    confirmation = receipt.get("fresh_confirmation", {})
    if not confirmation.get("source_gate_passed"):
        raise ValueError("Sokoban source effect gate did not pass")
    if not all(bool(value) for value in confirmation.get("gates", {}).values()):
        raise ValueError("Sokoban source confirmation gates are incomplete")
    rules = receipt.get("program", {}).get("rules", [])
    signatures = {(row.get("when"), row.get("select"), row.get("then")) for row in rules}
    required = {
        (
            "DIRECT_PROGRESS_AVAILABLE_OR_ASSIGNMENT_IMPROVEMENT_AVAILABLE",
            "COMMIT",
            "VERIFY_EXPECTED_EFFECT",
        ),
        ("EXPECTED_EFFECT_REFUTED", "REPLAN_OR_ABSTAIN", None),
    }
    if not required.issubset(signatures):
        raise ValueError("Sokoban VERIFY/REFUTED/REPLAN contract is missing")


def parse_executor_effect(raw_results: Sequence[str]) -> ExecutorEffectReceipt:
    if not raw_results or any(value not in {"yes", "no", "error"} for value in raw_results):
        raise ValueError("target executor receipt must contain yes/no/error results")
    errors = sum(value == "error" for value in raw_results)
    return ExecutorEffectReceipt(
        expected_effect_observed=errors == 0,
        expected_effect_refuted=errors > 0,
        raw_result_count=len(raw_results),
        error_count=errors,
    )


def authentic_recovery_decision(receipt: ExecutorEffectReceipt) -> bool:
    """Return True when the source contract authorizes target-native recovery."""

    if receipt.expected_effect_observed == receipt.expected_effect_refuted:
        raise ValueError("effect observation and refutation must be exclusive")
    return receipt.expected_effect_refuted


def exact_binomial_two_sided(wins: int, losses: int) -> float:
    if wins < 0 or losses < 0:
        raise ValueError("paired counts must be nonnegative")
    count = wins + losses
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, k) for k in range(min(wins, losses) + 1)) / 2**count
    return min(1.0, 2.0 * tail)


__all__ = [
    "ExecutorEffectReceipt",
    "authentic_recovery_decision",
    "exact_binomial_two_sided",
    "parse_executor_effect",
    "validate_source_receipt",
]
