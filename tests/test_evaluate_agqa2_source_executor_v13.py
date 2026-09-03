from __future__ import annotations

import importlib.util
from pathlib import Path

from motif_transfer.contracts import stable_hash


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_agqa2_source_executor_v13",
    REPO_ROOT / "scripts" / "evaluate_agqa2_source_executor_v13.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report(rows: list[dict]) -> dict:
    body = {
        "metrics": {
            "valid_runtime_rows": len(rows),
            "direct_correct": sum(row["direct_correct"] for row in rows),
            "route_accuracy": 1.0,
            "typed_vs_direct_wins": 0,
            "typed_vs_direct_losses": 0,
            "typed_fallback_correct": 0,
            "decisive_executions": 0,
        },
        "controls": {
            "source_permuted_abstentions": len(rows),
            "target_written_equivalent_matches": len(rows),
        },
        "qualification_gates": {"runtime_integrity": True},
        "reported_provider_cost_usd": 0.1,
        "rows": rows,
    }
    return body | {"report_sha256": stable_hash(body)}


def _row(*, direct: bool, typed: bool, events: int = 1, tiebreak: bool = False) -> dict:
    return {
        "direct_correct": direct,
        "decisive_correct": typed,
        "decisive_execution": True,
        "grounding_receipt": {"events": [{} for _ in range(events)]},
        "operand_runs": {"A": {"tiebreak_triggered": tiebreak}},
    }


def test_single_binding_rule_selects_only_unambiguous_single_event_rows() -> None:
    rows = [
        _row(direct=False, typed=True),
        _row(direct=True, typed=False, events=2),
        _row(direct=True, typed=False, tiebreak=True),
    ]
    protocol = {
        "claim_boundary": "TEST",
        "sample_count": 3,
        "applicability_rule": "SINGLE_EVENT_BINDING_AND_NO_CONFLICT_TIEBREAK",
        "formal_gate": {
            "minimum_wins": 1,
            "maximum_losses": 0,
            "minimum_net_gain": 1,
            "maximum_one_sided_exact_pvalue": 1.0,
            "minimum_route_accuracy": 1.0,
            "minimum_source_permuted_abstention_rate": 1.0,
            "minimum_target_written_equivalent_rate": 1.0,
            "maximum_cost_usd": 1.0,
        },
    }

    result = MODULE.evaluate(protocol, _report(rows))

    assert result["status"] == "PASSED"
    assert result["metrics"]["authorized_count"] == 1
    assert result["metrics"]["source_induced_correct"] == 3
    assert result["metrics"]["wins"] == 1
    assert result["metrics"]["losses"] == 0


def test_exact_one_sided_binomial_gate_rejects_more_losses_than_wins() -> None:
    rows = [
        _row(direct=False, typed=True),
        _row(direct=True, typed=False),
        _row(direct=True, typed=False),
    ]
    protocol = {
        "claim_boundary": "TEST",
        "sample_count": 3,
        "applicability_rule": "SINGLE_EVENT_BINDING_AND_NO_CONFLICT_TIEBREAK",
        "formal_gate": {
            "minimum_wins": 0,
            "maximum_losses": 3,
            "minimum_net_gain": -3,
            "maximum_one_sided_exact_pvalue": 0.5,
            "minimum_route_accuracy": 1.0,
            "minimum_source_permuted_abstention_rate": 1.0,
            "minimum_target_written_equivalent_rate": 1.0,
            "maximum_cost_usd": 1.0,
        },
    }

    result = MODULE.evaluate(protocol, _report(rows))

    assert result["metrics"]["one_sided_exact_binomial_pvalue"] == 0.875
    assert result["gates"]["maximum_one_sided_exact_pvalue"] is False
    assert result["status"] == "FAILED"
