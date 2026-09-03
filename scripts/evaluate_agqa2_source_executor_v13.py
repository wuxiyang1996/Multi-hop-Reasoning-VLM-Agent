#!/usr/bin/env python3
"""Apply the separately pre-registered net-gain gate to a base report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(protocol: dict, report: dict) -> dict:
    body = dict(report); claimed = body.pop("report_sha256", None)
    if stable_hash(body) != claimed:
        raise ValueError("base report hash mismatch")
    metrics, controls = report["metrics"], report["controls"]
    gate = protocol.get("qualification_gate") or protocol.get("formal_gate")
    if not isinstance(gate, dict):
        raise ValueError("protocol lacks qualification_gate or formal_gate")
    n = int(metrics["valid_runtime_rows"])
    policy = protocol.get("applicability_rule", "ALL_DECISIVE_TYPED_EXECUTIONS")
    if policy == "ALL_DECISIVE_TYPED_EXECUTIONS":
        wins = int(metrics["typed_vs_direct_wins"]); losses = int(metrics["typed_vs_direct_losses"])
        source_correct = int(metrics["typed_fallback_correct"])
        authorized_count = int(metrics["decisive_executions"])
    elif policy == "SINGLE_EVENT_BINDING_AND_NO_CONFLICT_TIEBREAK":
        wins = losses = source_correct = authorized_count = 0
        for row in report["rows"]:
            authorized = (
                bool(row["decisive_execution"])
                and len(row["grounding_receipt"]["events"]) == 1
                and not any(
                    bool(operand.get("tiebreak_triggered"))
                    for operand in row["operand_runs"].values()
                )
            )
            direct_correct = bool(row["direct_correct"])
            selected_correct = (
                bool(row["decisive_correct"]) if authorized else direct_correct
            )
            wins += int(selected_correct and not direct_correct)
            losses += int(direct_correct and not selected_correct)
            source_correct += int(selected_correct)
            authorized_count += int(authorized)
    else:
        raise ValueError(f"unknown applicability_rule: {policy}")
    discordant = wins + losses
    one_sided_p = (
        sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
        / (2 ** discordant)
        if discordant else 1.0
    )
    gates = {
        "required_sample_count": n == int(protocol["sample_count"]),
        "minimum_wins": wins >= int(gate["minimum_wins"]),
        "maximum_losses": losses <= int(gate["maximum_losses"]),
        "minimum_net_gain": wins - losses >= int(gate["minimum_net_gain"]),
        "minimum_route_accuracy": float(metrics["route_accuracy"]) >= float(gate["minimum_route_accuracy"]),
        "minimum_source_permuted_abstention_rate": controls["source_permuted_abstentions"] / n >= float(gate["minimum_source_permuted_abstention_rate"]),
        "minimum_target_written_equivalent_rate": controls["target_written_equivalent_matches"] / n >= float(gate["minimum_target_written_equivalent_rate"]),
        "maximum_cost_usd": float(report["reported_provider_cost_usd"]) <= float(gate["maximum_cost_usd"]),
        "collector_internal_gates": all(report["qualification_gates"].values()),
    }
    if "maximum_one_sided_exact_pvalue" in gate:
        gates["maximum_one_sided_exact_pvalue"] = one_sided_p <= float(
            gate["maximum_one_sided_exact_pvalue"]
        )
    passed = all(gates.values())
    result_body = {
        "schema_version": "agqa2-source-executor-evaluation-v1",
        "status": "PASSED" if passed else "FAILED",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": stable_hash(protocol),
        "base_report_sha256": report["report_sha256"],
        "metrics": {
            "sample_count": n,
            "neural_only_correct": int(metrics["direct_correct"]),
            "source_induced_correct": source_correct,
            "neural_only_accuracy": metrics["direct_correct"] / n,
            "source_induced_accuracy": source_correct / n,
            "authorized_count": authorized_count,
            "applicability_rule": policy,
            "wins": wins, "losses": losses, "net_gain": wins - losses,
            "accuracy_delta_pp": 100.0 * (wins - losses) / n,
            "one_sided_exact_binomial_pvalue": one_sided_p,
        },
        "controls": controls,
        "gates": gates,
        "reported_provider_cost_usd": report["reported_provider_cost_usd"],
    }
    return result_body | {"report_sha256": stable_hash(result_body)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True, type=Path); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--report", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    protocol, config, report = json.loads(args.protocol.read_text()), json.loads(args.config.read_text()), json.loads(args.report.read_text())
    if report.get("config_sha256") != stable_hash(config): raise ValueError("report belongs to a different activated config")
    if config.get("formal_protocol_file_sha256") != _sha256(args.protocol): raise ValueError("activated config belongs to a different formal protocol")
    result = evaluate(protocol, report); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True)); raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__": main()
