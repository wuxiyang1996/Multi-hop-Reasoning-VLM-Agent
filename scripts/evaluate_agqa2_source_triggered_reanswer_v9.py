#!/usr/bin/env python3
"""Evaluate a frozen source-triggered re-answer runtime against its base arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import _answer_matches  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_report_hash(report: Mapping[str, Any]) -> bool:
    body = dict(report)
    claimed = body.pop("report_sha256", None)
    return isinstance(claimed, str) and stable_hash(body) == claimed


def evaluate(
    config: Mapping[str, Any], base: Mapping[str, Any], runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if runtime.get("config_sha256") != stable_hash(config):
        raise ValueError("runtime belongs to a different frozen config")
    if runtime.get("base_report_sha256") != base.get("report_sha256"):
        raise ValueError("runtime belongs to a different base report")
    if not _valid_report_hash(base) or not _valid_report_hash(runtime):
        raise ValueError("input report hash mismatch")

    base_rows = {str(row["task_id"]): row for row in base["rows"]}
    runtime_rows = {str(row["task_id"]): row for row in runtime["rows"]}
    if len(base_rows) != len(base["rows"]) or set(base_rows) != set(runtime_rows):
        raise ValueError("base/runtime task sets are not identical and unique")

    rows = []
    for task_id in sorted(base_rows):
        original, tool = base_rows[task_id], runtime_rows[task_id]
        if str(tool["task_id"]) != task_id:
            raise ValueError(f"runtime task mismatch: {task_id}")
        triggered = bool(tool["source_triggered"])
        if triggered:
            if not str(tool.get("response") or "").strip():
                raise ValueError(f"empty triggered response: {task_id}")
            if any(bool(tool.get(key)) for key in (
                "answer_read", "program_read", "scene_graph_read",
                "source_identity_read",
            )):
                raise ValueError(f"forbidden runtime access: {task_id}")
        elif tool.get("response") is not None:
            raise ValueError(f"untriggered row has a response: {task_id}")
        direct_prediction = original["direct_response"]
        source_prediction = tool["response"] if triggered else direct_prediction
        gold = original["gold_answer_evaluator_only"]
        direct_correct = _answer_matches(direct_prediction, gold)
        source_correct = _answer_matches(source_prediction, gold)
        rows.append({
            "task_id": task_id,
            "source_triggered": triggered,
            "direct_prediction": direct_prediction,
            "source_prediction": source_prediction,
            "gold_answer_evaluator_only": gold,
            "direct_correct": direct_correct,
            "source_correct": source_correct,
            "win": source_correct and not direct_correct,
            "loss": direct_correct and not source_correct,
        })

    count = len(rows)
    triggered = sum(row["source_triggered"] for row in rows)
    direct_correct = sum(row["direct_correct"] for row in rows)
    source_correct = sum(row["source_correct"] for row in rows)
    wins = sum(row["win"] for row in rows)
    losses = sum(row["loss"] for row in rows)
    controls = base["controls"]
    route_accuracy = float(base["metrics"]["route_accuracy"])
    permuted_rate = float(controls["source_permuted_abstentions"]) / count
    equivalent_rate = float(controls["target_written_equivalent_matches"]) / count
    combined_cost = float(base["reported_provider_cost_usd"]) + float(
        runtime["reported_provider_cost_usd"]
    )
    gate_spec = config.get("qualification_gate") or config.get("formal_gate")
    if not isinstance(gate_spec, Mapping):
        raise ValueError("config lacks a qualification_gate or formal_gate")
    gates = {
        "minimum_wins": wins >= int(gate_spec["minimum_wins"]),
        "maximum_losses": losses <= int(gate_spec["maximum_losses"]),
        "minimum_net_gain": wins - losses >= int(gate_spec["minimum_net_gain"]),
        "minimum_route_accuracy": route_accuracy >= float(gate_spec["minimum_route_accuracy"]),
        "minimum_source_permuted_abstention_rate": permuted_rate >= float(
            gate_spec["minimum_source_permuted_abstention_rate"]
        ),
        "minimum_target_written_equivalent_rate": equivalent_rate >= float(
            gate_spec["minimum_target_written_equivalent_rate"]
        ),
        "maximum_combined_cost_usd": combined_cost <= float(
            gate_spec["maximum_combined_cost_usd"]
        ),
        "runtime_rows_complete": count == int(base["sample_count"]),
        "trigger_receipts_complete": triggered == int(runtime["triggered_count"]),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "agqa2-source-triggered-reanswer-evaluation-v1",
        "status": "PASSED" if passed else "FAILED",
        "claim_boundary": config["claim_boundary"],
        "config_sha256": stable_hash(config),
        "base_report_sha256": base["report_sha256"],
        "runtime_report_sha256": runtime["report_sha256"],
        "metrics": {
            "sample_count": count,
            "triggered_count": triggered,
            "direct_correct": direct_correct,
            "source_induced_correct": source_correct,
            "direct_accuracy": direct_correct / count,
            "source_induced_accuracy": source_correct / count,
            "wins": wins,
            "losses": losses,
            "net_gain": wins - losses,
            "accuracy_delta_pp": 100.0 * (source_correct - direct_correct) / count,
            "route_accuracy": route_accuracy,
        },
        "controls": {
            "source_permuted_abstention_rate": permuted_rate,
            "target_written_equivalent_match_rate": equivalent_rate,
        },
        "cost": {
            "base_reported_provider_cost_usd": float(base["reported_provider_cost_usd"]),
            "reanswer_reported_provider_cost_usd": float(runtime["reported_provider_cost_usd"]),
            "combined_reported_provider_cost_usd": combined_cost,
        },
        "gates": gates,
        "rows": rows,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base_path = REPO_ROOT / config["base_report"]
    if _sha256(base_path) != config["base_report_file_sha256"]:
        raise ValueError("frozen base report file hash mismatch")
    base, runtime = json.loads(base_path.read_text()), json.loads(args.runtime.read_text())
    result = evaluate(config, base, runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "metrics": result["metrics"],
        "controls": result["controls"], "cost": result["cost"],
        "gates": result["gates"], "report_sha256": result["report_sha256"],
    }, indent=2, sort_keys=True))
    if result["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
