#!/usr/bin/env python3
"""Run V23 paired success evaluation and apply its causal-only gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from run_real_source_relation_eval_v20 import main as run_v20  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V23 report: {args.output}")
    plan = _read(args.plan)
    if plan.get("schema_version") != "real-source-relation-causal-only-plan-v23":
        raise SystemExit("unexpected V23 plan schema")
    with tempfile.TemporaryDirectory(prefix="v23-eval-") as directory:
        raw_path = Path(directory) / "raw_v20_report.json"
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_real_source_relation_eval_v20.py",
                "--plan", str(args.plan),
                "--output", str(raw_path),
                "--alfworld-config", str(args.alfworld_config),
                "--alfworld-data", str(args.alfworld_data),
            ]
            _raw_status = run_v20()
        finally:
            sys.argv = original_argv
        if not raw_path.exists():
            raise SystemExit("V23 underlying matched evaluation produced no report")
        raw = _read(raw_path)
    primary = raw["policy_metrics"]["v23_causal_only"]
    gate_spec = plan["v23_gates"]
    invariants = all(
        all(row["invariants"].values()) for row in raw["forks"]
    )
    gates = {
        "minimum_opportunities": raw["opportunity_count"] >= int(
            gate_spec["minimum_opportunities"]
        ),
        "minimum_primary_admissions": primary["selected"] >= int(
            gate_spec["minimum_primary_admissions"]
        ),
        "minimum_primary_success_wins": primary["success_wins"] >= int(
            gate_spec["minimum_success_wins"]
        ),
        "primary_success_delta_strictly_positive": primary["success_delta"] > 0,
        "primary_exact_sign_test_passed": primary["one_sided_exact_sign_p"] <= float(
            gate_spec["one_sided_exact_sign_alpha"]
        ),
        "primary_selected_utility_strictly_positive": (
            primary["selected_incremental_utility"] > 0.0
        ),
        "source_event_recall_passed": primary["source_event_recall"] >= float(
            gate_spec["source_event_recall_at_least"]
        ),
        "all_exact_state_fork_invariants": invariants,
    }
    passed = all(gates.values())
    role = str(plan["role"])
    status = {
        ("development_gate", True): (
            "V23_DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
        ),
        ("development_gate", False): "V23_DEVELOPMENT_TRANSFER_GATE_FAILED_STOP",
        ("sealed_confirmation", True): (
            "V23_SEALED_CROSS_DOMAIN_TRANSFER_VALIDATED"
        ),
        ("sealed_confirmation", False): (
            "V23_SEALED_CROSS_DOMAIN_TRANSFER_NOT_VALIDATED"
        ),
    }[(role, passed)]
    body = dict(raw)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "real-source-relation-causal-only-report-v23",
        "status": status,
        "claim_boundary": (
            "REAL_GAME_SOURCE_BIND_TO_RELATE_GRAPH; TARGET_NATIVE_NEURAL_"
            "CAUSAL_SUCCESSOR GROUNDING; FROZEN_CAUSAL_ONLY_ADMISSION; FULL_"
            "DISJOINT_ALFWORLD_SPLIT_SUCCESS_RATE_AGAINST_GRAPH_ERASED_TARGET"
        ),
        "primary_policy": "v23_causal_only",
        "gates": gates,
        "all_gates_passed": passed,
        "confirmation_authorized": role == "development_gate" and passed,
        "cross_domain_transfer_validated": role == "sealed_confirmation" and passed,
        "v20_generic_selector_gates": raw["gates"],
        "v20_generic_selector_status_not_used": raw["status"],
    })
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "status": status,
        "role": role,
        "task_count": report["task_count"],
        "opportunity_count": report["opportunity_count"],
        "primary_metrics": primary,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
