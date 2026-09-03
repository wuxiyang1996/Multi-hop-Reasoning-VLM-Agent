#!/usr/bin/env python3
"""Run V24 sealed paired evaluation and apply neural-risk transfer gates."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V24 report: {args.output}")
    plan = _read(args.plan)
    plan_hash = _validate_hash(plan, "plan_sha256")
    if plan.get("schema_version") != "real-source-relation-neural-risk-plan-v24":
        raise SystemExit("unexpected V24 plan schema")
    if plan.get("role") != "sealed_confirmation":
        raise SystemExit("V24 runner only permits sealed confirmation")
    with tempfile.TemporaryDirectory(prefix="v24-eval-") as directory:
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
            raise SystemExit("V24 underlying matched evaluation produced no report")
        raw = _read(raw_path)
    metrics = raw["policy_metrics"]
    primary = metrics["v24_neural_risk"]
    always = metrics["always_source_edge"]
    lexical = metrics["lexical_move_relation"]
    late = metrics["late_step_heuristic"]
    old_selective = metrics["v20_selective"]
    gate_spec = plan["v24_gates"]
    invariants = all(all(row["invariants"].values()) for row in raw["forks"])
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
        "primary_loss_count_strictly_less_than_always_source": (
            primary["success_losses"] < always["success_losses"]
        ),
        "primary_net_delta_strictly_greater_than_lexical": (
            primary["success_delta"] > lexical["success_delta"]
        ),
        "primary_net_delta_strictly_greater_than_late_step": (
            primary["success_delta"] > late["success_delta"]
        ),
        "primary_net_delta_strictly_greater_than_v20_selective": (
            primary["success_delta"] > old_selective["success_delta"]
        ),
        "all_exact_state_fork_invariants": invariants,
    }
    passed = all(gates.values())
    status = (
        "V24_SEALED_CROSS_DOMAIN_TRANSFER_VALIDATED"
        if passed else "V24_SEALED_CROSS_DOMAIN_TRANSFER_NOT_VALIDATED"
    )
    body = dict(raw)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "real-source-relation-neural-risk-report-v24",
        "status": status,
        "claim_boundary": (
            "REAL_MINIGRID_AND_MINIWORLD_SOURCE_BIND_TO_RELATE_GRAPH; "
            "TARGET_NATIVE_NEURAL_CAUSAL_AND_RISK_GROUNDING; FROZEN_SMALL_"
            "MLP_ADMISSION; FULL_DISJOINT_ALFWORLD_SUCCESS_RATE_AGAINST_"
            "GRAPH_ERASED_TARGET_AND_FROZEN_NEGATIVE_CONTROLS"
        ),
        "primary_policy": "v24_neural_risk",
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan_hash,
        },
        "gates": gates,
        "all_gates_passed": passed,
        "confirmation_authorized": False,
        "cross_domain_transfer_validated": passed,
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
        "task_count": report["task_count"],
        "opportunity_count": report["opportunity_count"],
        "primary_metrics": primary,
        "negative_control_metrics": {
            name: metrics[name]
            for name in (
                "always_source_edge", "causal_effect_only",
                "lexical_move_relation", "late_step_heuristic", "v20_selective",
            )
        },
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
