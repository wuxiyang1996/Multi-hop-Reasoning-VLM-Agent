#!/usr/bin/env python3
"""Evaluate the operator-level recurrent AGQA temporal binding."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_aggregate_temporal_transfer import (  # noqa: E402
    bind_aggregate_temporal_pair_program,
    build_aggregate_temporal_harness,
    build_aggregate_temporal_route,
    decide_aggregate_temporal_relation,
    unified_aggregate_temporal_grounding,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_robust_temporal_v34_formal as core  # noqa: E402


def evaluate_aggregate(
    *, config_path: Path, base_report_path: Path, output_path: Path,
    formal: bool,
) -> dict:
    replacements = {
        "bind_robust_temporal_pair_program": (
            bind_aggregate_temporal_pair_program
        ),
        "build_temporal_harness": build_aggregate_temporal_harness,
        "build_temporal_route": build_aggregate_temporal_route,
        "decide_temporal_relation": decide_aggregate_temporal_relation,
        "unified_temporal_grounding": unified_aggregate_temporal_grounding,
    }
    originals = {name: getattr(core, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(core, name, value)
        result = core.evaluate(
            config_path=config_path,
            base_report_path=base_report_path,
            output_path=output_path,
        )
    finally:
        for name, value in originals.items():
            setattr(core, name, value)
    body = deepcopy(result)
    body.pop("report_sha256", None)
    qualified = all(body["qualification_gates"].values())
    version = "v39-formal" if formal else "v38-development"
    status = (
        "AGQA2_AGGREGATE_TEMPORAL_V39_FORMAL_QUALIFIED"
        if formal and qualified else
        "AGQA2_AGGREGATE_TEMPORAL_V39_FORMAL_NOT_QUALIFIED"
        if formal else
        "AGQA2_AGGREGATE_TEMPORAL_V38_DEVELOPMENT_METHOD_SELECTED"
        if qualified else
        "AGQA2_AGGREGATE_TEMPORAL_V38_DEVELOPMENT_METHOD_REJECTED"
    )
    body.update({
        "schema_version": f"agqa2-aggregate-temporal-{version}-report-v1",
        "status": status,
        "split": "fresh_formal" if formal else "consumed_development",
        "confirmatory_claim": bool(formal and qualified),
        "method_selected_after_v37_development_outcome_access": not formal,
        "binding_semantics": (
            "TYPED_BINARY_ARGUMENTS_PLUS_OPERATOR_LEVEL_RECURRENCE_PLUS_"
            "STRICT_CONSISTENT_ALL_PAIRS_RELATION"
        ),
        "v34_v35_v36_v37_reclassified_as_success": False,
    })
    final = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_aggregate_temporal_v38_development.json",
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=(
            REPO_ROOT
            / "runs/agqa2_robust_temporal_v36_development/base_report.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_aggregate_temporal_v38_development/report.json",
    )
    args = parser.parse_args()
    result = evaluate_aggregate(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
        formal=False,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
