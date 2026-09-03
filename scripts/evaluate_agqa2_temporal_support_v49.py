#!/usr/bin/env python3
"""Evaluate the frozen V48 temporal-support applicability rule."""

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
from motif_transfer.agqa_temporal_support_calibrator import (  # noqa: E402
    TemporalSupportRule,
    apply_temporal_support_rule,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_robust_temporal_v34_formal as core  # noqa: E402


def _load_rule(config: dict) -> tuple[TemporalSupportRule, dict]:
    spec = config["temporal_support_calibration"]
    artifact = json.loads((REPO_ROOT / spec["artifact"]).read_text())
    body = dict(artifact)
    claimed = body.pop("artifact_sha256")
    if stable_hash(body) != claimed or claimed != spec["artifact_sha256"]:
        raise ValueError("temporal-support artifact hash mismatch")
    rule = TemporalSupportRule.from_mapping(artifact["rule"])
    if rule.rule_sha256 != spec["rule_sha256"]:
        raise ValueError("temporal-support rule hash mismatch")
    return rule, artifact


def evaluate_calibrated(
    *, config_path: Path, base_report_path: Path, output_path: Path,
    formal: bool,
) -> dict:
    config = json.loads(config_path.read_text())
    rule, artifact = _load_rule(config)

    def calibrated_binding(**kwargs):
        binding = bind_aggregate_temporal_pair_program(**kwargs)
        return apply_temporal_support_rule(binding, rule)

    replacements = {
        "bind_robust_temporal_pair_program": calibrated_binding,
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
            config_path=config_path, base_report_path=base_report_path,
            output_path=output_path,
        )
    finally:
        for name, value in originals.items():
            setattr(core, name, value)
    body = deepcopy(result)
    body.pop("report_sha256", None)
    qualified = all(body["qualification_gates"].values())
    body.update({
        "schema_version": (
            "agqa2-temporal-support-v50-formal-report-v1" if formal else
            "agqa2-temporal-support-v49-qualification-report-v1"
        ),
        "status": (
            "AGQA2_TEMPORAL_SUPPORT_V50_FORMAL_QUALIFIED"
            if formal and qualified else
            "AGQA2_TEMPORAL_SUPPORT_V50_FORMAL_NOT_QUALIFIED"
            if formal else
            "AGQA2_TEMPORAL_SUPPORT_V49_QUALIFICATION_QUALIFIED"
            if qualified else
            "AGQA2_TEMPORAL_SUPPORT_V49_QUALIFICATION_NOT_QUALIFIED"
        ),
        "split": "fresh_formal" if formal else "fresh_train_qualification",
        "confirmatory_claim": bool(formal and qualified),
        "calibration_artifact_sha256": artifact["artifact_sha256"],
        "calibration_rule_sha256": rule.rule_sha256,
        "allowed_singleton_views": list(rule.allowed_singleton_views),
        "minimum_cross_pair_gap": rule.minimum_cross_pair_gap,
        "maximum_within_operand_endpoint_spread": (
            rule.maximum_within_operand_endpoint_spread
        ),
        "minimum_max_interval_span": rule.minimum_max_interval_span,
        "runtime_calibrator_authority": (
            "ABSTENTION_ONLY;NO_INTERVAL_RELATION_OR_BINDING_CREATION_OR_EDIT"
        ),
        "current_outcome_used_for_calibration": False,
        "prior_failed_splits_reclassified_as_success": False,
    })
    final = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_temporal_support_v49_qualification.json",
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=REPO_ROOT / "runs/agqa2_temporal_support_v49_qualification/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_temporal_support_v49_qualification/report.json",
    )
    args = parser.parse_args()
    result = evaluate_calibrated(
        config_path=args.config.resolve(), base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(), formal=False,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "rows", "source_executor_authorizations",
        "source_vs_target_native", "qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
