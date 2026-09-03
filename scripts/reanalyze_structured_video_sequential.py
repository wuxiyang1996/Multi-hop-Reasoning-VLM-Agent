#!/usr/bin/env python3
"""Reanalyze matched video probes under one- and two-TEST MDP budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.active_video_transfer import (  # noqa: E402
    build_source_value_models,
    source_test_feature_support,
)
from motif_transfer.structured_video_transfer import (  # noqa: E402
    FIXED_TEST_CONDITIONS,
    evaluate_fixed_test_budget,
)


def _aggregate(
    evaluated: Sequence[Mapping[str, Any]],
    *,
    minimum_contrast: int,
) -> dict[str, Any]:
    count = len(evaluated)
    baseline = sum(bool(row["baseline_correct"]) for row in evaluated)
    oracle = sum(bool(row["oracle_correct"]) for row in evaluated)
    coverage = sum(row["gold_answer"] in row["answer_space"] for row in evaluated)
    contrast = sum(bool(row["authentic_action_contrast"]) for row in evaluated)
    conditions = {
        condition: {
            "correct": sum(
                bool(row["conditions"][condition]["correct"]) for row in evaluated
            ),
            "accuracy": sum(
                bool(row["conditions"][condition]["correct"]) for row in evaluated
            ) / count,
        }
        for condition in FIXED_TEST_CONDITIONS
    }
    authentic = conditions["authentic_source_plus_target"]["correct"]
    gates = {
        "all_receipts_complete": count > 0,
        "gold_answer_world_coverage_at_least_75pct": coverage / count >= 0.75,
        "oracle_probe_headroom": oracle > baseline,
        "authentic_action_contrast": contrast >= minimum_contrast,
        "authentic_above_target_information_gain": authentic > conditions[
            "target_native_information_gain"
        ]["correct"],
        "authentic_above_target_expected_accuracy": authentic > conditions[
            "target_native_expected_accuracy"
        ]["correct"],
        "authentic_above_shuffled": authentic > conditions[
            "shuffled_source_plus_target"
        ]["correct"],
        "authentic_above_marginal": authentic > conditions[
            "source_marginal_plus_target"
        ]["correct"],
    }
    return {
        "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle": {"correct": oracle, "accuracy": oracle / count},
        "gold_answer_world_coverage": {
            "samples": coverage, "fraction": coverage / count,
        },
        "authentic_action_contrasts": contrast,
        "conditions": conditions,
        "gates": gates,
        "status": (
            "ADAPTATION_PREFLIGHT_PASS" if all(gates.values())
            else "ADAPTATION_PREFLIGHT_FAIL"
        ),
        "rows": list(evaluated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipts_path = args.run_dir / "receipts.json"
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    controlled = json.loads(Path(
        config["source"]["controlled_v3_config"]
    ).read_text(encoding="utf-8"))
    source_models = build_source_value_models(
        controlled,
        seed=int(config["source"]["model_seed"]),
        objective_test_cost=float(config["source"]["target_objective_test_cost"]),
    )
    budgets = {}
    for budget in (1, 2):
        evaluated = []
        for row in receipts:
            world_model, probe_receipts = runner._rehydrate(row)
            result = evaluate_fixed_test_budget(
                sample_id=str(row["sample_id"]),
                gold_answer=str(row["gold_answer"]),
                world_model=world_model,
                probe_receipts=probe_receipts,
                source_models=source_models,
                test_budget=budget,
            )
            result["family"] = str(row["family"])
            evaluated.append(result)
        budgets[str(budget)] = _aggregate(
            evaluated,
            minimum_contrast=int(config["adaptation_gates"][
                "minimum_authentic_action_contrasts"
            ]),
        )
    report = {
        "schema_version": 1,
        "benchmark": str(receipts[0]["benchmark"]),
        "status": budgets["2"]["status"],
        "primary_budget": 2,
        "budgets": budgets,
        "source_test_feature_support": source_test_feature_support(
            controlled, objective_test_cost=float(
                config["source"]["target_objective_test_cost"]
            ),
        ),
        "claim_boundary": (
            "Adaptation-only matched-probe sequential reanalysis; no new model calls, "
            "qualification, or held-out outcomes."
        ),
        "receipts": {
            "path": str(receipts_path.resolve()),
            "sha256": runner.media_helpers.file_sha256(receipts_path),
        },
    }
    output = args.run_dir / "adaptation_sequential_reanalysis_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "benchmark": report["benchmark"],
        "status": report["status"],
        "one_test": budgets["1"]["conditions"],
        "two_test": budgets["2"]["conditions"],
        "two_test_gates": budgets["2"]["gates"],
        "report": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
