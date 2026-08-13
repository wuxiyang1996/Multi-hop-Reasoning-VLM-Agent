#!/usr/bin/env python3
"""Evaluate variable-budget TEST/COMMIT transfer on saved video receipts."""

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
from motif_transfer.active_video_transfer import build_source_value_models  # noqa: E402
from motif_transfer.structured_video_transfer import evaluate_adaptive_test_commit  # noqa: E402


CONDITIONS = (
    "target_native_greedy_test_commit",
    "target_native_exact_dp_test_commit",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
)


def _aggregate(
    rows: Sequence[Mapping[str, Any]], *, minimum_contrast: int,
) -> dict[str, Any]:
    count = len(rows)
    conditions = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in rows) / count,
            "mean_tests": sum(
                int(row["conditions"][condition]["test_count"]) for row in rows
            ) / count,
            "mean_net_utility": sum(
                float(row["conditions"][condition]["net_utility"]) for row in rows
            ) / count,
        }
        for condition in CONDITIONS
    }
    authentic = conditions["authentic_source_plus_target"]["correct"]
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    oracle = sum(bool(row["oracle_correct"]) for row in rows)
    contrast = sum(bool(row["authentic_action_contrast"]) for row in rows)
    gates = {
        "all_rows_complete": count > 0,
        "complete_native_answer_coverage": all(
            row["gold_answer"] in row["answer_space"] for row in rows
        ),
        "oracle_probe_headroom": oracle > baseline,
        "authentic_action_contrast": contrast >= minimum_contrast,
        "authentic_above_target_greedy": authentic > conditions[
            "target_native_greedy_test_commit"
        ]["correct"],
        "authentic_above_target_exact_dp": authentic > conditions[
            "target_native_exact_dp_test_commit"
        ]["correct"],
        "authentic_above_shuffled": authentic > conditions[
            "shuffled_source_plus_target"
        ]["correct"],
        "authentic_above_marginal": authentic > conditions[
            "source_marginal_plus_target"
        ]["correct"],
    }
    return {
        "status": "ADAPTATION_ADAPTIVE_PASS" if all(gates.values()) else "ADAPTATION_ADAPTIVE_FAIL",
        "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle": {"correct": oracle, "accuracy": oracle / count},
        "authentic_action_contrasts": contrast,
        "conditions": conditions,
        "gates": gates,
        "rows": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-costs", default="0,0.025,0.05,0.1")
    args = parser.parse_args()
    costs = tuple(float(value) for value in args.test_costs.split(","))
    if not costs or any(not 0 <= value < 1 for value in costs):
        raise SystemExit("test costs must be in [0,1)")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    controlled = json.loads(Path(
        config["source"]["controlled_v3_config"]
    ).read_text(encoding="utf-8"))
    raw_rows = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    hydrated = [(row, *runner._rehydrate(row)) for row in raw_rows]
    max_tests = int(config["interventions"]["max_tests"])
    reports = {}
    for cost_index, test_cost in enumerate(costs):
        source_models = build_source_value_models(
            controlled,
            seed=int(config["source"]["model_seed"]) + 100 * cost_index,
            objective_test_cost=test_cost,
        )
        evaluated = []
        for row, world_model, receipts in hydrated:
            result = evaluate_adaptive_test_commit(
                sample_id=str(row["sample_id"]),
                gold_answer=str(row["gold_answer"]),
                world_model=world_model,
                probe_receipts=receipts,
                source_models=source_models,
                max_tests=max_tests,
                test_cost=test_cost,
            )
            result["family"] = str(row["family"])
            evaluated.append(result)
        reports[str(test_cost)] = _aggregate(
            evaluated,
            minimum_contrast=int(config["adaptation_gates"][
                "minimum_authentic_action_contrasts"
            ]),
        )
    output = {
        "schema_version": 1,
        "benchmark": str(raw_rows[0]["benchmark"]),
        "protocol": "ZERO_SHOT_SOURCE_TEST_COMMIT_WITH_TARGET_NATIVE_EXACT_DP_CONTROL",
        "max_tests": max_tests,
        "costs": reports,
        "any_cost_passed": any(row["status"].endswith("PASS") for row in reports.values()),
        "claim_boundary": (
            "Adaptation-only outcome analysis. Test cost is an explicit development "
            "hyperparameter; no qualification or held-out outcome is read."
        ),
    }
    output_path = args.run_dir / "adaptation_adaptive_test_commit_report.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "benchmark": output["benchmark"],
        "any_cost_passed": output["any_cost_passed"],
        "costs": {
            cost: {
                "status": row["status"],
                "baseline": row["baseline"],
                "oracle": row["oracle"],
                "conditions": row["conditions"],
                "gates": row["gates"],
            }
            for cost, row in reports.items()
        },
        "report": str(output_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
