#!/usr/bin/env python3
"""Leave-one-video-out target residual analysis for structured video transfer."""

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
    add_target_residual_to_source_models,
    build_source_value_models,
)
from motif_transfer.controlled_exploration_transfer import fit_value_ensemble  # noqa: E402
from motif_transfer.structured_video_transfer import evaluate_fixed_test_budget  # noqa: E402
from motif_transfer.video_target_residual import build_target_test_value_examples  # noqa: E402


CONDITIONS = (
    "target_native_information_gain",
    "target_native_expected_accuracy",
    "target_only_adaptation",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
    "deterministic_random_probe",
)


def _aggregate(
    rows: Sequence[Mapping[str, Any]], *, minimum_contrast: int,
) -> dict[str, Any]:
    count = len(rows)
    conditions = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in rows) / count,
        }
        for condition in CONDITIONS
    }
    authentic = conditions["authentic_source_plus_target"]["correct"]
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    oracle = sum(bool(row["oracle_correct"]) for row in rows)
    contrast = sum(bool(row["authentic_action_contrast"]) for row in rows)
    gates = {
        "all_crossfit_rows_complete": count > 0,
        "complete_native_answer_coverage": all(
            row["gold_answer"] in row["answer_space"] for row in rows
        ),
        "oracle_probe_headroom": oracle > baseline,
        "authentic_action_contrast": contrast >= minimum_contrast,
        "authentic_above_target_information_gain": authentic > conditions[
            "target_native_information_gain"
        ]["correct"],
        "authentic_above_target_expected_accuracy": authentic > conditions[
            "target_native_expected_accuracy"
        ]["correct"],
        "authentic_above_target_only_adaptation": authentic > conditions[
            "target_only_adaptation"
        ]["correct"],
        "authentic_above_shuffled": authentic > conditions[
            "shuffled_source_plus_target"
        ]["correct"],
        "authentic_above_marginal": authentic > conditions[
            "source_marginal_plus_target"
        ]["correct"],
    }
    return {
        "status": "ADAPTATION_CROSSFIT_PASS" if all(gates.values()) else "ADAPTATION_CROSSFIT_FAIL",
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
    parser.add_argument("--residual-scales", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    scales = tuple(float(value) for value in args.residual_scales.split(","))
    if not scales or any(not 0 <= value <= 1 for value in scales):
        raise SystemExit("residual scales must be in [0,1]")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    controlled = json.loads(Path(
        config["source"]["controlled_v3_config"]
    ).read_text(encoding="utf-8"))
    raw_rows = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    test_budget = int(config["interventions"]["max_tests"])
    hydrated = {}
    examples = {}
    for row in raw_rows:
        sample_id = str(row["sample_id"])
        world_model, receipts = runner._rehydrate(row)
        hydrated[sample_id] = (row, world_model, receipts)
        examples[sample_id] = build_target_test_value_examples(
            sample_id=sample_id,
            gold_answer=str(row["gold_answer"]),
            world_model=world_model,
            probe_receipts=receipts,
            test_budget=test_budget,
        )

    source_models = build_source_value_models(
        controlled,
        seed=int(config["source"]["model_seed"]),
        objective_test_cost=float(config["source"]["target_objective_test_cost"]),
    )
    evaluated_by_scale: dict[float, list[dict[str, Any]]] = {
        scale: [] for scale in scales
    }
    sample_ids = sorted(hydrated)
    model_config = controlled["model"]
    for fold_index, held_id in enumerate(sample_ids):
        train_examples = tuple(
            example
            for sample_id in sample_ids if sample_id != held_id
            for example in examples[sample_id]
        )
        target_only = fit_value_ensemble(
            (), train_examples,
            seed=int(config["source"]["model_seed"]) + 10000 + fold_index,
            ensemble_size=int(model_config["ensemble_size"]),
            alpha=float(model_config["residual_ridge_alpha"]),
            target_mass=1.0,
        )
        assert target_only is not None
        raw, world_model, receipts = hydrated[held_id]
        for scale in scales:
            models = add_target_residual_to_source_models(
                controlled,
                source_models,
                train_examples,
                seed=int(config["source"]["model_seed"]) + 20000 + fold_index,
                residual_scale=scale,
            )
            models["target_only_adaptation"] = target_only
            result = evaluate_fixed_test_budget(
                sample_id=held_id,
                gold_answer=str(raw["gold_answer"]),
                world_model=world_model,
                probe_receipts=receipts,
                source_models=models,
                test_budget=test_budget,
                conditions=CONDITIONS,
                action_contrast_reference="target_only_adaptation",
            )
            result["family"] = str(raw["family"])
            result["crossfit_train_sample_count"] = len(sample_ids) - 1
            result["held_sample_gold_excluded"] = True
            evaluated_by_scale[scale].append(result)

    minimum_contrast = int(config["adaptation_gates"][
        "minimum_authentic_action_contrasts"
    ])
    reports = {
        str(scale): _aggregate(rows, minimum_contrast=minimum_contrast)
        for scale, rows in evaluated_by_scale.items()
    }
    output = {
        "schema_version": 1,
        "benchmark": str(raw_rows[0]["benchmark"]),
        "protocol": "LEAVE_ONE_VIDEO_OUT_TARGET_RESIDUAL",
        "test_budget": test_budget,
        "target_label_contract": (
            "Adaptation gold labels train realized TEST action values; each held "
            "sample is excluded from its fold and no gold enters runtime features."
        ),
        "source_contract": (
            "All source conditions receive the identical target fold; authentic, "
            "within-state shuffled, and action-marginal source priors differ only "
            "in source value structure."
        ),
        "scales": reports,
        "any_scale_passed": any(row["status"].endswith("PASS") for row in reports.values()),
    }
    output_path = args.run_dir / "adaptation_crossfit_residual_report.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "benchmark": output["benchmark"],
        "any_scale_passed": output["any_scale_passed"],
        "scales": {
            scale: {
                "status": row["status"],
                "conditions": row["conditions"],
                "gates": row["gates"],
            }
            for scale, row in reports.items()
        },
        "report": str(output_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
