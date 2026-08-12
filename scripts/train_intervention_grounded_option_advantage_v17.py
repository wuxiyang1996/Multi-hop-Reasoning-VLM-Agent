#!/usr/bin/env python3
"""Train V17 only from receipt-derived source intervention effects."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.hierarchical_skill_transfer import (  # noqa: E402
    FEATURE_NAMES,
    HierarchicalValueExample,
    shuffled_value_control,
)
from motif_transfer.pairwise_option_advantage import (  # noqa: E402
    PairwiseAdvantageEnsemble,
    choose_option_against_fallback,
    conformal_error_quantile,
    effect_blind_rows,
    fit_pairwise_ensemble,
    intervention_grounded_rows,
    option_scores,
    pairwise_examples,
    serialize_pairwise_ensemble,
    within_state_effect_permutation,
)
from train_pairwise_option_advantage_v16 import (  # noqa: E402
    _pair_metrics,
    _sha256,
    _source_rows,
    _stratified_bootstrap_lower_bound,
)


CONDITIONS = (
    "authentic_intervention_effect",
    "within_state_effect_permutation",
    "within_state_value_shuffle",
    "effect_blind",
)


def _grounded_splits(config: Mapping) -> dict[str, tuple[HierarchicalValueExample, ...]]:
    source = config["source"]
    return {
        split: intervention_grounded_rows(
            _source_rows(config, split),
            probe_trials=int(source["matched_probe_trials_per_state_option"]),
            probe_seed=int(source["probe_seed"]),
        )
        for split in ("train", "calibration", "heldout")
    }


def _fit_models(
    train: Sequence[HierarchicalValueExample], config: Mapping
) -> dict[str, PairwiseAdvantageEnsemble]:
    source = config["source"]
    treatments = {
        "authentic_intervention_effect": tuple(train),
        "within_state_effect_permutation": within_state_effect_permutation(train),
        "within_state_value_shuffle": shuffled_value_control(
            train, seed=int(source["control_seed"])
        ),
        "effect_blind": effect_blind_rows(train),
    }
    return {
        condition: fit_pairwise_ensemble(
            pairwise_examples(rows),
            seed=int(source["ensemble_seed"]),
            ensemble_size=int(source["ensemble_size"]),
            alpha=float(source["ridge_alpha"]),
        )
        for condition, rows in treatments.items()
    }


def _condition_rows(
    rows: Sequence[HierarchicalValueExample], condition: str
) -> tuple[HierarchicalValueExample, ...]:
    return effect_blind_rows(rows) if condition == "effect_blind" else tuple(rows)


def _group_options(
    rows: Sequence[HierarchicalValueExample],
) -> dict[str, dict[str, HierarchicalValueExample]]:
    groups: dict[str, dict[str, HierarchicalValueExample]] = defaultdict(dict)
    for row in rows:
        groups[row.state_id][row.option] = row
    if any(len(group) != 5 for group in groups.values()):
        raise ValueError("V17 source state does not contain five options")
    return dict(groups)


def _metadata(state_id: str) -> tuple[str, str]:
    parts = state_id.split(":")
    if len(parts) != 4 or parts[0] != "heldout":
        raise ValueError(f"unexpected V17 heldout state: {state_id}")
    return parts[1], f"{parts[1]}:{parts[2]}"


def _decisions(
    models: Mapping[str, PairwiseAdvantageEnsemble],
    quantiles: Mapping[str, float],
    rows: Sequence[HierarchicalValueExample],
) -> list[dict]:
    original_groups = _group_options(rows)
    blind_groups = _group_options(effect_blind_rows(rows))
    output = []
    for state_id in sorted(original_groups):
        original = original_groups[state_id]
        blind = blind_groups[state_id]
        fallback_scores = option_scores(
            models["effect_blind"],
            {name: row.features for name, row in blind.items()},
        )
        fallback = max(fallback_scores, key=lambda name: (fallback_scores[name], name))
        baseline_value = float(original[fallback].value)
        surface, domain_id = _metadata(state_id)
        for condition in CONDITIONS:
            if condition == "effect_blind":
                selected = fallback
                admitted = False
            else:
                decision = choose_option_against_fallback(
                    models[condition],
                    {name: row.features for name, row in original.items()},
                    fallback_option=fallback,
                    conformal_error=float(quantiles[condition]),
                )
                selected = str(decision["option"])
                admitted = bool(decision["source_admitted"])
            output.append({
                "state_id": state_id,
                "surface": surface,
                "domain_id": domain_id,
                "condition": condition,
                "fallback_option": fallback,
                "selected_option": selected,
                "source_admitted": admitted,
                "true_utility": float(original[selected].value - baseline_value),
                "selected_is_optimal": bool(
                    original[selected].value
                    >= max(row.value for row in original.values()) - 1e-12
                ),
            })
    return output


def _metrics(rows: Sequence[dict]) -> dict[str, dict]:
    output = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        admitted = [row for row in subset if row["source_admitted"]]
        output[condition] = {
            "states": len(subset),
            "interventions": len(admitted),
            "intervention_rate": len(admitted) / len(subset),
            "positive_interventions": sum(row["true_utility"] > 1e-12 for row in admitted),
            "negative_interventions": sum(row["true_utility"] < -1e-12 for row in admitted),
            "mean_true_utility": float(np.mean([
                row["true_utility"] for row in subset
            ])),
            "optimal_selection_rate": float(np.mean([
                row["selected_is_optimal"] for row in subset
            ])),
            "mean_true_utility_by_surface": {
                surface: float(np.mean([
                    row["true_utility"]
                    for row in subset if row["surface"] == surface
                ]))
                for surface in sorted({row["surface"] for row in subset})
            },
        }
    return output


def _domain_means(rows: Sequence[dict], condition: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            grouped[row["domain_id"]].append(float(row["true_utility"]))
    return {domain: float(np.mean(values)) for domain, values in grouped.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite V17 source artifacts")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != (
        "FROZEN_BEFORE_V17_SOURCE_CALIBRATION_OR_HELDOUT_READ"
    ):
        raise SystemExit("V17 config was not frozen before source evaluation")
    if config.get("target_authorized_by_this_config"):
        raise SystemExit("V17 source config improperly authorizes target execution")

    splits = _grounded_splits(config)
    forbidden = set(map(str, config["forbidden_source_features"]))
    forbidden_indices = [
        index for index, name in enumerate(FEATURE_NAMES) if name in forbidden
    ]
    if len(forbidden_indices) != len(forbidden):
        raise RuntimeError("V17 forbidden feature list does not match schema")
    forbidden_values_zero = all(
        abs(row.features[index]) <= 1e-12
        for rows in splits.values()
        for row in rows
        for index in forbidden_indices
    )
    state_ids = {
        split: {row.state_id for row in rows}
        for split, rows in splits.items()
    }
    cross_split_overlap = any(
        state_ids[left] & state_ids[right]
        for left, right in (
            ("train", "calibration"),
            ("train", "heldout"),
            ("calibration", "heldout"),
        )
    )
    models = _fit_models(splits["train"], config)
    conformal_alpha = float(config["calibration"]["split_conformal_alpha"])
    quantiles = {
        condition: conformal_error_quantile(
            models[condition],
            pairwise_examples(_condition_rows(
                splits["calibration"], condition
            )),
            alpha=conformal_alpha,
        )
        for condition in CONDITIONS
    }
    pair_metrics = {
        condition: _pair_metrics(
            models[condition],
            _condition_rows(splits["heldout"], condition),
        )
        for condition in CONDITIONS
    }
    decisions = _decisions(models, quantiles, splits["heldout"])
    metrics = _metrics(decisions)
    gate_config = config["source_gate"]
    authentic_domains = _domain_means(
        decisions, "authentic_intervention_effect"
    )
    bootstrap_args = {
        "resamples": int(gate_config["bootstrap_resamples"]),
        "confidence_alpha": float(gate_config["confidence_alpha"]),
    }
    bootstrap_seed = int(gate_config["bootstrap_seed"])
    authentic_lower = _stratified_bootstrap_lower_bound(
        authentic_domains, seed=bootstrap_seed, **bootstrap_args
    )
    difference_bounds = {}
    for offset, control in enumerate(CONDITIONS[1:], start=1):
        control_domains = _domain_means(decisions, control)
        differences = {
            domain: authentic_domains[domain] - control_domains[domain]
            for domain in authentic_domains
        }
        difference_bounds[control] = _stratified_bootstrap_lower_bound(
            differences, seed=bootstrap_seed + offset, **bootstrap_args
        )
    gates = {
        "zero_forbidden_feature_values": forbidden_values_zero,
        "zero_invalid_or_cross_split_state_ids": not cross_split_overlap,
        "authentic_utility_stratified_bootstrap_lower_bound_gt_zero": (
            authentic_lower > 0.0
        ),
        "authentic_minus_effect_permutation_lower_bound_gt_zero": (
            difference_bounds["within_state_effect_permutation"] > 0.0
        ),
        "authentic_minus_value_shuffle_lower_bound_gt_zero": (
            difference_bounds["within_state_value_shuffle"] > 0.0
        ),
        "authentic_minus_effect_blind_lower_bound_gt_zero": (
            difference_bounds["effect_blind"] > 0.0
        ),
    }
    passed = all(gates.values())
    runtime = {
        "config": _sha256(args.config),
        "trainer": _sha256(Path(__file__)),
        "pairwise_module": _sha256(
            REPO / "src/motif_transfer/pairwise_option_advantage.py"
        ),
        "source_generator": _sha256(
            REPO / "src/motif_transfer/hierarchical_skill_transfer.py"
        ),
    }
    candidate_body = {
        "schema_version": "intervention-grounded-option-advantage-candidate-v17",
        "status": "SOURCE_GATE_PASSED" if passed else "SOURCE_GATE_FAILED_STOP",
        "claim_boundary": config["claim_boundary"],
        "config": {
            "path": str(args.config.resolve()),
            "file_sha256": runtime["config"],
        },
        "models": {
            condition: serialize_pairwise_ensemble(model)
            for condition, model in models.items()
        },
        "conformal": {
            "alpha": conformal_alpha,
            "overprediction_error_quantiles": quantiles,
        },
        "forbidden_source_features": sorted(forbidden),
        "source_gates": gates,
        "runtime_hashes": runtime,
        "target_authorized": passed,
    }
    candidate = candidate_body | {
        "candidate_sha256": stable_hash(candidate_body)
    }
    report_body = {
        "schema_version": "intervention-grounded-option-advantage-report-v17",
        "status": candidate["status"],
        "claim_boundary": config["claim_boundary"],
        "split_counts": {
            split: {
                "states": len(state_ids[split]),
                "option_rows": len(rows),
                "pairwise_rows": len(pairwise_examples(rows)),
            }
            for split, rows in splits.items()
        },
        "forbidden_feature_values_zero": forbidden_values_zero,
        "cross_split_state_id_overlap": cross_split_overlap,
        "heldout_pair_metrics": pair_metrics,
        "heldout_treatment_metrics": metrics,
        "bootstrap": {
            **bootstrap_args,
            "authentic_utility_lower_bound": authentic_lower,
            "authentic_minus_control_lower_bounds": difference_bounds,
        },
        "gates": gates,
        "candidate_sha256": candidate["candidate_sha256"],
        "runtime_hashes": runtime,
        "target_read_or_run": False,
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.candidate.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "forbidden_feature_values_zero": forbidden_values_zero,
        "heldout_treatment_metrics": metrics,
        "bootstrap": report["bootstrap"],
        "gates": gates,
        "candidate_sha256": candidate["candidate_sha256"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
