#!/usr/bin/env python3
"""Train and source-qualify the non-heuristic V16 pairwise option controller."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.hierarchical_skill_transfer import (  # noqa: E402
    HierarchicalValueExample,
    collect_source_examples,
    phase_permuted_control,
    shuffled_value_control,
)
from motif_transfer.pairwise_option_advantage import (  # noqa: E402
    PairwiseAdvantageEnsemble,
    choose_option_against_fallback,
    conformal_error_quantile,
    fit_pairwise_ensemble,
    option_scores,
    pairwise_examples,
    phase_blind_rows,
    serialize_pairwise_ensemble,
)


CONDITIONS = (
    "authentic",
    "phase_permuted",
    "within_state_value_shuffle",
    "phase_blind",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows(config: Mapping, split: str) -> tuple[HierarchicalValueExample, ...]:
    source = config["source"]
    workflow = source["workflow"]
    rows = collect_source_examples(
        surfaces=tuple(map(str, source["surfaces"])),
        domains_per_surface=int(source[f"{split}_domains_per_surface"]),
        states_per_domain=int(source["states_per_domain"]),
        seed=int(source[f"{split}_seed"]),
        minimum_budget=int(workflow["minimum_budget"]),
        maximum_budget=int(workflow["maximum_budget"]),
        completion_probability_range=workflow["completion_probability_range"],
        failure_cost_range=workflow["failure_cost_range"],
        progress_reward=float(workflow["progress_reward"]),
        invalid_option_cost=float(workflow["invalid_option_cost"]),
    )
    return tuple(replace(row, state_id=f"{split}:{row.state_id}") for row in rows)


def _fit_models(
    train: Sequence[HierarchicalValueExample], config: Mapping
) -> dict[str, PairwiseAdvantageEnsemble]:
    source = config["source"]
    treatments = {
        "authentic": tuple(train),
        "phase_permuted": phase_permuted_control(train),
        "within_state_value_shuffle": shuffled_value_control(
            train, seed=int(source["control_seed"])
        ),
        "phase_blind": phase_blind_rows(train),
    }
    return {
        name: fit_pairwise_ensemble(
            pairwise_examples(rows),
            seed=int(source["ensemble_seed"]),
            ensemble_size=int(source["ensemble_size"]),
            alpha=float(source["ridge_alpha"]),
        )
        for name, rows in treatments.items()
    }


def _calibration_rows(
    rows: Sequence[HierarchicalValueExample], condition: str
) -> tuple[HierarchicalValueExample, ...]:
    return phase_blind_rows(rows) if condition == "phase_blind" else tuple(rows)


def _pair_metrics(
    model: PairwiseAdvantageEnsemble,
    rows: Sequence[HierarchicalValueExample],
) -> dict[str, float]:
    pairs = pairwise_examples(rows)
    predictions, _ = model.predict([row.features for row in pairs])
    truth = np.asarray([row.advantage for row in pairs], dtype=np.float64)
    nonzero = np.abs(truth) > 1e-12
    return {
        "pairs": len(pairs),
        "rmse": float(np.sqrt(np.mean((predictions - truth) ** 2))),
        "sign_accuracy": float(np.mean(
            np.sign(predictions[nonzero]) == np.sign(truth[nonzero])
        )),
    }


def _group_options(
    rows: Sequence[HierarchicalValueExample],
) -> dict[str, dict[str, HierarchicalValueExample]]:
    groups: dict[str, dict[str, HierarchicalValueExample]] = defaultdict(dict)
    for row in rows:
        groups[row.state_id][row.option] = row
    if any(len(group) != 5 for group in groups.values()):
        raise ValueError("source state does not contain exactly five options")
    return dict(groups)


def _state_metadata(state_id: str) -> tuple[str, str]:
    parts = state_id.split(":")
    if len(parts) != 4 or parts[0] != "heldout":
        raise ValueError(f"unexpected heldout state identity: {state_id}")
    return parts[1], f"{parts[1]}:{parts[2]}"


def _evaluate_treatments(
    models: Mapping[str, PairwiseAdvantageEnsemble],
    quantiles: Mapping[str, float],
    rows: Sequence[HierarchicalValueExample],
) -> list[dict]:
    authentic_groups = _group_options(rows)
    blind_groups = _group_options(phase_blind_rows(rows))
    output = []
    for state_id in sorted(authentic_groups):
        original = authentic_groups[state_id]
        blind = blind_groups[state_id]
        fallback_scores = option_scores(
            models["phase_blind"],
            {name: row.features for name, row in blind.items()},
        )
        fallback = max(fallback_scores, key=lambda name: (fallback_scores[name], name))
        surface, domain_id = _state_metadata(state_id)
        baseline_value = float(original[fallback].value)
        for condition in CONDITIONS:
            if condition == "phase_blind":
                selected = fallback
                admitted = False
                lower_bound = None
                predicted_advantage = None
            else:
                decision = choose_option_against_fallback(
                    models[condition],
                    {name: row.features for name, row in original.items()},
                    fallback_option=fallback,
                    conformal_error=float(quantiles[condition]),
                )
                selected = str(decision["option"])
                admitted = bool(decision["source_admitted"])
                lower_bound = float(
                    decision["comparison"]["conformal_lower_bound"]
                )
                predicted_advantage = float(
                    decision["comparison"]["predicted_advantage"]
                )
            output.append({
                "state_id": state_id,
                "surface": surface,
                "domain_id": domain_id,
                "condition": condition,
                "fallback_option": fallback,
                "selected_option": selected,
                "source_admitted": admitted,
                "predicted_advantage": predicted_advantage,
                "conformal_lower_bound": lower_bound,
                "true_utility": float(original[selected].value - baseline_value),
                "selected_is_optimal": bool(
                    original[selected].value
                    >= max(row.value for row in original.values()) - 1e-12
                ),
            })
    return output


def _condition_metrics(rows: Sequence[dict]) -> dict[str, dict]:
    output = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        admitted = [row for row in subset if row["source_admitted"]]
        by_surface = {
            surface: float(np.mean([
                row["true_utility"] for row in subset if row["surface"] == surface
            ]))
            for surface in sorted({row["surface"] for row in subset})
        }
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
            "mean_true_utility_by_surface": by_surface,
        }
    return output


def _domain_means(rows: Sequence[dict], condition: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            grouped[row["domain_id"]].append(float(row["true_utility"]))
    return {domain: float(np.mean(values)) for domain, values in grouped.items()}


def _stratified_bootstrap_lower_bound(
    values: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    confidence_alpha: float,
) -> float:
    by_surface: dict[str, list[float]] = defaultdict(list)
    for domain_id, value in values.items():
        surface = domain_id.split(":", 1)[0]
        by_surface[surface].append(float(value))
    rng = np.random.default_rng(seed)
    samples = []
    for _index in range(resamples):
        surface_means = []
        for surface in sorted(by_surface):
            observed = np.asarray(by_surface[surface], dtype=np.float64)
            draw = rng.choice(observed, size=len(observed), replace=True)
            surface_means.append(float(np.mean(draw)))
        samples.append(float(np.mean(surface_means)))
    return float(np.quantile(samples, confidence_alpha / 2.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite V16 source artifacts")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != (
        "FROZEN_BEFORE_V16_SOURCE_CALIBRATION_OR_HELDOUT_READ"
    ):
        raise SystemExit("V16 source config was not frozen before evaluation")
    if config.get("target_authorized_by_this_config"):
        raise SystemExit("V16 source config improperly authorizes target execution")

    splits = {
        split: _source_rows(config, split)
        for split in ("train", "calibration", "heldout")
    }
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
            pairwise_examples(_calibration_rows(
                splits["calibration"], condition
            )),
            alpha=conformal_alpha,
        )
        for condition in CONDITIONS
    }
    pair_metrics = {
        condition: _pair_metrics(
            models[condition],
            _calibration_rows(splits["heldout"], condition),
        )
        for condition in CONDITIONS
    }
    decisions = _evaluate_treatments(models, quantiles, splits["heldout"])
    metrics = _condition_metrics(decisions)
    gate_config = config["source_gate"]
    resamples = int(gate_config["bootstrap_resamples"])
    bootstrap_seed = int(gate_config["bootstrap_seed"])
    confidence_alpha = float(gate_config["confidence_alpha"])
    authentic_domains = _domain_means(decisions, "authentic")
    lower_authentic = _stratified_bootstrap_lower_bound(
        authentic_domains,
        resamples=resamples,
        seed=bootstrap_seed,
        confidence_alpha=confidence_alpha,
    )
    difference_bounds = {}
    for offset, control in enumerate(CONDITIONS[1:], start=1):
        control_domains = _domain_means(decisions, control)
        differences = {
            domain: authentic_domains[domain] - control_domains[domain]
            for domain in authentic_domains
        }
        difference_bounds[control] = _stratified_bootstrap_lower_bound(
            differences,
            resamples=resamples,
            seed=bootstrap_seed + offset,
            confidence_alpha=confidence_alpha,
        )
    gates = {
        "zero_invalid_or_cross_split_state_ids": not cross_split_overlap,
        "stratified_bootstrap_lower_bound_authentic_utility_gt_zero": (
            lower_authentic > 0.0
        ),
        "stratified_bootstrap_lower_bound_authentic_minus_phase_permuted_gt_zero": (
            difference_bounds["phase_permuted"] > 0.0
        ),
        "stratified_bootstrap_lower_bound_authentic_minus_value_shuffle_gt_zero": (
            difference_bounds["within_state_value_shuffle"] > 0.0
        ),
        "stratified_bootstrap_lower_bound_authentic_minus_phase_blind_gt_zero": (
            difference_bounds["phase_blind"] > 0.0
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
        "schema_version": "pairwise-option-advantage-source-candidate-v16",
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
        "source_gates": gates,
        "runtime_hashes": runtime,
        "target_authorized": passed,
    }
    candidate = candidate_body | {
        "candidate_sha256": stable_hash(candidate_body)
    }
    report_body = {
        "schema_version": "pairwise-option-advantage-source-report-v16",
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
        "cross_split_state_id_overlap": cross_split_overlap,
        "heldout_pair_metrics": pair_metrics,
        "heldout_treatment_metrics": metrics,
        "bootstrap": {
            "resamples": resamples,
            "confidence_alpha": confidence_alpha,
            "authentic_utility_lower_bound": lower_authentic,
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
        "heldout_treatment_metrics": metrics,
        "bootstrap": report["bootstrap"],
        "gates": gates,
        "candidate_sha256": candidate["candidate_sha256"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
