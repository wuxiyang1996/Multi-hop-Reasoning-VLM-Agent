#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_neural_grounder import (  # noqa: E402
    grounder_features,
    mlp_score,
)
from motif_transfer.controlled_exploration_transfer import (  # noqa: E402
    MatchedValueExample,
    calibrate_target_grounder,
    collect_matched_examples,
    fit_value_ensemble,
    make_domain,
    marginal_value_control,
    shuffled_value_control,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_seed(value: object) -> int:
    return int(hashlib.sha256(repr(value).encode()).hexdigest()[:16], 16) % (2**32)


def _grounder_rows(
    episodes: Sequence[dict[str, Any]], feature_bins: int, negatives: int
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for episode in episodes:
        history: list[str] = []
        for transition in episode["transitions"]:
            expert = str(transition["expert_action"])
            candidates = [str(action) for action in transition["native_actions"]]
            ranked_negatives = sorted(
                (action for action in candidates if action != expert),
                key=lambda action: hashlib.sha256(
                    f"{episode['task_id']}\0{transition['step']}\0{action}".encode()
                ).hexdigest(),
            )[:negatives]
            for action, label in [(expert, 1), *((item, 0) for item in ranked_negatives)]:
                features.append(grounder_features(
                    goal=str(transition["goal"]),
                    observation=str(transition["before_observation"]),
                    action=action,
                    step=int(transition["step"]),
                    action_history=history,
                    feature_bins=feature_bins,
                ))
                labels.append(label)
            history.append(expert)
    return np.asarray(features), np.asarray(labels)


def _fit_grounder(episodes, config):
    features, labels = _grounder_rows(
        episodes,
        int(config["feature_bins"]),
        int(config["negative_candidates_per_state"]),
    )
    model = MLPClassifier(
        hidden_layer_sizes=(int(config["hidden_units"]),),
        activation="tanh",
        solver="lbfgs",
        alpha=0.1,
        max_iter=int(config["maximum_iterations"]),
        random_state=int(config["random_seed"]),
    )
    model.fit(features, labels)
    artifact = {
        "kind": "target-native-one-hidden-layer-mlp",
        "feature_version": "alfworld-grounder-features-v1",
        "feature_bins": int(config["feature_bins"]),
        "hidden_activation": "tanh",
        "layers": [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in zip(model.coefs_, model.intercepts_, strict=True)
        ],
        "training_examples": len(labels),
        "positive_examples": int(np.sum(labels)),
    }
    return model, artifact


def _top1(episodes, model_artifact):
    correct = 0
    states = 0
    random_mass = 0.0
    for episode in episodes:
        history: list[str] = []
        for transition in episode["transitions"]:
            candidates = list(map(str, transition["native_actions"]))
            scored = []
            for action in candidates:
                features = grounder_features(
                    goal=str(transition["goal"]),
                    observation=str(transition["before_observation"]),
                    action=action,
                    step=int(transition["step"]),
                    action_history=history,
                    feature_bins=int(model_artifact["feature_bins"]),
                )
                scored.append((mlp_score(features, model_artifact), action))
            predicted = max(scored, key=lambda item: (item[0], item[1]))[1]
            correct += int(predicted == transition["expert_action"])
            states += 1
            random_mass += 1.0 / len(candidates)
            history.append(str(transition["expert_action"]))
    return {
        "states": states,
        "top1_accuracy": correct / states,
        "random_top1_expectation": random_mass / states,
    }


def _source_examples(config, seed_key: str, state_count: int):
    rows: list[MatchedValueExample] = []
    domains = 0
    for family in config["families"]:
        for seed in family[seed_key]:
            domain = make_domain(
                seed=int(seed),
                surface=str(family["surface"]),
                hypothesis_count=int(family["hypothesis_count"]),
                test_count=int(family["test_count"]),
                max_tests=int(family["max_tests"]),
                test_cost=float(family["test_cost"]),
            )
            grounded = calibrate_target_grounder(
                domain,
                samples_per_cell=int(config["calibration_samples_per_cell"]),
                seed=_stable_seed(("source-grounder", family["surface"], seed)),
                beta_prior=float(config["calibration_beta_prior"]),
            )
            rows.extend(collect_matched_examples(
                domain,
                grounded,
                state_count=state_count,
                seed=_stable_seed(("source-states", family["surface"], seed)),
            ))
            domains += 1
    return tuple(rows), domains


def _source_model(rows, source, seed):
    model = fit_value_ensemble(
        rows,
        (),
        seed=seed,
        ensemble_size=int(source["ensemble_size"]),
        alpha=float(source["ridge_alpha"]),
        target_mass=1.0,
    )
    assert model is not None
    return model


def _mse(model, rows):
    predictions, _ = model.predict([row.features for row in rows])
    labels = np.asarray([row.value for row in rows])
    return float(np.mean((predictions - labels) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config["target"]
    receipts_path = (REPO / target["adaptation_receipts"]).resolve()
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    successful = [episode for episode in receipts["episodes"] if episode["official_success"]]
    if len(successful) < 3:
        raise RuntimeError("fewer than three successful target adaptation expert episodes")

    cv_rows = []
    for heldout in successful:
        training = [row for row in successful if row is not heldout]
        _, fold_artifact = _fit_grounder(training, config["grounder"])
        cv_rows.append(_top1([heldout], fold_artifact) | {"task_id": heldout["task_id"]})
    _, grounder = _fit_grounder(successful, config["grounder"])
    cv_states = sum(row["states"] for row in cv_rows)
    cv_accuracy = sum(row["top1_accuracy"] * row["states"] for row in cv_rows) / cv_states
    cv_random = sum(
        row["random_top1_expectation"] * row["states"] for row in cv_rows
    ) / cv_states
    cv_ratio = cv_accuracy / cv_random
    grounder_report = {
        "successful_adaptation_episodes": len(successful),
        "failed_adaptation_episodes": len(receipts["episodes"]) - len(successful),
        "leave_one_task_out": cv_rows,
        "top1_accuracy": cv_accuracy,
        "random_top1_expectation": cv_random,
        "top1_over_random_ratio": cv_ratio,
        "minimum_ratio": float(config["grounder"]["minimum_cv_top1_over_random_ratio"]),
        "passed": cv_ratio >= float(config["grounder"]["minimum_cv_top1_over_random_ratio"]),
    }

    source = config["source"]
    train_rows, train_domains = _source_examples(
        source, "train_seeds", int(source["states_per_domain"])
    )
    evaluation_rows, evaluation_domains = _source_examples(
        source, "evaluation_seeds", int(source["evaluation_states_per_domain"])
    )
    authentic = _source_model(train_rows, source, int(source["model_seed"]))
    shuffled = _source_model(
        shuffled_value_control(train_rows, seed=int(source["control_seed"])),
        source,
        int(source["model_seed"]),
    )
    marginal = _source_model(
        marginal_value_control(train_rows), source, int(source["model_seed"])
    )
    source_mse = {
        "authentic": _mse(authentic, evaluation_rows),
        "within_state_shuffled": _mse(shuffled, evaluation_rows),
        "source_marginal": _mse(marginal, evaluation_rows),
    }
    source_passed = source_mse["authentic"] < min(
        source_mse["within_state_shuffled"], source_mse["source_marginal"]
    )
    artifact = {
        "schema_version": "multisource-alfworld-neurosymbolic-candidate-v1",
        "status": (
            "QUALIFICATION_AUTHORIZED"
            if source_passed and grounder_report["passed"]
            else "BLOCKED_BEFORE_QUALIFICATION"
        ),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "adaptation_receipts_path": str(receipts_path),
        "adaptation_receipts_sha256": _sha256(receipts_path),
        "target_grounder": grounder,
        "target_grounder_gate": grounder_report,
        "source": {
            "surfaces": [family["surface"] for family in source["families"]],
            "train_domains": train_domains,
            "evaluation_domains": evaluation_domains,
            "train_examples": len(train_rows),
            "evaluation_examples": len(evaluation_rows),
            "heldout_value_mse": source_mse,
            "gate_passed": source_passed,
            "transferred_feature_names": [
                "is_test", "expected_information_gain", "expected_map_confidence_gain",
                "predicted_outcome_balance", "current_map_confidence", "current_entropy",
                "remaining_test_fraction", "candidate_hypothesis_probability",
                "action_repeat_fraction",
            ],
            "raw_action_tokens_transferred": False,
            "models": {
                "authentic_source_plus_target": asdict(authentic),
                "shuffled_source_plus_target": asdict(shuffled),
                "source_marginal_plus_target": asdict(marginal),
            },
        },
        "qualification_or_heldout_used_for_training": False,
        "cross_domain_transfer_supported": False,
    }
    output = (REPO / target["artifact"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": artifact["status"],
        "grounder_gate": grounder_report,
        "source_mse": source_mse,
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "QUALIFICATION_AUTHORIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
