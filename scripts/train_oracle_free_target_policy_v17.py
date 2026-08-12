#!/usr/bin/env python3
"""Train a target-native policy head with every workflow oracle excluded."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_hierarchical_grounder import action_option  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.oracle_free_target_grounder import (  # noqa: E402
    DENSE_FEATURE_NAMES,
    FORBIDDEN_SEMANTIC_INPUTS,
    policy_features,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(*values: object) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).hexdigest()


def _episode_rows(
    episodes: Sequence[Mapping[str, Any]],
    *,
    feature_bins: int,
    maximum_transitions: int,
    negative_candidates: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for episode in episodes:
        history: list[str] = []
        for index, transition in enumerate(episode["transitions"]):
            expert = str(transition["expert_action"])
            if index < maximum_transitions:
                candidates = [
                    str(action) for action in transition["native_actions"]
                    if action_option(str(action)) != "EXCLUDE"
                ]
                if expert in candidates:
                    negatives = sorted(
                        (action for action in candidates if action != expert),
                        key=lambda action: _rank(
                            episode["task_id"], transition["step"], action,
                            "oracle-free-policy-negative",
                        ),
                    )
                    if negative_candidates is not None:
                        negatives = negatives[:negative_candidates]
                    selected = (expert, *negatives)
                    state_key = f"{episode['task_id']}:{transition['step']}"
                    for action in selected:
                        features.append(policy_features(
                            goal=str(transition["goal"]),
                            observation=str(transition["before_observation"]),
                            action=action,
                            step=int(transition["step"]),
                            action_history=history,
                            feature_bins=feature_bins,
                        ))
                        labels.append(int(action == expert))
                        metadata.append({
                            "task_id": str(episode["task_id"]),
                            "state_key": state_key,
                            "action": action,
                            "candidate_count": len(selected),
                        })
            history.append(expert)
    return np.asarray(features), np.asarray(labels), metadata


def _permuted_expert_labels(
    labels: np.ndarray, metadata: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    result = np.zeros_like(labels)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row["state_key"])].append(index)
    for state_key, indices in groups.items():
        expert = next(index for index in indices if labels[index] == 1)
        alternatives = sorted(
            (index for index in indices if index != expert),
            key=lambda index: _rank(
                state_key, metadata[index]["action"], "permuted-expert-control",
            ),
        )
        result[alternatives[0] if alternatives else expert] = 1
    return result


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_units: int,
    maximum_iterations: int,
    seed: int,
    estimator: str,
) -> tuple[Any, dict[str, Any]]:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(f"one-class target policy rows: {counts.tolist()}")
    rng = np.random.default_rng(seed)
    count = int(np.max(counts))
    selected = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=count, replace=True)
        for label in (0, 1)
    ])
    rng.shuffle(selected)
    if estimator == "convex_logistic":
        model = LogisticRegression(
            C=5.0,
            solver="lbfgs",
            max_iter=maximum_iterations,
            random_state=seed,
        )
    elif estimator == "tanh_mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation="tanh",
            solver="lbfgs",
            alpha=0.2,
            max_iter=maximum_iterations,
            random_state=seed,
        )
    else:
        raise ValueError(f"unsupported target policy estimator: {estimator}")
    model.fit(features[selected], labels[selected])
    if estimator == "convex_logistic":
        layers = [{
            "weights": model.coef_.T.tolist(),
            "bias": model.intercept_.tolist(),
        }]
        iterations = int(model.n_iter_[0])
        kind = "target-native-oracle-free-logistic-neuron-v18"
    else:
        layers = [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in zip(model.coefs_, model.intercepts_)
        ]
        iterations = int(model.n_iter_)
        kind = "target-native-oracle-free-policy-mlp-v17"
    artifact = {
        "kind": kind,
        "hidden_activation": "tanh",
        "layers": layers,
        "raw_training_examples": len(labels),
        "raw_positive_examples": int(np.sum(labels)),
        "balanced_training_examples": len(selected),
    }
    artifact["iterations"] = iterations
    artifact["maximum_iterations"] = int(maximum_iterations)
    artifact["optimizer_converged_before_iteration_limit"] = bool(
        iterations < maximum_iterations
    )
    return model, artifact


def _state_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row["state_key"])].append(index)
    output = []
    for state_key, indices in sorted(groups.items()):
        expert = next(index for index in indices if labels[index] == 1)
        negatives = [index for index in indices if labels[index] == 0]
        expert_score = float(scores[expert])
        pairwise = float(np.mean([
            float(expert_score > scores[index])
            + 0.5 * float(expert_score == scores[index])
            for index in negatives
        ])) if negatives else 1.0
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), index))
        rank = ordered.index(expert) + 1
        output.append({
            "state_key": state_key,
            "task_id": str(metadata[expert]["task_id"]),
            "candidate_count": len(indices),
            "pairwise_auc": pairwise,
            "top1": float(rank == 1),
            "top3": float(rank <= 3),
            "random_top1": 1.0 / len(indices),
        })
    return output


def _cluster_bootstrap_lower(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    baseline_field: str | None,
    seed: int,
    samples: int,
    alpha: float,
) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if baseline_field is not None:
            value -= float(row[baseline_field])
        by_task[str(row["task_id"])].append(value)
    task_means = np.asarray([
        np.mean(by_task[task]) for task in sorted(by_task)
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.asarray([
        np.mean(rng.choice(task_means, size=len(task_means), replace=True))
        for _ in range(samples)
    ])
    return float(np.quantile(estimates, alpha))


def _paired_rows(
    authentic: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    control_by_state = {str(row["state_key"]): row for row in control}
    return [
        dict(row) | {
            "pairwise_control": float(control_by_state[str(row["state_key"])][
                "pairwise_auc"
            ])
        }
        for row in authentic
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen target grounder: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_ORACLE_FREE_POLICY_TRAINING":
        raise SystemExit("target policy config was not frozen before training")
    for name, receipt in config["implementation"].items():
        dependency = (REPO / receipt["path"]).resolve()
        if _sha256(dependency) != receipt["file_sha256"]:
            raise SystemExit(f"frozen {name} implementation hash mismatch")
    target = config["target_adaptation"]
    receipts = (REPO / target["receipts"]["path"]).resolve()
    if _sha256(receipts) != target["receipts"]["file_sha256"]:
        raise SystemExit("adaptation receipt hash mismatch")
    payload = json.loads(receipts.read_text(encoding="utf-8"))
    if payload.get("qualification_or_heldout_read"):
        raise SystemExit("adaptation receipt crossed target boundary")
    train = [
        row for row in payload["episodes"]
        if row["partition"] == "adaptation_train"
    ]
    validation = [
        row for row in payload["episodes"]
        if row["partition"] == "adaptation_validation"
    ]
    train_ids = {str(row["task_id"]) for row in train}
    validation_ids = {str(row["task_id"]) for row in validation}
    if train_ids & validation_ids:
        raise SystemExit("target adaptation train/validation identity overlap")
    model_config = config["model"]
    train_rows = _episode_rows(
        train,
        feature_bins=int(model_config["feature_bins"]),
        maximum_transitions=int(model_config["maximum_transitions_per_episode"]),
        negative_candidates=int(model_config["negative_candidates_per_positive"]),
    )
    validation_rows = _episode_rows(
        validation,
        feature_bins=int(model_config["feature_bins"]),
        maximum_transitions=int(model_config["maximum_transitions_per_episode"]),
        negative_candidates=None,
    )
    authentic, authentic_head = _fit(
        train_rows[0], train_rows[1],
        hidden_units=int(model_config["hidden_units"]),
        maximum_iterations=int(model_config["maximum_iterations"]),
        seed=int(model_config["seed"]),
        estimator=str(model_config.get("estimator", "tanh_mlp")),
    )
    permuted_labels = _permuted_expert_labels(train_rows[1], train_rows[2])
    control, _control_head = _fit(
        train_rows[0], permuted_labels,
        hidden_units=int(model_config["hidden_units"]),
        maximum_iterations=int(model_config["maximum_iterations"]),
        seed=int(model_config["seed"]) + 1,
        estimator=str(model_config.get("estimator", "tanh_mlp")),
    )
    authentic_metrics = _state_metrics(
        validation_rows[1], authentic.predict_proba(validation_rows[0])[:, 1],
        validation_rows[2],
    )
    control_metrics = _state_metrics(
        validation_rows[1], control.predict_proba(validation_rows[0])[:, 1],
        validation_rows[2],
    )
    paired = _paired_rows(authentic_metrics, control_metrics)
    bootstrap = config["bootstrap"]
    lower_top1 = _cluster_bootstrap_lower(
        authentic_metrics, field="top1", baseline_field="random_top1",
        seed=int(bootstrap["seed"]), samples=int(bootstrap["samples"]),
        alpha=float(bootstrap["lower_tail_alpha"]),
    )
    lower_pairwise_control = _cluster_bootstrap_lower(
        paired, field="pairwise_auc", baseline_field="pairwise_control",
        seed=int(bootstrap["seed"]) + 1, samples=int(bootstrap["samples"]),
        alpha=float(bootstrap["lower_tail_alpha"]),
    )
    summary = {
        "validation_states": len(authentic_metrics),
        "authentic_pairwise_auc": float(np.mean([
            row["pairwise_auc"] for row in authentic_metrics
        ])),
        "permuted_expert_control_pairwise_auc": float(np.mean([
            row["pairwise_auc"] for row in control_metrics
        ])),
        "authentic_top1": float(np.mean([row["top1"] for row in authentic_metrics])),
        "authentic_top3": float(np.mean([row["top3"] for row in authentic_metrics])),
        "random_top1": float(np.mean([
            row["random_top1"] for row in authentic_metrics
        ])),
        "cluster_bootstrap_lower_top1_minus_random": lower_top1,
        "cluster_bootstrap_lower_pairwise_auc_minus_permuted_control": (
            lower_pairwise_control
        ),
    }
    requirements = config["gates"]
    gates = {
        "minimum_validation_episodes": (
            len(validation) >= int(requirements["minimum_validation_episodes"])
        ),
        "disjoint_adaptation_task_ids": not bool(train_ids & validation_ids),
        "minimum_authentic_pairwise_auc": (
            summary["authentic_pairwise_auc"]
            >= float(requirements["minimum_authentic_pairwise_auc"])
        ),
        "top1_minus_random_bootstrap_lower_bound_gt_zero": lower_top1 > 0.0,
        "authentic_minus_permuted_control_bootstrap_lower_bound_gt_zero": (
            lower_pairwise_control > 0.0
        ),
        "authentic_optimizer_converged_before_iteration_limit": bool(
            authentic_head["optimizer_converged_before_iteration_limit"]
        ),
        "control_optimizer_converged_before_iteration_limit": bool(
            _control_head["optimizer_converged_before_iteration_limit"]
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "oracle-free-target-policy-grounder-v17",
        "status": "TARGET_GROUNDER_GATE_PASSED" if passed else "TARGET_GROUNDER_GATE_FAILED",
        "claim_boundary": config["claim_boundary"],
        "config": {
            "path": str(args.config.resolve()),
            "file_sha256": _sha256(args.config),
        },
        "adaptation_receipts": {
            "path": str(receipts),
            "file_sha256": _sha256(receipts),
            "qualification_or_heldout_read": False,
        },
        "feature_bins": int(model_config["feature_bins"]),
        "dense_feature_names": list(DENSE_FEATURE_NAMES),
        "forbidden_semantic_inputs": list(FORBIDDEN_SEMANTIC_INPUTS),
        "required_option_or_workflow_features_used": False,
        "training_supervision": "expert_action_identity_only",
        "reward_success_completion_fields_consumed": False,
        "policy_head": authentic_head,
        "validation_summary": summary,
        "gates": gates,
        "runtime_hashes": {
            "trainer": _sha256(Path(__file__).resolve()),
            "feature_module": _sha256(
                REPO / "src/motif_transfer/oracle_free_target_grounder.py"
            ),
        },
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": artifact["status"],
        "validation_summary": summary,
        "gates": gates,
        "artifact_sha256": artifact["artifact_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
