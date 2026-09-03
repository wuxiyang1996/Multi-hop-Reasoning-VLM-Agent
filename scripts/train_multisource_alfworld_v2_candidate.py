#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_hierarchical_grounder import (  # noqa: E402
    action_option,
    completion_label,
    goal_binding_label,
    grounder_features,
    infer_required_option,
)
from motif_transfer.hierarchical_skill_transfer import (  # noqa: E402
    FEATURE_NAMES,
    OPTION_NAMES,
    HierarchicalValueExample,
    collect_source_examples,
    fit_value_ensemble,
    marginal_value_control,
    phase_permuted_control,
    serialize_ensemble,
    shuffled_value_control,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_rank(*values: object) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).hexdigest()


def _transition_metadata(episode: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[str] = []
    result = []
    for transition in episode["transitions"]:
        goal = str(transition["goal"])
        native = list(map(str, transition["native_actions"]))
        action = str(transition["expert_action"])
        required = infer_required_option(
            goal=goal, native_actions=native, action_history=history,
        )
        completed = completion_label(
            goal=goal,
            before_native_actions=native,
            action_history=history,
            action=action,
            after_native_actions=list(map(str, transition["after_native_actions"])),
            official_success_after=bool(transition["official_success_after"]),
        )
        result.append({
            "transition": transition,
            "history": tuple(history),
            "required": required,
            "completion": completed,
        })
        history.append(action)
    return result


def _selected_metadata(
    episode: dict[str, Any], maximum: int,
) -> list[dict[str, Any]]:
    rows = _transition_metadata(episode)
    positives = [row for row in rows if row["completion"]]
    negatives = sorted(
        (row for row in rows if not row["completion"]),
        key=lambda row: _stable_rank(
            episode["task_id"], row["transition"]["step"], "completion-cap",
        ),
    )
    chosen = positives + negatives[: max(0, maximum - len(positives))]
    return sorted(chosen, key=lambda row: int(row["transition"]["step"]))


def _candidate_rows(
    episodes: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    by_head: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_bins = int(config["feature_bins"])
    negatives = int(config["negative_candidates_per_positive"])
    maximum = int(config["maximum_transitions_per_episode"])
    for episode in episodes:
        for row in _selected_metadata(episode, maximum):
            transition = row["transition"]
            goal = str(transition["goal"])
            expert = str(transition["expert_action"])
            candidates = [
                str(action) for action in transition["native_actions"]
                if action_option(str(action)) != "EXCLUDE"
            ]
            features_by_action = {
                action: grounder_features(
                    goal=goal,
                    observation=str(transition["before_observation"]),
                    action=action,
                    required_option=str(row["required"]),
                    step=int(transition["step"]),
                    action_history=row["history"],
                    feature_bins=feature_bins,
                )
                for action in candidates
            }
            policy_features_by_action = {
                action: grounder_features(
                    goal=goal,
                    observation=str(transition["before_observation"]),
                    action=action,
                    required_option=str(row["required"]),
                    step=int(transition["step"]),
                    action_history=row["history"],
                    feature_bins=feature_bins,
                    mask_required_option=True,
                )
                for action in candidates
            }
            expert_option = action_option(expert)
            applicability_positive = [
                action for action in candidates if action_option(action) == expert_option
            ]
            applicability_negative = sorted(
                (action for action in candidates if action_option(action) != expert_option),
                key=lambda action: _stable_rank(
                    episode["task_id"], transition["step"], action, "applicability",
                ),
            )[:negatives]
            for action in (*applicability_positive, *applicability_negative):
                by_head["applicability"].append(features_by_action[action])
                labels["applicability"].append(int(action_option(action) == expert_option))
                metadata["applicability"].append({
                    "task_id": episode["task_id"],
                    "required": row["required"],
                    "option": action_option(action),
                })

            binding_positive = [
                action for action in candidates if goal_binding_label(goal, action)
            ]
            binding_negative = sorted(
                (action for action in candidates if not goal_binding_label(goal, action)),
                key=lambda action: _stable_rank(
                    episode["task_id"], transition["step"], action, "binding",
                ),
            )[: max(negatives, negatives * len(binding_positive))]
            for action in (*binding_positive, *binding_negative):
                by_head["binding"].append(features_by_action[action])
                labels["binding"].append(goal_binding_label(goal, action))
                metadata["binding"].append({
                    "task_id": episode["task_id"],
                    "state_key": f"{episode['task_id']}:{transition['step']}",
                    "action": action,
                })

            if expert not in features_by_action:
                continue
            policy_negatives = sorted(
                (action for action in candidates if action != expert),
                key=lambda action: _stable_rank(
                    episode["task_id"], transition["step"], action, "policy",
                ),
            )[:negatives]
            for action in (expert, *policy_negatives):
                by_head["policy"].append(policy_features_by_action[action])
                labels["policy"].append(int(action == expert))
                metadata["policy"].append({
                    "task_id": episode["task_id"],
                    "state_key": f"{episode['task_id']}:{transition['step']}",
                    "action": action,
                })
            by_head["completion"].append(features_by_action[expert])
            labels["completion"].append(int(row["completion"]))
            metadata["completion"].append({
                "task_id": episode["task_id"],
                "required": row["required"],
            })
    return {
        name: (np.asarray(by_head[name]), np.asarray(labels[name]), metadata[name])
        for name in ("applicability", "binding", "completion", "policy")
    }


def _balanced_training_rows(
    features: np.ndarray, labels: np.ndarray, *, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(f"grounder head has one label class: {counts.tolist()}")
    rng = np.random.default_rng(seed)
    target_count = int(np.max(counts))
    selected = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=target_count, replace=True)
        for label in (0, 1)
    ])
    rng.shuffle(selected)
    return features[selected], labels[selected]


def _fit_head(features: np.ndarray, labels: np.ndarray, config, seed_offset: int):
    balanced_features, balanced_labels = _balanced_training_rows(
        features, labels, seed=int(config["random_seed"]) + seed_offset,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(int(config["hidden_units"]),),
        activation="tanh",
        solver="lbfgs",
        alpha=0.2,
        max_iter=int(config["maximum_iterations"]),
        random_state=int(config["random_seed"]) + seed_offset,
    )
    model.fit(balanced_features, balanced_labels)
    artifact = {
        "kind": "target-native-independent-binary-mlp-v2",
        "hidden_activation": "tanh",
        "layers": [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in zip(model.coefs_, model.intercepts_)
        ],
        "raw_training_examples": len(labels),
        "raw_positive_examples": int(np.sum(labels)),
        "balanced_training_examples": len(balanced_labels),
        "class_balanced_bootstrap_training": True,
    }
    return model, artifact


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    wins = sum(
        float(np.sum(score > negatives)) + 0.5 * float(np.sum(score == negatives))
        for score in positives
    )
    return wins / (len(positives) * len(negatives))


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predicted = scores >= 0.5
    recall_positive = float(np.mean(predicted[labels == 1]))
    recall_negative = float(np.mean(~predicted[labels == 0]))
    return 0.5 * (recall_positive + recall_negative)


def _binding_recall_at_3(
    labels: np.ndarray, scores: np.ndarray, metadata: Sequence[dict[str, Any]]
) -> tuple[float, int]:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for label, score, row in zip(labels, scores, metadata):
        groups[str(row["state_key"])].append((float(score), int(label)))
    eligible = [rows for rows in groups.values() if any(label for _, label in rows)]
    hits = sum(
        any(label for _, label in sorted(rows, reverse=True)[:3]) for rows in eligible
    )
    return hits / len(eligible), len(eligible)


def _source_rows(config, *, evaluation: bool) -> tuple[HierarchicalValueExample, ...]:
    workflow = config["workflow"]
    domains = (
        int(config["evaluation_domains_per_surface"])
        if evaluation else int(config["train_domains_per_surface"])
    )
    return collect_source_examples(
        surfaces=tuple(map(str, config["surfaces"])),
        domains_per_surface=domains,
        states_per_domain=int(config["states_per_domain"]),
        seed=int(config["model_seed"]) + (100000 if evaluation else 0),
        minimum_budget=int(workflow["minimum_budget"]),
        maximum_budget=int(workflow["maximum_budget"]),
        completion_probability_range=workflow["completion_probability_range"],
        failure_cost_range=workflow["failure_cost_range"],
        progress_reward=float(workflow["progress_reward"]),
        invalid_option_cost=float(workflow["invalid_option_cost"]),
    )


def _mse(model, rows: Sequence[HierarchicalValueExample]) -> float:
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
    if receipts["qualification_or_heldout_read"]:
        raise RuntimeError("adaptation receipts crossed the frozen boundary")
    train_episodes = [
        row for row in receipts["episodes"] if row["partition"] == "adaptation_train"
    ]
    validation_episodes = [
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_validation"
    ]
    if len(validation_episodes) < int(config["grounder"]["minimum_validation_episodes"]):
        raise RuntimeError("too few frozen validation episodes")
    train_rows = _candidate_rows(train_episodes, config["grounder"])
    validation_rows = _candidate_rows(validation_episodes, config["grounder"])
    heads = {}
    models = {}
    for offset, name in enumerate(("applicability", "binding", "completion", "policy")):
        features, labels, _ = train_rows[name]
        model, artifact = _fit_head(features, labels, config["grounder"], offset)
        models[name] = model
        heads[name] = artifact

    validation_scores = {
        name: models[name].predict_proba(validation_rows[name][0])[:, 1]
        for name in models
    }
    role_aucs = {}
    role_labels = validation_rows["applicability"][1]
    for option in OPTION_NAMES:
        indices = np.asarray([
            row["required"] == option
            for row in validation_rows["applicability"][2]
        ])
        if np.any(indices) and len(np.unique(role_labels[indices])) == 2:
            role_aucs[option] = _auc(
                role_labels[indices], validation_scores["applicability"][indices],
            )
    macro_role_auc = float(np.mean(list(role_aucs.values())))
    binding_recall, binding_states = _binding_recall_at_3(
        validation_rows["binding"][1],
        validation_scores["binding"],
        validation_rows["binding"][2],
    )
    completion_balanced = _balanced_accuracy(
        validation_rows["completion"][1], validation_scores["completion"],
    )
    policy_auc = _auc(
        validation_rows["policy"][1], validation_scores["policy"],
    )
    target_gate = {
        "frozen_train_episodes": len(train_episodes),
        "frozen_validation_episodes": len(validation_episodes),
        "successful_train_episodes": sum(row["official_success"] for row in train_episodes),
        "successful_validation_episodes": sum(
            row["official_success"] for row in validation_episodes
        ),
        "training_rows": {name: len(rows[1]) for name, rows in train_rows.items()},
        "validation_rows": {name: len(rows[1]) for name, rows in validation_rows.items()},
        "macro_role_auc": macro_role_auc,
        "per_required_option_auc": role_aucs,
        "minimum_macro_role_auc": float(config["grounder"]["minimum_macro_role_auc"]),
        "goal_binding_recall_at_3": binding_recall,
        "goal_binding_states": binding_states,
        "minimum_goal_binding_recall_at_3": float(
            config["grounder"]["minimum_goal_binding_recall_at_3"]
        ),
        "completion_balanced_accuracy": completion_balanced,
        "completion_validation_positive_rate": float(
            np.mean(validation_rows["completion"][1])
        ),
        "neural_only_policy_auc": policy_auc,
        "minimum_effect_balanced_accuracy": float(
            config["grounder"]["minimum_effect_balanced_accuracy"]
        ),
    }
    target_gate["passed"] = bool(
        macro_role_auc >= target_gate["minimum_macro_role_auc"]
        and binding_recall >= target_gate["minimum_goal_binding_recall_at_3"]
        and completion_balanced >= target_gate["minimum_effect_balanced_accuracy"]
    )

    source_config = config["source"]
    source_train = _source_rows(source_config, evaluation=False)
    source_evaluation = _source_rows(source_config, evaluation=True)
    conditions = {
        "authentic_source_plus_target": source_train,
        "shuffled_source_plus_target": shuffled_value_control(
            source_train, seed=int(source_config["control_seed"]),
        ),
        "source_marginal_plus_target": marginal_value_control(source_train),
        "phase_permuted_source_plus_target": phase_permuted_control(source_train),
    }
    source_models = {
        name: fit_value_ensemble(
            rows,
            seed=int(source_config["model_seed"]),
            ensemble_size=int(source_config["ensemble_size"]),
            alpha=float(source_config["ridge_alpha"]),
        )
        for name, rows in conditions.items()
    }
    source_mse = {
        name: _mse(model, source_evaluation) for name, model in source_models.items()
    }
    authentic_mse = source_mse["authentic_source_plus_target"]
    source_ordering_passed = all(
        authentic_mse < value
        for name, value in source_mse.items()
        if name != "authentic_source_plus_target"
    )
    relative_improvements = {
        name: (value - authentic_mse) / max(value, 1e-12)
        for name, value in source_mse.items()
        if name != "authentic_source_plus_target"
    }
    minimum_source_improvement = float(
        source_config["minimum_relative_mse_improvement_over_each_control"]
    )
    source_passed = source_ordering_passed and all(
        improvement >= minimum_source_improvement
        for improvement in relative_improvements.values()
    )
    artifact = {
        "schema_version": "multisource-alfworld-neurosymbolic-candidate-v2",
        "status": (
            "QUALIFICATION_AUTHORIZED"
            if target_gate["passed"] and source_passed
            else "BLOCKED_BEFORE_QUALIFICATION"
        ),
        "claim_boundary": config["claim_boundary"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "adaptation_receipts_path": str(receipts_path),
        "adaptation_receipts_sha256": _sha256(receipts_path),
        "target_grounder": {
            "kind": "target-native-hierarchical-event-grounder-v2",
            "feature_bins": int(config["grounder"]["feature_bins"]),
            "applicability_head": heads["applicability"],
            "binding_head": heads["binding"],
            "completion_head": heads["completion"],
            "policy_head": heads["policy"],
        },
        "target_grounder_gate": target_gate,
        "source": {
            "surfaces": source_config["surfaces"],
            "train_domains": len(source_config["surfaces"])
            * int(source_config["train_domains_per_surface"]),
            "evaluation_domains": len(source_config["surfaces"])
            * int(source_config["evaluation_domains_per_surface"]),
            "train_examples": len(source_train),
            "evaluation_examples": len(source_evaluation),
            "transferred_feature_names": list(FEATURE_NAMES),
            "raw_action_tokens_transferred": False,
            "heldout_value_mse": source_mse,
            "relative_mse_improvement_over_control": relative_improvements,
            "minimum_relative_mse_improvement_over_each_control": (
                minimum_source_improvement
            ),
            "strict_mse_ordering_passed": source_ordering_passed,
            "gate_passed": source_passed,
            "models": {
                name: serialize_ensemble(model) for name, model in source_models.items()
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
        "target_grounder_gate": target_gate,
        "source_mse": source_mse,
        "source_gate_passed": source_passed,
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "QUALIFICATION_AUTHORIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
