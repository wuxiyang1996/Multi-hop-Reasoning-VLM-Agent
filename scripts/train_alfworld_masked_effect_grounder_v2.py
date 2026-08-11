#!/usr/bin/env python3
"""Train ALFWorld neural grounding heads with target stage features masked."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier

from motif_transfer.alfworld_hierarchical_grounder import (
    OPTION_NAMES,
    action_option,
    completion_label,
    goal_binding_label,
    grounder_features,
    infer_required_option,
)
from motif_transfer.alfworld_masked_effect_grounder import ARTIFACT_VERSION
from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(*values: object) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()


def _metadata(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    history: list[str] = []
    rows = []
    for transition in episode["transitions"]:
        goal = str(transition["goal"])
        native = tuple(map(str, transition["native_actions"]))
        action = str(transition["expert_action"])
        required = infer_required_option(
            goal=goal, native_actions=native, action_history=history,
        )
        completed = completion_label(
            goal=goal,
            before_native_actions=native,
            action_history=history,
            action=action,
            after_native_actions=tuple(map(str, transition["after_native_actions"])),
            official_success_after=bool(transition["official_success_after"]),
        )
        rows.append({
            "transition": transition,
            "history": tuple(history),
            "required": required,
            "completion": completed,
        })
        history.append(action)
    return rows


def _selected_metadata(
    episode: Mapping[str, Any], maximum: int,
) -> list[dict[str, Any]]:
    rows = _metadata(episode)
    positives = [row for row in rows if row["completion"]]
    negatives = sorted(
        (row for row in rows if not row["completion"]),
        key=lambda row: _rank(
            episode["task_id"], row["transition"]["step"], "completion-cap",
        ),
    )
    return sorted(
        [*positives, *negatives[: max(0, maximum - len(positives))]],
        key=lambda row: int(row["transition"]["step"]),
    )


def _rows(
    episodes: Sequence[Mapping[str, Any]], *, feature_bins: int,
    negative_candidates: int, maximum_transitions: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        for row in _selected_metadata(episode, maximum_transitions):
            transition = row["transition"]
            goal = str(transition["goal"])
            expert = str(transition["expert_action"])
            candidates = [
                str(action) for action in transition["native_actions"]
                if action_option(str(action)) != "EXCLUDE"
            ]
            features = {
                action: grounder_features(
                    goal=goal,
                    observation=str(transition["before_observation"]),
                    action=action,
                    required_option="SEARCH",
                    step=int(transition["step"]),
                    action_history=row["history"],
                    feature_bins=feature_bins,
                    mask_required_option=True,
                )
                for action in candidates
            }
            expert_option = action_option(expert)
            app_positive = [
                action for action in candidates if action_option(action) == expert_option
            ]
            app_negative = sorted(
                (action for action in candidates if action_option(action) != expert_option),
                key=lambda action: _rank(
                    episode["task_id"], transition["step"], action, "applicability",
                ),
            )[:negative_candidates]
            for action in (*app_positive, *app_negative):
                values["applicability"].append(features[action])
                labels["applicability"].append(int(action_option(action) == expert_option))
                metadata["applicability"].append({
                    "required": row["required"], "option": action_option(action),
                })

            binding_positive = [
                action for action in candidates if goal_binding_label(goal, action)
            ]
            binding_negative = sorted(
                (action for action in candidates if not goal_binding_label(goal, action)),
                key=lambda action: _rank(
                    episode["task_id"], transition["step"], action, "binding",
                ),
            )[: max(negative_candidates, negative_candidates * len(binding_positive))]
            for action in (*binding_positive, *binding_negative):
                values["binding"].append(features[action])
                labels["binding"].append(goal_binding_label(goal, action))
                metadata["binding"].append({
                    "state_key": f"{episode['task_id']}:{transition['step']}",
                    "action": action,
                })

            if expert not in features:
                continue
            policy_negative = sorted(
                (action for action in candidates if action != expert),
                key=lambda action: _rank(
                    episode["task_id"], transition["step"], action, "policy",
                ),
            )[:negative_candidates]
            for action in (expert, *policy_negative):
                values["policy"].append(features[action])
                labels["policy"].append(int(action == expert))
                metadata["policy"].append({
                    "state_key": f"{episode['task_id']}:{transition['step']}",
                    "action": action,
                })
            values["completion"].append(features[expert])
            labels["completion"].append(int(row["completion"]))
            metadata["completion"].append({"required": row["required"]})
    return {
        name: (
            np.asarray(values[name]), np.asarray(labels[name]), metadata[name],
        )
        for name in ("applicability", "binding", "completion", "policy")
    }


def _fit_head(
    features: np.ndarray, labels: np.ndarray, *, seed: int,
    hidden_units: int, maximum_iterations: int,
) -> tuple[MLPClassifier, dict[str, Any]]:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(f"one-class target head: {counts.tolist()}")
    rng = np.random.default_rng(seed)
    count = int(np.max(counts))
    selected = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=count, replace=True)
        for label in (0, 1)
    ])
    rng.shuffle(selected)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,), activation="tanh", solver="lbfgs",
        alpha=0.2, max_iter=maximum_iterations, random_state=seed,
    )
    model.fit(features[selected], labels[selected])
    return model, {
        "kind": "target-native-stage-masked-binary-mlp-v2",
        "hidden_activation": "tanh",
        "layers": [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in zip(model.coefs_, model.intercepts_)
        ],
        "raw_training_examples": len(labels),
        "raw_positive_examples": int(np.sum(labels)),
        "balanced_training_examples": len(selected),
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    wins = sum(
        float(np.sum(score > negatives)) + 0.5 * float(np.sum(score == negatives))
        for score in positives
    )
    return wins / (len(positives) * len(negatives))


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predicted = scores >= 0.5
    return 0.5 * (
        float(np.mean(predicted[labels == 1]))
        + float(np.mean(~predicted[labels == 0]))
    )


def _binding_recall_at_3(
    labels: np.ndarray, scores: np.ndarray, metadata: Sequence[Mapping[str, Any]],
) -> tuple[float, int]:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for label, score, row in zip(labels, scores, metadata):
        groups[str(row["state_key"])].append((float(score), int(label)))
    eligible = [rows for rows in groups.values() if any(label for _, label in rows)]
    hits = sum(
        any(label for _, label in sorted(rows, reverse=True)[:3]) for rows in eligible
    )
    return hits / len(eligible), len(eligible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-bins", type=int, default=192)
    parser.add_argument("--hidden-units", type=int, default=48)
    parser.add_argument("--seed", type=int, default=96201)
    parser.add_argument("--maximum-iterations", type=int, default=800)
    parser.add_argument("--maximum-transitions", type=int, default=48)
    parser.add_argument("--negative-candidates", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen grounder: {args.output}")
    receipts = json.loads(args.receipts.read_text(encoding="utf-8"))
    if receipts.get("qualification_or_heldout_read"):
        raise SystemExit("target receipts crossed the adaptation boundary")
    train = [row for row in receipts["episodes"] if row["partition"] == "adaptation_train"]
    validation = [
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_validation"
    ]
    train_rows = _rows(
        train, feature_bins=args.feature_bins,
        negative_candidates=args.negative_candidates,
        maximum_transitions=args.maximum_transitions,
    )
    validation_rows = _rows(
        validation, feature_bins=args.feature_bins,
        negative_candidates=args.negative_candidates,
        maximum_transitions=args.maximum_transitions,
    )
    heads = {}
    models = {}
    for offset, name in enumerate(("applicability", "binding", "completion", "policy")):
        models[name], heads[name] = _fit_head(
            train_rows[name][0], train_rows[name][1], seed=args.seed + offset,
            hidden_units=args.hidden_units,
            maximum_iterations=args.maximum_iterations,
        )
    scores = {
        name: models[name].predict_proba(validation_rows[name][0])[:, 1]
        for name in models
    }
    labels = validation_rows["applicability"][1]
    option_auc = {}
    for option in OPTION_NAMES:
        selected = np.asarray([
            row["required"] == option for row in validation_rows["applicability"][2]
        ])
        if np.any(selected) and len(np.unique(labels[selected])) == 2:
            option_auc[option] = _auc(labels[selected], scores["applicability"][selected])
    binding_recall, binding_states = _binding_recall_at_3(
        validation_rows["binding"][1], scores["binding"],
        validation_rows["binding"][2],
    )
    metrics = {
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "training_rows": {name: len(row[1]) for name, row in train_rows.items()},
        "validation_rows": {name: len(row[1]) for name, row in validation_rows.items()},
        "macro_applicability_auc": float(np.mean(list(option_auc.values()))),
        "per_required_option_applicability_auc": option_auc,
        "binding_recall_at_3": binding_recall,
        "binding_states": binding_states,
        "completion_balanced_accuracy": _balanced_accuracy(
            validation_rows["completion"][1], scores["completion"],
        ),
        "policy_auc": _auc(validation_rows["policy"][1], scores["policy"]),
    }
    gates = {
        "macro_applicability_auc": metrics["macro_applicability_auc"] >= 0.70,
        "binding_recall_at_3": binding_recall >= 0.75,
        "completion_balanced_accuracy": metrics["completion_balanced_accuracy"] >= 0.65,
        "policy_auc": metrics["policy_auc"] >= 0.70,
    }
    passed = all(gates.values())
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "status": "ADAPTATION_GATE_PASSED" if passed else "ADAPTATION_GATE_FAILED",
        "claim_boundary": (
            "TARGET_ADAPTATION_ONLY; REQUIRED_OPTION_MASKED_FOR_EVERY_HEAD;_"
            "NO_QUALIFICATION_OR_HELDOUT_TASK_USED"
        ),
        "required_option_masked_for_every_head": True,
        "receipts": {
            "path": str(args.receipts.resolve()),
            "file_sha256": _sha256(args.receipts),
        },
        "feature_bins": args.feature_bins,
        "applicability_head": heads["applicability"],
        "binding_head": heads["binding"],
        "completion_head": heads["completion"],
        "policy_head": heads["policy"],
        "validation_metrics": metrics,
        "gates": gates,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "status": artifact["status"],
        "validation_metrics": metrics,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
