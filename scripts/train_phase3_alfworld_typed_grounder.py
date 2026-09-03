#!/usr/bin/env python3
"""Train and qualify an outcome-blind ALFWorld grounder for Phase-3 IR."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_hierarchical_grounder import action_option  # noqa: E402
from motif_transfer.alfworld_masked_effect_grounder import (  # noqa: E402
    validate_artifact as validate_policy_artifact,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_typed_grounder import (  # noqa: E402
    ARTIFACT_VERSION,
    EFFECT_TYPES,
    masked_features,
)
from motif_transfer.phase3_source_portfolio import (  # noqa: E402
    permute_selected_effect_binding,
    select_source_program_portfolio,
)
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    target_trial_order,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(*values: object) -> str:
    return stable_hash(list(map(str, values)))


def _episode_states(
    episodes: Sequence[Mapping[str, Any]], *, feature_bins: int,
    label_granularity: str = "exact_action",
) -> list[dict[str, Any]]:
    """Create labels without touching reward or official-success fields."""

    states: list[dict[str, Any]] = []
    for episode in episodes:
        history: list[str] = []
        transitions = list(episode["transitions"])
        for index, transition in enumerate(transitions):
            actions = tuple(
                str(action) for action in transition["native_actions"]
                if action_option(str(action)) != "EXCLUDE"
            )
            future_actions = tuple(
                str(row["expert_action"])
                for row in transitions[index:index + 8]
            )
            future_options = tuple(map(action_option, future_actions))
            successor_sets = tuple(
                set(map(str, row["after_native_actions"]))
                for row in transitions[index:index + 4]
            )
            successor_option_sets = tuple(
                {action_option(value) for value in values}
                for values in successor_sets
            )
            rows = []
            for action in actions:
                if label_granularity == "exact_action":
                    continuation = future_actions
                    candidate_value = action
                    persistence_sets = successor_sets
                elif label_granularity == "target_native_option":
                    continuation = future_options
                    candidate_value = action_option(action)
                    persistence_sets = successor_option_sets
                else:
                    raise ValueError(
                        f"unknown target label granularity: {label_granularity}"
                    )
                labels = {
                    "EFFECT_BY_TRANSITION_1": int(candidate_value in continuation[:1]),
                    "EFFECT_BY_TRANSITION_4": int(candidate_value in continuation[:4]),
                    "EFFECT_BY_TRANSITION_8": int(candidate_value in continuation[:8]),
                    "EXECUTABLE_TRANSITION_PERSISTENCE": int(
                        bool(persistence_sets)
                        and all(candidate_value in values for values in persistence_sets)
                    ),
                }
                rows.append({
                    "action": action,
                    "action_sha256": stable_hash({"target_native_action": action}),
                    "features": masked_features(
                        goal=str(transition["goal"]),
                        observation=str(transition["before_observation"]),
                        action=action,
                        step=int(transition["step"]),
                        action_history=history,
                        feature_bins=feature_bins,
                    ),
                    "labels": labels,
                })
            states.append({
                "task_id": str(episode["task_id"]),
                "step": int(transition["step"]),
                "expert_action": str(transition["expert_action"]),
                "rows": rows,
            })
            history.append(str(transition["expert_action"]))
    return states


def _training_matrix(
    states: Sequence[Mapping[str, Any]], *, effect: str,
    maximum_per_class_per_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for state in states:
        for label in (0, 1):
            rows = sorted(
                (row for row in state["rows"] if row["labels"][effect] == label),
                key=lambda row: _rank(
                    state["task_id"], state["step"], row["action"], effect,
                ),
            )[:maximum_per_class_per_state]
            selected.extend(rows)
    features = np.asarray([row["features"] for row in selected])
    labels = np.asarray([row["labels"][effect] for row in selected], dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError(f"typed head has one label class: {effect}")
    return features, labels


def _fit_head(
    features: np.ndarray, labels: np.ndarray, *, seed: int,
    hidden_units: int, alpha: float, maximum_iterations: int,
) -> tuple[MLPClassifier, dict[str, Any]]:
    counts = np.bincount(labels, minlength=2)
    target = int(np.max(counts))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(np.flatnonzero(labels == value), target, replace=True)
        for value in (0, 1)
    ])
    rng.shuffle(chosen)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="tanh",
        solver="lbfgs",
        alpha=alpha,
        max_iter=maximum_iterations,
        random_state=seed,
    )
    model.fit(features[chosen], labels[chosen])
    artifact = {
        "kind": "target-native-binary-mlp-v1",
        "hidden_activation": "tanh",
        "layers": [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in zip(model.coefs_, model.intercepts_)
        ],
        "raw_training_examples": len(labels),
        "raw_positive_examples": int(np.sum(labels)),
        "balanced_training_examples": len(chosen),
    }
    return model, artifact


def _effect_predictions(
    state: Mapping[str, Any], *, models: Mapping[str, MLPClassifier],
    policy_head: Mapping[str, Any], policy_exponent: float,
) -> tuple[list[str], list[dict[str, float]], list[float]]:
    from motif_transfer.alfworld_hierarchical_grounder import mlp_probability

    actions = [str(row["action"]) for row in state["rows"]]
    policy = [
        mlp_probability(row["features"], policy_head) for row in state["rows"]
    ]
    matrix = np.asarray([row["features"] for row in state["rows"]])
    horizon = {
        effect: models[effect].predict_proba(matrix)[:, 1]
        for effect in EFFECT_TYPES
    }
    effects = [{
        effect: float(horizon[effect][index] * policy[index] ** policy_exponent)
        for effect in EFFECT_TYPES
    } for index in range(len(actions))]
    return actions, effects, policy


def _portfolio_metrics(
    states: Sequence[Mapping[str, Any]], *, models: Mapping[str, MLPClassifier],
    policy_head: Mapping[str, Any], policy_exponent: float,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    selected_effects = Counter()
    for state in states:
        actions, effects, policy = _effect_predictions(
            state, models=models, policy_head=policy_head,
            policy_exponent=policy_exponent,
        )
        ids = [
            stable_hash({"target_native_action": action}) for action in actions
        ]
        receipt = select_source_program_portfolio(
            source_artifacts,
            candidate_ids=ids,
            candidate_effects=effects,
            target_grounding_sha256=stable_hash(effects),
        )
        counts["states"] += 1
        expert = str(state["expert_action"])
        neural_index = max(range(len(actions)), key=lambda i: (policy[i], actions[i]))
        generic_index = max(range(len(actions)), key=lambda i: (
            sum(effects[i].values()) / len(EFFECT_TYPES), actions[i],
        ))
        counts["neural_hits"] += int(actions[neural_index] == expert)
        counts["generic_hits"] += int(actions[generic_index] == expert)
        selected_sha = receipt["selected_artifact_sha256"]
        if selected_sha is None:
            continue
        counts["applicable"] += 1
        artifact = next(
            row for row in source_artifacts
            if row["artifact_sha256"] == selected_sha
        )
        program = artifact["typed_effect_program"]
        order, reason = target_trial_order(program, effects)
        if reason is not None:
            raise RuntimeError(f"selected source program could not bind: {reason}")
        source_index = order[0]
        counts["source_hits"] += int(actions[source_index] == expert)
        selected_effects[str(program["selected_effect_type"])] += 1
        permuted, control = permute_selected_effect_binding(
            program, candidate_ids=ids, candidate_effects=effects,
        )
        if not control["nonidentity"]:
            raise RuntimeError("effect-binding control was an identity")
        permuted_order, reason = target_trial_order(program, permuted)
        if reason is not None:
            raise RuntimeError(f"permuted source program could not bind: {reason}")
        permuted_index = permuted_order[0]
        counts["permuted_hits"] += int(actions[permuted_index] == expert)
        counts["source_permuted_action_contrasts"] += int(
            source_index != permuted_index
        )
    total = counts["states"]
    applicable = counts["applicable"]
    return {
        "states": total,
        "applicable_states": applicable,
        "applicability_rate": applicable / total,
        "neural_only_expert_action_top1": counts["neural_hits"] / total,
        "source_induced_expert_action_top1": counts["source_hits"] / total,
        "source_permuted_expert_action_top1": counts["permuted_hits"] / total,
        "generic_scaffold_expert_action_top1": counts["generic_hits"] / total,
        "source_permuted_action_contrast_rate": (
            counts["source_permuted_action_contrasts"] / max(1, applicable)
        ),
        "selected_effect_counts": dict(selected_effects),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite grounder artifact: {args.output}")
    config_path = args.config.resolve()
    config = _read(config_path)
    receipts_path = (REPO / config["target_adaptation"]["receipts"]).resolve()
    receipts = _read(receipts_path)
    if receipts.get("qualification_or_heldout_read"):
        raise SystemExit("target adaptation receipts crossed evaluation boundary")
    base_path = (REPO / config["target_adaptation"]["policy_grounder"]).resolve()
    base = _read(base_path)
    validate_policy_artifact(base)
    source_paths = [
        (REPO / value).resolve() for value in config["source_programs"]
    ]
    source_artifacts = [_read(path) for path in source_paths]
    train_episodes = [
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_train"
    ]
    validation_episodes = [
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_validation"
    ]
    feature_bins = int(base["feature_bins"])
    if feature_bins != int(config["model"]["feature_bins"]):
        raise SystemExit(
            "configured feature bins do not match the frozen target policy head"
        )
    label_granularity = str(
        config["model"].get("label_granularity", "exact_action")
    )
    train_states = _episode_states(
        train_episodes,
        feature_bins=feature_bins,
        label_granularity=label_granularity,
    )
    validation_states = _episode_states(
        validation_episodes,
        feature_bins=feature_bins,
        label_granularity=label_granularity,
    )
    models = {}
    heads = {}
    head_metrics = {}
    for offset, effect in enumerate(EFFECT_TYPES):
        train_x, train_y = _training_matrix(
            train_states,
            effect=effect,
            maximum_per_class_per_state=int(
                config["model"]["maximum_per_class_per_state"]
            ),
        )
        model, head = _fit_head(
            train_x,
            train_y,
            seed=int(config["model"]["seed"]) + offset,
            hidden_units=int(config["model"]["hidden_units"]),
            alpha=float(config["model"]["alpha"]),
            maximum_iterations=int(config["model"]["maximum_iterations"]),
        )
        validation_x, validation_y = _training_matrix(
            validation_states,
            effect=effect,
            maximum_per_class_per_state=int(
                config["model"]["maximum_per_class_per_state"]
            ),
        )
        scores = model.predict_proba(validation_x)[:, 1]
        head_metrics[effect] = {
            "validation_examples": len(validation_y),
            "validation_positive_rate": float(np.mean(validation_y)),
            "validation_auc": float(roc_auc_score(validation_y, scores)),
        }
        models[effect] = model
        heads[effect] = head

    candidate_exponents = tuple(map(
        float, config["qualification"]["policy_support_exponent_grid"]
    ))
    train_grid = [{
        "policy_support_exponent": exponent,
        "metrics": _portfolio_metrics(
            train_states,
            models=models,
            policy_head=base["policy_head"],
            policy_exponent=exponent,
            source_artifacts=source_artifacts,
        ),
    } for exponent in candidate_exponents]
    eligible = [
        row for row in train_grid
        if row["metrics"]["source_permuted_action_contrast_rate"]
        >= float(config["qualification"]["training_minimum_contrast_rate"])
    ]
    if not eligible:
        raise SystemExit("no policy-support exponent retained source contrast")
    selected = max(eligible, key=lambda row: (
        row["metrics"]["source_induced_expert_action_top1"],
        row["metrics"]["source_permuted_action_contrast_rate"],
        row["policy_support_exponent"],
    ))
    exponent = float(selected["policy_support_exponent"])
    validation = _portfolio_metrics(
        validation_states,
        models=models,
        policy_head=base["policy_head"],
        policy_exponent=exponent,
        source_artifacts=source_artifacts,
    )
    thresholds = config["qualification"]["frozen_thresholds"]
    gates = {
        "all_typed_heads_auc": all(
            row["validation_auc"] >= float(thresholds["minimum_each_head_auc"])
            for row in head_metrics.values()
        ),
        "portfolio_applicability": validation["applicability_rate"] >= float(
            thresholds["minimum_portfolio_applicability_rate"]
        ),
        "source_permuted_action_contrast": (
            validation["source_permuted_action_contrast_rate"] >= float(
                thresholds["minimum_source_permuted_action_contrast_rate"]
            )
        ),
        "source_action_top1_noninferior": (
            validation["source_induced_expert_action_top1"]
            >= validation["neural_only_expert_action_top1"] - float(
                thresholds["maximum_source_top1_drop_vs_neural"]
            )
        ),
        "source_action_top1_noninferior_to_generic": (
            validation["source_induced_expert_action_top1"]
            >= validation["generic_scaffold_expert_action_top1"] - float(
                thresholds.get("maximum_source_top1_drop_vs_generic", 0.0)
            )
        ),
    }
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "status": (
            "ALFWORLD_TYPED_GROUNDING_QUALIFIED"
            if all(gates.values()) else "ALFWORLD_TYPED_GROUNDING_BLOCKED"
        ),
        "config_path": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "effect_types": list(EFFECT_TYPES),
        "label_granularity": label_granularity,
        "label_contract": {
            "EFFECT_BY_TRANSITION_1": (
                f"{label_granularity.upper()}_APPEARS_IN_EXPERT_CONTINUATION_BY_1"
            ),
            "EFFECT_BY_TRANSITION_4": (
                f"{label_granularity.upper()}_APPEARS_IN_EXPERT_CONTINUATION_BY_4"
            ),
            "EFFECT_BY_TRANSITION_8": (
                f"{label_granularity.upper()}_APPEARS_IN_EXPERT_CONTINUATION_BY_8"
            ),
            "EXECUTABLE_TRANSITION_PERSISTENCE": (
                f"{label_granularity.upper()}_REMAINS_NATIVE_EXECUTABLE_"
                "ACROSS_UP_TO_4_SUCCESSORS"
            ),
        },
        "feature_bins": feature_bins,
        "required_option_masked_for_every_head": True,
        "formal_success_read_for_training_or_qualification": False,
        "target_adaptation_receipts": {
            "path": str(receipts_path),
            "file_sha256": _sha256(receipts_path),
            "train_episodes": len(train_episodes),
            "validation_episodes": len(validation_episodes),
        },
        "target_policy_grounder": {
            "path": str(base_path),
            "file_sha256": _sha256(base_path),
            "artifact_sha256": base["artifact_sha256"],
        },
        "target_policy_head": base["policy_head"],
        "typed_effect_heads": heads,
        "policy_support_exponent": exponent,
        "policy_support_exponent_selection": {
            "partition": "adaptation_train",
            "grid": train_grid,
        },
        "head_validation": head_metrics,
        "portfolio_validation": validation,
        "frozen_qualification_thresholds": thresholds,
        "qualification_gates": gates,
        "source_programs": [
            {"path": str(path), "file_sha256": _sha256(path),
             "artifact_sha256": artifact["artifact_sha256"]}
            for path, artifact in zip(source_paths, source_artifacts)
        ],
        "target_outcome_used_for_source_program_selection": False,
        "source_identity_used_as_grounder_feature": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
        "policy_support_exponent": exponent,
        "head_validation": head_metrics,
        "portfolio_validation": validation,
        "qualification_gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
