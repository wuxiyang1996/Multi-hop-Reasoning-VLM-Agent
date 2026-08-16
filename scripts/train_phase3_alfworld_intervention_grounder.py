#!/usr/bin/env python3
"""Train neural typed-effect heads from ALFWorld intervention forks."""

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

from motif_transfer.alfworld_hierarchical_grounder import mlp_probability  # noqa: E402
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
from motif_transfer.phase3_typed_effect_induction import target_trial_order  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _states(
    receipts: Mapping[str, Any], *, partition: str,
    feature_bins: int, policy_head: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for episode in receipts["episodes"]:
        if str(episode["partition"]) != partition:
            continue
        for snapshot in episode["snapshots"]:
            history = tuple(map(str, snapshot["prefix_actions"]))
            rows = []
            for candidate in snapshot["candidates"]:
                action = str(candidate["candidate_action"])
                features = masked_features(
                    goal=str(snapshot["goal"]),
                    observation=str(snapshot["before_observation"]),
                    action=action,
                    step=int(snapshot["step"]),
                    action_history=history,
                    feature_bins=feature_bins,
                )
                rows.append({
                    "option": str(candidate["candidate_option"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    "action": action,
                    "features": features,
                    "policy": mlp_probability(features, policy_head),
                    "actual_effects": {
                        name: float(candidate["normalized_typed_effects"][name])
                        for name in EFFECT_TYPES
                    },
                })
            if len(rows) >= 2:
                result.append({
                    "task_id": str(snapshot["task_id"]),
                    "step": int(snapshot["step"]),
                    "snapshot_sha256": str(snapshot["snapshot_sha256"]),
                    "rows": rows,
                })
    return result


def _matrix(
    states: Sequence[Mapping[str, Any]], *, effect: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    features = []
    labels = []
    varying = 0
    for state in states:
        values = [float(row["actual_effects"][effect]) for row in state["rows"]]
        winners = [index for index, value in enumerate(values) if value == max(values)]
        if len(winners) != 1 or max(values) <= min(values):
            continue
        varying += 1
        for index, row in enumerate(state["rows"]):
            features.append(row["features"])
            labels.append(int(index == winners[0]))
    if not features or len(set(labels)) != 2:
        raise RuntimeError(f"intervention head has insufficient labels: {effect}")
    return np.asarray(features), np.asarray(labels, dtype=np.int64), varying


def _fit(
    features: np.ndarray, labels: np.ndarray, *, seed: int,
    hidden_units: int, alpha: float, maximum_iterations: int,
) -> tuple[MLPClassifier, dict[str, Any]]:
    counts = np.bincount(labels, minlength=2)
    target = int(np.max(counts))
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(labels == value), target, replace=True)
        for value in (0, 1)
    ])
    rng.shuffle(selected)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,), activation="tanh", solver="lbfgs",
        alpha=alpha, max_iter=maximum_iterations, random_state=seed,
    )
    model.fit(features[selected], labels[selected])
    head = {
        "kind": "target-native-intervention-binary-mlp-v1",
        "hidden_activation": "tanh",
        "layers": [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in zip(model.coefs_, model.intercepts_)
        ],
        "raw_training_examples": len(labels),
        "raw_positive_examples": int(np.sum(labels)),
        "balanced_training_examples": len(selected),
    }
    return model, head


def _predictions(
    state: Mapping[str, Any], *, heads: Mapping[str, Mapping[str, Any]],
    exponent: float,
) -> tuple[list[dict[str, float]], list[float]]:
    policy = [float(row["policy"]) for row in state["rows"]]
    effects = [{
        effect: mlp_probability(row["features"], heads[effect])
        * policy[index] ** exponent
        for effect in EFFECT_TYPES
    } for index, row in enumerate(state["rows"])]
    return effects, policy


def _metrics(
    states: Sequence[Mapping[str, Any]], *,
    heads: Mapping[str, Mapping[str, Any]], exponent: float,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    utilities = Counter()
    selected_effects = Counter()
    selected_programs = Counter()
    for state in states:
        effects, policy = _predictions(state, heads=heads, exponent=exponent)
        rows = list(state["rows"])
        ids = [str(row["candidate_id"]) for row in rows]
        neural = max(range(len(rows)), key=lambda i: (policy[i], rows[i]["option"]))
        generic = max(range(len(rows)), key=lambda i: (
            sum(effects[i].values()) / len(EFFECT_TYPES), policy[i], rows[i]["option"],
        ))
        receipt = select_source_program_portfolio(
            source_artifacts,
            candidate_ids=ids,
            candidate_effects=effects,
            target_grounding_sha256=stable_hash(effects),
        )
        source = neural
        permuted = neural
        selected_sha = receipt["selected_artifact_sha256"]
        if selected_sha is not None:
            counts["applicable"] += 1
            artifact = next(
                row for row in source_artifacts
                if row["artifact_sha256"] == selected_sha
            )
            program = artifact["typed_effect_program"]
            order, reason = target_trial_order(program, effects)
            if reason is not None:
                raise RuntimeError(f"source intervention binding failed: {reason}")
            source = order[0]
            shuffled, _ = permute_selected_effect_binding(
                program, candidate_ids=ids, candidate_effects=effects,
            )
            shuffled_order, reason = target_trial_order(program, shuffled)
            if reason is not None:
                raise RuntimeError(f"permuted intervention binding failed: {reason}")
            permuted = shuffled_order[0]
            effect = str(program["selected_effect_type"])
            selected_effects[effect] += 1
            selected_programs[str(program["program_sha256"])] += 1
            actual = [float(row["actual_effects"][effect]) for row in rows]
            winners = [i for i, value in enumerate(actual) if value == max(actual)]
            if len(winners) == 1 and max(actual) > min(actual):
                counts["type_evaluable"] += 1
                counts["source_type_hits"] += int(source == winners[0])
        counts["states"] += 1
        counts["source_neural_changes"] += int(source != neural)
        counts["source_permuted_contrasts"] += int(source != permuted)
        for name, index in (
            ("neural", neural), ("source", source),
            ("permuted", permuted), ("generic", generic),
        ):
            utilities[name] += float(
                rows[index]["actual_effects"]["EFFECT_BY_TRANSITION_8"]
            )
    total = counts["states"]
    return {
        "states": total,
        "applicability_rate": counts["applicable"] / max(1, total),
        "source_neural_option_change_rate": (
            counts["source_neural_changes"] / max(1, total)
        ),
        "source_permuted_option_contrast_rate": (
            counts["source_permuted_contrasts"] / max(1, total)
        ),
        "source_selected_effect_accuracy": (
            counts["source_type_hits"] / max(1, counts["type_evaluable"])
        ),
        "source_selected_effect_evaluable_states": counts["type_evaluable"],
        "mean_h8_utility": {
            name: float(utilities[name]) / max(1, total)
            for name in ("neural", "source", "permuted", "generic")
        },
        "selected_effect_counts": dict(selected_effects),
        "selected_program_counts": dict(selected_programs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite intervention grounder: {args.output}")
    config_path = args.config.resolve()
    config = _read(config_path)
    receipts_path = (REPO / config["intervention_receipts"]).resolve()
    receipts = _read(receipts_path)
    if receipts.get("formal_success_read") is not False:
        raise SystemExit("intervention receipts did not attest outcome blindness")
    if receipts.get("qualification_or_formal_target_reset") is not False:
        raise SystemExit("intervention receipts crossed evaluation boundary")
    policy_path = Path(receipts["target_policy_artifact"]["path"]).resolve()
    policy = _read(policy_path)
    source_paths = [(REPO / value).resolve() for value in config["source_programs"]]
    source_artifacts = [_read(path) for path in source_paths]
    feature_bins = int(policy["feature_bins"])
    train = _states(
        receipts, partition="adaptation_train", feature_bins=feature_bins,
        policy_head=policy["policy_head"],
    )
    validation = _states(
        receipts, partition="adaptation_validation", feature_bins=feature_bins,
        policy_head=policy["policy_head"],
    )
    heads = {}
    head_validation = {}
    for offset, effect in enumerate(EFFECT_TYPES):
        train_x, train_y, train_varying = _matrix(train, effect=effect)
        model, head = _fit(
            train_x, train_y,
            seed=int(config["model"]["seed"]) + offset,
            hidden_units=int(config["model"]["hidden_units"]),
            alpha=float(config["model"]["alpha"]),
            maximum_iterations=int(config["model"]["maximum_iterations"]),
        )
        validation_x, validation_y, validation_varying = _matrix(
            validation, effect=effect,
        )
        scores = model.predict_proba(validation_x)[:, 1]
        head_validation[effect] = {
            "train_varying_snapshots": train_varying,
            "validation_varying_snapshots": validation_varying,
            "validation_examples": len(validation_y),
            "validation_auc": float(roc_auc_score(validation_y, scores)),
        }
        heads[effect] = head
    grid = []
    for exponent in map(float, config["policy_support_exponent_grid"]):
        metrics = _metrics(
            train, heads=heads, exponent=exponent,
            source_artifacts=source_artifacts,
        )
        grid.append({"policy_support_exponent": exponent, "metrics": metrics})
    selection = config["train_selection"]
    eligible = [row for row in grid if (
        row["metrics"]["source_neural_option_change_rate"]
        >= float(selection["minimum_source_neural_option_change_rate"])
        and row["metrics"]["source_permuted_option_contrast_rate"]
        >= float(selection["minimum_source_permuted_option_contrast_rate"])
    )]
    if not eligible:
        raise SystemExit("no intervention-grounded policy exponent qualified on train")
    selected = max(eligible, key=lambda row: (
        row["metrics"]["mean_h8_utility"]["source"],
        row["metrics"]["source_selected_effect_accuracy"],
        row["policy_support_exponent"],
    ))
    exponent = float(selected["policy_support_exponent"])
    validation_metrics = _metrics(
        validation, heads=heads, exponent=exponent,
        source_artifacts=source_artifacts,
    )
    thresholds = config["frozen_validation_thresholds"]
    utilities = validation_metrics["mean_h8_utility"]
    gates = {
        "all_intervention_heads_auc": all(
            row["validation_auc"] >= float(thresholds["minimum_each_head_auc"])
            for row in head_validation.values()
        ),
        "portfolio_applicability": validation_metrics["applicability_rate"]
        >= float(thresholds["minimum_applicability_rate"]),
        "source_effect_prediction": (
            validation_metrics["source_selected_effect_accuracy"]
            >= float(thresholds["minimum_source_selected_effect_accuracy"])
        ),
        "source_h8_noninferior_to_neural": utilities["source"] >= (
            utilities["neural"] - float(thresholds["maximum_h8_drop_vs_neural"])
        ),
        "source_change_nontrivial": (
            validation_metrics["source_neural_option_change_rate"]
            >= float(thresholds["minimum_source_neural_option_change_rate"])
        ),
        "permuted_contrast": (
            validation_metrics["source_permuted_option_contrast_rate"]
            >= float(thresholds["minimum_source_permuted_option_contrast_rate"])
        ),
        "multiple_source_effects": len(validation_metrics["selected_effect_counts"])
        >= int(thresholds["minimum_selected_effect_types"]),
    }
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "status": (
            "ALFWORLD_INTERVENTION_GROUNDER_QUALIFIED"
            if all(gates.values()) else "ALFWORLD_INTERVENTION_GROUNDER_BLOCKED"
        ),
        "binding_level": "target_native_option",
        "label_granularity": "target_native_option_intervention",
        "effect_types": list(EFFECT_TYPES),
        "feature_bins": feature_bins,
        "typed_effect_heads": heads,
        "target_policy_head": policy["policy_head"],
        "policy_support_exponent": exponent,
        "minimum_source_policy_support_ratio": 0.0,
        "effect_observation_protocol": "TARGET_NATIVE_MACRO_ROLLOUT_V1",
        "required_option_masked_for_every_head": True,
        "formal_success_read_for_training_or_qualification": False,
        "source_identity_used_as_grounder_feature": False,
        "target_outcome_used_for_source_program_selection": False,
        "source_programs": [{
            "path": str(path),
            "file_sha256": _sha256(path),
            "artifact_sha256": artifact["artifact_sha256"],
        } for path, artifact in zip(source_paths, source_artifacts)],
        "target_adaptation_receipts": {
            "path": str(receipts_path), "file_sha256": _sha256(receipts_path),
            "train_snapshots": len(train), "validation_snapshots": len(validation),
        },
        "grounding_supervision": (
            "DEVELOPMENT_ONLY_TARGET_NATIVE_OPTION_INTERVENTION_FORKS_"
            "WITH_H1_H4_H8_TRANSITION_EFFECTS"
        ),
        "head_validation": head_validation,
        "policy_support_exponent_selection": {
            "partition": "adaptation_train", "grid": grid, "selected": selected,
        },
        "intervention_qualification": {
            "validation": validation_metrics,
            "frozen_thresholds": thresholds,
            "gates": gates,
        },
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
        "head_validation": head_validation,
        "selected_policy_support_exponent": exponent,
        "validation": validation_metrics,
        "gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
