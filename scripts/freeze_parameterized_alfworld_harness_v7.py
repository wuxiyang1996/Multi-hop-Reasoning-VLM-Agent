#!/usr/bin/env python3
"""Train and freeze the adaptation-only parameterized ALFWorld Harness V7."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier

from motif_transfer.alfworld_hierarchical_grounder import action_option
from motif_transfer.alfworld_masked_effect_grounder import (
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.parameterized_alfworld_harness import (
    PROPERTY_CLASSES,
    choose_parameterized_action,
    parameterize_source_ir,
    property_label_from_actions,
    property_router_features,
    property_router_probabilities,
    target_effect_receipt,
    validate_parameterized_source_ir,
    validate_property_router,
)
from motif_transfer.typed_alfworld_harness import target_effect


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_goal(episode: Mapping[str, Any]) -> str:
    goals = {str(row["goal"]) for row in episode["transitions"]}
    if len(goals) != 1:
        raise ValueError("adaptation episode does not have one stable target goal")
    return next(iter(goals))


def _episode_property(episode: Mapping[str, Any]) -> str:
    return property_label_from_actions(
        [str(row["expert_action"]) for row in episode["transitions"]]
    )


def _fit_property_router(
    episodes: Sequence[Mapping[str, Any]],
    *,
    feature_bins: int,
    hidden_units: int,
    maximum_iterations: int,
    seed: int,
) -> tuple[MLPClassifier, dict[str, Any]]:
    features = np.asarray([
        property_router_features(_episode_goal(row), feature_bins=feature_bins)
        for row in episodes
    ])
    labels = np.asarray([_episode_property(row) for row in episodes])
    if set(labels) != set(PROPERTY_CLASSES):
        raise RuntimeError("adaptation train does not cover every property class")
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="tanh",
        solver="lbfgs",
        alpha=0.2,
        max_iter=maximum_iterations,
        random_state=seed,
    )
    model.fit(features, labels)
    class_order = [list(map(str, model.classes_)).index(name) for name in PROPERTY_CLASSES]
    weights = [np.asarray(value) for value in model.coefs_]
    biases = [np.asarray(value) for value in model.intercepts_]
    weights[-1] = weights[-1][:, class_order]
    biases[-1] = biases[-1][class_order]
    body = {
        "schema_version": "target-native-property-router-v7",
        "training_authority": "TARGET_ADAPTATION_EXPERT_ACTIONS_ONLY",
        "input_authority": "TARGET_GOAL_TEXT_HASH_FEATURES_ONLY",
        "feature_bins": feature_bins,
        "hidden_activation": "tanh",
        "classes": list(PROPERTY_CLASSES),
        "layers": [
            {"weights": weight.tolist(), "bias": bias.tolist()}
            for weight, bias in zip(weights, biases)
        ],
        "training_episodes": len(episodes),
        "training_class_counts": dict(Counter(map(str, labels))),
        "random_seed": seed,
    }
    return model, body | {"artifact_sha256": stable_hash(body)}


def _router_metrics(
    episodes: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for episode in episodes:
        label = _episode_property(episode)
        probabilities = property_router_probabilities(_episode_goal(episode), artifact)
        predicted = max(
            PROPERTY_CLASSES,
            key=lambda name: (float(probabilities[name]), name),
        )
        rows.append((label, predicted, float(probabilities[predicted])))
    per_class = {}
    for name in PROPERTY_CLASSES:
        relevant = [row for row in rows if row[0] == name]
        per_class[name] = {
            "episodes": len(relevant),
            "recall": sum(row[1] == name for row in relevant) / len(relevant),
            "mean_predicted_confidence": sum(row[2] for row in relevant) / len(relevant),
        }
    return {
        "episodes": len(rows),
        "accuracy": sum(label == predicted for label, predicted, _ in rows) / len(rows),
        "minimum_predicted_confidence": min(row[2] for row in rows),
        "per_class": per_class,
    }


def _cache_states(
    episodes: Sequence[Mapping[str, Any]],
    *,
    target: Mapping[str, Any],
    router: Mapping[str, Any],
    minimum_role_binding: float,
) -> list[dict[str, Any]]:
    result = []
    for episode in episodes:
        history: list[str] = []
        receipts: list[str] = []
        goal = _episode_goal(episode)
        probabilities = property_router_probabilities(goal, router)
        required_property = max(
            PROPERTY_CLASSES,
            key=lambda name: (float(probabilities[name]), name),
        )
        for transition in episode["transitions"]:
            grounded = score_actions(
                goal=goal,
                observation=str(transition["before_observation"]),
                native_actions=tuple(map(str, transition["native_actions"])),
                step=int(transition["step"]),
                action_history=history,
                artifact=target,
            )
            expert = str(transition["expert_action"])
            result.append({
                "task_id": str(episode["task_id"]),
                "partition": str(episode["partition"]),
                "step": int(transition["step"]),
                "goal": goal,
                "grounded": grounded,
                "history": tuple(history),
                "effect_receipts": tuple(receipts),
                "property_probabilities": probabilities,
                "expert_action": expert,
                "expert_effect": target_effect(action_option(expert)),
            })
            if expert in grounded:
                receipt = target_effect_receipt(
                    action=expert,
                    grounding=grounded[expert],
                    required_property=required_property,
                    minimum_role_binding=minimum_role_binding,
                )
            else:
                receipt = "IGNORE"
            history.append(expert)
            receipts.append(receipt)
    return result


def _harness_metrics(
    states: Sequence[Mapping[str, Any]],
    *,
    source_ir: Mapping[str, Any],
    minimum_property_confidence: float,
    minimum_role_binding: float,
    minimum_realization_score: float,
    minimum_target_policy_ratio: float,
    include_changed_examples: bool = False,
) -> dict[str, Any]:
    counts = {
        "target_only": Counter(),
        "authentic_parameterized_ir": Counter(),
    }
    changed_tasks = {condition: set() for condition in counts}
    changed_families = {condition: set() for condition in counts}
    per_effect: dict[str, Counter[str]] = {}
    changed_examples: list[dict[str, Any]] = []
    for state in states:
        expert_effect = str(state["expert_effect"])
        per_effect.setdefault(expert_effect, Counter())
        for condition, row in counts.items():
            decision = choose_parameterized_action(
                condition=condition,
                grounded=state["grounded"],
                history=state["history"],
                effect_receipts=state["effect_receipts"],
                source_ir=source_ir,
                property_probabilities=state["property_probabilities"],
                minimum_property_confidence=minimum_property_confidence,
                minimum_role_binding=minimum_role_binding,
                minimum_realization_score=minimum_realization_score,
                minimum_target_policy_ratio=minimum_target_policy_ratio,
            )
            row["states"] += 1
            row["effect_hits"] += int(
                decision["target_realized_effect"] == expert_effect
            )
            row["action_hits"] += int(decision["action"] == state["expert_action"])
            row["source_admissions"] += int(decision["source_admitted"])
            row["changed_effects"] += int(decision["changed_effect"])
            if decision["changed_effect"]:
                changed_tasks[condition].add(str(state["task_id"]))
                changed_families[condition].add(
                    str(state["task_id"]).split("-", 1)[0]
                )
            if (
                include_changed_examples
                and condition == "authentic_parameterized_ir"
                and decision["changed_effect"]
            ):
                changed_examples.append({
                    "task_id": state["task_id"],
                    "partition": state["partition"],
                    "step": state["step"],
                    "goal": state["goal"],
                    "expert_action": state["expert_action"],
                    "expert_effect": expert_effect,
                    "fallback_action": decision["fallback_action"],
                    "fallback_effect": decision["fallback_effect"],
                    "selected_action": decision["action"],
                    "selected_effect": decision["target_realized_effect"],
                    "requested_source_effect": decision.get(
                        "requested_source_effect"
                    ),
                    "required_property": decision.get("required_property"),
                    "target_cycle_state": decision["target_cycle_state"],
                    "target_policy_ratio": decision.get("target_policy_ratio"),
                    "best_realization_score": decision.get(
                        "best_realization_score"
                    ),
                })
            if expert_effect not in {"POSITION", "EXCLUDE"}:
                row["nonposition_states"] += 1
                row["nonposition_hits"] += int(
                    decision["target_realized_effect"] == expert_effect
                )
            effect_row = per_effect[expert_effect]
            effect_row[f"{condition}_states"] += 1
            effect_row[f"{condition}_hits"] += int(
                decision["target_realized_effect"] == expert_effect
            )
    summaries = {}
    for condition, row in counts.items():
        summaries[condition] = {
            "states": row["states"],
            "effect_accuracy": row["effect_hits"] / row["states"],
            "expert_action_top1": row["action_hits"] / row["states"],
            "nonposition_effect_recall": (
                row["nonposition_hits"] / row["nonposition_states"]
            ),
            "source_admission_rate": row["source_admissions"] / row["states"],
            "changed_effect_rate": row["changed_effects"] / row["states"],
            "changed_effect_count": row["changed_effects"],
            "changed_task_count": len(changed_tasks[condition]),
            "changed_task_family_count": len(changed_families[condition]),
        }
    result = {
        "minimum_realization_score": minimum_realization_score,
        "minimum_target_policy_ratio": minimum_target_policy_ratio,
        "conditions": summaries,
        "per_expert_effect": {
            effect: {
                condition: (
                    row[f"{condition}_hits"] / row[f"{condition}_states"]
                )
                for condition in counts
            }
            for effect, row in sorted(per_effect.items())
        },
    }
    if include_changed_examples:
        result["changed_effect_examples"] = changed_examples
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--target-grounder", type=Path, required=True)
    parser.add_argument("--adaptation-receipts", type=Path, required=True)
    parser.add_argument("--adaptation-gate-receipts", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-bins", type=int, default=96)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--maximum-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=97201)
    parser.add_argument("--minimum-property-confidence", type=float, default=0.8)
    parser.add_argument("--minimum-role-binding", type=float, default=0.5)
    parser.add_argument("--realization-grid", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--policy-ratio-grid", default="0.05,0.1,0.25,0.5,0.75,0.9,0.95")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen Harness: {args.output}")

    source = _read(args.source_report)
    target = _read(args.target_grounder)
    receipts = _read(args.adaptation_receipts)
    gate_receipts = _read(args.adaptation_gate_receipts)
    manifest = _read(args.confirmation_manifest)
    if source.get("overall_status") != "SOURCE_TYPED_GATE_PASSED":
        raise SystemExit("real-source V4 typed gate did not pass")
    if source.get("edge_replication_gate", {}).get("status") != "EDGE_REPLICATION_GATE_PASSED":
        raise SystemExit("real-source edge replication gate did not pass")
    validate_target_artifact(target)
    if target.get("status") != "ADAPTATION_GATE_PASSED":
        raise SystemExit("target-native masked grounder gate did not pass")
    if receipts.get("qualification_or_heldout_read"):
        raise SystemExit("adaptation receipts crossed an evaluation boundary")
    if gate_receipts.get("confirmation_or_heldout_read"):
        raise SystemExit("adaptation gate crossed the fresh evaluation boundary")
    if gate_receipts.get("selection_used_target_outcomes"):
        raise SystemExit("adaptation-gate task selection used target outcomes")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET":
        raise SystemExit("fresh confirmation manifest was not frozen before reset")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("fresh confirmation selection used target outcomes")
    source_ir = parameterize_source_ir(source["effect_ir"])
    validate_parameterized_source_ir(source_ir)
    train_episodes = [
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_train"
    ]
    validation_episodes = [
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_validation"
    ]
    gate_episodes = list(gate_receipts["episodes"])
    original_ids = {str(row["task_id"]) for row in receipts["episodes"]}
    gate_ids = {str(row["task_id"]) for row in gate_episodes}
    confirmation_ids = set(map(
        str, manifest["splits"]["fresh_confirmation"]
    ))
    if original_ids & gate_ids or gate_ids & confirmation_ids:
        raise SystemExit("target adaptation/confirmation task identity overlap")
    if any(row.get("partition") != "adaptation_gate" for row in gate_episodes):
        raise SystemExit("new target gate receipts have a wrong partition")
    _, router = _fit_property_router(
        train_episodes,
        feature_bins=args.feature_bins,
        hidden_units=args.hidden_units,
        maximum_iterations=args.maximum_iterations,
        seed=args.seed,
    )
    validate_property_router(router)
    router_train = _router_metrics(train_episodes, router)
    router_validation = _router_metrics(validation_episodes, router)
    router_gate = _router_metrics(gate_episodes, router)
    selection_states = _cache_states(
        [*train_episodes, *validation_episodes],
        target=target,
        router=router,
        minimum_role_binding=args.minimum_role_binding,
    )
    gate_states = _cache_states(
        gate_episodes,
        target=target,
        router=router,
        minimum_role_binding=args.minimum_role_binding,
    )
    realization_grid = tuple(map(float, args.realization_grid.split(",")))
    policy_grid = tuple(map(float, args.policy_ratio_grid.split(",")))
    training_grid = [
        _harness_metrics(
            selection_states,
            source_ir=source_ir,
            minimum_property_confidence=args.minimum_property_confidence,
            minimum_role_binding=args.minimum_role_binding,
            minimum_realization_score=realization,
            minimum_target_policy_ratio=ratio,
        )
        for realization in realization_grid
        for ratio in policy_grid
    ]
    eligible = []
    for row in training_grid:
        authentic = row["conditions"]["authentic_parameterized_ir"]
        baseline = row["conditions"]["target_only"]
        if (
            row["minimum_realization_score"] >= 0.05
            and authentic["changed_effect_count"] >= 12
            and authentic["changed_task_family_count"] >= 2
            and 0.03 <= authentic["source_admission_rate"] <= 0.30
            and authentic["effect_accuracy"] >= baseline["effect_accuracy"] - 0.01
        ):
            eligible.append(row)
    if not eligible:
        raise SystemExit("no adaptation-train candidate retained safe nontrivial transfer")
    selected = max(
        eligible,
        key=lambda row: (
            row["conditions"]["authentic_parameterized_ir"]["effect_accuracy"],
            row["conditions"]["authentic_parameterized_ir"]["expert_action_top1"],
            row["conditions"]["authentic_parameterized_ir"]["nonposition_effect_recall"],
            -row["minimum_target_policy_ratio"],
            row["minimum_realization_score"],
        ),
    )
    adaptation_gate = _harness_metrics(
        gate_states,
        source_ir=source_ir,
        minimum_property_confidence=args.minimum_property_confidence,
        minimum_role_binding=args.minimum_role_binding,
        minimum_realization_score=float(selected["minimum_realization_score"]),
        minimum_target_policy_ratio=float(selected["minimum_target_policy_ratio"]),
        include_changed_examples=True,
    )
    authentic = adaptation_gate["conditions"]["authentic_parameterized_ir"]
    baseline = adaptation_gate["conditions"]["target_only"]
    gates = {
        "source_target_and_manifest_prerequisites": True,
        "property_router_adaptation_gate_accuracy": router_gate["accuracy"] >= 0.90,
        "property_router_adaptation_gate_every_class_recall": all(
            row["recall"] >= 0.50
            for row in router_gate["per_class"].values()
        ),
        "adaptation_gate_effect_accuracy_noninferior": (
            authentic["effect_accuracy"] >= baseline["effect_accuracy"] - 0.01
        ),
        "adaptation_gate_action_top1_noninferior": (
            authentic["expert_action_top1"] >= baseline["expert_action_top1"] - 0.01
        ),
        "adaptation_gate_changed_effect_nontrivial": (
            authentic["changed_effect_count"] >= 2
            and authentic["changed_task_count"] >= 2
        ),
        "adaptation_gate_source_admission_nonconstant": (
            0.01 <= authentic["source_admission_rate"] <= 0.30
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "parameterized-real-source-alfworld-harness-v7",
        "status": (
            "FRESH_CONFIRMATION_AUTHORIZED"
            if passed else "BLOCKED_BEFORE_FRESH_CONFIRMATION"
        ),
        "claim_boundary": (
            "TARGET_ADAPTATION_ONLY_ROUTER_AND_CALIBRATION; FRESH_CONFIRMATION "
            "PERMITTED_ONLY_FOR_THIS_HASH; EXISTING_VALID_UNSEEN_HELDOUT_FORBIDDEN"
        ),
        "source_report": {
            "path": str(args.source_report.resolve()),
            "file_sha256": _sha256(args.source_report),
            "parent_ir_sha256": source["effect_ir"]["ir_sha256"],
        },
        "parameterized_source_ir": source_ir,
        "target_grounder": {
            "path": str(args.target_grounder.resolve()),
            "file_sha256": _sha256(args.target_grounder),
            "artifact_sha256": target["artifact_sha256"],
        },
        "adaptation_receipts": {
            "path": str(args.adaptation_receipts.resolve()),
            "file_sha256": _sha256(args.adaptation_receipts),
        },
        "adaptation_gate_receipts": {
            "path": str(args.adaptation_gate_receipts.resolve()),
            "file_sha256": _sha256(args.adaptation_gate_receipts),
            "episodes": len(gate_episodes),
            "successful_expert_episodes": sum(
                bool(row["official_success"]) for row in gate_episodes
            ),
        },
        "confirmation_manifest": {
            "path": str(args.confirmation_manifest.resolve()),
            "file_sha256": _sha256(args.confirmation_manifest),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "property_router": router,
        "property_router_metrics": {
            "adaptation_train": router_train,
            "adaptation_validation": router_validation,
            "fresh_adaptation_gate": router_gate,
        },
        "thresholds": {
            "selection_partition": (
                "adaptation_train_plus_consumed_adaptation_validation"
            ),
            "minimum_property_confidence": args.minimum_property_confidence,
            "minimum_role_binding": args.minimum_role_binding,
            "realization_grid": list(realization_grid),
            "policy_ratio_grid": list(policy_grid),
            "selected_minimum_realization_score": selected["minimum_realization_score"],
            "selected_minimum_target_policy_ratio": selected["minimum_target_policy_ratio"],
            "training_grid": training_grid,
        },
        "fresh_adaptation_gate": adaptation_gate,
        "gates": gates,
        "permissions": {
            "source_ir": [
                "SELECT_PARAMETERIZED_EFFECT",
                "ORDER_GOAL_CONDITIONED_EFFECTS",
            ],
            "target_router": [
                "PREDICT_REQUIRED_PROPERTY_FROM_TARGET_GOAL",
                "GROUND_TARGET_OBJECT_ROLE",
                "ABSTAIN_TO_EXACT_TARGET_POLICY",
            ],
            "forbidden": [
                "SOURCE_ACTION_OR_COORDINATE_AT_RUNTIME",
                "SOURCE_TASK_ID_AT_RUNTIME",
                "TARGET_REQUIRED_OPTION_DIAGNOSTIC",
                "TARGET_QUALIFICATION_OR_HELDOUT_RESET",
                "TARGET_CONFIRMATION_OUTCOME_DURING_SELECTION",
            ],
        },
    }
    payload = body | {"harness_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": payload["status"],
        "harness_sha256": payload["harness_sha256"],
        "property_router_validation": router_validation,
        "property_router_fresh_adaptation_gate": router_gate,
        "selected": {
            "minimum_realization_score": selected["minimum_realization_score"],
            "minimum_target_policy_ratio": selected["minimum_target_policy_ratio"],
        },
        "fresh_adaptation_gate": adaptation_gate,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
