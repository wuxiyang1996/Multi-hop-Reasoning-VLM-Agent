#!/usr/bin/env python3
"""Train/qualify a target-native structural grounder and target function."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    validate_structural_program,
)
from motif_transfer.target_structural_induction import (  # noqa: E402
    DISCOVERYWORLD_OPERATOR_IDS,
    REMOVE_ENTITY_SLOT,
    discoveryworld_core_sequence,
    discoveryworld_transition_operator_ids,
    export_mlp_grounder,
    grounded_operator_ids,
    induce_target_partial_order_program,
    source_sequence_support,
    target_action_features,
    target_action_labels,
    target_program_supports,
)


TRAIN_SEEDS = (45, 46, 47)
QUALIFICATION_SEEDS = (48, 49, 50)
SOURCE_PROGRAM_ROOT = REPO / "configs/source_structural_v5c_frozen/programs"
MATCHED_ROOT = (
    REPO / "runs/phase3_discoveryworld_consumed_development_"
    "v14_fail_closed_acquisition_typed"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _source_programs() -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted(SOURCE_PROGRAM_ROOT.glob("*.json")):
        program = _read(path)
        validate_structural_program(program)
        output[path.stem] = program
    if len(output) != 3:
        raise ValueError("expected the three fresh-validated source programs")
    return output


def _target_success_path(seed: int) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any],
]:
    root = MATCHED_ROOT / f"proteomics.easy.seed{seed}"
    config_path = root / "config.json"
    result_path = root / "matched_result.json"
    config = _read(config_path)
    result = _read(result_path)
    _self_hash(result, "result_sha256")
    reference_path = REPO / str(config["reference_episode"])
    reference = _read(reference_path)
    _self_hash(reference, "episode_sha256")
    fork_step = int(result["fork_after_episode_step"])
    if (
        reference["steps"][fork_step - 1]["transition"][
            "after_policy_state_sha256"
        ]
        != result["fork_policy_state_sha256"]
    ):
        raise ValueError(f"target development fork mismatch: seed {seed}")
    ceiling = result["conditions"]["target_native_ceiling"]
    if ceiling.get("official_success") is not True:
        raise ValueError(f"target-native development ceiling failed: seed {seed}")
    steps = [
        *reference["steps"][:fork_step],
        *ceiling["recovery"],
    ]
    receipt = {
        "seed": seed,
        "reference_path": str(reference_path.relative_to(REPO)),
        "reference_file_sha256": _sha(reference_path),
        "reference_episode_sha256": reference["episode_sha256"],
        "fork_step": fork_step,
        "matched_result_path": str(result_path.relative_to(REPO)),
        "matched_result_file_sha256": _sha(result_path),
        "matched_result_sha256": result["result_sha256"],
        "development_success_selection_read": True,
        "formal_target_data": False,
    }
    return steps, receipt, dict(result["target_binding"])


def _learn_commit_guard(
    seeds: Sequence[int], all_steps: Mapping[int, Sequence[Mapping[str, Any]]],
    bindings: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    observations = []
    remove_type = REMOVE_ENTITY_SLOT["operator_type_id"]
    for seed in seeds:
        target_uuid = int(bindings[seed]["target_uuid"])
        matches = []
        for step in all_steps[seed]:
            if remove_type not in discoveryworld_transition_operator_ids(step):
                continue
            before = step.get("before_target_native_facts") or {}
            relations = [
                row for row in before.get("salient_relative_objects") or ()
                if isinstance(row, Mapping) and row.get("uuid") == target_uuid
            ]
            if len(relations) != 1:
                raise ValueError(f"commit target relation is not unique: seed {seed}")
            inventory = [
                row for row in before.get("inventory") or ()
                if isinstance(row, Mapping)
            ]
            matches.append({
                "target_relation_from_agent": str(
                    relations[0]["relation_from_agent"]
                ),
                "target_distance": int(relations[0]["distance"]),
                "minimum_inventory_cardinality": len(inventory),
            })
        if len(matches) != 1:
            raise ValueError(f"expected one successful remove transition: seed {seed}")
        observations.extend(matches)
    relation_values = {row["target_relation_from_agent"] for row in observations}
    distance_values = {row["target_distance"] for row in observations}
    if len(relation_values) != 1 or len(distance_values) != 1:
        raise ValueError("development commit relation is not stable")
    return {
        "operator_type_id": remove_type,
        "binding": "TARGET_NATIVE_BOUND_GOAL_ENTITY",
        "target_relation_from_agent": relation_values.pop(),
        "target_distance": distance_values.pop(),
        "minimum_inventory_cardinality": min(
            row["minimum_inventory_cardinality"] for row in observations
        ),
        "induction": "INTERSECTION_OVER_SUCCESSFUL_TARGET_DEVELOPMENT_TRANSITIONS",
        "formal_target_data_read": False,
    }


def _guard_support(
    guard: Mapping[str, Any], steps: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
) -> bool:
    remove_type = str(guard["operator_type_id"])
    target_uuid = int(binding["target_uuid"])
    for step in steps:
        if remove_type not in discoveryworld_transition_operator_ids(step):
            continue
        before = step.get("before_target_native_facts") or {}
        inventory = [row for row in before.get("inventory") or () if isinstance(row, Mapping)]
        relations = [
            row for row in before.get("salient_relative_objects") or ()
            if isinstance(row, Mapping) and row.get("uuid") == target_uuid
        ]
        return (
            len(relations) == 1
            and str(relations[0].get("relation_from_agent"))
            == str(guard["target_relation_from_agent"])
            and int(relations[0].get("distance")) == int(guard["target_distance"])
            and len(inventory) >= int(guard["minimum_inventory_cardinality"])
        )
    return False


def _balanced_training_rows(
    features: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    positives = labels.sum(axis=0)
    maximum = max(int(value) for value in positives if value > 0)
    repeats = []
    for row in labels:
        present = [int(positives[index]) for index, value in enumerate(row) if value]
        repeat = max(
            (int(math.ceil(maximum / value)) for value in present),
            default=1,
        )
        repeats.append(min(repeat, 16))
    indices = np.repeat(np.arange(len(features)), repeats)
    return features[indices], labels[indices], repeats


def _metrics(
    truth: Sequence[Sequence[int]], predictions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    actual = np.asarray(truth, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    true_positive = int(np.logical_and(actual == 1, predicted == 1).sum())
    false_positive = int(np.logical_and(actual == 0, predicted == 1).sum())
    false_negative = int(np.logical_and(actual == 1, predicted == 0).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_type = {}
    for index, type_id in enumerate(DISCOVERYWORLD_OPERATOR_IDS):
        tp = int(np.logical_and(actual[:, index] == 1, predicted[:, index] == 1).sum())
        fp = int(np.logical_and(actual[:, index] == 0, predicted[:, index] == 1).sum())
        fn = int(np.logical_and(actual[:, index] == 1, predicted[:, index] == 0).sum())
        per_type[type_id] = {
            "support": int(actual[:, index].sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }
    return {
        "transitions": len(actual),
        "exact_match_accuracy": float(np.all(actual == predicted, axis=1).mean()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "per_operator": per_type,
    }


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        next_row = [i]
        for j, b in enumerate(right, start=1):
            next_row.append(min(
                next_row[-1] + 1,
                row[j] + 1,
                row[j - 1] + int(a != b),
            ))
        row = next_row
    return row[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")

    source_programs = _source_programs()
    all_steps: dict[int, list[dict[str, Any]]] = {}
    bindings: dict[int, dict[str, Any]] = {}
    receipts = []
    for seed in (*TRAIN_SEEDS, *QUALIFICATION_SEEDS):
        all_steps[seed], receipt, bindings[seed] = _target_success_path(seed)
        receipts.append(receipt)
    train_steps = [step for seed in TRAIN_SEEDS for step in all_steps[seed]]
    qualification_steps = [
        step for seed in QUALIFICATION_SEEDS for step in all_steps[seed]
    ]
    x_train = np.asarray([target_action_features(row) for row in train_steps])
    y_train = np.asarray([target_action_labels(row) for row in train_steps])
    balanced_x, balanced_y, repeats = _balanced_training_rows(x_train, y_train)
    model = MLPClassifier(
        hidden_layer_sizes=(16,), activation="relu", solver="lbfgs",
        alpha=1e-4, max_iter=4000, random_state=314159,
    )
    model.fit(balanced_x, balanced_y)
    grounder = export_mlp_grounder(model, threshold=0.5)

    predicted = []
    for step in qualification_steps:
        grounded = set(grounded_operator_ids(grounder, step))
        predicted.append([
            int(type_id in grounded) for type_id in DISCOVERYWORLD_OPERATOR_IDS
        ])
    grounding_metrics = _metrics(
        [target_action_labels(row) for row in qualification_steps], predicted,
    )

    development_sequences = [
        discoveryworld_core_sequence(all_steps[seed]) for seed in TRAIN_SEEDS
    ]
    qualification_sequences = [
        discoveryworld_core_sequence(all_steps[seed])
        for seed in QUALIFICATION_SEEDS
    ]
    commit_guard = _learn_commit_guard(TRAIN_SEEDS, all_steps, bindings)
    target_program = induce_target_partial_order_program(
        development_sequences,
        development_receipts_sha256=stable_hash([
            row for row in receipts if row["seed"] in TRAIN_SEEDS
        ]),
        learned_target_guards=(commit_guard,),
    )
    target_support = sum(
        target_program_supports(target_program, row)
        for row in qualification_sequences
    )
    qualification_guard_support = sum(
        _guard_support(commit_guard, all_steps[seed], bindings[seed])
        for seed in QUALIFICATION_SEEDS
    )
    source_support = {}
    for task, program in source_programs.items():
        sequence = tuple(program["induced_sequence"])
        source_support[task] = {
            "development_support": sum(
                source_sequence_support(sequence, row)
                for row in development_sequences
            ),
            "qualification_support": sum(
                source_sequence_support(sequence, row)
                for row in qualification_sequences
            ),
            "sequence": list(sequence),
            "program_sha256": program["program_sha256"],
        }
    ordering = sorted(
        source_support,
        key=lambda task: (
            source_support[task]["development_support"],
            source_support[task]["qualification_support"],
            -len(source_support[task]["sequence"]),
            stable_hash(task),
        ),
        reverse=True,
    )
    selected = ordering[0]
    selected_sequence = source_support[selected]["sequence"]
    controls = [task for task in source_support if task != selected]
    permuted = max(
        controls,
        key=lambda task: (
            _edit_distance(selected_sequence, source_support[task]["sequence"]),
            stable_hash(task),
        ),
    )
    best_development = source_support[selected]["development_support"]
    unique_best = sum(
        row["development_support"] == best_development
        for row in source_support.values()
    ) == 1

    substantive = [
        row for type_id, row in grounding_metrics["per_operator"].items()
        if type_id != DISCOVERYWORLD_OPERATOR_IDS[0] and row["support"] > 0
    ]
    target_type_ids = {
        row["operator_type_id"] for row in target_program["operator_requirements"]
    }
    source_type_ids = set(selected_sequence)
    gates = {
        "qualification_exact_transition_accuracy": grounding_metrics[
            "exact_match_accuracy"
        ] >= 0.90,
        "qualification_micro_f1": grounding_metrics["micro_f1"] >= 0.90,
        "all_observed_substantive_operator_recalls": bool(substantive) and all(
            row["recall"] >= 0.80 for row in substantive
        ),
        "target_function_supports_all_heldout_development_paths": (
            target_support == len(qualification_sequences)
        ),
        "learned_commit_guard_supports_all_heldout_development_paths": (
            qualification_guard_support == len(QUALIFICATION_SEEDS)
        ),
        "source_motif_selected_uniquely_without_target_formal_data": unique_best,
        "selected_source_motif_supports_all_qualification_paths": (
            source_support[selected]["qualification_support"]
            == len(qualification_sequences)
        ),
        "permuted_source_motif_has_zero_qualification_support": (
            source_support[permuted]["qualification_support"] == 0
        ),
        "target_function_contains_target_specific_operator": bool(
            target_type_ids - source_type_ids
        ),
        "target_function_is_not_copied_source_body": target_program[
            "source_program_copied_as_target_body"
        ] is False,
        "no_formal_target_data": all(
            row["formal_target_data"] is False for row in receipts
        ),
    }
    body = {
        "schema_version": "discoveryworld-structural-grounder-development-v1",
        "status": (
            "DISCOVERYWORLD_STRUCTURAL_GROUNDER_QUALIFIED"
            if all(gates.values()) else
            "DISCOVERYWORLD_STRUCTURAL_GROUNDER_FAILED"
        ),
        "splits": {
            "training_seeds": list(TRAIN_SEEDS),
            "qualification_seeds": list(QUALIFICATION_SEEDS),
            "formal_seeds_read": [],
        },
        "development_receipts": receipts,
        "grounder": grounder,
        "training": {
            "raw_transitions": len(train_steps),
            "balanced_transitions": len(balanced_x),
            "positive_labels": {
                type_id: int(y_train[:, index].sum())
                for index, type_id in enumerate(DISCOVERYWORLD_OPERATOR_IDS)
            },
            "maximum_balance_repeat": max(repeats),
            "fixed_grounding_threshold": 0.5,
        },
        "qualification_grounding": grounding_metrics,
        "target_program": target_program,
        "development_target_sequences": [list(row) for row in development_sequences],
        "qualification_target_sequences": [list(row) for row in qualification_sequences],
        "qualification_target_program_support": target_support,
        "qualification_commit_guard_support": qualification_guard_support,
        "source_motif_support": source_support,
        "selected_source_program": selected,
        "source_permuted_control": permuted,
        "selected_source_program_sha256": source_programs[selected][
            "program_sha256"
        ],
        "source_permuted_program_sha256": source_programs[permuted][
            "program_sha256"
        ],
        "gates": gates,
        "claim_boundary": (
            "CONSUMED_DISCOVERYWORLD_DEVELOPMENT_ONLY;TARGET_NATIVE_NEURAL_"
            "DELTA_GROUNDER_AND_DOMAIN_FUNCTION;NO_NEW_FORMAL_TRANSFER_CLAIM"
        ),
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "qualification_grounding": grounding_metrics,
        "selected_source_program": selected,
        "source_permuted_control": permuted,
        "source_motif_support": source_support,
        "target_program": target_program,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
