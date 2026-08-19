#!/usr/bin/env python3
"""Qualify Normal source grounding and measure source information value."""

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
from motif_transfer.discoveryworld_normal_transfer import (  # noqa: E402
    export_neural_grounder,
    induce_target_only_program,
    predict_grounding,
    source_role_operator_ids,
    target_grounding_features,
    target_grounding_label,
    trace_conforms,
    typed_trace,
)
from motif_transfer.source_goal_acquisition_induction import (  # noqa: E402
    validate_goal_acquisition_program,
)


PROTOCOL = REPO / "configs/discoveryworld_normal_source_value_v26_protocol.json"
SOURCE = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"
SOURCE_CONFIRMATION = REPO / "runs/sokoban_goal_acquisition_v1/fresh_confirmation_report.json"
DEVELOPMENT = REPO / "runs/discoveryworld_normal_source_value_v26_development"
QUALIFICATION = REPO / "runs/discoveryworld_normal_source_value_v26_qualification"


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


def _episodes(root: Path, expected_seeds: Sequence[int]) -> list[dict[str, Any]]:
    output = []
    for seed in expected_seeds:
        path = root / f"proteomics.normal.seed{seed}.json"
        episode = _read(path)
        _self_hash(episode, "episode_sha256")
        if episode["task"] != {"scenario": "Proteomics", "difficulty": "Normal", "seed": seed}:
            raise ValueError(f"task identity mismatch: {path}")
        if episode.get("policy_runtime_saw_oracle_scorecard") is not False:
            raise ValueError(f"oracle scorecard exposure: {path}")
        output.append(episode)
    return output


def _balanced_indices(labels: Sequence[str]) -> list[int]:
    counts = Counter(labels)
    maximum = max(counts.values())
    output = []
    for index, label in enumerate(labels):
        repeats = min(32, int(math.ceil(maximum / counts[label])))
        output.extend([index] * repeats)
    return output


def _classification_metrics(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    labels = sorted(set(truth) | set(predicted))
    per_class = {}
    for label in labels:
        tp = sum(left == label and right == label for left, right in zip(truth, predicted))
        fp = sum(left != label and right == label for left, right in zip(truth, predicted))
        fn = sum(left == label and right != label for left, right in zip(truth, predicted))
        per_class[label] = {
            "support": sum(value == label for value in truth),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }
    return {
        "transitions": len(truth),
        "exact_accuracy": sum(left == right for left, right in zip(truth, predicted)) / len(truth),
        "per_class": per_class,
    }


def _program_matches_source(target: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    program = target.get("program")
    if not isinstance(program, Mapping):
        return False
    roles = source_role_operator_ids(source)
    return (
        set(map(str, program["acquisition_operator_type_ids"]))
        == {str(roles["ACQUISITION_ENTITY"]), str(roles["ACQUISITION_CONTROL"])}
        and str(program["binding_operator_type_id"]) == str(roles["BINDING"])
        and str(program["relation_operator_type_id"]) == str(roles["RELATION"])
        and program["binding_to_relation"] is True
    )


def _target_program_supports(target: Mapping[str, Any], sequence: Sequence[str]) -> bool:
    program = target.get("program")
    if not isinstance(program, Mapping) or len(sequence) < 3:
        return False
    binding = str(program["binding_operator_type_id"])
    relation = str(program["relation_operator_type_id"])
    acquisition = set(map(str, program["acquisition_operator_type_ids"]))
    values = tuple(map(str, sequence))
    return (
        values[-2:] == (binding, relation)
        and values.count(binding) == 1
        and values.count(relation) == 1
        and all(value in acquisition for value in values[:-2])
    )


def build_report() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read(PROTOCOL)
    if protocol["status"] != "FROZEN_BEFORE_ANY_V26_TARGET_RESET_OR_OUTCOME":
        raise ValueError("V26 protocol was not prospectively frozen")
    source = _read(SOURCE)
    validate_goal_acquisition_program(source)
    confirmation = _read(SOURCE_CONFIRMATION)
    _self_hash(confirmation, "report_sha256")
    if confirmation.get("source_gate_passed") is not True:
        raise ValueError("source acquisition program lacks fresh held-out confirmation")
    roles = protocol["task_roles"]
    development_seeds = list(map(int, roles["development_seeds"]))
    qualification_seeds = list(map(int, roles["qualification_seeds"]))
    formal_seeds = list(map(int, roles["sealed_formal_reserve_seeds"]))
    if set(development_seeds) & set(qualification_seeds) or (
        set(development_seeds) | set(qualification_seeds)
    ) & set(formal_seeds):
        raise ValueError("V26 seed roles overlap")
    development = _episodes(DEVELOPMENT, development_seeds)
    qualification = _episodes(QUALIFICATION, qualification_seeds)
    development_steps = [step for episode in development for step in episode["steps"]]
    qualification_steps = [step for episode in qualification for step in episode["steps"]]

    train_labels = [target_grounding_label(step) for step in development_steps]
    train_features = np.asarray([
        target_grounding_features(step) for step in development_steps
    ], dtype=float)
    indices = _balanced_indices(train_labels)
    model = MLPClassifier(
        hidden_layer_sizes=(16,), activation="relu", solver="lbfgs",
        alpha=1e-5, max_iter=5000, random_state=2601,
    )
    model.fit(train_features[indices], np.asarray(train_labels)[indices])
    grounder = export_neural_grounder(model)
    truth = [target_grounding_label(step) for step in qualification_steps]
    predictions = [predict_grounding(grounder, step)[0] for step in qualification_steps]
    grounding = _classification_metrics(truth, predictions)

    authentic_sequences = [typed_trace(episode["steps"], source) for episode in qualification]
    neural_sequences = []
    role_ids = source_role_operator_ids(source)
    for episode in qualification:
        values = []
        for step in episode["steps"]:
            label, _ = predict_grounding(grounder, step)
            operator_id = role_ids[label]
            if operator_id is not None:
                values.append(str(operator_id))
        neural_sequences.append(tuple(values))
    authentic_support = sum(trace_conforms(row, source) for row in authentic_sequences)
    neural_support = sum(trace_conforms(row, source) for row in neural_sequences)
    shuffled_sequences = [row[1:] + row[:1] for row in authentic_sequences]
    shuffled_support = sum(trace_conforms(row, source) for row in shuffled_sequences)
    permuted_sequences = [row[:-2] + (row[-1], row[-2]) for row in authentic_sequences]
    permuted_support = sum(trace_conforms(row, source) for row in permuted_sequences)

    development_sequences = [typed_trace(episode["steps"], source) for episode in development]
    target_curve = []
    first_matching_budget = None
    for budget in map(int, protocol["target_only_curve"]["complete_ordered_target_trajectory_budgets"]):
        target = induce_target_only_program(development_sequences, budget=budget)
        matches = _program_matches_source(target, source)
        if matches and first_matching_budget is None:
            first_matching_budget = budget
        target_curve.append({
            "complete_ordered_target_trajectory_budget": budget,
            "status": target["status"],
            "matches_source_phase_program": matches,
            "qualification_support": sum(
                _target_program_supports(target, row) for row in authentic_sequences
            ),
            "program_sha256": target["program_sha256"],
        })
    complete_target_trajectories_replaced = int(first_matching_budget or 0)
    relation_metrics = grounding["per_class"]["RELATION"]
    binding_metrics = grounding["per_class"]["BINDING"]
    official_successes = sum(
        bool(episode["evaluation"]["official_success"]) for episode in qualification
    )
    thresholds = protocol["qualification_gates"]
    gates = {
        "development_trajectory_count": len(development) >= int(thresholds["development_trajectory_count_at_least"]),
        "qualification_trajectory_count": len(qualification) >= int(thresholds["qualification_trajectory_count_at_least"]),
        "qualification_official_success_rate": official_successes / len(qualification) >= float(thresholds["qualification_official_success_rate_at_least"]),
        "neural_grounder_exact_accuracy": grounding["exact_accuracy"] >= float(thresholds["neural_grounder_exact_accuracy_at_least"]),
        "binding_precision_recall": binding_metrics["precision"] == binding_metrics["recall"] == float(thresholds["binding_and_relation_precision_recall_equal"]),
        "relation_precision_recall": relation_metrics["precision"] == relation_metrics["recall"] == float(thresholds["binding_and_relation_precision_recall_equal"]),
        "authentic_trace_conformance": authentic_support / len(qualification) >= float(thresholds["authentic_trace_conformance_rate_at_least"]),
        "neural_trace_conformance": neural_support / len(qualification) >= float(thresholds["source_zero_target_demo_support_rate_at_least"]),
        "shuffled_trace_rejected": shuffled_support / len(qualification) <= float(thresholds["shuffled_trace_conformance_rate_at_most"]),
        "source_permuted_trace_rejected": permuted_support == 0,
        "target_only_zero_demo_abstains": target_curve[0]["status"] == "ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY",
        "source_replaces_complete_target_trajectory": complete_target_trajectories_replaced >= int(thresholds["minimum_complete_target_trajectories_replaced"]),
        "source_heldout_confirmation": confirmation["source_gate_passed"] is True,
        "formal_reserve_still_sealed": not any(
            (REPO / f"runs/discoveryworld_normal_source_value_v26_formal/proteomics.normal.seed{seed}.json").exists()
            for seed in formal_seeds
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "discoveryworld-normal-source-value-v26-qualification-report",
        "status": "DISCOVERYWORLD_NORMAL_V26_QUALIFIED" if passed else "DISCOVERYWORLD_NORMAL_V26_FAILED",
        "formal_reserve_authorized": passed,
        "claim_boundary": (
            "Fresh qualification of program identification and target-native neural grounding. "
            "The source condition uses zero complete ordered target trajectories and replaces the "
            "minimum target-only demo needed to identify the same phase program. The source-blind "
            "survey controller is shared and already succeeds, so this does not claim an incremental "
            "success-rate gain over that controller or magical provenance for extensionally identical programs."
        ),
        "estimand": protocol["estimand"],
        "metrics": {
            "development_trajectories": len(development),
            "development_transitions": len(development_steps),
            "qualification_trajectories": len(qualification),
            "qualification_transitions": len(qualification_steps),
            "qualification_official_successes": official_successes,
            "qualification_official_success_rate": official_successes / len(qualification),
            "grounding": grounding,
            "authentic_trace_support": authentic_support,
            "neural_trace_support": neural_support,
            "shuffled_trace_support": shuffled_support,
            "source_permuted_trace_support": permuted_support,
            "complete_target_trajectories_replaced": complete_target_trajectories_replaced,
        },
        "target_only_induction_curve": target_curve,
        "identifiability_statement": {
            "source_provenance_identifiable_from_extensional_behavior_alone": False,
            "measured_source_information_value": "ONE_COMPLETE_ORDERED_SUCCESSFUL_TARGET_TRAJECTORY",
            "isomorphic_target_written_program": "ORACLE_CEILING",
        },
        "gates": gates,
        "all_qualification_gates_passed": passed,
        "lineage": {
            "source_artifact_sha256": source["artifact_sha256"],
            "source_confirmation_sha256": confirmation["report_sha256"],
            "grounder_sha256": grounder["grounder_sha256"],
            "protocol_file_sha256": _sha(PROTOCOL),
            "development_summary_sha256": _read(DEVELOPMENT / "acquisition_summary.json")["summary_sha256"],
            "qualification_summary_sha256": _read(QUALIFICATION / "acquisition_summary.json")["summary_sha256"],
            "formal_seeds_read": [],
        },
    }
    return body | {"report_sha256": stable_hash(body)}, grounder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "qualification_report.json"
    grounder_path = args.output_dir / "neural_grounder.json"
    if report_path.exists() or grounder_path.exists():
        raise SystemExit("refusing to overwrite V26 qualification artifacts")
    report, grounder = build_report()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    grounder_path.write_text(json.dumps(grounder, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "metrics": report["metrics"],
        "failed_gates": sorted(key for key, value in report["gates"].items() if not value),
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if report["all_qualification_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
