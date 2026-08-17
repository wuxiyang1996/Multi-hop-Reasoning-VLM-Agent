#!/usr/bin/env python3
"""Train and qualify the ALFWorld multiplicity structural grounder on development only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_structural_induction import (  # noqa: E402
    ADD_ID,
    OPERATOR_IDS,
    REMOVE_ID,
    binding_labels,
    episode_structural_sequence,
    export_binary_mlp,
    induce_target_sequence_program,
    infer_demonstrated_bindings,
    observed_transition_operator_ids,
    repeated_source_support,
    target_candidate_features,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.alfworld_hierarchical_grounder import tokens  # noqa: E402
from motif_transfer.structural_delta_induction import validate_structural_program  # noqa: E402


DEFAULT_RECEIPTS = REPO / "runs/multisource_alfworld_neurosymbolic_v2/adaptation_expert_receipts.json"
DEFAULT_SOURCES = REPO / "configs/source_structural_v5c_frozen/programs"
DEFAULT_ARTIFACT = REPO / "artifacts/alfworld_structural_grounder_v1/artifact.json.gz"
DEFAULT_REPORT = REPO / "runs/alfworld_structural_grounder_v1_development/report.json"
FEATURE_BINS = 128
THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _examples(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    operator_x: list[np.ndarray] = []
    operator_y: dict[str, list[int]] = {key: [] for key in OPERATOR_IDS}
    candidate_x: list[np.ndarray] = []
    candidate_y: dict[str, list[int]] = {
        "ENTITY_BINDING": [], "DESTINATION_BINDING": [], "BEHAVIOR": [],
    }
    audits = []
    for episode in episodes:
        bindings = infer_demonstrated_bindings(episode)
        history: list[str] = []
        for transition in episode.get("transitions") or ():
            goal = str(transition.get("goal") or "")
            observation = str(transition.get("before_observation") or "")
            expert = str(transition.get("expert_action") or "")
            step = int(transition.get("step") or 0)
            selected_features = target_candidate_features(
                goal=goal, observation=observation, action=expert, step=step,
                action_history=history, feature_bins=FEATURE_BINS,
            )
            observed = set(observed_transition_operator_ids(transition))
            operator_x.append(selected_features)
            for type_id in OPERATOR_IDS:
                operator_y[type_id].append(int(type_id in observed))
            for raw_action in transition.get("native_actions") or ():
                action = str(raw_action)
                candidate_x.append(target_candidate_features(
                    goal=goal, observation=observation, action=action, step=step,
                    action_history=history, feature_bins=FEATURE_BINS,
                ))
                entity, destination = binding_labels(action, bindings)
                candidate_y["ENTITY_BINDING"].append(entity)
                candidate_y["DESTINATION_BINDING"].append(destination)
                candidate_y["BEHAVIOR"].append(int(action == expert))
            history.append(expert)
        audits.append({
            "task_id": episode["task_id"],
            "partition": episode["partition"],
            "induced_bindings": {key: list(value) for key, value in bindings.items()},
            "structural_sequence": list(episode_structural_sequence(episode)),
        })
    operator_matrix = np.asarray(operator_x, dtype=np.float64)
    candidate_matrix = np.asarray(candidate_x, dtype=np.float64)
    output = {
        key: (operator_matrix, np.asarray(values, dtype=np.int64))
        for key, values in operator_y.items()
    }
    output.update({
        key: (candidate_matrix, np.asarray(values, dtype=np.int64))
        for key, values in candidate_y.items()
    })
    return output, audits


def _induce_verb_operator_map(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    votes: dict[str, dict[str, int]] = {}
    for episode in episodes:
        for transition in episode.get("transitions") or ():
            values = tokens(str(transition.get("expert_action") or ""))
            if not values:
                continue
            for type_id in observed_transition_operator_ids(transition):
                votes.setdefault(values[0], {}).setdefault(type_id, 0)
                votes[values[0]][type_id] += 1
    return {
        verb: max(counts, key=lambda type_id: (counts[type_id], type_id))
        for verb, counts in votes.items()
    }


def _candidate_operator_examples(
    episodes: Sequence[Mapping[str, Any]], verb_map: Mapping[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    features = []
    labels: dict[str, list[int]] = {type_id: [] for type_id in OPERATOR_IDS}
    for episode in episodes:
        history: list[str] = []
        for transition in episode.get("transitions") or ():
            goal = str(transition.get("goal") or "")
            observation = str(transition.get("before_observation") or "")
            step = int(transition.get("step") or 0)
            for raw_action in transition.get("native_actions") or ():
                action = str(raw_action)
                features.append(target_candidate_features(
                    goal=goal, observation=observation, action=action, step=step,
                    action_history=history, feature_bins=FEATURE_BINS,
                ))
                values = tokens(action)
                induced = verb_map.get(values[0], "") if values else ""
                for type_id in OPERATOR_IDS:
                    labels[type_id].append(int(induced == type_id))
            history.append(str(transition.get("expert_action") or ""))
    matrix = np.asarray(features, dtype=np.float64)
    return {
        type_id: (matrix, np.asarray(values, dtype=np.int64))
        for type_id, values in labels.items()
    }


def _balanced(x: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("binary neural head requires both classes")
    rng = np.random.default_rng(seed)
    target = max(len(positive), min(len(negative), len(positive) * 4))
    negative_selected = rng.choice(negative, size=target, replace=len(negative) < target)
    positive_selected = rng.choice(positive, size=target, replace=len(positive) < target)
    indices = np.concatenate((positive_selected, negative_selected))
    rng.shuffle(indices)
    return x[indices], y[indices]


def _fit(x: np.ndarray, y: np.ndarray, *, seed: int, label: str) -> dict[str, Any]:
    bx, by = _balanced(x, y, seed=seed)
    model = MLPClassifier(
        hidden_layer_sizes=(24,), activation="relu", solver="adam",
        alpha=1e-4, batch_size=min(64, len(by)), learning_rate_init=0.003,
        max_iter=800, random_state=seed, early_stopping=False,
    )
    model.fit(bx, by)
    exported = export_binary_mlp(model, label=label)
    body = dict(exported)
    body.pop("head_sha256")
    body.update({
        "raw_training_examples": int(len(y)),
        "balanced_training_examples": int(len(by)),
        "raw_positive_examples": int(y.sum()),
    })
    return body | {"head_sha256": stable_hash(body)}


def _probability(head: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    hidden = np.asarray(x, dtype=np.float64)
    for index, (weights, bias) in enumerate(zip(head["coefs"], head["intercepts"])):
        hidden = hidden @ np.asarray(weights, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        if index + 1 == len(head["coefs"]):
            hidden = 1.0 / (1.0 + np.exp(-np.clip(hidden, -50.0, 50.0)))
        else:
            hidden = np.maximum(hidden, 0.0)
    return hidden.reshape(-1)


def _metrics(head: Mapping[str, Any], x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predicted = (_probability(head, x) >= THRESHOLD).astype(np.int64)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "examples": int(len(y)), "positive_examples": int(y.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / max(1, len(y)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def _source_rows(source_dir: Path, target_sequence: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for path in sorted(source_dir.glob("*.json")):
        program = _read(path)
        validate_structural_program(program)
        output.append({
            "source_name": path.stem,
            "path": str(path.relative_to(REPO)),
            "file_sha256": _sha256(path),
            "program_sha256": program["program_sha256"],
            "source_sequence": list(program["induced_sequence"]),
            "applicability": repeated_source_support(
                program["induced_sequence"], target_sequence,
            ),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--source-programs", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    receipts_path = args.receipts.resolve()
    receipts = _read(receipts_path)
    all_success = [row for row in receipts["episodes"] if bool(row.get("official_success"))]
    multiplicity = [
        row for row in receipts["episodes"]
        if row.get("task_family") == "pick_two_obj_and_place"
        and bool(row.get("official_success"))
    ]
    train = [row for row in multiplicity if row.get("partition") == "adaptation_train"]
    qualification = [
        row for row in multiplicity if row.get("partition") == "adaptation_validation"
    ]
    if len(train) < 4 or len(qualification) < 2:
        raise SystemExit("insufficient successful multiplicity development episodes")
    train_examples, train_audit = _examples(train)
    qualification_examples, qualification_audit = _examples(qualification)
    # Operator semantics are shared within the target domain, so all consumed
    # successful development trajectories provide transition-delta labels.
    # Entity/destination bindings and the target function remain strictly on
    # the multiplicity family selected for this replication.
    operator_train = [
        row for row in all_success if row.get("partition") == "adaptation_train"
    ]
    operator_qualification = [
        row for row in all_success if row.get("partition") == "adaptation_validation"
    ]
    executed_operator_train_examples, _ = _examples(operator_train)
    executed_operator_qualification_examples, _ = _examples(operator_qualification)
    verb_operator_map = _induce_verb_operator_map(operator_train)
    operator_train_examples = _candidate_operator_examples(
        operator_train, verb_operator_map,
    )
    operator_qualification_examples = _candidate_operator_examples(
        operator_qualification, verb_operator_map,
    )
    for type_id in OPERATOR_IDS:
        train_examples[type_id] = operator_train_examples[type_id]
        qualification_examples[type_id] = operator_qualification_examples[type_id]
    heads = {
        name: _fit(x, y, seed=6100 + index, label=name)
        for index, (name, (x, y)) in enumerate(train_examples.items())
    }
    metrics = {
        name: _metrics(heads[name], *qualification_examples[name])
        for name in heads
    }
    executed_operator_metrics = {
        type_id: _metrics(
            heads[type_id], *executed_operator_qualification_examples[type_id],
        )
        for type_id in OPERATOR_IDS
    }

    # A shuffled transition-effect control leaves the binding/policy labels
    # intact and destroys only the causal operator correspondence.
    rng = np.random.default_rng(91601)
    shuffled_metrics = {}
    for index, type_id in enumerate(OPERATOR_IDS):
        x, y = train_examples[type_id]
        shuffled_y = y[rng.permutation(len(y))]
        shuffled_head = _fit(x, shuffled_y, seed=7100 + index, label=type_id)
        shuffled_metrics[type_id] = _metrics(
            shuffled_head, *qualification_examples[type_id],
        )

    train_paths = [episode_structural_sequence(row) for row in train]
    target_program = induce_target_sequence_program(
        train_paths, development_receipts_sha256=_sha256(receipts_path),
    )
    target_sequence = tuple(target_program["induced_sequence"])
    qualification_support = sum(
        episode_structural_sequence(row) == target_sequence for row in qualification
    ) / len(qualification)
    sources = _source_rows(args.source_programs.resolve(), target_sequence)
    applicable = [row for row in sources if row["applicability"]["applicable"]]

    authentic_operator_macro_f1 = float(np.mean([
        metrics[type_id]["f1"] for type_id in (ADD_ID, REMOVE_ID)
    ]))
    shuffled_operator_macro_f1 = float(np.mean([
        shuffled_metrics[type_id]["f1"] for type_id in (ADD_ID, REMOVE_ID)
    ]))
    gates = {
        "development_only_partitions": all(
            row["partition"] in {"adaptation_train", "adaptation_validation"}
            for row in (*train_audit, *qualification_audit)
        ),
        "fixed_grounding_threshold": THRESHOLD == 0.5,
        "add_remove_grounding_f1": all(
            metrics[type_id]["f1"] >= 0.90 for type_id in (ADD_ID, REMOVE_ID)
        ),
        "executed_add_remove_grounding_f1": all(
            executed_operator_metrics[type_id]["f1"] >= 0.90
            for type_id in (ADD_ID, REMOVE_ID)
        ),
        "entity_binding_f1": metrics["ENTITY_BINDING"]["f1"] >= 0.80,
        "destination_binding_f1": metrics["DESTINATION_BINDING"]["f1"] >= 0.80,
        "shuffled_effect_gap": (
            authentic_operator_macro_f1 - shuffled_operator_macro_f1 >= 0.35
        ),
        "heldout_target_sequence_support": qualification_support == 1.0,
        "unique_source_structural_match": len(applicable) == 1,
        "put_near_selected_without_source_identity_feature": (
            len(applicable) == 1 and applicable[0]["source_name"] == "put_near"
        ),
    }
    qualified = all(gates.values())
    artifact_body = {
        "schema_version": "alfworld-target-native-structural-grounder-v1",
        "status": "QUALIFIED" if qualified else "ABSTAINING",
        "feature_bins": FEATURE_BINS,
        "scalar_feature_count": 14,
        "threshold": THRESHOLD,
        "heads": heads,
        "target_program": target_program,
        "source_applicability": sources,
        "selected_source_program_sha256": (
            applicable[0]["program_sha256"] if len(applicable) == 1 else None
        ),
        "selected_source_name_evaluator_label_only": (
            applicable[0]["source_name"] if len(applicable) == 1 else None
        ),
        "training_authority": "CONSUMED_ALFWORLD_DEVELOPMENT_EXPERT_TRANSITIONS_ONLY",
        "training_task_ids": [row["task_id"] for row in train],
        "qualification_task_ids": [row["task_id"] for row in qualification],
        "operator_training_task_ids": [row["task_id"] for row in operator_train],
        "operator_qualification_task_ids": [
            row["task_id"] for row in operator_qualification
        ],
        "development_receipts_path": str(receipts_path.relative_to(REPO)),
        "development_receipts_sha256": _sha256(receipts_path),
        "outcome_fields_used_at_inference": False,
        "entity_and_receptacle_identity_tokens_masked": True,
        "operator_verb_mapping_induced_from_observed_target_deltas": verb_operator_map,
        "formal_target_outcome_read": False,
        "qualification_thresholds_frozen_before_fresh_execution": True,
    }
    artifact = artifact_body | {"grounder_sha256": stable_hash(artifact_body)}
    report_body = {
        "schema_version": "alfworld-structural-grounder-development-report-v1",
        "status": "ALFWORLD_STRUCTURAL_GROUNDER_QUALIFIED" if qualified else "ALFWORLD_STRUCTURAL_GROUNDER_FAILED",
        "artifact_path": str(args.artifact.resolve().relative_to(REPO)),
        "grounder_sha256": artifact["grounder_sha256"],
        "development_receipts_sha256": _sha256(receipts_path),
        "train_episode_audit": train_audit,
        "qualification_episode_audit": qualification_audit,
        "qualification_metrics": metrics,
        "executed_transition_qualification_metrics": executed_operator_metrics,
        "shuffled_effect_metrics": shuffled_metrics,
        "authentic_add_remove_macro_f1": authentic_operator_macro_f1,
        "shuffled_add_remove_macro_f1": shuffled_operator_macro_f1,
        "target_program": target_program,
        "heldout_target_sequence_support": qualification_support,
        "source_applicability": sources,
        "gates": gates,
        "formal_target_outcome_read": False,
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.artifact, "wt", encoding="utf-8") as handle:
        json.dump(artifact, handle, sort_keys=True, separators=(",", ":"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "gates": gates,
        "metrics": metrics,
        "executed_operator_metrics": executed_operator_metrics,
        "authentic_add_remove_macro_f1": authentic_operator_macro_f1,
        "shuffled_add_remove_macro_f1": shuffled_operator_macro_f1,
        "target_sequence": list(target_sequence),
        "applicable_sources": [row["source_name"] for row in applicable],
        "artifact": str(args.artifact), "report": str(args.report),
    }, indent=2, sort_keys=True))
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
