#!/usr/bin/env python3
"""Run a frozen CPU neural-probe transfer feasibility experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import TransitionReceipt, stable_hash  # noqa: E402
from motif_transfer.instrumented_import import import_native_source_batch  # noqa: E402
from motif_transfer.neurosymbolic_probe_experiment import (  # noqa: E402
    FEATURE_NAMES,
    LABEL_NAMES,
    OperationalProbeExample,
    OperationalTransition,
    build_operational_probe_examples,
    split_source_examples,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_examples(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transitions = []
    input_files = [{
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
    }]
    for item in manifest["inputs"]:
        evidence_dir = Path(item["evidence_dir"])
        episodes = import_native_source_batch(evidence_dir)
        for name in ("manifest.json", "events.jsonl", "episodes.jsonl"):
            path = evidence_dir / name
            input_files.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        for episode in episodes:
            for record in episode.records:
                transitions.append(OperationalTransition(
                    episode.episode_id,
                    record.step,
                    tuple(record.before.native_actions),
                    record.action,
                    tuple(record.after.native_actions),
                    record.reward,
                    record.after.terminal,
                ))
    examples = build_operational_probe_examples(transitions)
    return split_source_examples(examples), input_files


def _target_examples(input_dir: Path, offsets: Sequence[int]):
    by_offset = {}
    input_files = []
    for offset in offsets:
        path = input_dir / f"task_{offset}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("collection_split") != "adaptation":
            raise ValueError(f"target task {offset} is not adaptation data")
        if raw.get("condition") != "BASE_DECISION_TARGET_ONLY":
            raise ValueError(f"target task {offset} is not target-only")
        if raw.get("harness_used") or raw.get("source_motif_used"):
            raise ValueError(f"target task {offset} contains source assistance")
        transitions = []
        for step, record in enumerate(raw.get("records") or ()):
            receipt = TransitionReceipt(**record["transition"])
            if not receipt.validate():
                raise ValueError(f"target task {offset} has an invalid receipt")
            before_actions = tuple(map(str, record["before"]["native_actions"]))
            after_actions = tuple(map(str, record["after"]["native_actions"]))
            action = str(receipt.action)
            if action not in before_actions:
                raise ValueError(f"target task {offset} executed a non-native action")
            if receipt.done != bool(record["after"]["terminal"]):
                raise ValueError(f"target task {offset} terminal receipt mismatch")
            if receipt.official_success != bool(record["after"]["official_success"]):
                raise ValueError(f"target task {offset} success receipt mismatch")
            transitions.append(OperationalTransition(
                str(raw["task_id"]), step, before_actions, action, after_actions,
                float(record["reward"]), bool(record["after"]["terminal"]),
            ))
        expected_steps = int((raw.get("metrics") or {}).get("steps", -1))
        if not transitions or len(transitions) != expected_steps:
            raise ValueError(f"target task {offset} has incomplete transitions")
        by_offset[offset] = build_operational_probe_examples(transitions)
        input_files.append({
            "task_offset": offset,
            "task_id": raw["task_id"],
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "examples": len(transitions),
            "official_success": bool((raw.get("metrics") or {}).get("official_success")),
        })
    return by_offset, input_files


def _arrays(examples: Sequence[OperationalProbeExample]):
    return (
        np.asarray([row.features for row in examples], dtype=np.float64),
        np.asarray([row.labels for row in examples], dtype=np.int64),
    )


def _fit_predict(
    train: Sequence[OperationalProbeExample],
    evaluation: Sequence[OperationalProbeExample],
    *,
    seed: int,
    model_config: dict[str, Any],
    train_labels: np.ndarray | None = None,
) -> np.ndarray:
    x_train, y_train = _arrays(train)
    x_eval, _ = _arrays(evaluation)
    if train_labels is not None:
        y_train = train_labels
    hidden = tuple(int(value) for value in model_config["hidden_layer_sizes"])
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=str(model_config["activation"]),
            solver=str(model_config["solver"]),
            alpha=float(model_config["alpha"]),
            max_iter=int(model_config["max_iter"]),
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_eval), dtype=np.float64)
    if probabilities.shape != (len(evaluation), len(LABEL_NAMES)):
        raise ValueError(f"unexpected neural probe output shape: {probabilities.shape}")
    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def _shuffle_labels(
    examples: Sequence[OperationalProbeExample], seed: int,
) -> np.ndarray:
    labels = _arrays(examples)[1]
    shuffled = labels.copy()
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        by_episode[row.episode_id].append(index)
    for episode_id, indices in sorted(by_episode.items()):
        local_seed = int(stable_hash({
            "seed": seed, "episode_id": episode_id,
            "control": "WITHIN_EPISODE_JOINT_LABEL_SHUFFLE_V1",
        })[:16], 16)
        permutation = np.random.default_rng(local_seed).permutation(indices)
        shuffled[indices] = labels[permutation]
    return shuffled


def _balanced_training_rows(
    source: Sequence[OperationalProbeExample],
    target: Sequence[OperationalProbeExample],
) -> tuple[OperationalProbeExample, ...]:
    if not target:
        return tuple(source)
    repeats = max(1, math.ceil(len(source) / len(target)))
    return tuple(source) + tuple(target) * repeats


def _metrics(
    probabilities: np.ndarray,
    evaluation: Sequence[OperationalProbeExample],
    *,
    refuted_max: float,
    supported_min: float,
) -> dict[str, Any]:
    _, labels = _arrays(evaluation)
    squared = (probabilities - labels) ** 2
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    log_loss = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    per_label = {}
    balanced_scores = []
    for index, name in enumerate(LABEL_NAMES):
        class_scores = []
        for value in (0, 1):
            mask = labels[:, index] == value
            if np.any(mask):
                class_scores.append(float(np.mean(squared[mask, index])))
        balanced = float(np.mean(class_scores))
        balanced_scores.append(balanced)
        per_label[name] = {
            "brier": float(np.mean(squared[:, index])),
            "balanced_brier": balanced,
            "log_loss": float(np.mean(log_loss[:, index])),
            "positive_rate": float(np.mean(labels[:, index])),
        }
    selected = (probabilities <= refuted_max) | (probabilities >= supported_min)
    predicted = probabilities >= supported_min
    selective_total = int(np.sum(selected))
    episode_brier = {}
    for episode_id in sorted({row.episode_id for row in evaluation}):
        mask = np.asarray([row.episode_id == episode_id for row in evaluation])
        episode_brier[episode_id] = float(np.mean(squared[mask]))
    return {
        "examples": len(evaluation),
        "macro_brier": float(np.mean(squared)),
        "macro_balanced_brier": float(np.mean(balanced_scores)),
        "macro_log_loss": float(np.mean(log_loss)),
        "exact_vector_accuracy": float(np.mean(np.all(
            (probabilities >= 0.5) == labels, axis=1,
        ))),
        "selective_coverage": selective_total / labels.size,
        "selective_accuracy": (
            float(np.mean(predicted[selected] == labels[selected]))
            if selective_total else None
        ),
        "per_label": per_label,
        "per_episode_macro_brier": episode_brier,
    }


def _constant_probabilities(
    source: Sequence[OperationalProbeExample],
    target: Sequence[OperationalProbeExample],
    evaluation_count: int,
) -> np.ndarray:
    _, source_labels = _arrays(source)
    source_mean = np.mean(source_labels, axis=0)
    if target:
        _, target_labels = _arrays(target)
        probabilities = (source_mean + np.mean(target_labels, axis=0)) / 2
    else:
        probabilities = source_mean
    return np.tile(np.clip(probabilities, 1e-6, 1 - 1e-6), (evaluation_count, 1))


def _summarize(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["adaptation_k"], row["condition"])].append(row["metrics"])
    summary = []
    metric_names = (
        "macro_brier", "macro_balanced_brier", "macro_log_loss",
        "exact_vector_accuracy", "selective_coverage", "selective_accuracy",
    )
    for (adaptation_k, condition), metrics_rows in sorted(grouped.items()):
        metrics = {}
        for name in metric_names:
            values = [row[name] for row in metrics_rows if row[name] is not None]
            metrics[name] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
            }
        summary.append({
            "adaptation_k": adaptation_k,
            "condition": condition,
            "seeds": len(metrics_rows),
            "metrics": metrics,
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen game-to-ALFWorld operational probe feasibility"
    )
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs" / "neurosymbolic_transfer_probe_v0.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_FIRST_RUN":
        raise ValueError("experiment config is not frozen for the first run")
    if tuple(config["before_state_features"]) != FEATURE_NAMES:
        raise ValueError("config feature contract differs from implementation")
    if tuple(config["operational_labels"]) != LABEL_NAMES:
        raise ValueError("config label contract differs from implementation")

    source_manifest = REPO / config["source"]["manifest"]
    source_splits, source_files = _source_examples(source_manifest)
    support_offsets = tuple(config["target"]["support_task_offsets"])
    eval_offsets = tuple(config["target"]["internal_evaluation_task_offsets"])
    target_by_offset, target_files = _target_examples(
        REPO / config["target"]["input_dir"], support_offsets + eval_offsets,
    )
    evaluation = tuple(
        row for offset in eval_offsets for row in target_by_offset[offset]
    )
    source_train = source_splits[config["source"]["training_split"]]
    source_diagnostic = source_splits[config["source"]["diagnostic_split"]]
    thresholds = config["selective_thresholds"]
    model_config = config["model"]
    seeds = tuple(int(seed) for seed in model_config["seeds"])

    runs = []
    source_diagnostics = []
    source_predictions = {}
    shuffled_source_predictions = {}
    shuffled_labels_by_seed = {}
    for seed in seeds:
        shuffled_labels = _shuffle_labels(source_train, seed)
        shuffled_labels_by_seed[seed] = shuffled_labels
        source_predictions[seed] = _fit_predict(
            source_train, evaluation, seed=seed, model_config=model_config,
        )
        shuffled_source_predictions[seed] = _fit_predict(
            source_train, evaluation, seed=seed, model_config=model_config,
            train_labels=shuffled_labels,
        )
        source_diagnostics.append({
            "seed": seed,
            "authentic": _metrics(
                _fit_predict(
                    source_train, source_diagnostic,
                    seed=seed, model_config=model_config,
                ),
                source_diagnostic, **thresholds,
            ),
            "shuffled": _metrics(
                _fit_predict(
                    source_train, source_diagnostic,
                    seed=seed, model_config=model_config,
                    train_labels=shuffled_labels,
                ),
                source_diagnostic, **thresholds,
            ),
        })

    for adaptation_k in config["target"]["adaptation_k"]:
        target_train = tuple(
            row
            for offset in support_offsets[:adaptation_k]
            for row in target_by_offset[offset]
        )
        for seed in seeds:
            if target_train:
                target_only = _fit_predict(
                    target_train, evaluation, seed=seed,
                    model_config=model_config,
                )
            else:
                target_only = np.full(
                    (len(evaluation), len(LABEL_NAMES)), 0.5,
                    dtype=np.float64,
                )
            combined = _balanced_training_rows(source_train, target_train)
            authentic_hybrid = _fit_predict(
                combined, evaluation, seed=seed, model_config=model_config,
            )
            if target_train:
                source_repeats = len(combined) - len(source_train)
                _, target_labels = _arrays(target_train)
                hybrid_shuffled_labels = np.concatenate((
                    shuffled_labels_by_seed[seed],
                    np.tile(target_labels, (
                        math.ceil(source_repeats / len(target_train)), 1,
                    ))[:source_repeats],
                ))
            else:
                hybrid_shuffled_labels = shuffled_labels_by_seed[seed]
            shuffled_hybrid = _fit_predict(
                combined, evaluation, seed=seed, model_config=model_config,
                train_labels=hybrid_shuffled_labels,
            )
            condition_probabilities = {
                "target_only": target_only,
                "source_neural_zero_shot": source_predictions[seed],
                "authentic_source_plus_target": authentic_hybrid,
                "shuffled_source_plus_target": shuffled_hybrid,
                "source_marginal_plus_target": _constant_probabilities(
                    source_train, target_train, len(evaluation),
                ),
            }
            for condition, probabilities in condition_probabilities.items():
                runs.append({
                    "adaptation_k": adaptation_k,
                    "seed": seed,
                    "condition": condition,
                    "target_training_examples": len(target_train),
                    "metrics": _metrics(
                        probabilities, evaluation, **thresholds,
                    ),
                })

    summary = _summarize(runs)
    lookup = {
        (row["adaptation_k"], row["condition"]):
            row["metrics"][config["primary_metric"]]["mean"]
        for row in summary
    }
    comparisons = []
    gate_passed = True
    authentic_name = "authentic_source_plus_target"
    for adaptation_k in config["positive_gate"]["required_k"]:
        authentic = lookup[(adaptation_k, authentic_name)]
        for control in config["positive_gate"]["authentic_must_beat"]:
            control_value = lookup[(adaptation_k, control)]
            passed = authentic < control_value
            gate_passed &= passed
            comparisons.append({
                "adaptation_k": adaptation_k,
                "authentic": authentic,
                "control": control,
                "control_value": control_value,
                "passed": passed,
            })
    source_diag_summary = {
        condition: {
            "macro_brier_mean": float(np.mean([
                row[condition]["macro_brier"] for row in source_diagnostics
            ])),
            "macro_brier_std": float(np.std([
                row[condition]["macro_brier"] for row in source_diagnostics
            ])),
        }
        for condition in ("authentic", "shuffled")
    }
    payload = {
        "schema_version": 1,
        "experiment": "NEUROSYMBOLIC_OPERATIONAL_PROBE_TRANSFER_V0",
        "config": config,
        "config_sha256": _sha256(args.config),
        "implementation_sha256": _sha256(Path(__file__)),
        "libraries": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source": {
            "split_counts": {
                name: len(rows) for name, rows in source_splits.items()
            },
            "input_files": source_files,
            "held_out_diagnostic": source_diag_summary,
        },
        "target": {
            "support_offsets": support_offsets,
            "evaluation_offsets": eval_offsets,
            "evaluation_examples": len(evaluation),
            "input_files": target_files,
        },
        "summary": summary,
        "positive_gate": {
            "passed": gate_passed,
            "comparisons": comparisons,
        },
        "runs": runs,
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "source_split_counts": payload["source"]["split_counts"],
        "source_held_out_diagnostic": source_diag_summary,
        "target_evaluation_examples": len(evaluation),
        "positive_gate": payload["positive_gate"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
