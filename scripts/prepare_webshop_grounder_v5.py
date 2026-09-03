#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.real_game_latent_options import (  # noqa: E402
    extract_structural_episode,
    reward_normalizer,
)
from motif_transfer.real_game_multitarget_manifest import file_sha256, stable_hash  # noqa: E402
from motif_transfer.webshop_neural_grounder_v5 import (  # noqa: E402
    TARGET_FEATURE_NAMES,
    finite_json,
    nearest_source_options,
    target_action_features,
)


SEED = 20260810


def _rank(value: str) -> str:
    return hashlib.sha256(f"webshop-grounder-v5\0{value}".encode()).hexdigest()


def _canonical(step: dict) -> str:
    metadata = step.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("schema_canonical") or metadata.get("schema") or step.get("state") or "")
    return str(step.get("state") or "")


def _url(step: dict) -> str:
    metadata = step.get("metadata")
    if isinstance(metadata, dict) and metadata.get("url"):
        return str(metadata["url"])
    interface = step.get("interface")
    if isinstance(interface, dict) and interface.get("url"):
        return str(interface["url"])
    return ""


def _episodes(manifest: dict) -> list[dict]:
    target = manifest["targets"]["webshop"]
    task_ids = target["partition"]["roles"]["adaptation"]
    rows = []
    for root_text in target["historical_rollout_roots"]:
        root = Path(root_text)
        for task_id in task_ids:
            path = root / task_id / "episode_000.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            experiences = payload.get("experiences")
            if not isinstance(experiences, list) or not experiences:
                raise ValueError(f"empty adaptation episode: {path}")
            rows.append({
                "task_id": task_id,
                "path": str(path),
                "sha256": file_sha256(path),
                "experiences": experiences,
            })
    return rows


def _sanitized(episode: dict) -> list[dict]:
    experiences = episode["experiences"]
    output = []
    for index, step in enumerate(experiences):
        next_text = _canonical(experiences[index + 1]) if index + 1 < len(experiences) else str(
            step.get("next_state") or ""
        )
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        next_metadata = (
            experiences[index + 1].get("metadata")
            if index + 1 < len(experiences)
            and isinstance(experiences[index + 1].get("metadata"), dict)
            else {}
        )
        output.append({
            "state": _canonical(step),
            "action": str(step.get("action") or ""),
            "reward": step.get("reward") or 0.0,
            "next_state": next_text,
            "done": bool(step.get("done")),
            "available_actions": list(metadata.get("candidate_actions") or []),
            "next_available_actions": list(next_metadata.get("candidate_actions") or []),
        })
    return output


def _dataset(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sanitized = [_sanitized(episode) for episode in episodes]
    mean, scale = reward_normalizer(sanitized)
    features = []
    labels = []
    task_ids = []
    for episode, clean in zip(episodes, sanitized, strict=True):
        structural = extract_structural_episode(
            clean,
            game="webshop",
            episode_id=episode["path"],
            reward_mean=mean,
            reward_scale=scale,
        )
        previous_action = None
        maximum_steps = len(clean)
        for index, (raw, row) in enumerate(zip(episode["experiences"], structural, strict=True)):
            features.append(target_action_features(
                observation_text=_canonical(raw),
                url=_url(raw),
                goal=str(raw.get("goal") or raw.get("tasks") or ""),
                action=str(raw.get("action") or ""),
                step_index=index,
                maximum_steps=maximum_steps,
                previous_action=previous_action,
            ))
            labels.append(row.effect_features)
            task_ids.append(episode["task_id"])
            previous_action = str(raw.get("action") or "")
    return np.asarray(features), np.asarray(labels), task_ids


def _fit(
    features: np.ndarray, labels: np.ndarray, *, maximum_iterations: int
) -> tuple[StandardScaler, MLPRegressor]:
    scaler = StandardScaler().fit(features)
    model = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        solver="lbfgs",
        alpha=1e-3,
        max_iter=maximum_iterations,
        max_fun=50000,
        random_state=SEED,
    ).fit(scaler.transform(features), labels)
    return scaler, model


def _round(value: object) -> object:
    if isinstance(value, np.ndarray):
        return np.round(value.astype(np.float64), 12).tolist()
    if isinstance(value, float):
        return round(value, 12)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/real_game_multitarget_v5_manifest.json",
    )
    parser.add_argument(
        "--source-candidate", type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/source_development/frozen_candidate.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/webshop_grounder",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    source_candidate = json.loads(args.source_candidate.read_text())
    episodes = _episodes(manifest)
    features, labels, task_ids = _dataset(episodes)
    unique_tasks = sorted(set(task_ids), key=lambda task_id: (_rank(task_id), task_id))
    validation_tasks = set(unique_tasks[-2:])
    train_mask = np.asarray([task_id not in validation_tasks for task_id in task_ids])
    validation_mask = ~train_mask
    baseline = np.repeat(np.mean(labels[train_mask], axis=0)[None, :], np.sum(validation_mask), axis=0)
    expected_options = nearest_source_options(labels[validation_mask], source_candidate)
    baseline_options = nearest_source_options(baseline, source_candidate)
    marginal_effect_mse = float(mean_squared_error(labels[validation_mask], baseline))
    marginal_option_accuracy = float(
        balanced_accuracy_score(expected_options, baseline_options)
    )
    validation_candidates = []
    for maximum_iterations in (250, 500, 1000, 2000, 3000):
        validation_scaler, validation_model = _fit(
            features[train_mask], labels[train_mask],
            maximum_iterations=maximum_iterations,
        )
        predicted = validation_model.predict(
            validation_scaler.transform(features[validation_mask])
        )
        predicted_options = nearest_source_options(predicted, source_candidate)
        validation_candidates.append({
            "maximum_iterations": maximum_iterations,
            "actual_iterations": int(validation_model.n_iter_),
            "effect_mse": float(mean_squared_error(labels[validation_mask], predicted)),
            "latent_option_balanced_accuracy": float(
                balanced_accuracy_score(expected_options, predicted_options)
            ),
        })
    eligible_candidates = [
        row for row in validation_candidates
        if row["effect_mse"] < marginal_effect_mse
        and row["latent_option_balanced_accuracy"] > marginal_option_accuracy
    ]
    selected = min(
        eligible_candidates or validation_candidates,
        key=lambda row: (row["effect_mse"], -row["latent_option_balanced_accuracy"]),
    )
    validation = {
        "tasks": sorted(validation_tasks),
        "rows": int(np.sum(validation_mask)),
        "hyperparameter_candidates": validation_candidates,
        "selected_maximum_iterations": selected["maximum_iterations"],
        "effect_mse": selected["effect_mse"],
        "marginal_effect_mse": marginal_effect_mse,
        "latent_option_balanced_accuracy": selected["latent_option_balanced_accuracy"],
        "marginal_latent_option_balanced_accuracy": marginal_option_accuracy,
    }
    scaler, model = _fit(
        features, labels, maximum_iterations=selected["maximum_iterations"]
    )
    artifact = {
        "schema_version": 1,
        "artifact_role": "WEBSHOP_TARGET_NATIVE_NEURAL_GROUNDER_V5",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_candidate_artifact_sha256": source_candidate["artifact_sha256"],
        "adaptation_task_ids": manifest["targets"]["webshop"]["partition"]["roles"]["adaptation"],
        "adaptation_episode_receipts": [
            {key: episode[key] for key in ("task_id", "path", "sha256")} for episode in episodes
        ],
        "input_feature_names": list(TARGET_FEATURE_NAMES),
        "input_scaler": {"mean": _round(scaler.mean_), "scale": _round(scaler.scale_)},
        "mlp": {
            "activation": "relu",
            "hidden_layer_sizes": [48, 24],
            "solver": "lbfgs",
            "alpha": 0.001,
            "max_iter": selected["maximum_iterations"],
            "n_iter": int(model.n_iter_),
            "loss": round(float(model.loss_), 12),
            "coefficients": [_round(value) for value in model.coefs_],
            "intercepts": [_round(value) for value in model.intercepts_],
        },
        "validation": validation,
        "semantic_source_fields_accessed": [],
    }
    if not finite_json(artifact):
        raise ValueError("grounder artifact contains non-finite values")
    artifact["artifact_sha256"] = stable_hash(artifact)
    report = {
        "schema_version": 1,
        "status": "GROUNDER_GATE_PASS"
        if validation["effect_mse"] < validation["marginal_effect_mse"]
        and validation["latent_option_balanced_accuracy"]
        > validation["marginal_latent_option_balanced_accuracy"]
        else "GROUNDER_GATE_FAIL",
        "adaptation_tasks": len(set(task_ids)),
        "adaptation_episodes": len(episodes),
        "adaptation_rows": len(features),
        "validation": validation,
        "grounder_artifact_sha256": artifact["artifact_sha256"],
    }
    report["report_sha256"] = stable_hash(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_grounder.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "grounder_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
