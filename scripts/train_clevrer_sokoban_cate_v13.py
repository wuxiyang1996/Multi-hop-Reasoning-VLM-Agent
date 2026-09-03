#!/usr/bin/env python3
"""Train and cross-fit the CLEVRER target-native paired recovery uplift head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_transfer.sokoban_video_recovery import exact_binomial_two_sided  # noqa: E402
from motif_transfer.video_recovery_cate import FEATURE_NAMES, build_features  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fold(sample_id: str, folds: int) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16], 16) % folds


def _rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for spec in config["input_reports"]:
        path = Path(spec["path"])
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"input report hash mismatch: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["rows"]:
            explicit = row["conditions"]["target_explicit_no_recovery"]
            trajectory = row["conditions"]["target_trajectory_only"]
            features = build_features(
                family=str(row["family"]),
                question_program=row["compiled_question_program"],
                choice_programs=row["compiled_choice_programs"],
                explicit_answer=str(explicit["answer"]),
                trajectory_answer=str(trajectory["answer"]),
                explicit_error_count=int(row["typed_effect_receipt"]["error_count"]),
            )
            output.append({
                "sample_id": str(row["sample_id"]),
                "source_batch": str(spec["name"]),
                "family": str(row["family"]),
                "features": features,
                "uplift": int(bool(trajectory["correct"])) - int(bool(explicit["correct"])),
                "explicit_correct": bool(explicit["correct"]),
            })
    if len({row["sample_id"] for row in output}) != len(output):
        raise ValueError("CATE training reports contain duplicate sample IDs")
    return output


def _fit(
    matrix: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[Any, bool]:
    max_iter = int(config["max_iter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(int(config["hidden_units"]),),
                activation="tanh",
                solver="lbfgs",
                alpha=float(config["alpha"]),
                max_iter=max_iter,
                random_state=int(config["seed"]),
            ),
        )
        pipeline.fit(matrix, labels)
    model = pipeline.named_steps["mlpregressor"]
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    converged &= int(model.n_iter_) < max_iter
    return pipeline, bool(converged)


def _serialize(pipeline: Any) -> dict[str, Any]:
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["mlpregressor"]
    if len(model.coefs_) != 2 or model.n_outputs_ != 1:
        raise ValueError("expected a one-hidden-layer scalar MLP")
    return {
        "feature_mean": list(map(float, scaler.mean_)),
        "feature_scale": list(map(float, scaler.scale_)),
        "input_weights": [list(map(float, row)) for row in model.coefs_[0]],
        "hidden_bias": list(map(float, model.intercepts_[0])),
        "output_weights": list(map(float, model.coefs_[1].reshape(-1))),
        "output_bias": float(model.intercepts_[1][0]),
        "n_iter": int(model.n_iter_),
        "loss": float(model.loss_),
    }


def _policy_metrics(
    rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, threshold: float,
) -> dict[str, Any]:
    selected = predictions > threshold
    labels = np.asarray([row["uplift"] for row in rows], dtype=np.int64)
    wins = int(np.sum(labels[selected] == 1))
    losses = int(np.sum(labels[selected] == -1))
    batches = {}
    for batch in sorted({str(row["source_batch"]) for row in rows}):
        mask = np.asarray([str(row["source_batch"]) == batch for row in rows])
        batches[batch] = {
            "selected": int(np.sum(selected[mask])),
            "wins": int(np.sum(labels[mask][selected[mask]] == 1)),
            "losses": int(np.sum(labels[mask][selected[mask]] == -1)),
        }
        batches[batch]["net_wins"] = batches[batch]["wins"] - batches[batch]["losses"]
    return {
        "selected": int(np.sum(selected)),
        "wins": wins,
        "losses": losses,
        "ties_selected": int(np.sum(labels[selected] == 0)),
        "net_wins": wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        "by_source_batch": batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = _rows(config)
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["uplift"] for row in rows], dtype=np.float64)
    folds = int(config["cross_fit_folds"])
    assignments = np.asarray([_fold(row["sample_id"], folds) for row in rows])

    rng = np.random.default_rng(int(config["control_seed"]))
    permuted = labels.copy()
    for family in sorted({row["family"] for row in rows}):
        indices = np.flatnonzero([row["family"] == family for row in rows])
        permuted[indices] = permuted[indices[rng.permutation(len(indices))]]

    predictions = {"authentic": np.zeros(len(rows)), "permuted": np.zeros(len(rows))}
    convergence = {"authentic": [], "permuted": []}
    for fold in range(folds):
        train = assignments != fold
        test = assignments == fold
        if not np.any(test) or not np.any(train):
            raise ValueError("empty CATE cross-fit fold")
        for name, target in (("authentic", labels), ("permuted", permuted)):
            model, converged = _fit(matrix[train], target[train], config["model"])
            convergence[name].append(converged)
            predictions[name][test] = model.predict(matrix[test])

    threshold = float(config["decision_threshold"])
    oof = {
        name: _policy_metrics(rows, value, threshold)
        for name, value in predictions.items()
    }
    authentic_model, authentic_converged = _fit(matrix, labels, config["model"])
    permuted_model, permuted_converged = _fit(matrix, permuted, config["model"])
    gates = {
        "all_optimizers_converged": all(
            convergence["authentic"] + convergence["permuted"]
            + [authentic_converged, permuted_converged]
        ),
        "minimum_selected": oof["authentic"]["selected"]
        >= int(config["gates"]["minimum_selected"]),
        "minimum_net_wins": oof["authentic"]["net_wins"]
        >= int(config["gates"]["minimum_net_wins"]),
        "maximum_exact_p": oof["authentic"]["exact_two_sided_p"]
        <= float(config["gates"]["maximum_exact_p"]),
        "positive_in_every_consumed_batch": all(
            value["net_wins"] >= int(config["gates"]["minimum_net_wins_per_batch"])
            for value in oof["authentic"]["by_source_batch"].values()
        ),
        "authentic_above_permuted": oof["authentic"]["net_wins"]
        > oof["permuted"]["net_wins"],
    }
    artifact_body = {
        "schema_version": 1,
        "status": "FROZEN_TARGET_NATIVE_PAIRED_UPLIFT_GROUNDER",
        "estimand": "E[success(trajectory_recovery)-success(explicit_relation)|pre-commit target receipts]",
        "feature_names": list(FEATURE_NAMES),
        "decision_threshold": threshold,
        "model": _serialize(authentic_model),
        "permuted_control_model": _serialize(permuted_model),
        "training_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["sample_id"] for row in rows)).encode("utf-8")
        ).hexdigest(),
        "input_report_sha256": {
            str(Path(spec["path"]).resolve()): spec["sha256"]
            for spec in config["input_reports"]
        },
        "config_sha256": _sha256(args.config),
    }
    artifact = artifact_body | {"artifact_sha256": _content_hash(artifact_body)}
    report = {
        "schema_version": 1,
        "status": "CATE_DEVELOPMENT_GATE_PASSED" if all(gates.values()) else "CATE_DEVELOPMENT_GATE_FAILED",
        "samples": len(rows),
        "uplift_label_counts": {
            str(value): int(np.sum(labels == value)) for value in (-1, 0, 1)
        },
        "cross_fit_folds": folds,
        "decision_threshold": threshold,
        "oof": oof,
        "convergence": convergence | {
            "final_authentic": authentic_converged,
            "final_permuted": permuted_converged,
        },
        "gates": gates,
        "artifact_sha256": artifact["artifact_sha256"],
        "claim_boundary": config["claim_boundary"],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "samples": len(rows), "oof": oof,
        "gates": gates, "artifact": str(args.artifact.resolve()),
        "report": str(args.report.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
