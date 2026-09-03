#!/usr/bin/env python3
"""Train/freeze the V36 matched-model source-proof uplift ensemble."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.natural_video_matched_cate import (  # noqa: E402
    SOURCE_CONTRACT,
    artifact_content_hash,
    cross_video_binding_rotation,
)
from motif_transfer.natural_video_recovery import (  # noqa: E402
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
)
from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
    validate_source_receipt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit(
    matrix: np.ndarray, labels: np.ndarray, model_config: Mapping[str, Any], seed: int,
) -> tuple[Any, bool]:
    maximum = int(model_config["max_iter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(int(model_config["hidden_units"]),),
                activation="tanh", solver="lbfgs",
                alpha=float(model_config["alpha"]), max_iter=maximum,
                random_state=seed,
            ),
        )
        pipeline.fit(matrix, labels)
    model = pipeline.named_steps["mlpregressor"]
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    converged &= int(model.n_iter_) < maximum
    return pipeline, bool(converged)


def _serialize(pipeline: Any) -> dict[str, Any]:
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["mlpregressor"]
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


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, threshold: float,
) -> dict[str, Any]:
    selected = predictions > threshold
    uplift = np.asarray([int(row["uplift"]) for row in rows])

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        chosen = selected & mask
        values = uplift[chosen]
        wins = int(np.sum(values == 1))
        losses = int(np.sum(values == -1))
        return {
            "questions": int(np.sum(mask)), "selected": int(np.sum(chosen)),
            "wins": wins, "losses": losses,
            "ties_selected": int(np.sum(values == 0)), "net_wins": wins - losses,
            "result_correct": int(sum(
                int(row["proof_correct"] if selected[index] else row["direct_correct"])
                for index, row in enumerate(rows) if mask[index]
            )),
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }

    result = summarize(np.ones(len(rows), dtype=bool))
    for field in ("batch", "benchmark", "family"):
        result[f"by_{field}"] = {
            value: summarize(np.asarray([str(row[field]) == value for row in rows]))
            for value in sorted({str(row[field]) for row in rows})
        }
    return result


def _permuted_labels(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    output = labels.copy()
    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(str(row["batch"]), str(row["benchmark"]), str(row["family"]))].append(index)
    for cell in sorted(cells):
        indices = sorted(cells[cell], key=lambda i: str(rows[i]["sample_id"]))
        values = [float(labels[index]) for index in indices]
        rng.shuffle(values)
        output[indices] = values
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(config["source_receipt"])
    adaptation_path = Path(config["adaptation_receipts"])
    if sha256(source_path) != config["frozen_lineage"]["source_receipt_sha256"]:
        raise ValueError("V36 source receipt lineage mismatch")
    if sha256(adaptation_path) != config["frozen_lineage"]["adaptation_receipts_sha256"]:
        raise ValueError("V36 adaptation receipt lineage mismatch")
    validate_source_receipt(json.loads(source_path.read_text()))
    receipt = json.loads(adaptation_path.read_text())
    if receipt.get("status") != "V36_MATCHED_MODEL_ADAPTATION_COMPILED":
        raise ValueError("V36 adaptation compilation did not complete")
    if tuple(receipt.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("V36 adaptation feature schema mismatch")
    rows = receipt["rows"]
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    labels = np.asarray([row["uplift"] for row in rows], dtype=float)
    groups = np.asarray([f"{row['benchmark']}:{row['video_id']}" for row in rows])
    base_count = len(BASE_FEATURE_NAMES)
    binding = cross_video_binding_rotation(
        rows, cell_fields=("batch", "benchmark", "family"),
    )
    bound_matrix = np.concatenate(
        (matrix[:, :base_count], matrix[np.asarray(binding), base_count:]), axis=1,
    )
    permuted = _permuted_labels(rows, labels, int(config["controls"]["permutation_seed"]))
    specifications = {
        "source_proof": (matrix, labels),
        "base_only": (matrix[:, :base_count], labels),
        "permuted_uplift": (matrix, permuted),
        "cross_video_binding": (bound_matrix, labels),
    }
    seeds = list(map(int, config["model"]["seeds"]))
    threshold = float(config["decision_threshold"])
    splitter = GroupKFold(n_splits=int(config["validation"]["group_folds"]))
    folds = list(splitter.split(matrix, labels, groups))
    heads = {
        name: np.zeros((len(seeds), len(rows)), dtype=float) for name in specifications
    }
    convergence: dict[str, list[bool]] = {name: [] for name in specifications}
    fold_audit = []
    for fold, (train, test) in enumerate(folds):
        overlap = set(groups[train]) & set(groups[test])
        if overlap:
            raise AssertionError("V36 video group leaked across OOF fold")
        fold_audit.append({
            "fold": fold, "train_rows": len(train), "test_rows": len(test),
            "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])),
            "group_overlap": 0,
        })
        for name, (features, targets) in specifications.items():
            for head, seed in enumerate(seeds):
                model, converged = _fit(features[train], targets[train], config["model"], seed)
                heads[name][head, test] = model.predict(features[test])
                convergence[name].append(converged)
    predictions = {name: values.mean(axis=0) for name, values in heads.items()}
    metrics = {name: _metrics(rows, values, threshold) for name, values in predictions.items()}
    final_models: dict[str, list[dict[str, Any]]] = {}
    for name, (features, targets) in specifications.items():
        final_models[name] = []
        for seed in seeds:
            model, converged = _fit(features, targets, config["model"], seed)
            convergence[name].append(converged)
            final_models[name].append(_serialize(model))
    source = metrics["source_proof"]
    gate = config["gates"]
    gates = {
        "all_optimizers_converged": all(value for values in convergence.values() for value in values),
        "minimum_selected": source["selected"] >= int(gate["minimum_selected"]),
        "minimum_net_wins": source["net_wins"] >= int(gate["minimum_net_wins"]),
        "maximum_exact_p": source["exact_two_sided_p"] <= float(gate["maximum_exact_p"]),
        "minimum_net_each_benchmark": all(
            value["net_wins"] >= int(gate["minimum_net_wins_each_benchmark"])
            for value in source["by_benchmark"].values()
        ),
        "nonnegative_each_batch": all(
            value["net_wins"] >= int(gate["minimum_net_wins_each_batch"])
            for value in source["by_batch"].values()
        ),
        "source_above_raw_proof": source["result_correct"] > int(receipt["audit"]["proof_correct"]),
        "source_above_base_only": source["net_wins"] > metrics["base_only"]["net_wins"],
        "source_above_permuted_uplift": source["net_wins"] > metrics["permuted_uplift"]["net_wins"],
        "source_above_cross_video_binding": source["net_wins"] > metrics["cross_video_binding"]["net_wins"],
    }
    passed = all(gates.values())
    artifact_body = {
        "schema_version": 36,
        "status": (
            "FROZEN_MATCHED_MODEL_NATURAL_VIDEO_SOURCE_CATE"
            if passed else "DEVELOPMENT_ONLY_MATCHED_MODEL_NATURAL_VIDEO_SOURCE_CATE_FAILED"
        ),
        "estimand": "E[success(typed_proof)-success(matched_direct)|source-typed target receipts]",
        "source_contract": list(SOURCE_CONTRACT),
        "feature_names": list(FEATURE_NAMES),
        "base_feature_count": base_count,
        "decision_threshold": threshold,
        "model_seeds": seeds,
        "source_proof_models": final_models["source_proof"],
        "base_only_control_models": final_models["base_only"],
        "permuted_uplift_control_models": final_models["permuted_uplift"],
        "cross_video_binding_control_models": final_models["cross_video_binding"],
        "adaptation_receipts_sha256": sha256(adaptation_path),
        "config_sha256": sha256(args.config),
        "trainer_sha256": sha256(Path(__file__).resolve()),
    }
    artifact = artifact_body | {"artifact_sha256": artifact_content_hash(artifact_body)}
    report = {
        "schema_version": 36,
        "status": "V36_MATCHED_CATE_DEVELOPMENT_GATE_PASSED" if passed else "V36_MATCHED_CATE_DEVELOPMENT_GATE_FAILED",
        "samples": len(rows), "video_groups": len(set(groups)),
        "feature_count": matrix.shape[1], "base_feature_count": base_count,
        "decision_threshold": threshold,
        "uplift_counts": receipt["audit"]["uplift_counts"],
        "raw_direct_correct": receipt["audit"]["direct_correct"],
        "raw_proof_correct": receipt["audit"]["proof_correct"],
        "validation": "10-fold grouped OOF by (benchmark,video_id)",
        "fold_audit": fold_audit,
        "oof": metrics,
        "convergence": {
            name: {"fits": len(values), "converged": sum(values)}
            for name, values in convergence.items()
        },
        "binding_audit": {
            "rows": len(rows), "same_sample": sum(i == j for i, j in enumerate(binding)),
            "same_video": sum(rows[i]["video_id"] == rows[j]["video_id"] for i, j in enumerate(binding)),
        },
        "gates": gates, "artifact_sha256": artifact["artifact_sha256"],
        "claim_boundary": config["claim_boundary"],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "oof": metrics, "gates": gates,
        "artifact": str(args.artifact.resolve()), "report": str(args.report.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
