#!/usr/bin/env python3
"""Train/freeze the V14 CLEVRER proof-receipt neural uplift ensemble."""

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

from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
    validate_source_receipt,
)
from motif_transfer.video_proof_grounder import (  # noqa: E402
    V14_FEATURE_NAMES,
    artifact_content_hash,
)
from motif_transfer.video_recovery_cate import FEATURE_NAMES  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit(
    matrix: np.ndarray, labels: np.ndarray, model_config: Mapping[str, Any], seed: int,
) -> tuple[Any, bool]:
    max_iter = int(model_config["max_iter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(int(model_config["hidden_units"]),),
                activation="tanh",
                solver="lbfgs",
                alpha=float(model_config["alpha"]),
                max_iter=max_iter,
                random_state=seed,
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
        raise ValueError("expected one-hidden-layer scalar proof MLP")
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
    labels = np.asarray([int(row["uplift"]) for row in rows], dtype=np.int64)

    def group(mask: np.ndarray) -> dict[str, int]:
        chosen = selected & mask
        values = labels[chosen]
        wins = int(np.sum(values == 1))
        losses = int(np.sum(values == -1))
        return {
            "selected": int(np.sum(chosen)),
            "wins": wins,
            "losses": losses,
            "ties_selected": int(np.sum(values == 0)),
            "net_wins": wins - losses,
        }

    all_metrics = group(np.ones(len(rows), dtype=bool))
    all_metrics["exact_two_sided_p"] = exact_binomial_two_sided(
        all_metrics["wins"], all_metrics["losses"],
    )
    all_metrics["by_source_batch"] = {
        batch: group(np.asarray([row["source_batch"] == batch for row in rows]))
        for batch in sorted({str(row["source_batch"]) for row in rows})
    }
    all_metrics["by_family"] = {
        family: group(np.asarray([row["family"] == family for row in rows]))
        for family in sorted({str(row["family"]) for row in rows})
    }
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lineage_paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "trainer_sha256": Path(__file__).resolve(),
        "proof_grounder_module_sha256": REPO / "src/motif_transfer/video_proof_grounder.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V14 training lineage mismatch for {key}: {path}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    receipt_path = Path(config["development_receipts"])
    if _sha256(receipt_path) != config["development_receipts_sha256"]:
        raise ValueError("V14 development receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "CLEVRER_V14_PROOF_DEVELOPMENT_COLLECTED":
        raise ValueError("V14 proof development collection did not complete")
    if tuple(receipt.get("feature_names", ())) != V14_FEATURE_NAMES:
        raise ValueError("V14 development feature schema mismatch")
    if not receipt.get("compiler_exact_on_all_rows"):
        raise ValueError("V14 compiler exactness preflight failed")
    rows = receipt["rows"]
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["uplift"] for row in rows], dtype=np.float64)
    batches = np.asarray([row["source_batch"] for row in rows])
    seeds = list(map(int, config["model"]["seeds"]))
    threshold = float(config["decision_threshold"])

    rng = np.random.default_rng(int(config["permuted_control_seed"]))
    permuted = labels.copy()
    for batch in sorted(set(batches)):
        for family in sorted({row["family"] for row in rows}):
            indices = np.flatnonzero([
                row["source_batch"] == batch and row["family"] == family
                for row in rows
            ])
            permuted[indices] = permuted[indices[rng.permutation(len(indices))]]

    specifications = {
        "proof": (matrix, labels),
        "base_only": (matrix[:, : len(FEATURE_NAMES)], labels),
        "permuted": (matrix, permuted),
    }
    oof_heads = {
        name: np.zeros((len(seeds), len(rows)), dtype=np.float64)
        for name in specifications
    }
    convergence = {name: [] for name in specifications}
    unique_batches = sorted(set(batches))
    for held_out in unique_batches:
        train = batches != held_out
        test = batches == held_out
        for name, (features, targets) in specifications.items():
            for head, seed in enumerate(seeds):
                model, converged = _fit(features[train], targets[train], config["model"], seed)
                convergence[name].append(converged)
                oof_heads[name][head, test] = model.predict(features[test])

    oof_predictions = {name: heads.mean(axis=0) for name, heads in oof_heads.items()}
    oof = {
        name: _policy_metrics(rows, predictions, threshold)
        for name, predictions in oof_predictions.items()
    }
    final_models: dict[str, list[dict[str, Any]]] = {}
    for name, (features, targets) in specifications.items():
        final_models[name] = []
        for seed in seeds:
            model, converged = _fit(features, targets, config["model"], seed)
            convergence[name].append(converged)
            final_models[name].append(_serialize(model))

    gates_config = config["gates"]
    proof = oof["proof"]
    gates = {
        "all_optimizers_converged": all(
            value for values in convergence.values() for value in values
        ),
        "minimum_selected": proof["selected"] >= int(gates_config["minimum_selected"]),
        "minimum_net_wins": proof["net_wins"] >= int(gates_config["minimum_net_wins"]),
        "maximum_exact_p": proof["exact_two_sided_p"] <= float(gates_config["maximum_exact_p"]),
        "positive_in_every_batch": all(
            value["net_wins"] >= int(gates_config["minimum_net_wins_per_batch"])
            for value in proof["by_source_batch"].values()
        ),
        "positive_in_every_family": all(
            value["net_wins"] >= int(gates_config["minimum_net_wins_per_family"])
            for value in proof["by_family"].values()
        ),
        "proof_above_base_only": proof["net_wins"] > oof["base_only"]["net_wins"],
        "proof_above_permuted": proof["net_wins"] > oof["permuted"]["net_wins"],
    }
    artifact_body = {
        "schema_version": 14,
        "status": "FROZEN_CLEVRER_PROOF_PAIRED_UPLIFT_ENSEMBLE",
        "estimand": "E[success(trajectory_recovery)-success(explicit_relation)|typed target proof receipts]",
        "feature_names": list(V14_FEATURE_NAMES),
        "base_feature_count": len(FEATURE_NAMES),
        "decision_threshold": threshold,
        "decision_rule": "mean(five frozen neural uplift heads) > decision_threshold",
        "model_seeds": seeds,
        "proof_models": final_models["proof"],
        "base_only_control_models": final_models["base_only"],
        "permuted_uplift_control_models": final_models["permuted"],
        "training_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["sample_id"] for row in rows)).encode("utf-8")
        ).hexdigest(),
        "development_receipts_file_sha256": config["development_receipts_sha256"],
        "config_sha256": _sha256(args.config),
    }
    artifact = artifact_body | {"artifact_sha256": artifact_content_hash(artifact_body)}
    report = {
        "schema_version": 14,
        "status": "V14_PROOF_GROUNDER_DEVELOPMENT_GATE_PASSED" if all(gates.values()) else "V14_PROOF_GROUNDER_DEVELOPMENT_GATE_FAILED",
        "samples": len(rows),
        "feature_count": matrix.shape[1],
        "model_seeds": seeds,
        "held_out_batches": unique_batches,
        "decision_threshold": threshold,
        "uplift_label_counts": {
            str(value): int(np.sum(labels == value)) for value in (-1, 0, 1)
        },
        "leave_one_batch_out": oof,
        "convergence": convergence,
        "gates": gates,
        "artifact_sha256": artifact["artifact_sha256"],
        "claim_boundary": config["claim_boundary"],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "samples": len(rows),
        "leave_one_batch_out": oof,
        "gates": gates,
        "artifact": str(args.artifact.resolve()),
        "report": str(args.report.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
