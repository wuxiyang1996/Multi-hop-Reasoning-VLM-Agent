#!/usr/bin/env python3
"""Fit a video-group-held-out CATE for Sokoban-to-STAR/NExT-QA transfer."""

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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.natural_video_proof_cate import (  # noqa: E402
    SOURCE_CONTRACT,
    artifact_content_hash,
    compile_v19_features,
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
    matrix: np.ndarray,
    labels: np.ndarray,
    model_config: Mapping[str, Any],
    seed: int,
) -> tuple[Any, bool]:
    maximum = int(model_config["max_iter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(int(model_config["hidden_units"]),),
                activation="tanh",
                solver="lbfgs",
                alpha=float(model_config["alpha"]),
                max_iter=maximum,
                random_state=seed,
            ),
        )
        pipeline.fit(matrix, labels)
    model = pipeline.named_steps["mlpregressor"]
    converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    converged &= int(model.n_iter_) < maximum
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


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, threshold: float,
) -> dict[str, Any]:
    selected = predictions > threshold
    uplift = np.asarray([
        int(row["proof_correct"]) - int(row["primary_correct"]) for row in rows
    ])

    def summarize(indices: np.ndarray) -> dict[str, Any]:
        chosen = selected & indices
        values = uplift[chosen]
        wins = int(np.sum(values == 1))
        losses = int(np.sum(values == -1))
        return {
            "questions": int(np.sum(indices)),
            "selected": int(np.sum(chosen)),
            "wins": wins,
            "losses": losses,
            "ties_selected": int(np.sum(values == 0)),
            "net_wins": wins - losses,
            "result_correct": int(sum(
                int(row["proof_correct"] if selected[index] else row["primary_correct"])
                for index, row in enumerate(rows) if indices[index]
            )),
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }

    result = summarize(np.ones(len(rows), dtype=bool))
    result["by_benchmark"] = {
        value: summarize(np.asarray([str(row["benchmark"]) == value for row in rows]))
        for value in sorted({str(row["benchmark"]) for row in rows})
    }
    result["by_family"] = {
        f"{benchmark}:{family}": summarize(np.asarray([
            str(row["benchmark"]) == benchmark and str(row["family"]) == family
            for row in rows
        ]))
        for benchmark, family in sorted({
            (str(row["benchmark"]), str(row["family"])) for row in rows
        })
    }
    return result


def _proof_shuffle(
    rows: Sequence[Mapping[str, Any]], seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Outcome-blind typed-proof binding control within benchmark/family."""

    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(str(row["benchmark"]), str(row["family"]))].append(index)
    mapping = list(range(len(rows)))
    for cell in sorted(cells):
        indices = sorted(cells[cell], key=lambda i: str(rows[i]["sample_id"]))
        if len(indices) <= 1:
            continue
        candidates = []
        for _ in range(1000):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            score = (
                sum(a == b for a, b in zip(indices, shuffled)),
                sum(
                    str(rows[a]["video_id"]) == str(rows[b]["video_id"])
                    for a, b in zip(indices, shuffled)
                ),
            )
            candidates.append((score, shuffled))
            if score == (0, 0):
                break
        shuffled = min(candidates, key=lambda item: item[0])[1]
        for source, target in zip(indices, shuffled):
            mapping[source] = target
    return np.asarray(mapping), {
        "fixed_points": sum(index == target for index, target in enumerate(mapping)),
        "same_video_bindings": sum(
            str(rows[index]["video_id"]) == str(rows[target]["video_id"])
            for index, target in enumerate(mapping)
        ),
        "mapping_sha256": hashlib.sha256(
            ",".join(map(str, mapping)).encode("utf-8")
        ).hexdigest(),
    }


def _permuted_uplift(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    output = labels.copy()
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(str(row["benchmark"]), str(row["family"]))].append(index)
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

    paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "development_receipts_sha256": Path(config["development_receipts"]),
        "feature_compiler_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
    }
    for key, path in paths.items():
        if sha256(path) != str(config["frozen_lineage"].get(key, "")):
            raise ValueError(f"V34 frozen lineage mismatch: {key}")
    validate_source_receipt(json.loads(paths["source_receipt_sha256"].read_text()))
    rows = json.loads(paths["development_receipts_sha256"].read_text())
    if not rows or len({(row["benchmark"], row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("V34 receipts are empty or have duplicate identities")
    matrix = np.asarray([compile_v19_features(row) for row in rows], dtype=float)
    labels = np.asarray([
        int(row["proof_correct"]) - int(row["primary_correct"]) for row in rows
    ], dtype=float)
    groups = np.asarray([
        f"{row['benchmark']}:{row['video_id']}" for row in rows
    ])
    base_count = len(BASE_FEATURE_NAMES)
    shuffle_index, shuffle_audit = _proof_shuffle(
        rows, int(config["controls"]["proof_shuffle_seed"]),
    )
    shuffled_matrix = np.concatenate(
        (matrix[:, :base_count], matrix[shuffle_index, base_count:]), axis=1,
    )
    permuted_labels = _permuted_uplift(
        rows, labels, int(config["controls"]["uplift_permutation_seed"]),
    )
    specifications = {
        "source_proof": (matrix, labels),
        "base_only": (matrix[:, :base_count], labels),
        "permuted_uplift": (matrix, permuted_labels),
        "shuffled_proof": (shuffled_matrix, labels),
    }
    seeds = list(map(int, config["model"]["seeds"]))
    threshold = float(config["decision_threshold"])
    folds = list(LeaveOneGroupOut().split(matrix, labels, groups))
    heads = {
        name: np.zeros((len(seeds), len(rows)), dtype=float)
        for name in specifications
    }
    convergence: dict[str, list[bool]] = {name: [] for name in specifications}
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("video group leaked across V34 fold")
        for name, (features, targets) in specifications.items():
            for head, seed in enumerate(seeds):
                model, converged = _fit(features[train], targets[train], config["model"], seed)
                heads[name][head, test] = model.predict(features[test])
                convergence[name].append(converged)
    predictions = {name: values.mean(axis=0) for name, values in heads.items()}
    metrics = {
        name: _metrics(rows, values, threshold) for name, values in predictions.items()
    }

    final_models: dict[str, list[dict[str, Any]]] = {}
    for name, (features, targets) in specifications.items():
        final_models[name] = []
        for seed in seeds:
            model, converged = _fit(features, targets, config["model"], seed)
            convergence[name].append(converged)
            final_models[name].append(_serialize(model))

    authentic = metrics["source_proof"]
    primary_correct = sum(bool(row["primary_correct"]) for row in rows)
    always_proof_correct = sum(bool(row["proof_correct"]) for row in rows)
    gate = config["gates"]
    gates = {
        "all_optimizers_converged": all(
            item for values in convergence.values() for item in values
        ),
        "minimum_selected": authentic["selected"] >= int(gate["minimum_selected"]),
        "minimum_net_wins": authentic["net_wins"] >= int(gate["minimum_net_wins"]),
        "maximum_exact_p": authentic["exact_two_sided_p"] <= float(gate["maximum_exact_p"]),
        "positive_each_benchmark": all(
            value["net_wins"] >= int(gate["minimum_net_wins_each_benchmark"])
            for value in authentic["by_benchmark"].values()
        ),
        "source_proof_above_primary": authentic["result_correct"] > primary_correct,
        "source_proof_above_always_proof": authentic["result_correct"] > always_proof_correct,
        "source_proof_above_base_only": authentic["net_wins"] > metrics["base_only"]["net_wins"],
        "source_proof_above_permuted_uplift": authentic["net_wins"] > metrics["permuted_uplift"]["net_wins"],
        "source_proof_above_shuffled_proof": authentic["net_wins"] > metrics["shuffled_proof"]["net_wins"],
    }
    passed = all(gates.values())
    artifact_status = (
        "FROZEN_NATURAL_VIDEO_SOURCE_PROOF_CATE"
        if passed else "DEVELOPMENT_ONLY_NATURAL_VIDEO_SOURCE_PROOF_CATE_FAILED"
    )
    artifact_body = {
        "schema_version": 34,
        "status": artifact_status,
        "estimand": "E[success(target_proof)-success(target_direct)|Sokoban-typed verification receipts]",
        "source_contract": list(SOURCE_CONTRACT),
        "feature_names": list(FEATURE_NAMES),
        "base_feature_count": base_count,
        "decision_threshold": threshold,
        "decision_rule": "mean(frozen neural uplift heads) > 0 triggers REPLAN to target proof answer",
        "model_seeds": seeds,
        "source_proof_models": final_models["source_proof"],
        "base_only_control_models": final_models["base_only"],
        "permuted_uplift_control_models": final_models["permuted_uplift"],
        "shuffled_proof_control_models": final_models["shuffled_proof"],
        "development_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["sample_id"]) for row in rows)).encode()
        ).hexdigest(),
        "development_receipts_sha256": config["frozen_lineage"]["development_receipts_sha256"],
        "config_sha256": sha256(args.config),
        "trainer_sha256": sha256(Path(__file__).resolve()),
    }
    artifact = artifact_body | {"artifact_sha256": artifact_content_hash(artifact_body)}
    report = {
        "schema_version": 34,
        "status": "V34_NATURAL_VIDEO_CATE_DEVELOPMENT_GATE_PASSED" if passed else "V34_NATURAL_VIDEO_CATE_DEVELOPMENT_GATE_FAILED",
        "samples": len(rows),
        "video_groups": len(set(groups)),
        "benchmark_counts": {
            value: sum(str(row["benchmark"]) == value for row in rows)
            for value in sorted({str(row["benchmark"]) for row in rows})
        },
        "feature_count": matrix.shape[1],
        "base_feature_count": base_count,
        "uplift_label_counts": {
            str(value): int(np.sum(labels == value)) for value in (-1, 0, 1)
        },
        "validation": "leave-one-(benchmark,video_id)-out; no video group crosses train/test",
        "decision_threshold": threshold,
        "leave_one_video_out": metrics,
        "raw_controls": {
            "primary_correct": primary_correct,
            "always_proof_correct": always_proof_correct,
            "oracle_correct": sum(
                max(bool(row["primary_correct"]), bool(row["proof_correct"])) for row in rows
            ),
        },
        "proof_shuffle_audit": shuffle_audit,
        "convergence": {
            name: {"fits": len(values), "converged": sum(values)}
            for name, values in convergence.items()
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
        "status": report["status"],
        "metrics": metrics,
        "gates": gates,
        "artifact": str(args.artifact.resolve()),
        "report": str(args.report.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
