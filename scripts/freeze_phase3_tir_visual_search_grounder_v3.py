#!/usr/bin/env python3
"""Freeze the TIR V3 target-native grounder before a new holdout is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_tir_nonmaze import (  # noqa: E402
    BASELINE_FEATURE_NAMES,
    EFFECT_HORIZONS,
    FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
    baseline_feature_map,
    candidate_feature_map,
    evaluate_matched_receipts,
    observation_feature_map,
    validate_grounder_artifact,
)
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    TYPED_EFFECTS,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_receipt(row: dict) -> None:
    body = dict(row)
    claimed = str(body.pop("receipt_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"receipt hash mismatch: {row.get('sample_id')}")
    if row.get("formal_outcome_exposed_to_neural_calls") is not False:
        raise ValueError("target outcome was exposed to neural collection")
    if row.get("source_program_or_identity_exposed_to_neural_calls") is not False:
        raise ValueError("source information was exposed to target grounder")


REGULARIZATION_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _fit_grouped_logistic_head(
    features: list[list[float]], labels: list[bool], groups: list[str], *,
    feature_names: tuple[str, ...], selection_rows: tuple[int, ...],
) -> dict:
    """Choose regularization by source-blind target prediction log loss.

    ``selection_rows`` contains only development-train rows.  Once C is
    selected, the final coefficients use all consumed development rows.  No
    transfer condition, source identity, or held-out success enters fitting.
    """

    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=int)
    group_array = np.asarray(groups)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    standardized = (matrix - means) / scales
    selection = np.asarray(selection_rows, dtype=int)
    selection_groups = group_array[selection]
    folds = GroupKFold(n_splits=min(4, len(set(selection_groups))))
    losses: dict[float, float] = {}
    for regularization in REGULARIZATION_GRID:
        fold_losses = []
        for train_indices, test_indices in folds.split(
            standardized[selection], target[selection], selection_groups,
        ):
            train = selection[train_indices]
            test = selection[test_indices]
            model = LogisticRegression(
                C=regularization, solver="liblinear", random_state=0,
                max_iter=1000,
            ).fit(standardized[train], target[train])
            probabilities = model.predict_proba(standardized[test])[:, 1]
            fold_losses.append(log_loss(
                target[test], probabilities, labels=(0, 1),
            ))
        losses[regularization] = float(np.mean(fold_losses))
    selected_c = min(REGULARIZATION_GRID, key=lambda value: (losses[value], value))
    model = LogisticRegression(
        C=selected_c, solver="liblinear", random_state=0, max_iter=1000,
    ).fit(standardized, target)
    return {
        "feature_names": list(feature_names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training": {
            "estimator": "GROUPED_CV_L2_LOGISTIC_REGRESSION",
            "selected_C": selected_c,
            "C_grid": list(REGULARIZATION_GRID),
            "development_train_grouped_cv_log_loss": {
                str(key): losses[key] for key in REGULARIZATION_GRID
            },
            "selection_examples": len(selection),
            "final_fit_examples": len(labels),
            "positive_examples": int(target.sum()),
            "final_fit_task_groups": len(set(groups)),
            "selection_objective": "TARGET_NATIVE_PREDICTION_LOG_LOSS_ONLY",
        },
    }


def _fit_effect_heads(
    rows: list[dict], *, development_train_ids: set[str],
) -> dict[str, dict]:
    output = {}
    for effect_type in TYPED_EFFECTS:
        features: list[list[float]] = []
        labels: list[bool] = []
        groups: list[str] = []
        selection_rows = []
        for receipt in rows:
            for candidate in receipt["candidates"]:
                row = candidate_feature_map(
                    candidate, effect_type=effect_type,
                    image_size=receipt["image_size"],
                    routing=receipt["wrapper_routing"],
                )
                features.append([row[name] for name in FEATURE_NAMES])
                horizon = str(EFFECT_HORIZONS[effect_type])
                labels.append(
                    candidate["endpoints"][horizon]["answer"]
                    == receipt["gold_answer"]
                )
                groups.append(str(receipt["sample_id"]))
                if str(receipt["sample_id"]) in development_train_ids:
                    selection_rows.append(len(features) - 1)
        head = _fit_grouped_logistic_head(
            features, labels, groups, feature_names=FEATURE_NAMES,
            selection_rows=tuple(selection_rows),
        )
        head["training"]["label"] = (
            f"H{EFFECT_HORIZONS[effect_type]}_ENDPOINT_ANSWER_CORRECT_"
            "ON_CONSUMED_DEVELOPMENT_ONLY"
        )
        output[effect_type] = head
    return output


def _fit_observation_head(
    rows: list[dict], *, development_train_ids: set[str],
) -> dict:
    features: list[list[float]] = []
    labels: list[bool] = []
    groups: list[str] = []
    selection_rows = []
    for receipt in rows:
        for candidate in receipt["candidates"]:
            for effect_type in TYPED_EFFECTS:
                row = observation_feature_map(
                    receipt, candidate, effect_type=effect_type,
                )
                features.append([
                    row[name] for name in OBSERVATION_FEATURE_NAMES
                ])
                horizon = str(EFFECT_HORIZONS[effect_type])
                labels.append(
                    candidate["endpoints"][horizon]["answer"]
                    == receipt["gold_answer"]
                )
                groups.append(str(receipt["sample_id"]))
                if str(receipt["sample_id"]) in development_train_ids:
                    selection_rows.append(len(features) - 1)
    head = _fit_grouped_logistic_head(
        features, labels, groups,
        feature_names=OBSERVATION_FEATURE_NAMES,
        selection_rows=tuple(selection_rows),
    )
    head["training"]["label"] = (
        "EXECUTED_ENDPOINT_ANSWER_CORRECT_ON_CONSUMED_DEVELOPMENT_ONLY"
    )
    return head


def _fit_baseline_head(
    rows: list[dict], *, development_train_ids: set[str],
) -> dict:
    # The independent verifier already emits the quantity used at runtime.
    # Preserve it directly rather than tuning a second calibrator on scarce
    # target labels.  Development labels are deliberately not read here.
    for receipt in rows:
        baseline_feature_map(receipt)
    return {
        "calibration": "DIRECT_INDEPENDENT_VERIFIER_SUPPORT_V1",
        "verifier_field": "support_probability",
        "target_outcome_used_for_calibration": False,
        "development_receipts_schema_checked": len(rows),
        "development_train_ids_ignored_for_calibration": len(
            development_train_ids
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipts-dir", type=Path,
        default=REPO / "runs/phase3_tir_visual_search_v11_verified_development",
    )
    parser.add_argument(
        "--parent-splits", type=Path,
        default=REPO / "configs/phase3_tir_visual_search_v2_splits.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/phase3_tir_visual_search_v12_verified_frozen",
    )
    parser.add_argument(
        "--design-receipts", type=Path,
        default=(
            REPO / "runs/phase3_tir_visual_search_v11_verified_development/"
            "development_holdout_receipts.json"
        ),
    )
    args = parser.parse_args()
    parent = json.loads(args.parent_splits.read_text())
    parent_body = dict(parent)
    parent_claimed = str(parent_body.pop("config_sha256", ""))
    if not parent_claimed or stable_hash(parent_body) != parent_claimed:
        raise SystemExit("parent TIR split manifest hash mismatch")
    paths = {
        stage: args.receipts_dir / f"{stage}_receipts.json"
        for stage in ("development_train", "development_validation")
    }
    split_rows = {
        stage: json.loads(path.read_text()) for stage, path in paths.items()
    }
    for stage, rows in split_rows.items():
        expected = list(map(str, parent["splits"][stage]))
        observed = [str(row["sample_id"]) for row in rows]
        if observed != expected:
            raise SystemExit(f"{stage} receipt order does not match frozen split")
        for row in rows:
            _validate_receipt(row)
    contracts = {
        stage: {str(row["collection_contract_sha256"]) for row in rows}
        for stage, rows in split_rows.items()
    }
    if any(len(values) != 1 for values in contracts.values()):
        raise SystemExit("a development split spans collection contracts")
    forbidden = set(map(str, parent["splits"]["qualification"])) | set(
        map(str, parent["splits"]["formal"])
    )
    acquisition = [*split_rows["development_train"],
                   *split_rows["development_validation"]]
    if forbidden & {str(row["sample_id"]) for row in acquisition}:
        raise SystemExit("acquisition touched a locked qualification/formal ID")

    train_ids = set(map(str, parent["splits"]["development_train"]))
    effect_heads = _fit_effect_heads(
        acquisition, development_train_ids=train_ids,
    )
    observation_head = _fit_observation_head(
        acquisition, development_train_ids=train_ids,
    )
    baseline_head = _fit_baseline_head(
        acquisition, development_train_ids=train_ids,
    )
    body = {
        "schema_version": "phase3-tir-nonmaze-grounder-v2",
        "status": "DEVELOPMENT_GROUNDER_FROZEN_BEFORE_NEW_HOLDOUT",
        "formal_outcome_read_for_training_or_calibration": False,
        "source_program_updated": False,
        "heads": effect_heads,
        "observation_head": observation_head,
        "baseline_head": baseline_head,
        "thresholds": {
            # These are semantic probability thresholds, not transfer-success
            # hyperparameters.  They are fixed before the next holdout.
            "baseline_commit_confidence": 0.9,
            "evidence_high_probability": 0.5,
            "minimum_predicted_advantage": 0.0,
        },
        "training_audit": {
            "development_tasks": len(acquisition),
            "development_ids_sha256": stable_hash([
                str(row["sample_id"]) for row in acquisition
            ]),
            "receipt_file_sha256": {
                stage: file_sha256(path) for stage, path in paths.items()
            },
            "collection_contract_sha256_by_stage": {
                stage: next(iter(values))
                for stage, values in contracts.items()
            },
            "parent_split_config_sha256": parent["config_sha256"],
            "qualification_tasks_read": 0,
            "formal_tasks_read": 0,
            "source_identity_used_as_feature": False,
            "grounder_selection_objective": (
                "GROUPED_TARGET_NATIVE_PREDICTION_LOG_LOSS_ONLY"
            ),
            "transfer_condition_success_used_for_model_selection": False,
            "thresholds_frozen_before_new_holdout": True,
        },
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    validate_grounder_artifact(artifact)
    sources = [
        json.loads((REPO / row["path"]).read_text())
        for row in parent["source_programs"]
    ]
    design_rows = json.loads(args.design_receipts.read_text())
    for row in design_rows:
        _validate_receipt(row)
    if {str(row["sample_id"]) for row in design_rows} & {
        str(row["sample_id"]) for row in acquisition
    }:
        raise SystemExit("consumed design receipts overlap grounder acquisition")
    design_report = evaluate_matched_receipts(
        design_rows,
        grounder_artifact=artifact, source_artifacts=sources,
        role="development_design",
        gates={
            "expected_tasks": len(design_rows),
            "minimum_ceiling_successes": 6,
            "minimum_source_action_contrasts": 3,
            "minimum_permuted_action_contrasts": 3,
            "minimum_selected_effect_types": 2,
            "maximum_negative_transfer_rate": 0.0,
            "required_gate_names": [
                "expected_task_count", "target_native_ceiling_capable",
                "source_changes_target_policy",
                "authentic_differs_from_permuted",
                "multiple_source_effect_types_selected",
                "maximum_negative_transfer", "source_not_below_neural",
                "source_strictly_beats_neural",
                "source_strictly_beats_permuted",
                "source_strictly_beats_generic",
            ],
        },
    )
    design_body = dict(design_report)
    design_body.pop("report_sha256", None)
    design_body["claim_boundary"] = (
        "CONSUMED_INDEPENDENT_DEVELOPMENT_DESIGN_EVIDENCE_ONLY;"
        "NEW_HOLDOUT_REQUIRED"
    )
    design_body["design_receipts_file_sha256"] = file_sha256(
        args.design_receipts
    )
    design_report = design_body | {"report_sha256": stable_hash(design_body)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "artifact.json"
    report_path = args.output_dir / "consumed_design_report.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    report_path.write_text(json.dumps(design_report, indent=2) + "\n")
    print(json.dumps({
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
        "design_status": design_report["status"],
        "design_successes": design_report["successes"],
        "artifact": str(artifact_path.resolve()),
        "report": str(report_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
