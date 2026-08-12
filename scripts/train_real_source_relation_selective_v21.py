#!/usr/bin/env python3
"""Train a prospective V21 selective-risk utility candidate from V20 only."""

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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_source_relation_causal_v20 import (  # noqa: E402
    linear_value,
)
from train_real_source_relation_causal_v20 import (  # noqa: E402
    _effect_metrics,
    _fit_logistic,
    _fit_ridge,
    _fork_row,
    _read,
    _validate_hash,
    _with_effect_probabilities,
)


L2_GRID = (10.0, 100.0, 1000.0)
THRESHOLD_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
FOLD_COUNT = 5
MINIMUM_OOF_SELECTED = 24
MINIMUM_OOF_WINS = 4
MAXIMUM_OOF_LOSSES = 0
MAXIMUM_OOF_SIGN_P = 0.10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold(fork_id: str) -> int:
    digest = hashlib.sha256(
        f"v21-selective-risk:{fork_id}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % FOLD_COUNT


def _sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return float(sum(
        math.comb(discordant, value)
        for value in range(wins, discordant + 1)
    ) / (2 ** discordant))


def _metrics(
    predictions: Sequence[tuple[Mapping[str, Any], float]], threshold: float
) -> dict[str, Any]:
    selected = [row for row, score in predictions if score > threshold]
    wins = sum(int(row["success_delta"]) > 0 for row in selected)
    losses = sum(int(row["success_delta"]) < 0 for row in selected)
    return {
        "threshold": float(threshold),
        "selected": len(selected),
        "success_wins": wins,
        "success_losses": losses,
        "success_delta": wins - losses,
        "one_sided_exact_sign_p": _sign_p(wins, losses),
        "selected_utility": float(sum(float(row["utility"]) for row in selected)),
        "selected_positive_utility": sum(
            float(row["utility"]) > 1e-12 for row in selected
        ),
        "selected_negative_utility": sum(
            float(row["utility"]) < -1e-12 for row in selected
        ),
        "selected_fork_ids": sorted(str(row["fork_id"]) for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-report", type=Path, required=True)
    parser.add_argument("--failed-v20-candidate", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V21 candidate: {args.output}")
    report = _read(args.fork_report)
    report_hash = _validate_hash(report, "report_sha256")
    if report.get("status") != (
        "MATCHED_CAUSAL_ADAPTATION_CALIBRATION_FORKS_COMPLETE"
    ) or not report.get("all_matched_invariants_passed"):
        raise SystemExit("V21 requires complete invariant-passing V20 forks")
    failed = _read(args.failed_v20_candidate)
    failed_hash = _validate_hash(failed, "candidate_sha256")
    if failed.get("status") != "TARGET_CAUSAL_OR_UTILITY_GATE_FAILED_STOP":
        raise SystemExit("V21 expects the fail-closed V20 candidate")
    if failed["fork_report"]["report_sha256"] != report_hash:
        raise SystemExit("V21 V20 candidate and fork report differ")
    source = _read(args.source_summary)
    if source.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise SystemExit("V21 source typed gate did not pass")
    rows = [_fork_row(row, int(report["max_steps"])) for row in report["forks"]]
    effect_head = _fit_logistic([
        row for row in rows if row["role"] == "causal_adaptation"
    ])
    rows = _with_effect_probabilities(rows, effect_head)
    effect_metrics = _effect_metrics(rows)
    folds = Counter(_fold(str(row["fork_id"])) for row in rows)
    audit = []
    eligible = []
    for l2 in L2_GRID:
        predictions: list[tuple[Mapping[str, Any], float]] = []
        for fold in range(FOLD_COUNT):
            training = [row for row in rows if _fold(str(row["fork_id"])) != fold]
            testing = [row for row in rows if _fold(str(row["fork_id"])) == fold]
            model = _fit_ridge(training, l2)
            predictions.extend(
                (row, linear_value(model, row["utility_features"]))
                for row in testing
            )
        if len(predictions) != len(rows):
            raise RuntimeError("V21 OOF predictions are incomplete")
        squared_error = float(np.mean([
            (score - float(row["utility"])) ** 2
            for row, score in predictions
        ]))
        for threshold in THRESHOLD_GRID:
            metrics = _metrics(predictions, threshold)
            gates = {
                "minimum_oof_selected": metrics["selected"] >= MINIMUM_OOF_SELECTED,
                "minimum_oof_success_wins": metrics["success_wins"] >= MINIMUM_OOF_WINS,
                "maximum_oof_success_losses": (
                    metrics["success_losses"] <= MAXIMUM_OOF_LOSSES
                ),
                "oof_success_delta_positive": metrics["success_delta"] > 0,
                "oof_selected_utility_positive": metrics["selected_utility"] > 0.0,
                "oof_sign_test_passed": (
                    metrics["one_sided_exact_sign_p"] <= MAXIMUM_OOF_SIGN_P
                ),
            }
            cell = {
                "l2": l2,
                "oof_mean_squared_error": squared_error,
                "metrics": metrics,
                "gates": gates,
                "passed": all(gates.values()),
            }
            audit.append(cell)
            if cell["passed"]:
                eligible.append(cell)
    if not eligible:
        raise SystemExit("V21 has no prospective selective-risk candidate")
    selected = min(
        eligible,
        key=lambda cell: (
            -int(cell["metrics"]["success_delta"]),
            -int(cell["metrics"]["success_wins"]),
            int(cell["metrics"]["selected"]),
            float(cell["oof_mean_squared_error"]),
            float(cell["l2"]),
            float(cell["metrics"]["threshold"]),
        ),
    )
    utility_head = _fit_ridge(rows, float(selected["l2"]))
    utility_signs = Counter(
        "positive" if row["utility"] > 1e-12 else
        "negative" if row["utility"] < -1e-12 else "neutral"
        for row in rows
    )
    gates = {
        "source_typed_gate_passed": source["status"] == "SOURCE_TYPED_GATE_PASSED",
        "all_v20_matched_invariants": bool(report["all_matched_invariants_passed"]),
        "effect_balanced_accuracy_at_least_0p95": (
            effect_metrics["balanced_accuracy_at_0p5"] >= 0.95
        ),
        "effect_brier_at_most_0p05": effect_metrics["brier_score"] <= 0.05,
        "effect_source_recall_at_least_0p95": (
            effect_metrics["source_event_recall_at_0p5"] >= 0.95
        ),
        "five_nonempty_oof_folds": len(folds) == FOLD_COUNT and min(folds.values()) > 0,
        "selective_risk_cell_passed": bool(selected["passed"]),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "real-source-relation-selective-risk-candidate-v21",
        "status": (
            "PROSPECTIVE_UTILITY_REQUALIFICATION_AUTHORIZED"
            if passed else "V21_SELECTIVE_RISK_TRAINING_FAILED_STOP"
        ),
        "claim_boundary": (
            "REAL_SOURCE_TYPED_BIND_TO_RELATE_GRAPH; TARGET_NATIVE_CAUSAL_"
            "SUCCESSOR_GROUNDING; V20_ADAPTATION_CALIBRATION_RECLASSIFIED_AS_"
            "MODEL_DEVELOPMENT_AFTER_FAIL_CLOSED_GATE; STABLE_FIVE_FOLD_OUT_"
            "OF_FOLD_SELECTIVE_RISK_ONLY; FRESH_V21_REQUALIFICATION_UNREAD; "
            "V20_DEVELOPMENT_CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "fork_report": {
            "path": str(args.fork_report.resolve()),
            "file_sha256": _sha256(args.fork_report),
            "report_sha256": report_hash,
        },
        "failed_v20_candidate": {
            "path": str(args.failed_v20_candidate.resolve()),
            "file_sha256": _sha256(args.failed_v20_candidate),
            "candidate_sha256": failed_hash,
            "failure_reason": "RESIDUAL_CONFORMAL_THRESHOLD_CAUSED_ZERO_ADMISSIONS",
        },
        "source_summary": {
            "path": str(args.source_summary.resolve()),
            "file_sha256": _sha256(args.source_summary),
            "ir_sha256": source["effect_ir"]["ir_sha256"],
            "matched_source_forks": int(source["matched_forks"]),
        },
        "parent_candidate": failed["parent_candidate"],
        "target_causal_effect_head": effect_head,
        "target_causal_effect_metrics": effect_metrics,
        "target_incremental_utility_head": utility_head,
        "selective_risk_calibration": {
            "authority": (
                "STABLE_FIVE_FOLD_OUT_OF_FOLD_V20_MODEL_DEVELOPMENT_ONLY"
            ),
            "fold_function": "sha256(v21-selective-risk:fork_id) modulo 5",
            "fold_counts": dict(folds),
            "l2_grid": list(L2_GRID),
            "threshold_grid": list(THRESHOLD_GRID),
            "selection_constraints": {
                "minimum_selected": MINIMUM_OOF_SELECTED,
                "minimum_success_wins": MINIMUM_OOF_WINS,
                "maximum_success_losses": MAXIMUM_OOF_LOSSES,
                "maximum_one_sided_exact_sign_p": MAXIMUM_OOF_SIGN_P,
            },
            "selected_l2": float(selected["l2"]),
            "admission_threshold": float(selected["metrics"]["threshold"]),
            "selected_oof_metrics": selected["metrics"],
            "audit": audit,
        },
        "utility_sign_counts": dict(utility_signs),
        "score_contract": {
            "predicted_successor_event": "RELATE_SLOT_CLOSED",
            "causal_successor_effect_certified_on_v20": passed,
            "entity_conditioned_action_binding": True,
            "incremental_utility_is_paired_source_minus_fallback": True,
            "outcome_fields_consumed_at_live_inference": False,
        },
        "gates": gates,
        "utility_requalification_authorized": passed,
        "development_authorized": False,
        "confirmation_authorized": False,
        "fresh_v21_requalification_read_or_run": False,
        "v20_development_or_confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    candidate = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": candidate["status"],
        "candidate_sha256": candidate["candidate_sha256"],
        "effect_metrics": effect_metrics,
        "utility_sign_counts": dict(utility_signs),
        "selected_l2": selected["l2"],
        "admission_threshold": selected["metrics"]["threshold"],
        "selected_oof_metrics": selected["metrics"],
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
