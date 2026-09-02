#!/usr/bin/env python3
"""Evaluator-only gold opening for frozen CLEVRER five-arm predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(path)
    return value


def _one_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0: return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    wins = sum(row["correct"][left] and not row["correct"][right] for row in rows)
    losses = sum(not row["correct"][left] and row["correct"][right] for row in rows)
    return {"wins": wins, "losses": losses, "ties": len(rows)-wins-losses,
            "net_wins": wins-losses, "one_sided_exact_p": _one_sided(wins, losses)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = _read(args.predictions); protocol = _read(args.preregistration)
    frozen_body = dict(frozen); claimed = frozen_body.pop("predictions_sha256")
    if stable_hash(frozen_body) != claimed: raise ValueError("prediction artifact hash mismatch")
    if frozen.get("status") != "CLEVRER_FIVE_ARM_PREDICTIONS_FROZEN":
        raise ValueError("predictions were not frozen")
    if frozen.get("answers_read") or frozen.get("official_programs_read"):
        raise ValueError("prediction stage crossed evaluator boundary")

    annotations = {
        int(scene["scene_index"]): {
            int(q["question_id"]): q for q in scene["questions"]
        } for scene in json.loads(args.annotations.read_text(encoding="utf-8"))
    }
    rows = []
    for frozen_row in frozen["rows"]:
        question = annotations[int(frozen_row["video_id"])][int(frozen_row["question_id"])]
        family = str(frozen_row["question_family"])
        if family == "descriptive":
            gold = str(question["answer"])
        else:
            gold = "".join(
                "1" if choice["answer"] == "correct" else "0"
                for choice in question["choices"]
            )
        rows.append({
            "task_id": frozen_row["task_id"], "question_family": family,
            "prediction_receipt_sha256": frozen_row["prediction_receipt_sha256"],
            "source_commit": frozen_row["source_commit"],
            "permuted_commit": frozen_row["permuted_commit"],
            "correct": {name: value == gold for name, value in frozen_row["predictions"].items()},
        })

    arms = tuple(frozen["arms"]); n = len(rows)
    metrics = {
        arm: {"correct": sum(row["correct"][arm] for row in rows),
              "accuracy": sum(row["correct"][arm] for row in rows) / n}
        for arm in arms
    }
    family_metrics = {}
    for family in sorted({row["question_family"] for row in rows}):
        subset = [row for row in rows if row["question_family"] == family]
        family_metrics[family] = {
            arm: {"correct": sum(row["correct"][arm] for row in subset),
                  "count": len(subset),
                  "accuracy": sum(row["correct"][arm] for row in subset)/len(subset)}
            for arm in arms
        }
    vs_neural = _paired(rows, "source_induced", "neural_only")
    vs_permuted = _paired(rows, "source_induced", "source_permuted")
    gates_cfg = protocol["formal_gates"]
    iso_equivalence = sum(
        frozen_row["predictions"]["source_induced"]
        == frozen_row["predictions"]["target_written_isomorphic"]
        for frozen_row in frozen["rows"]
    ) / n
    shared_receipts = len({row["prediction_receipt_sha256"] for row in rows}) == n
    loss_fraction = vs_neural["losses"] / n
    gates = {
        "source_vs_neural_significant": vs_neural["one_sided_exact_p"] <= gates_cfg["source_vs_neural_one_sided_exact_p_maximum"],
        "source_vs_permuted_significant": vs_permuted["one_sided_exact_p"] <= gates_cfg["source_vs_permuted_one_sided_exact_p_maximum"],
        "source_vs_neural_positive": vs_neural["net_wins"] >= gates_cfg["source_vs_neural_net_wins_minimum"],
        "source_vs_permuted_positive": vs_permuted["net_wins"] >= gates_cfg["source_vs_permuted_net_wins_minimum"],
        "negative_transfer_controlled": loss_fraction <= gates_cfg["negative_transfer_loss_fraction_maximum"],
        "target_written_isomorphic_equivalence": iso_equivalence == gates_cfg["target_written_isomorphic_prediction_equivalence"],
        "shared_receipt_fraction": float(shared_receipts) == gates_cfg["shared_grounder_receipt_fraction"],
        "minimum_source_commits": sum(row["source_commit"] for row in rows) >= gates_cfg["minimum_source_symbolic_commits"],
    }
    body = {
        "schema_version": "clevrer-five-arm-formal-report-v1",
        "status": "CLEVRER_FULL_LAYER_B_TRANSFER_VALIDATED" if all(gates.values()) else "CLEVRER_FULL_LAYER_B_TRANSFER_NOT_VALIDATED",
        "predictions_sha256": claimed, "task_count": n,
        "metrics": metrics, "family_metrics": family_metrics,
        "paired": {"source_vs_neural": vs_neural, "source_vs_permuted": vs_permuted},
        "negative_transfer_loss_fraction": loss_fraction,
        "target_written_isomorphic_prediction_equivalence": iso_equivalence,
        "gates": gates,
        "claim_boundary": "Raw frames are content-bound to cached off-the-shelf NS-DR outputs; this is not live raw-video inference. Generic and isomorphic arms are ceilings, not provenance controls.",
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({k: body[k] for k in ("status", "task_count", "metrics", "paired", "negative_transfer_loss_fraction", "gates")}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__": raise SystemExit(main())
