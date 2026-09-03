#!/usr/bin/env python3
"""Evaluate the preregistered V26 source-vs-target-only paired endpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue, target_only_ontology_decision,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import _answer_matches  # noqa: E402
import scripts.collect_agqa2_query_object_v24 as v24  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def _evaluation_core(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_evaluation_protocol_sha256": config[
            "expected_evaluation_protocol_sha256"
        ],
        "source_specific_evaluation": config["source_specific_evaluation"],
        "split": config["split"],
        "manifest": config["manifest"],
    }


def _paired_evaluation(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    spec = config["source_specific_evaluation"]
    thresholds = spec["minimum_ontology_confidences"]
    rows = []
    wins = losses = target_decisive = source_correct = target_correct = 0
    for original in result["rows"]:
        row = deepcopy(original)
        decision = target_only_ontology_decision(
            row["object_ontology_receipts"], thresholds,
        )
        target_prediction = decision or row["direct_response"]
        gold = row["gold_answer_evaluator_only"]
        target_row_correct = _answer_matches(target_prediction, gold)
        source_row_correct = _answer_matches(
            row["unified_harness_prediction"], gold,
        )
        outcome = "TIE"
        if source_row_correct and not target_row_correct:
            outcome = "SOURCE_WIN"
            wins += 1
        elif target_row_correct and not source_row_correct:
            outcome = "SOURCE_LOSS"
            losses += 1
        target_decisive += decision is not None
        source_correct += source_row_correct
        target_correct += target_row_correct
        row.update({
            "target_only_ontology_decision": decision,
            "target_only_prediction": target_prediction,
            "target_only_decisive": decision is not None,
            "target_only_correct": target_row_correct,
            "source_vs_target_only_outcome": outcome,
            "target_only_prediction_frozen_without_gold_or_source_view": True,
        })
        rows.append(row)
    discordant = wins + losses
    return {
        "rows": rows,
        "metrics": {
            "valid_paired_rows": len(rows),
            "source_harness_correct": source_correct,
            "target_only_correct": target_correct,
            "source_minus_target_only_correct": source_correct - target_correct,
            "target_only_decisive": target_decisive,
            "source_vs_target_only_wins": wins,
            "source_vs_target_only_losses": losses,
            "source_vs_target_only_ties": len(rows) - discordant,
            "discordant_pairs": discordant,
            "exact_one_sided_pvalue": exact_one_sided_pvalue(
                source_wins=wins, source_losses=losses,
            ),
        },
    }


def collect(**kwargs) -> dict[str, Any]:
    config_path = Path(kwargs["config_path"])
    output_path = Path(kwargs["output_path"])
    config = json.loads(config_path.read_text())
    spec = config["source_specific_evaluation"]
    for label in ("module", "collector"):
        path = REPO_ROOT / spec[label]
        if _sha256(path) != spec[f"{label}_sha256"]:
            raise ValueError(f"V26 source-specific evaluator {label} hash mismatch")
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    if evaluation_sha256 != config[
        "expected_source_specific_evaluation_protocol_sha256"
    ]:
        raise ValueError("V26 source-specific evaluation protocol changed")

    base = v24.collect(**kwargs)
    if base["grounder_sha256"] != config["expected_grounder_sha256"]:
        raise ValueError("V26 changed the frozen V24 neural grounder")
    paired = _paired_evaluation(base, config)
    metrics = paired["metrics"]
    gate = spec["qualification_gates"]
    source_gates = {
        "base_mechanism_gates_passed": bool(base["grounder_qualified"]),
        "required_valid_paired_rows": (
            metrics["valid_paired_rows"] >= gate["required_valid_paired_rows"]
        ),
        "minimum_target_only_decisive": (
            metrics["target_only_decisive"] >= gate["minimum_target_only_decisive"]
        ),
        "minimum_source_vs_target_only_wins": (
            metrics["source_vs_target_only_wins"]
            >= gate["minimum_source_vs_target_only_wins"]
        ),
        "maximum_source_vs_target_only_losses": (
            metrics["source_vs_target_only_losses"]
            <= gate["maximum_source_vs_target_only_losses"]
        ),
        "minimum_source_minus_target_only_correct": (
            metrics["source_minus_target_only_correct"]
            >= gate["minimum_source_minus_target_only_correct"]
        ),
        "maximum_exact_one_sided_pvalue": (
            metrics["exact_one_sided_pvalue"]
            <= gate["maximum_exact_one_sided_pvalue"]
        ),
        "target_only_prediction_is_candidate_and_source_blind": all(
            row["target_only_prediction_frozen_without_gold_or_source_view"]
            for row in paired["rows"]
        ),
    }
    qualified = all(source_gates.values())
    body = deepcopy(base)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "agqa2-query-object-source-specific-report-v26",
        "status": (
            "AGQA2_QUERY_OBJECT_V26_SOURCE_SPECIFIC_QUALIFIED"
            if qualified else
            "AGQA2_QUERY_OBJECT_V26_SOURCE_SPECIFIC_NOT_QUALIFIED"
        ),
        "rows": paired["rows"],
        "source_specific_metrics": metrics,
        "source_specific_qualification_gates": source_gates,
        "source_specific_transfer_qualified": qualified,
        "source_specific_evaluation_protocol_sha256": evaluation_sha256,
        "target_only_baseline_policy": spec["policy"],
        "target_only_baseline_direct_is_fallback_not_vote": True,
        "source_provenance_claim": qualified,
    })
    final = body | {"report_sha256": stable_hash(body)}
    output_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "metrics", "source_specific_metrics",
        "qualification_gates", "source_specific_qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
