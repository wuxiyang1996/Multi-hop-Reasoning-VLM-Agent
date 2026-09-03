#!/usr/bin/env python3
"""Audit source-recurrent third-view acquisition on consumed AGQA receipts.

The generic control uses exactly two independent anchor views.  The authentic
source contract may apply its recurrent transition once more only when the
two-view terminal/consensus guard is false.  The audit re-executes consensus
from frozen answer-blind view receipts; it never selects rows by correctness.
It is retrospective mechanism evidence and cannot create a fresh claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    consensus_anchor_interval,
)
from motif_transfer.clevrer_unified_goal_relation import (  # noqa: E402
    source_goal_relation_contract,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.source_controlled_grounding import (  # noqa: E402
    GroundingControlVerdict,
    SourceControlledGroundingPolicy,
    TypedGroundingControlState,
)


DEFAULT_REPORTS = (
    "runs/agqa2_temporal_localized_query_v59_development/report.json",
    "runs/agqa2_temporal_localized_query_v60_qualification/report.json",
    "runs/agqa2_full_distribution_v62/base_report.json",
)


def _normalize(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _equivalent(left: Any, right: Any) -> bool:
    predicted, expected = _normalize(left), _normalize(right)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def _exact_two_sided(wins: int, losses: int) -> float:
    from math import comb
    n = wins + losses
    if not n:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, k) for k in range(tail + 1)) / (2 ** n))


def _load_policy() -> SourceControlledGroundingPolicy:
    artifact = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    ).read_text())
    confirmation = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    ).read_text())
    return SourceControlledGroundingPolicy(
        source_goal_relation_contract(artifact, confirmation)
    )


def _group_views(row: Mapping[str, Any]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for view in row.get("anchor_views", ()):
        grouped[int(view.get("anchor_index", 0))].append(view)
    for rows in grouped.values():
        order = {"anchor_primary": 0, "anchor_secondary": 1, "anchor_tiebreak": 2}
        rows.sort(key=lambda value: order[str(value["view"])])
    return dict(grouped)


def audit_report(
    report_path: Path, *, minimum_confidence: float,
    maximum_endpoint_spread: int,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    policy = _load_policy()
    rows_out = []
    for row in report["rows"]:
        grouped = _group_views(row)
        stored = list(row.get("anchor_consensus_receipts") or ())
        if len(grouped) != len(stored):
            raise ValueError("anchor view/consensus arity mismatch")
        recurrent_authorizations = []
        fixed_two_authorized = True
        full_receipts_exact = True
        source_authorization_receipts = []
        for anchor_index, views in sorted(grouped.items()):
            if len(views) not in {2, 3}:
                raise ValueError("cached anchor needs two or three independent views")
            first_two = consensus_anchor_interval(
                [value["receipt"] for value in views[:2]],
                minimum_confidence=minimum_confidence,
                maximum_endpoint_spread=maximum_endpoint_spread,
            )
            final = consensus_anchor_interval(
                [value["receipt"] for value in views],
                minimum_confidence=minimum_confidence,
                maximum_endpoint_spread=maximum_endpoint_spread,
            )
            if final.receipt_sha256 != stored[anchor_index]["receipt_sha256"]:
                full_receipts_exact = False
            needs_third = len(views) == 3
            if needs_third != (not first_two.authorized):
                raise ValueError("third view was not caused by failed two-view consensus")
            fixed_two_authorized &= first_two.authorized
            if needs_third:
                state = TypedGroundingControlState(
                    task_id=str(row["task_id"]), target_domain="agqa2",
                    target_state_sha256=first_two.receipt_sha256,
                    transition_guard_observable=True,
                    transition_guard_satisfied=True,
                    transition_effect_authenticated=True,
                    terminal_guard_observable=True,
                    terminal_guard_satisfied=False,
                    abstention_guard_satisfied=False,
                    interventions_used=1, intervention_budget=2,
                    formal_outcome_read=False,
                )
                authorization = policy.decide(state)
                source_authorization_receipts.append(
                    authorization.authorization_sha256
                )
                recurrent_authorizations.append(
                    authorization.verdict == GroundingControlVerdict.APPLY_TRANSITION
                )
        tiebreak_used = bool(recurrent_authorizations)
        if recurrent_authorizations and not all(recurrent_authorizations):
            raise ValueError("source recurrence failed to authorize a required view")
        direct = str(row["direct_response"])
        candidate = row.get("candidate_typed_prediction")
        source_prediction = str(row["source_prediction"])
        expected_source = str(candidate or direct)
        if source_prediction != expected_source:
            raise ValueError("source prediction differs from frozen typed fallback")
        generic_prediction = str(candidate or direct) if fixed_two_authorized else direct
        gold = row["gold_answer_evaluator_only"]
        source_correct = _equivalent(source_prediction, gold)
        generic_correct = _equivalent(generic_prediction, gold)
        rows_out.append({
            "task_id_sha256": stable_hash(row["task_id"]),
            "tiebreak_used": tiebreak_used,
            "source_recurrent_authorized": all(recurrent_authorizations),
            "fixed_two_authorized": fixed_two_authorized,
            "candidate_available": candidate is not None,
            "source_prediction_sha256": stable_hash(source_prediction),
            "generic_prediction_sha256": stable_hash(generic_prediction),
            "source_correct": source_correct,
            "generic_correct": generic_correct,
            "full_consensus_receipts_exact": full_receipts_exact,
            "source_authorization_receipts": source_authorization_receipts,
            "runtime_answer_read": False,
        })
    wins = sum(row["source_correct"] and not row["generic_correct"] for row in rows_out)
    losses = sum(row["generic_correct"] and not row["source_correct"] for row in rows_out)
    return {
        "report_path": str(report_path.relative_to(REPO)),
        "source_status": report["status"],
        "rows": len(rows_out),
        "tiebreak_rows": sum(row["tiebreak_used"] for row in rows_out),
        "tiebreak_rows_with_candidate": sum(
            row["tiebreak_used"] and row["candidate_available"] for row in rows_out
        ),
        "source_correct": sum(row["source_correct"] for row in rows_out),
        "generic_fixed_two_correct": sum(row["generic_correct"] for row in rows_out),
        "source_vs_generic": {
            "wins": wins,
            "losses": losses,
            "ties": len(rows_out) - wins - losses,
            "exact_two_sided_p": _exact_two_sided(wins, losses),
        },
        "gates": {
            "all_full_consensus_receipts_reexecuted_exactly": all(
                row["full_consensus_receipts_exact"] for row in rows_out
            ),
            "all_tiebreaks_follow_failed_two_view_terminal": True,
            "all_required_tiebreaks_source_authorized": all(
                row["source_recurrent_authorized"] for row in rows_out
            ),
            "all_runtime_rows_answer_blind": all(
                row["runtime_answer_read"] is False for row in rows_out
            ),
        },
        "row_receipts_sha256": stable_hash(rows_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports", nargs="*", type=Path,
        default=[REPO / value for value in DEFAULT_REPORTS],
    )
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/agqa2_temporal_localized_query_v59_development.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/agqa2_source_controlled_tiebreak_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    calibration = config["calibration"]
    audits = [
        audit_report(
            path.resolve(),
            minimum_confidence=float(calibration["anchor_minimum_confidence"]),
            maximum_endpoint_spread=int(calibration["anchor_maximum_endpoint_spread"]),
        )
        for path in args.reports
    ]
    gates = {
        "all_receipt_reexecution_gates_pass": all(
            all(audit["gates"].values()) for audit in audits
        ),
        "development_no_negative_transfer": audits[0]["source_vs_generic"]["losses"] == 0,
        "qualification_positive_action_utility": (
            audits[1]["source_vs_generic"]["wins"] > audits[1]["source_vs_generic"]["losses"]
        ),
        "consumed_formal_diagnostic_positive": (
            audits[2]["source_vs_generic"]["wins"] > audits[2]["source_vs_generic"]["losses"]
        ),
    }
    body = {
        "schema_version": "agqa2-source-controlled-tiebreak-audit-v1",
        "status": (
            "AGQA2_SOURCE_CONTROLLED_TIEBREAK_RETROSPECTIVE_SUPPORTED_FRESH_REQUIRED"
            if all(gates.values()) else
            "AGQA2_SOURCE_CONTROLLED_TIEBREAK_RETROSPECTIVE_NOT_SUPPORTED"
        ),
        "fresh_evidence": False,
        "claim_boundary": (
            "Re-executes consumed answer-blind grounding receipts. It validates "
            "mechanism and counterfactual fixed-two controls but adds no fresh "
            "AGQA success claim."
        ),
        "source_policy": (
            "The qualified recurrent game program authorizes a third independent "
            "anchor view only after the two-view terminal/consensus guard fails."
        ),
        "generic_control": (
            "A target-written fixed-two-view scaffold with the same maximum "
            "budget but no recurrent transition."
        ),
        "audits": audits,
        "gates": gates,
        "next_step": (
            "Wire this source authorization into a fresh collector before the "
            "third-view call, freeze new video-disjoint qualification, and rerun."
        ),
    }
    result = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "audits": audits, "gates": gates,
        "report_sha256": result["report_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
