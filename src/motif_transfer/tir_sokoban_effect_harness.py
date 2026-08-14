"""Typed Sokoban effect-program transfer to TIR evidence operations.

The source program controls only POSITION / COMMIT / REPLAN.  TIR owns the
neural proposal scores, concrete wrapper calls, and answer-slot grounding.
Gold answers are used only by :func:`evaluate_tir_effect_transfer` after every
condition has emitted an answer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .active_video_transfer import exact_binomial_two_sided
from .contracts import stable_hash


AUTHENTIC = "authentic_sokoban_effect_plus_target"
CONDITIONS = (
    "raw_target_only",
    "target_native_one_test",
    AUTHENTIC,
    "within_state_candidate_order_shuffle",
    "inverted_verify_control",
    "position_until_budget_control",
    "phase_permuted_commit_first",
)


def validate_source_receipt(receipt: Mapping[str, Any]) -> None:
    body = dict(receipt)
    claimed = str(body.pop("compact_receipt_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("invalid Sokoban compact receipt self hash")
    if receipt.get("artifact_version") != "SOKOBAN_EFFECT_PROGRAM_V2":
        raise ValueError("TIR harness requires SOKOBAN_EFFECT_PROGRAM_V2")
    confirmation = receipt.get("fresh_confirmation") or {}
    if not confirmation.get("source_gate_passed"):
        raise ValueError("Sokoban effect source gate did not pass")
    rules = (receipt.get("program") or {}).get("rules") or []
    if [row.get("select") for row in rules[:2]] != ["COMMIT", "POSITION"]:
        raise ValueError("Sokoban POSITION/COMMIT rules changed")
    if not any(row.get("select") == "REPLAN_OR_ABSTAIN" for row in rules):
        raise ValueError("Sokoban effect program lacks the refutation edge")


def _native_candidate_order(
    receipt: Mapping[str, Any],
    *,
    shuffled: bool,
    shuffle_seed: str,
) -> list[Mapping[str, Any]]:
    candidates = list(receipt.get("candidates") or ())
    if not candidates:
        raise ValueError("TIR receipt has no target-native evidence candidates")
    if shuffled:
        sample_id = str(receipt["sample_id"])
        return sorted(
            candidates,
            key=lambda row: stable_hash({
                "seed": shuffle_seed,
                "sample_id": sample_id,
                "candidate_id": str(row["candidate_id"]),
            }),
        )
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["planner_score"]), str(row["candidate_id"]),
        ),
    )


def _answer(candidate: Mapping[str, Any]) -> str:
    value = str(candidate["answer"]["answer"])
    if not value:
        raise ValueError("candidate answer slot is empty")
    return value


def _commit(
    answer: str,
    *,
    decisions: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    decisions.append({
        "source_option": "COMMIT",
        "target_native_action": "commit_answer_slot",
        "answer_slot": answer,
        "reason": reason,
    })
    return {
        "committed_answer": answer,
        "tests": sum(
            row["source_option"] == "POSITION"
            for row in decisions
        ),
        "source_decisions": decisions,
    }


def _position(
    candidate: Mapping[str, Any],
    *,
    decisions: list[dict[str, Any]],
    effect: str,
) -> str:
    answer = _answer(candidate)
    decisions.append({
        "source_option": "POSITION",
        "target_native_action": str(candidate["wrapper_receipt"]["tool"]),
        "candidate_id": str(candidate["candidate_id"]),
        "planner_score": float(candidate["planner_score"]),
        "observed_answer_slot": answer,
        "effect_predicate": effect,
    })
    return answer


def _authentic_program(
    receipt: Mapping[str, Any],
    *,
    shuffled: bool,
    shuffle_seed: str,
) -> dict[str, Any]:
    baseline = str(receipt["baseline"]["answer"]["answer"])
    candidates = _native_candidate_order(
        receipt, shuffled=shuffled, shuffle_seed=shuffle_seed,
    )
    decisions: list[dict[str, Any]] = []
    first = _position(
        candidates[0], decisions=decisions, effect="EFFECT_PENDING_VERIFICATION",
    )
    if first == baseline:
        decisions[-1]["effect_predicate"] = "EXPECTED_EFFECT_OBSERVED"
        return _commit(
            baseline, decisions=decisions,
            reason="TARGET_EVIDENCE_CORROBORATES_CURRENT_COMMIT",
        )

    # A changed target hypothesis is not accepted from one crop.  The source
    # VERIFY edge keeps selecting POSITION while the target grounder owns which
    # concrete evidence operation comes next.
    for candidate in candidates[1:]:
        observed = _position(
            candidate, decisions=decisions,
            effect="VERIFY_CANDIDATE_EFFECT",
        )
        if observed == first:
            decisions[-1]["effect_predicate"] = "EXPECTED_EFFECT_OBSERVED"
            return _commit(
                first, decisions=decisions,
                reason="INDEPENDENT_TARGET_EVIDENCE_CORROBORATES_NEW_COMMIT",
            )
    decisions.append({
        "source_option": "REPLAN_OR_ABSTAIN",
        "target_native_action": "abstain_to_target_baseline",
        "effect_predicate": "EXPECTED_EFFECT_REFUTED",
    })
    return _commit(
        baseline, decisions=decisions,
        reason="UNVERIFIED_TARGET_CHANGE_REJECTED",
    )


def execute_condition(
    receipt: Mapping[str, Any],
    *,
    condition: str,
    shuffle_seed: str,
) -> dict[str, Any]:
    """Execute one condition without reading the evaluator-only gold answer."""

    if condition not in CONDITIONS:
        raise ValueError(f"unsupported TIR transfer condition: {condition}")
    baseline = str(receipt["baseline"]["answer"]["answer"])
    candidates = _native_candidate_order(
        receipt, shuffled=False, shuffle_seed=shuffle_seed,
    )
    if condition == AUTHENTIC:
        result = _authentic_program(
            receipt, shuffled=False, shuffle_seed=shuffle_seed,
        )
    elif condition == "within_state_candidate_order_shuffle":
        result = _authentic_program(
            receipt, shuffled=True, shuffle_seed=shuffle_seed,
        )
    elif condition in ("raw_target_only", "phase_permuted_commit_first"):
        result = _commit(
            baseline, decisions=[], reason=(
                "NO_SOURCE" if condition == "raw_target_only"
                else "PHASE_PERMUTED_COMMIT_BEFORE_POSITION"
            ),
        )
    elif condition == "target_native_one_test":
        decisions: list[dict[str, Any]] = []
        observed = _position(
            candidates[0], decisions=decisions,
            effect="TARGET_NATIVE_NO_SOURCE_VERIFY",
        )
        result = _commit(
            observed, decisions=decisions, reason="ONE_TEST_TARGET_POLICY",
        )
    elif condition == "inverted_verify_control":
        decisions = []
        first = _position(
            candidates[0], decisions=decisions,
            effect="INVERTED_EFFECT_GUARD",
        )
        if first != baseline:
            result = _commit(
                first, decisions=decisions,
                reason="INVERTED_ACCEPTS_UNCORROBORATED_CHANGE",
            )
        else:
            changed = None
            for candidate in candidates[1:]:
                value = _position(
                    candidate, decisions=decisions,
                    effect="INVERTED_REJECTS_CORROBORATION",
                )
                if value != first:
                    changed = value
                    break
            result = _commit(
                changed or baseline, decisions=decisions,
                reason="INVERTED_VERIFY_CONTROL",
            )
    else:
        decisions = []
        votes = [baseline]
        for candidate in candidates:
            votes.append(_position(
                candidate, decisions=decisions,
                effect="POSITION_PRIOR_IGNORES_STOP_EDGE",
            ))
        counts = Counter(votes)
        committed = max(
            votes, key=lambda value: (counts[value], -votes.index(value)),
        )
        result = _commit(
            committed, decisions=decisions,
            reason="EXHAUSTIVE_TARGET_MAJORITY_AFTER_POSITION_BUDGET",
        )
    body = {
        "sample_id": str(receipt["sample_id"]),
        "family": str(receipt.get("family") or ""),
        "condition": condition,
        "baseline_answer": baseline,
        **result,
    }
    body["trace_sha256"] = stable_hash(body)
    return body


def evaluate_tir_effect_transfer(
    receipts: Sequence[Mapping[str, Any]],
    *,
    source_receipt: Mapping[str, Any],
    expected_ids: Sequence[str],
    claim_boundary: str,
    evidence_tier: str,
) -> dict[str, Any]:
    validate_source_receipt(source_receipt)
    ids = tuple(str(row["sample_id"]) for row in receipts)
    if ids != tuple(map(str, expected_ids)):
        raise ValueError("TIR receipt order/coverage differs from frozen IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate TIR receipt sample IDs")
    shuffle_seed = str(
        source_receipt["fresh_confirmation"]["report_sha256"]
    )
    traces: list[dict[str, Any]] = []
    for receipt in receipts:
        # Conditions are executed before the evaluator-only label is consulted.
        condition_rows = [
            execute_condition(
                receipt, condition=condition, shuffle_seed=shuffle_seed,
            )
            for condition in CONDITIONS
        ]
        gold = str(receipt["gold_answer"])
        for row in condition_rows:
            body = dict(row)
            body.pop("trace_sha256")
            body.update({
                "gold_answer_evaluator_only": gold,
                "correct_evaluator_only": row["committed_answer"] == gold,
            })
            body["trace_sha256"] = stable_hash(body)
            traces.append(body)

    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in CONDITIONS
    }
    summaries = {
        condition: {
            "tasks": len(rows),
            "successes": sum(row["correct_evaluator_only"] for row in rows),
            "success_rate": (
                sum(row["correct_evaluator_only"] for row in rows) / len(rows)
            ),
            "tests": sum(int(row["tests"]) for row in rows),
            "action_changes_vs_raw": sum(
                row["committed_answer"] != row["baseline_answer"] for row in rows
            ),
        }
        for condition, rows in by_condition.items()
    }
    authentic_index = {
        row["sample_id"]: row for row in by_condition[AUTHENTIC]
    }
    paired: dict[str, Any] = {}
    for comparator in CONDITIONS:
        if comparator == AUTHENTIC:
            continue
        other = {row["sample_id"]: row for row in by_condition[comparator]}
        wins = losses = 0
        for sample_id in ids:
            a = bool(authentic_index[sample_id]["correct_evaluator_only"])
            b = bool(other[sample_id]["correct_evaluator_only"])
            wins += a and not b
            losses += b and not a
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(ids) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }

    raw = summaries["raw_target_only"]
    authentic = summaries[AUTHENTIC]
    exhaustive = summaries["position_until_budget_control"]
    destructive = ("inverted_verify_control", "phase_permuted_commit_first")
    gates = {
        "source_receipt_valid_and_fresh_confirmed": True,
        "receipt_matrix_complete": len(receipts) == len(expected_ids),
        "authentic_nontrivial_action_contrast": (
            authentic["action_changes_vs_raw"] >= 2
        ),
        "authentic_success_gain_vs_raw": (
            authentic["successes"] > raw["successes"]
        ),
        "authentic_zero_negative_transfer_vs_raw": (
            paired["raw_target_only"]["losses"] == 0
        ),
        "authentic_beats_one_test_target_policy": (
            authentic["successes"]
            > summaries["target_native_one_test"]["successes"]
        ),
        "authentic_strictly_beats_source_structure_controls": all(
            authentic["successes"] > summaries[name]["successes"]
            for name in destructive
        ),
        "authentic_order_shuffle_robustness": (
            authentic["successes"]
            >= summaries["within_state_candidate_order_shuffle"]["successes"]
            and authentic["tests"]
            <= summaries["within_state_candidate_order_shuffle"]["tests"]
        ),
        "authentic_matches_exhaustive_success_with_fewer_tests": (
            authentic["successes"] >= exhaustive["successes"]
            and authentic["tests"] < exhaustive["tests"]
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "tir-sokoban-effect-harness-v6",
        "status": (
            "DEVELOPMENT_GATE_PASSED_FREEZE_QUALIFICATION"
            if passed and evidence_tier == "CONSUMED_DEVELOPMENT"
            else "QUALIFICATION_GATE_PASSED_FREEZE_HELDOUT"
            if passed and evidence_tier == "FRESH_QUALIFICATION"
            else "FORMAL_CONFIRMATION_PASSED"
            if passed and evidence_tier == "FRESH_FORMAL_CONFIRMATION"
            else "TRANSFER_GATE_FAILED"
        ),
        "claim_boundary": claim_boundary,
        "evidence_tier": evidence_tier,
        "source_artifact_sha256": str(
            source_receipt["artifact"]["artifact_sha256"]
        ),
        "source_confirmation_sha256": shuffle_seed,
        "mapping": {
            "source_POSITION": "target_native_evidence_operation",
            "source_COMMIT": "target_native_answer_commit",
            "source_VERIFY_EXPECTED_EFFECT": (
                "corroborate_changed_answer_with_independent_target_evidence"
            ),
            "source_EXPECTED_EFFECT_REFUTED": "replan_or_abstain_to_target",
        },
        "tasks": list(ids),
        "families": dict(sorted(Counter(
            str(row.get("family") or "") for row in receipts
        ).items())),
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "traces": traces,
    }
    body["report_sha256"] = stable_hash(body)
    return body


__all__ = [
    "AUTHENTIC",
    "CONDITIONS",
    "evaluate_tir_effect_transfer",
    "execute_condition",
    "validate_source_receipt",
]
