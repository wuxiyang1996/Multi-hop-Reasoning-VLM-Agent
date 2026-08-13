"""Candidate-factorized BIND->MUTATE programs for natural video.

The neural target grounder proposes and measures candidate-specific state
transitions.  The transferred symbolic edge only controls whether a mutation
measurement may consume an identity-bound observation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CANDIDATE_MUTATION_CONDITIONS = (
    "target_unbound_mutation_verification",
    "authentic_bound_mutation_program",
    "reversed_mutation_then_bind",
    "wrong_guard_bound_mutation",
    "node_only_bind",
    "source_marginal_bind_mutate",
    "shuffled_bind_mutate_correspondence",
)


def _probability(value: Any, field: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0,1]")
    return result


def _bind(candidate: Mapping[str, Any]) -> float:
    return _probability(
        candidate["identity_verification"]["identity_match_probability"],
        "identity_verification.identity_match_probability",
    )


def _mutation(candidate: Mapping[str, Any], key: str) -> float:
    return _probability(candidate[key]["support_probability"], f"{key}.support_probability")


def _commit(scores: Sequence[float], slots: Sequence[str], threshold: float) -> str:
    if len(scores) != len(slots) or not scores:
        raise ValueError("mutation scores and candidate slots must align")
    return str(slots[max(range(len(scores)), key=lambda index: (scores[index], -index))])


def evaluate_candidate_mutation_program(
    *, sample_id: str, gold_answer: str, baseline_answer: str,
    fork: Mapping[str, Any], threshold: float = 0.5,
    guard_threshold: float = 0.5, minimum_action_delta: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 < threshold < 1.0 or not 0.0 < guard_threshold < 1.0:
        raise ValueError("thresholds must be in (0,1)")
    if not 0.0 <= minimum_action_delta <= 1.0:
        raise ValueError("minimum_action_delta must be in [0,1]")
    candidates = list(fork.get("candidates") or ())
    if not candidates or not bool(fork.get("complete")):
        raise ValueError("candidate mutation fork must be complete and nonempty")
    slots = [str(row["slot"]) for row in candidates]
    if len(set(slots)) != len(slots):
        raise ValueError("candidate slots must be unique")
    bind = [_bind(row) for row in candidates]
    unbound = [_mutation(row, "unbound_mutation") for row in candidates]
    bound = [_mutation(row, "bound_mutation") for row in candidates]
    wrong = [_mutation(row, "wrong_guard_mutation") for row in candidates]
    identity_guards = [value >= guard_threshold for value in bind]
    guards = [
        identity_guards[index]
        and abs(bound[index] - unbound[index]) >= minimum_action_delta
        for index in range(len(bind))
    ]
    wrong_guards = [
        identity_guards[index]
        and abs(wrong[index] - unbound[index]) >= minimum_action_delta
        for index in range(len(bind))
    ]
    shuffled = [
        bind[(index + 1) % len(bind)] >= guard_threshold
        and abs(bound[index] - unbound[index]) >= minimum_action_delta
        for index in range(len(bind))
    ]
    marginal = (
        sum(bind) / len(bind) >= guard_threshold
        and any(abs(bound[index] - unbound[index]) >= minimum_action_delta for index in range(len(bind)))
    )
    scores = {
        "target_unbound_mutation_verification": unbound,
        "authentic_bound_mutation_program": [
            bound[index] if guards[index] else unbound[index]
            for index in range(len(bind))
        ],
        "reversed_mutation_then_bind": unbound,
        "wrong_guard_bound_mutation": [
            wrong[index] if wrong_guards[index] else unbound[index]
            for index in range(len(bind))
        ],
        "node_only_bind": bind,
        "source_marginal_bind_mutate": [
            bound[index]
            if marginal and abs(bound[index] - unbound[index]) >= minimum_action_delta
            else unbound[index]
            for index in range(len(bind))
        ],
        "shuffled_bind_mutate_correspondence": [
            bound[index] if shuffled[index] else unbound[index]
            for index in range(len(bind))
        ],
    }
    conditions = {}
    for name in CANDIDATE_MUTATION_CONDITIONS:
        answer = _commit(scores[name], slots, threshold)
        conditions[name] = {
            "scores": list(map(float, scores[name])),
            "committed_answer": answer,
            "correct": answer == gold_answer,
        }
    gold_index = slots.index(gold_answer)
    oracle_scores = []
    for index in range(len(slots)):
        available = (unbound[index], bind[index] * bound[index], bind[index] * wrong[index], bind[index])
        oracle_scores.append(max(available) if index == gold_index else min(available))
    oracle_answer = _commit(oracle_scores, slots, threshold)
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == gold_answer,
        "slots": slots,
        "bind_probabilities": bind,
        "identity_guard_passed": identity_guards,
        "authentic_guard_passed": guards,
        "minimum_action_delta": minimum_action_delta,
        "conditions": conditions,
        "oracle_scores": oracle_scores,
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "bound_unbound_changed_candidates": sum(
            abs(bound[index] - unbound[index]) >= 0.05 for index in range(len(bind))
        ),
        "authentic_action_contrast": any(
            guards[index]
            for index in range(len(bind))
        ),
    }


__all__ = ["CANDIDATE_MUTATION_CONDITIONS", "evaluate_candidate_mutation_program"]
