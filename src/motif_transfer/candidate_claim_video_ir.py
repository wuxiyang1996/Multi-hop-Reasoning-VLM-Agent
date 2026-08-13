"""Candidate-factorized BIND->RELATE programs for video transfer.

The target compiler turns each native answer candidate into a binary visual
claim.  The source-transferred edge contributes only the executable guard:
RELATE evidence is admissible after a candidate-specific carrier was BINDed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CANDIDATE_CLAIM_CONDITIONS = (
    "target_unbound_claim_verification",
    "authentic_bound_claim_program",
    "reversed_claim_then_bind",
    "wrong_guard_bound_claim",
    "node_only_bind",
    "source_marginal_bind",
    "shuffled_bind_correspondence",
)


def _probability(value: Any, *, field: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0,1]")
    return result


def _bind_probability(candidate: Mapping[str, Any]) -> float:
    if "identity_verification" in candidate:
        return _probability(
            candidate["identity_verification"]["identity_match_probability"],
            field="identity_verification.identity_match_probability",
        )
    track = candidate["track"]
    reliability = _probability(
        track["sensor_reliability"], field="track.sensor_reliability",
    )
    if reliability < 0.5:
        raise ValueError("track reliability must be in [0.5,1]")
    return reliability if bool(track["observed_true"]) else 1.0 - reliability


def _relation_probability(candidate: Mapping[str, Any], key: str) -> float:
    return _probability(
        candidate[key]["support_probability"],
        field=f"{key}.support_probability",
    )


def _commit(
    scores: Sequence[float], *, slots: Sequence[str], answer_contract: str,
    threshold: float,
) -> str:
    if len(scores) != len(slots) or not scores:
        raise ValueError("candidate scores and slots must be nonempty and aligned")
    if answer_contract == "binary_vector":
        return "".join("1" if score >= threshold else "0" for score in scores)
    if answer_contract == "single_choice":
        return slots[max(range(len(scores)), key=lambda index: (scores[index], -index))]
    raise ValueError(f"unsupported answer contract: {answer_contract}")


def evaluate_candidate_claim_program(
    *, sample_id: str, gold_answer: str, baseline_answer: str,
    fork: Mapping[str, Any], threshold: float = 0.5,
    guard_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate matched candidate programs without consulting gold for policy."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if not 0.0 < guard_threshold < 1.0:
        raise ValueError("guard_threshold must be in (0,1)")
    candidates = list(fork.get("candidates") or ())
    if not candidates:
        raise ValueError("candidate fork is empty")
    if not bool(fork.get("complete")):
        raise ValueError("candidate fork is incomplete")
    slots = [str(row["slot"]) for row in candidates]
    if len(set(slots)) != len(slots):
        raise ValueError("candidate slots must be unique")
    contract = str(fork["answer_contract"])
    bind = [_bind_probability(row) for row in candidates]
    unbound = [_relation_probability(row, "unbound_relation") for row in candidates]
    bound = [_relation_probability(row, "bound_relation") for row in candidates]
    wrong = [_relation_probability(row, "wrong_guard_relation") for row in candidates]
    count = len(candidates)
    mean_bind = sum(bind) / count
    authentic_guard = [value >= guard_threshold for value in bind]
    shuffled_guard = [bind[(index + 1) % count] >= guard_threshold for index in range(count)]
    marginal_guard = mean_bind >= guard_threshold
    condition_scores = {
        "target_unbound_claim_verification": unbound,
        "authentic_bound_claim_program": [
            bound[index] if authentic_guard[index] else unbound[index]
            for index in range(count)
        ],
        "reversed_claim_then_bind": unbound,
        "wrong_guard_bound_claim": [
            wrong[index] if authentic_guard[index] else unbound[index]
            for index in range(count)
        ],
        "node_only_bind": bind,
        "source_marginal_bind": [
            bound[index] if marginal_guard else unbound[index]
            for index in range(count)
        ],
        "shuffled_bind_correspondence": [
            bound[index] if shuffled_guard[index] else unbound[index]
            for index in range(count)
        ],
    }
    conditions = {}
    for name in CANDIDATE_CLAIM_CONDITIONS:
        scores = condition_scores[name]
        answer = _commit(
            scores, slots=slots, answer_contract=contract, threshold=threshold,
        )
        conditions[name] = {
            "scores": list(map(float, scores)),
            "committed_answer": answer,
            "correct": answer == gold_answer,
        }

    # Evaluator-only attainable headroom from the already matched receipts.
    if contract == "binary_vector":
        oracle_scores = []
        for index in range(count):
            available = (
                unbound[index], bind[index] * bound[index],
                bind[index] * wrong[index], bind[index],
            )
            oracle_scores.append(
                max(available) if gold_answer[index] == "1" else min(available)
            )
    else:
        gold_index = slots.index(gold_answer)
        oracle_scores = []
        for index in range(count):
            available = (
                unbound[index], bind[index] * bound[index],
                bind[index] * wrong[index], bind[index],
            )
            oracle_scores.append(max(available) if index == gold_index else min(available))
    oracle_answer = _commit(
        oracle_scores, slots=slots, answer_contract=contract, threshold=threshold,
    )
    changed = sum(
        abs(bound[index] - unbound[index]) >= 0.05 for index in range(count)
    )
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "answer_contract": contract,
        "slots": slots,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == gold_answer,
        "bind_probabilities": bind,
        "authentic_guard_passed": authentic_guard,
        "guard_threshold": guard_threshold,
        "conditions": conditions,
        "oracle_scores": oracle_scores,
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "bound_unbound_changed_candidates": changed,
        "authentic_action_contrast": any(
            authentic_guard[index]
            and abs(bound[index] - unbound[index]) >= 0.05
            for index in range(count)
        ),
    }


__all__ = [
    "CANDIDATE_CLAIM_CONDITIONS", "evaluate_candidate_claim_program",
]
