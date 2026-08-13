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


def _commit(
    scores: Sequence[float], slots: Sequence[str], threshold: float,
    answer_contract: str,
) -> str:
    if len(scores) != len(slots) or not scores:
        raise ValueError("mutation scores and candidate slots must align")
    if answer_contract == "binary_vector":
        return "".join("1" if value >= threshold else "0" for value in scores)
    if answer_contract != "single_choice":
        raise ValueError(f"unsupported answer contract: {answer_contract}")
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
    decoy_bind = [
        _probability(
            row["decoy_identity_verification"]["identity_match_probability"],
            "decoy_identity_verification.identity_match_probability",
        ) if "decoy_identity_verification" in row else None
        for row in candidates
    ]
    strict_guard_noop = all(value is not None for value in decoy_bind)
    answer_contract = str(fork.get("answer_contract", "single_choice"))
    if answer_contract == "binary_vector":
        if len(baseline_answer) != len(slots) or set(baseline_answer) - {"0", "1"}:
            raise ValueError("binary baseline must align with mutation candidates")
        baseline_scores = [float(value) for value in baseline_answer]
    else:
        if baseline_answer not in slots:
            raise ValueError("single-choice baseline must be a candidate slot")
        baseline_scores = [float(slot == baseline_answer) for slot in slots]
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
        (
            decoy_bind[index] >= guard_threshold
            if decoy_bind[index] is not None else identity_guards[index]
        )
        and abs(wrong[index] - unbound[index]) >= minimum_action_delta
        for index in range(len(bind))
    ]
    shuffled = [
        (
            decoy_bind[(index + 1) % len(bind)] >= guard_threshold
            if decoy_bind[index] is not None
            else bind[(index + 1) % len(bind)] >= guard_threshold
        )
        and abs(
            wrong[(index + 1) % len(bind)] - unbound[index]
        ) >= minimum_action_delta
        for index in range(len(bind))
    ]
    marginal = (
        sum(bind) / len(bind) >= guard_threshold
        and any(abs(bound[index] - unbound[index]) >= minimum_action_delta for index in range(len(bind)))
    )
    fallback = baseline_scores if strict_guard_noop else unbound
    scores = {
        "target_unbound_mutation_verification": unbound,
        "authentic_bound_mutation_program": [
            bound[index] if guards[index] else fallback[index]
            for index in range(len(bind))
        ],
        "reversed_mutation_then_bind": unbound,
        "wrong_guard_bound_mutation": [
            wrong[index] if wrong_guards[index] else fallback[index]
            for index in range(len(bind))
        ],
        "node_only_bind": bind,
        "source_marginal_bind_mutate": [
            bound[index]
            if marginal and abs(bound[index] - unbound[index]) >= minimum_action_delta
            else fallback[index]
            for index in range(len(bind))
        ],
        "shuffled_bind_mutate_correspondence": [
            wrong[(index + 1) % len(bind)] if shuffled[index] else fallback[index]
            for index in range(len(bind))
        ],
    }
    conditions = {}
    for name in CANDIDATE_MUTATION_CONDITIONS:
        answer = _commit(scores[name], slots, threshold, answer_contract)
        conditions[name] = {
            "scores": list(map(float, scores[name])),
            "committed_answer": answer,
            "correct": answer == gold_answer,
        }
    oracle_scores = []
    if answer_contract == "binary_vector":
        for index in range(len(slots)):
            available = (
                unbound[index], bind[index] * bound[index],
                bind[index] * wrong[index], bind[index],
            )
            oracle_scores.append(
                max(available) if gold_answer[index] == "1" else min(available)
            )
    else:
        gold_index = slots.index(gold_answer)
        for index in range(len(slots)):
            available = (
                unbound[index], bind[index] * bound[index],
                bind[index] * wrong[index], bind[index],
            )
            oracle_scores.append(
                max(available) if index == gold_index else min(available)
            )
    oracle_answer = _commit(oracle_scores, slots, threshold, answer_contract)
    distinct_wrong_controls = sum(
        bool(candidate.get("decoy_entity_visual_description"))
        and candidate["wrong_guard_mutation"].get("panel_sha256")
        != candidate["bound_mutation"].get("panel_sha256")
        for candidate in candidates
    )
    distinct_shuffled_controls = sum(
        candidates[(index + 1) % len(candidates)]["wrong_guard_mutation"].get(
            "panel_sha256"
        ) != candidate["bound_mutation"].get("panel_sha256")
        for index, candidate in enumerate(candidates)
    )
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == gold_answer,
        "slots": slots,
        "bind_probabilities": bind,
        "decoy_bind_probabilities": decoy_bind,
        "answer_contract": answer_contract,
        "guard_failure_transition": (
            "NOOP_TO_BASELINE" if strict_guard_noop else "LEGACY_UNBOUND_FALLBACK"
        ),
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
        "distinct_wrong_control_candidates": distinct_wrong_controls,
        "wrong_control_action_contrast": distinct_wrong_controls > 0,
        "distinct_shuffled_control_candidates": distinct_shuffled_controls,
        "shuffled_control_action_contrast": distinct_shuffled_controls > 0,
    }


__all__ = ["CANDIDATE_MUTATION_CONDITIONS", "evaluate_candidate_mutation_program"]
