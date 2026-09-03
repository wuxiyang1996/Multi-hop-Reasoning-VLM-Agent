"""Replay TIR maze candidate search with the V16 source automaton.

TIR owns option strings, image-derived passability, neural direction/color
bindings, and the answer slot.  The source contributes only EXPLORE,
BACKTRACK, and COMMIT routing over target-native evidence candidates.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .active_video_transfer import exact_binomial_two_sided
from .contracts import stable_hash
from .search_automaton_transfer_v16 import (
    AttemptLedger,
    OUTCOME_NONTERMINAL_EFFECT,
    OUTCOME_REFUTED,
    SourceSearchAutomaton,
    bind_native_action,
    ground_target_event,
)
from .sokoban_search_automaton_v16 import BACKTRACK, COMMIT, EXPLORE
from .tir_maze_topology import (
    _color_centroid,
    _execute,
    parse_maze_options,
    validate_neural_binding,
)


RAW = "raw_target_only"
AUTHENTIC = "authentic_search_automaton_plus_target"
PERMUTED = "event_binding_permuted_control"
LEDGER_BLIND = "ledger_blind_repeat_first_control"
COMMIT_AVAILABLE = "commit_availability_only_control"
EXHAUSTIVE = "target_native_exhaustive_ceiling"
CONDITIONS = (
    RAW,
    AUTHENTIC,
    PERMUTED,
    LEDGER_BLIND,
    COMMIT_AVAILABLE,
    EXHAUSTIVE,
)


def _target_candidates(
    image: Image.Image,
    prompt: str,
    neural_binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grounded = validate_neural_binding(neural_binding)
    options = parse_maze_options(prompt)
    array = np.asarray(image.convert("RGB"))
    start_channel = int(grounded["start_channel"])
    goal_channel = int(grounded["goal_channel"])
    start = _color_centroid(array, channel=start_channel)
    goal = _color_centroid(array, channel=goal_channel)
    start_other = [index for index in range(3) if index != start_channel]
    goal_other = [index for index in range(3) if index != goal_channel]
    start_mask = (
        (array[:, :, start_channel] > 60)
        & (array[:, :, start_channel] > array[:, :, start_other[0]] * 1.5)
        & (array[:, :, start_channel] > array[:, :, start_other[1]] * 1.5)
    )
    goal_mask = (
        (array[:, :, goal_channel] > 60)
        & (array[:, :, goal_channel] > array[:, :, goal_other[0]] * 1.5)
        & (array[:, :, goal_channel] > array[:, :, goal_other[1]] * 1.5)
    )
    passable = (array.mean(axis=2) > 100) | start_mask | goal_mask
    candidates = []
    for node_count in range(3, 102, 2):
        for answer_slot, sequence in sorted(options.items()):
            reached, receipt = _execute(
                sequence,
                node_count=node_count,
                start=start,
                goal=goal,
                passable=passable,
                deltas=grounded["move_deltas"],
                check_edges=True,
            )
            candidates.append({
                "candidate_id": f"grid-{node_count}:answer-{answer_slot}",
                "answer_slot": answer_slot,
                "sequence_sha256": stable_hash(sequence),
                "goal_reached": bool(reached),
                "execution_receipt": receipt,
            })
    audit = {
        "target_candidate_count": len(candidates),
        "target_option_count": len(options),
        "grounded_start_pixel": list(start),
        "grounded_goal_pixel": list(goal),
        "neural_binding_sha256": stable_hash(neural_binding),
    }
    return candidates, audit


def _route(
    source: SourceSearchAutomaton,
    *,
    domain_event: str,
    episode_id: str,
    decision_index: int,
    evidence_kind: str,
    evidence_payload: Mapping[str, Any],
    abstract_action: str,
    native_action_id: str,
    native_action: Any,
    confidence: float,
) -> dict[str, Any]:
    event = ground_target_event(
        domain="tirbench",
        episode_id=episode_id,
        decision_index=decision_index,
        untried_candidate_available=domain_event == "UNBOUND",
        active_candidate_refuted=domain_event == "REFUTED",
        terminal_commit_verified=domain_event == "VERIFIED",
        evidence_kind=evidence_kind,
        evidence_payload=evidence_payload,
        grounding_confidence=confidence,
    )
    if event is None:
        raise RuntimeError("TIR event unexpectedly abstained")
    binding = bind_native_action(
        event,
        abstract_action=abstract_action,
        native_action_id=native_action_id,
        native_action=native_action,
        grounding_confidence=confidence,
    )
    return asdict(source.route(event, {abstract_action: binding}))


def execute_tir_maze_search(
    *,
    image: Image.Image,
    prompt: str,
    sample_id: str,
    baseline_answer: str,
    neural_binding: Mapping[str, Any],
    source: SourceSearchAutomaton,
    condition: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown TIR V16 condition: {condition}")
    candidates, target_audit = _target_candidates(image, prompt, neural_binding)
    confidence = float(neural_binding.get("confidence", 0.0))
    if condition == RAW:
        return {
            "condition": condition,
            "selected_answer": baseline_answer,
            "target_audit": target_audit,
            "source_decisions": [],
            "tested_target_candidates": 0,
        }
    successful_slots = sorted({
        str(row["answer_slot"]) for row in candidates if row["goal_reached"]
    })
    exhaustive_answer = successful_slots[0] if len(successful_slots) == 1 else None
    if condition == EXHAUSTIVE:
        return {
            "condition": condition,
            "selected_answer": exhaustive_answer or baseline_answer,
            "target_audit": target_audit,
            "source_decisions": [],
            "tested_target_candidates": len(candidates),
        }
    if condition == COMMIT_AVAILABLE:
        trace = _route(
            source,
            domain_event="VERIFIED",
            episode_id=sample_id,
            decision_index=0,
            evidence_kind="answer_slot_exists_not_effect_verified_control",
            evidence_payload={"answer_slots_available": True},
            abstract_action=COMMIT,
            native_action_id="commit_target_baseline",
            native_action={"answer_slot": baseline_answer},
            confidence=confidence,
        )
        return {
            "condition": condition,
            "selected_answer": baseline_answer,
            "target_audit": target_audit,
            "source_decisions": [trace],
            "tested_target_candidates": 0,
        }
    if condition == PERMUTED:
        # The real target UNBOUND evidence is deliberately mislabeled REFUTED.
        trace = _route(
            source,
            domain_event="REFUTED",
            episode_id=sample_id,
            decision_index=0,
            evidence_kind="permuted_unbound_as_refuted_control",
            evidence_payload={"untried_target_candidates": len(candidates)},
            abstract_action=BACKTRACK,
            native_action_id="backtrack_without_active_candidate",
            native_action={"operation": "abstain_to_baseline"},
            confidence=confidence,
        )
        return {
            "condition": condition,
            "selected_answer": baseline_answer,
            "target_audit": target_audit,
            "source_decisions": [trace],
            "tested_target_candidates": 0,
        }

    ledger = AttemptLedger()
    ledger.begin_scope(stable_hash({
        "sample_id": sample_id,
        "prompt": prompt,
        "binding": neural_binding,
    }))
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    by_id = {str(row["candidate_id"]): row for row in candidates}
    source_decisions: list[dict[str, Any]] = []
    observed_successful_slots: set[str] = set()
    tested = 0
    for decision_index in range(len(candidates)):
        available_ids = (
            [candidate_ids[0]] * len(candidate_ids)
            if condition == LEDGER_BLIND
            else candidate_ids
        )
        event_name = ledger.unbound_event(available_ids)
        if event_name is None:
            break
        candidate_id = (
            candidate_ids[0]
            if condition == LEDGER_BLIND
            else next(
                value for value in candidate_ids if value not in ledger.tried
            )
        )
        explore = _route(
            source,
            domain_event="UNBOUND",
            episode_id=sample_id,
            decision_index=len(source_decisions),
            evidence_kind="target_native_untried_path_candidate",
            evidence_payload={
                "candidate_id": candidate_id,
                "remaining_candidate_count": len(candidates) - tested,
            },
            abstract_action=EXPLORE,
            native_action_id="execute_target_path_candidate",
            native_action={"candidate_id": candidate_id},
            confidence=confidence,
        )
        source_decisions.append(explore)
        if not explore["admitted"]:
            break
        if condition == LEDGER_BLIND:
            # Deliberately erase memory after each target attempt.
            ledger.begin_scope(f"ledger-blind-{decision_index}")
        selected_id = ledger.next_untried([candidate_id])
        if selected_id is None:
            break
        tested += 1
        candidate = by_id[selected_id]
        if candidate["goal_reached"]:
            observed_successful_slots.add(str(candidate["answer_slot"]))
            ledger.observe(selected_id, OUTCOME_NONTERMINAL_EFFECT)
            continue
        ledger.observe(selected_id, OUTCOME_REFUTED)
        refuted = _route(
            source,
            domain_event="REFUTED",
            episode_id=sample_id,
            decision_index=len(source_decisions),
            evidence_kind="target_path_candidate_execution_refuted",
            evidence_payload={
                "candidate_id": selected_id,
                "execution_receipt": candidate["execution_receipt"],
            },
            abstract_action=BACKTRACK,
            native_action_id="reject_candidate_and_replan",
            native_action={"candidate_id": selected_id},
            confidence=confidence,
        )
        source_decisions.append(refuted)
        if not refuted["admitted"]:
            break

    selected_answer = baseline_answer
    if condition == AUTHENTIC and len(observed_successful_slots) == 1:
        answer = next(iter(observed_successful_slots))
        commit = _route(
            source,
            domain_event="VERIFIED",
            episode_id=sample_id,
            decision_index=len(source_decisions),
            evidence_kind="target_exhaustive_unique_answer_verification",
            evidence_payload={
                "verified_answer_slot": answer,
                "tested_candidate_count": tested,
                "all_target_candidates_exhausted": tested == len(candidates),
            },
            abstract_action=COMMIT,
            native_action_id="commit_target_answer_slot",
            native_action={"answer_slot": answer},
            confidence=confidence,
        )
        source_decisions.append(commit)
        if commit["admitted"] and tested == len(candidates):
            selected_answer = answer
    return {
        "condition": condition,
        "selected_answer": selected_answer,
        "target_audit": target_audit,
        "source_decisions": source_decisions,
        "tested_target_candidates": tested,
        "successful_target_answer_slots": sorted(observed_successful_slots),
        "ledger": ledger.as_dict(),
    }


def evaluate_tir_maze_search(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_artifact_sha256: str,
    evidence_tier: str,
) -> dict[str, Any]:
    traces = []
    for row in rows:
        gold = str(row["gold_answer_evaluator_only"])
        for condition in CONDITIONS:
            result = dict(row["conditions"][condition])
            body = {
                "sample_id": str(row["sample_id"]),
                "condition": condition,
                "baseline_answer": str(row["baseline_answer"]),
                **result,
                "gold_answer_evaluator_only": gold,
                "correct_evaluator_only": str(result["selected_answer"]) == gold,
            }
            traces.append(body | {"trace_sha256": stable_hash(body)})
    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in CONDITIONS
    }
    summaries = {
        condition: {
            "tasks": len(condition_rows),
            "successes": sum(row["correct_evaluator_only"] for row in condition_rows),
            "success_rate": (
                sum(row["correct_evaluator_only"] for row in condition_rows)
                / len(condition_rows)
            ),
            "changed_answers_vs_raw": sum(
                row["selected_answer"] != row["baseline_answer"]
                for row in condition_rows
            ),
            "target_candidate_tests": sum(
                int(row["tested_target_candidates"]) for row in condition_rows
            ),
        }
        for condition, condition_rows in by_condition.items()
    }
    authentic = {row["sample_id"]: row for row in by_condition[AUTHENTIC]}
    paired = {}
    for comparator in CONDITIONS:
        if comparator == AUTHENTIC:
            continue
        other = {row["sample_id"]: row for row in by_condition[comparator]}
        wins = losses = 0
        for sample_id, authentic_row in authentic.items():
            a = bool(authentic_row["correct_evaluator_only"])
            b = bool(other[sample_id]["correct_evaluator_only"])
            wins += a and not b
            losses += b and not a
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(authentic) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }
    authentic_summary = summaries[AUTHENTIC]
    source_actions = {
        decision["source_action"]
        for row in by_condition[AUTHENTIC]
        for decision in row["source_decisions"]
        if decision["admitted"]
    }
    gates = {
        "complete_matched_condition_matrix": all(
            len(rows_for_condition) == len(rows) and len(rows) > 0
            for rows_for_condition in by_condition.values()
        ),
        "all_target_neural_bindings_valid": all(
            bool(row["neural_binding_valid"]) for row in rows
        ),
        "all_three_source_actions_exercised": source_actions
        == {BACKTRACK, COMMIT, EXPLORE},
        "nontrivial_answer_changes": (
            authentic_summary["changed_answers_vs_raw"] >= 2
        ),
        "success_gain_over_raw_target": (
            authentic_summary["successes"] > summaries[RAW]["successes"]
        ),
        "zero_negative_transfer_vs_raw": paired[RAW]["losses"] == 0,
        "strictly_beats_destructive_controls": all(
            authentic_summary["successes"] > summaries[name]["successes"]
            for name in (PERMUTED, LEDGER_BLIND, COMMIT_AVAILABLE)
        ),
        "matches_isomorphic_target_ceiling": (
            authentic_summary["successes"] == summaries[EXHAUSTIVE]["successes"]
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "tir-search-automaton-transfer-v16",
        "status": (
            "CONSUMED_FORMAL_REANALYSIS_MECHANISM_REPRODUCED"
            if passed else "TRANSFER_GATE_FAILED"
        ),
        "evidence_tier": evidence_tier,
        "claim_boundary": (
            "REANALYSIS_OF_PREVIOUSLY_CONSUMED_TIR_MAZE_FORMAL_RECEIPTS; "
            "REUSES_FROZEN_TARGET_NEURAL_BINDINGS; DOES_NOT_CREATE_NEW_FRESH_EVIDENCE"
        ),
        "source_artifact_sha256": source_artifact_sha256,
        "mapping": {
            "EXPLORE_UNTRIED": "execute_target_path_candidate",
            "BACKTRACK_REPLAN": "reject_candidate_and_replan",
            "COMMIT_VERIFY": "commit_unique_target_answer_slot",
        },
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "traces": traces,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "AUTHENTIC",
    "CONDITIONS",
    "evaluate_tir_maze_search",
    "execute_tir_maze_search",
]
