"""Executable BIND-handle -> RELATE observation MDP for video transfer."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .structured_video_transfer import ParsedTargetWorldModel
from .structured_video_transfer import _target_exact_plan_value
from .video_dynamics_mdp import (
    PredicateProbeReceipt,
    answer_belief,
    apply_probe_receipt,
    initial_state,
    source_compatible_action_features,
)


BOUND_VIDEO_CONDITIONS = (
    "target_native_expected_accuracy",
    "target_native_exact_dp",
    "authentic_bound_bind_relate_ir",
    "authentic_unbound_relation_ablation",
    "reversed_relate_bind_ir",
    "wrong_guard_bound_ir",
    "node_only_bind_bind_ir",
    "deterministic_random_global",
)


def _best(indices, features) -> int:
    if not indices:
        raise ValueError("no admissible action under typed edge")
    return max(indices, key=lambda index: (features[index][2], features[index][1], -index))


def _rehydrate_receipt(payload: Mapping[str, Any]) -> PredicateProbeReceipt:
    return PredicateProbeReceipt(**dict(payload))


def evaluate_bound_bind_relate_transfer(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    global_receipts: Mapping[str, PredicateProbeReceipt],
    fork_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    particles, probes = world_model.particles, world_model.probes
    bind = [i for i, probe in enumerate(probes) if probe.target_event_role == "BIND"]
    relate = [i for i, probe in enumerate(probes) if probe.target_event_role == "RELATE"]
    if len(bind) != 2 or len(relate) != 3:
        raise ValueError("bound evaluator requires exactly two BIND and three RELATE")
    tracks = fork_receipt["tracks"]
    matrix = fork_receipt["bound_relation_receipts"]
    if set(tracks) != {probes[index].probe_id for index in bind}:
        raise ValueError("bound fork lacks a complete BIND track matrix")
    if any(
        set(matrix[bind_id]) != {probes[index].probe_id for index in relate}
        for bind_id in matrix
    ):
        raise ValueError("bound fork lacks the 2x3 RELATE matrix")
    bind_receipts = {
        probe_id: _rehydrate_receipt(payload["bind_receipt"])
        for probe_id, payload in tracks.items()
    }
    bound_receipts = {
        (bind_id, relate_id): _rehydrate_receipt(payload["receipt"])
        for bind_id, rows in matrix.items()
        for relate_id, payload in rows.items()
    }
    initial = initial_state(particles, probes, max_tests=2)
    _, _, answer_space = source_compatible_action_features(
        particles, initial, probes, max_tests=2,
    )
    initial_belief = answer_belief(particles, initial, answer_space=answer_space)
    baseline = answer_space[int(np.argmax(initial_belief))]
    stable = sum((index + 1) * ord(value) for index, value in enumerate(sample_id))
    conditions = {}

    def finish(name, state, actions, receipts_used, handle_changed=False):
        posterior = answer_belief(particles, state, answer_space=answer_space)
        committed = answer_space[int(np.argmax(posterior))]
        conditions[name] = {
            "actions": actions,
            "receipts_used": receipts_used,
            "committed_answer": committed,
            "correct": committed == gold_answer,
            "gold_probability_after": (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
            "handle_changed_relation_observation": bool(handle_changed),
        }

    # Strong target-only control: two sequential global probes.
    state = initial
    selected = []
    used = []
    for _ in range(2):
        tests, _, _ = source_compatible_action_features(
            particles, state, probes, max_tests=2, answer_space=answer_space,
        )
        index = _best([i for i in range(len(probes)) if i not in selected], tests)
        selected.append(index)
        receipt = global_receipts[probes[index].probe_id]
        state = apply_probe_receipt(particles, state, probes, receipt)
        used.append(f"global:{probes[index].probe_id}")
    finish("target_native_expected_accuracy", state, [probes[i].probe_id for i in selected], used)

    # Complete target-native control: solve the two-step TEST/COMMIT policy under
    # the target model, then execute that policy against the held receipt stream.
    # A transferred controller must beat this condition, not merely a greedy one.
    state = initial
    selected = set()
    actions = []
    used = []
    while state.remaining_tests > 0 and len(selected) < len(probes):
        _, choice = _target_exact_plan_value(
            particles, probes, state, answer_space=answer_space,
            selected=frozenset(selected), test_cost=0.0,
        )
        if choice[0] == "COMMIT":
            break
        index = choice[1]
        selected.add(index)
        probe_id = probes[index].probe_id
        actions.append(probe_id)
        used.append(f"global:{probe_id}")
        state = apply_probe_receipt(
            particles, state, probes, global_receipts[probe_id],
        )
    finish("target_native_exact_dp", state, actions, used)

    def choose_bind(state):
        tests, _, _ = source_compatible_action_features(
            particles, state, probes, max_tests=2, answer_space=answer_space,
        )
        return _best(bind, tests)

    # Authentic and unbound ablation share the exact first BIND and relation.
    first = choose_bind(initial)
    bind_id = probes[first].probe_id
    bind_receipt = bind_receipts[bind_id]
    after_bind = apply_probe_receipt(particles, initial, probes, bind_receipt)
    tests, _, _ = source_compatible_action_features(
        particles, after_bind, probes, max_tests=2, answer_space=answer_space,
    )
    primary_entity = str(tracks[bind_id]["primary_entity_ref"])
    guarded = [
        index for index in relate if primary_entity in probes[index].entity_refs
    ]
    relation = _best(guarded or relate, tests)
    relate_id = probes[relation].probe_id
    if bind_receipt.observed_true:
        authentic_receipt = bound_receipts[(bind_id, relate_id)]
        authentic_state = apply_probe_receipt(
            particles, after_bind, probes, authentic_receipt,
        )
        authentic_actions = [f"BIND:{bind_id}", f"BOUND_RELATE:{relate_id}"]
        authentic_used = [f"track:{bind_id}", f"bound:{bind_id}->{relate_id}"]
    else:
        second = next(index for index in bind if index != first)
        second_id = probes[second].probe_id
        authentic_state = apply_probe_receipt(
            particles, after_bind, probes, bind_receipts[second_id],
        )
        authentic_actions = [f"BIND:{bind_id}", f"RECOVER_BIND:{second_id}"]
        authentic_used = [f"track:{bind_id}", f"track:{second_id}"]
    global_relation = global_receipts[relate_id]
    changed = (
        bind_receipt.observed_true
        and authentic_receipt.observed_true != global_relation.observed_true
    ) if bind_receipt.observed_true else False
    finish(
        "authentic_bound_bind_relate_ir", authentic_state,
        authentic_actions, authentic_used, changed,
    )
    unbound_state = apply_probe_receipt(
        particles, after_bind, probes, global_relation,
    )
    finish(
        "authentic_unbound_relation_ablation", unbound_state,
        [f"BIND:{bind_id}", f"GLOBAL_RELATE:{relate_id}"],
        [f"track:{bind_id}", f"global:{relate_id}"],
    )

    # Reversed edge: the relation is measured globally before a BIND handle exists.
    tests, _, _ = source_compatible_action_features(
        particles, initial, probes, max_tests=2, answer_space=answer_space,
    )
    reverse_relation = _best(relate, tests)
    reverse_relate_id = probes[reverse_relation].probe_id
    reverse_state = apply_probe_receipt(
        particles, initial, probes, global_receipts[reverse_relate_id],
    )
    tests, _, _ = source_compatible_action_features(
        particles, reverse_state, probes, max_tests=2, answer_space=answer_space,
    )
    reverse_bind = _best(bind, tests)
    reverse_bind_id = probes[reverse_bind].probe_id
    reverse_state = apply_probe_receipt(
        particles, reverse_state, probes, bind_receipts[reverse_bind_id],
    )
    finish(
        "reversed_relate_bind_ir", reverse_state,
        [f"GLOBAL_RELATE:{reverse_relate_id}", f"BIND:{reverse_bind_id}"],
        [f"global:{reverse_relate_id}", f"track:{reverse_bind_id}"],
    )

    # Wrong guard: use the selected handle to crop a relation not involving it.
    tests, _, _ = source_compatible_action_features(
        particles, after_bind, probes, max_tests=2, answer_space=answer_space,
    )
    wrong = [index for index in relate if primary_entity not in probes[index].entity_refs]
    wrong_relation = _best(wrong or relate, tests)
    wrong_id = probes[wrong_relation].probe_id
    wrong_receipt = bound_receipts[(bind_id, wrong_id)]
    wrong_state = apply_probe_receipt(particles, after_bind, probes, wrong_receipt)
    finish(
        "wrong_guard_bound_ir", wrong_state,
        [f"BIND:{bind_id}", f"WRONG_HANDLE_RELATE:{wrong_id}"],
        [f"track:{bind_id}", f"bound:{bind_id}->{wrong_id}"],
        wrong_receipt.observed_true != global_receipts[wrong_id].observed_true,
    )

    second_bind = next(index for index in bind if index != first)
    second_bind_id = probes[second_bind].probe_id
    node_state = apply_probe_receipt(
        particles, after_bind, probes, bind_receipts[second_bind_id],
    )
    finish(
        "node_only_bind_bind_ir", node_state,
        [f"BIND:{bind_id}", f"BIND:{second_bind_id}"],
        [f"track:{bind_id}", f"track:{second_bind_id}"],
    )

    random_indices = list(range(len(probes)))
    first_random = random_indices[stable % len(random_indices)]
    random_indices.remove(first_random)
    second_random = random_indices[(stable // 7) % len(random_indices)]
    random_state = apply_probe_receipt(
        particles, initial, probes, global_receipts[probes[first_random].probe_id],
    )
    random_state = apply_probe_receipt(
        particles, random_state, probes, global_receipts[probes[second_random].probe_id],
    )
    finish(
        "deterministic_random_global", random_state,
        [f"GLOBAL:{probes[first_random].probe_id}", f"GLOBAL:{probes[second_random].probe_id}"],
        [f"global:{probes[first_random].probe_id}", f"global:{probes[second_random].probe_id}"],
    )

    # Evaluator-only matched oracle over all authentic handle pairs and global pairs.
    oracle_rows = [(
        baseline == gold_answer,
        float(initial_belief[answer_space.index(gold_answer)]),
        [], initial_belief,
    )]
    for left in range(len(probes)):
        for right in range(len(probes)):
            if left == right:
                continue
            state = apply_probe_receipt(
                particles, initial, probes, global_receipts[probes[left].probe_id],
            )
            state = apply_probe_receipt(
                particles, state, probes, global_receipts[probes[right].probe_id],
            )
            posterior = answer_belief(particles, state, answer_space=answer_space)
            oracle_rows.append((
                answer_space[int(np.argmax(posterior))] == gold_answer,
                float(posterior[answer_space.index(gold_answer)]),
                [f"global:{probes[left].probe_id}", f"global:{probes[right].probe_id}"],
                posterior,
            ))
    for bind_index in bind:
        bind_probe_id = probes[bind_index].probe_id
        state = apply_probe_receipt(
            particles, initial, probes, bind_receipts[bind_probe_id],
        )
        for relate_index in relate:
            relate_probe_id = probes[relate_index].probe_id
            bound_state = apply_probe_receipt(
                particles, state, probes,
                bound_receipts[(bind_probe_id, relate_probe_id)],
            )
            posterior = answer_belief(
                particles, bound_state, answer_space=answer_space,
            )
            oracle_rows.append((
                answer_space[int(np.argmax(posterior))] == gold_answer,
                float(posterior[answer_space.index(gold_answer)]),
                [f"track:{bind_probe_id}", f"bound:{bind_probe_id}->{relate_probe_id}"],
                posterior,
            ))
    _, _, oracle_actions, oracle_belief = max(
        oracle_rows, key=lambda row: (row[0], row[1]),
    )
    oracle_answer = answer_space[int(np.argmax(oracle_belief))]
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "answer_space": list(answer_space),
        "baseline_answer": baseline,
        "baseline_correct": baseline == gold_answer,
        "conditions": conditions,
        "oracle_actions": oracle_actions,
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "authentic_action_contrast": conditions[
            "authentic_bound_bind_relate_ir"
        ]["actions"] != conditions["target_native_expected_accuracy"]["actions"],
        "authentic_guard_obeyed": (
            (bind_receipt.observed_true and authentic_actions[1].startswith("BOUND_RELATE"))
            or (not bind_receipt.observed_true and authentic_actions[1].startswith("RECOVER_BIND"))
        ),
    }


__all__ = ["BOUND_VIDEO_CONDITIONS", "evaluate_bound_bind_relate_transfer"]
