"""Target-adaptation residual supervision for structured video TEST policies.

Gold answers are used only to label actions in the adaptation fold. Runtime
features remain the anonymous source-compatible symbolic belief features; the
held sample's label is excluded by leave-one-sample-out evaluation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .controlled_exploration_transfer import AbstractAction, MatchedValueExample
from .structured_video_transfer import ParsedTargetWorldModel
from .video_dynamics_mdp import (
    PredicateProbeReceipt,
    answer_belief,
    apply_probe_receipt,
    initial_state,
    source_compatible_action_features,
)


def _gold_probability(
    world_model: ParsedTargetWorldModel,
    state,
    *,
    gold_answer: str,
    answer_space: Sequence[str],
) -> float:
    if gold_answer not in answer_space:
        return 0.0
    belief = answer_belief(
        world_model.particles, state, answer_space=answer_space,
    )
    return float(belief[tuple(answer_space).index(gold_answer)])


def _best_realized_terminal_probability(
    world_model: ParsedTargetWorldModel,
    receipts: Mapping[str, PredicateProbeReceipt],
    state,
    *,
    selected: frozenset[int],
    remaining: int,
    gold_answer: str,
    answer_space: Sequence[str],
) -> float:
    if remaining == 0:
        return _gold_probability(
            world_model, state, gold_answer=gold_answer,
            answer_space=answer_space,
        )
    values = []
    for index, probe in enumerate(world_model.probes):
        if index in selected:
            continue
        updated = apply_probe_receipt(
            world_model.particles, state, world_model.probes,
            receipts[probe.probe_id],
        )
        values.append(_best_realized_terminal_probability(
            world_model,
            receipts,
            updated,
            selected=selected | {index},
            remaining=remaining - 1,
            gold_answer=gold_answer,
            answer_space=answer_space,
        ))
    if not values:
        raise ValueError("target value recursion exhausted the probe set")
    return max(values)


def build_target_test_value_examples(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
    test_budget: int,
) -> tuple[MatchedValueExample, ...]:
    """Create dense adaptation-only action values from matched probe forks."""

    particles = world_model.particles
    probes = world_model.probes
    if not 1 <= test_budget <= len(probes):
        raise ValueError("test_budget must fit the matched probe set")
    if set(probe_receipts) != {probe.probe_id for probe in probes}:
        raise ValueError("matched receipts must cover every probe")
    initial = initial_state(particles, probes, max_tests=test_budget)
    _, _, answer_space = source_compatible_action_features(
        particles, initial, probes, max_tests=test_budget,
    )
    rows: list[MatchedValueExample] = []
    for prefix_length in range(test_budget):
        for selected_tuple in combinations(range(len(probes)), prefix_length):
            state = initial
            for index in selected_tuple:
                probe = probes[index]
                state = apply_probe_receipt(
                    particles, state, probes, probe_receipts[probe.probe_id],
                )
            selected = frozenset(selected_tuple)
            tests, _, _ = source_compatible_action_features(
                particles, state, probes, max_tests=test_budget,
                answer_space=answer_space,
            )
            state_id = f"{sample_id}|tested={','.join(map(str, selected_tuple))}"
            for index, probe in enumerate(probes):
                if index in selected:
                    continue
                updated = apply_probe_receipt(
                    particles, state, probes, probe_receipts[probe.probe_id],
                )
                value = _best_realized_terminal_probability(
                    world_model,
                    probe_receipts,
                    updated,
                    selected=selected | {index},
                    remaining=test_budget - prefix_length - 1,
                    gold_answer=gold_answer,
                    answer_space=answer_space,
                )
                rows.append(MatchedValueExample(
                    state_id=state_id,
                    action=AbstractAction("TEST", index, probe.probe_id),
                    features=tuple(map(float, tests[index])),
                    value=float(value),
                ))
    if not rows or not np.all(np.isfinite([row.value for row in rows])):
        raise ValueError("target action-value labels are empty or non-finite")
    return tuple(rows)


__all__ = ["build_target_test_value_examples"]
