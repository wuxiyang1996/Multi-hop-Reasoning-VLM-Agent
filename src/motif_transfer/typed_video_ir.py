"""Compile the source-validated BIND->RELATE edge into video probe policies."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from .structured_video_transfer import ParsedTargetWorldModel
from .video_dynamics_mdp import (
    PredicateProbe,
    PredicateProbeReceipt,
    answer_belief,
    apply_probe_receipt,
    initial_state,
    source_compatible_action_features,
)


TYPED_VIDEO_CONDITIONS = (
    "target_native_expected_accuracy",
    "authentic_bind_relate_ir",
    "reversed_relate_bind_ir",
    "wrong_guard_bind_relate_ir",
    "node_only_bind_bind_ir",
    "deterministic_random_probe",
)

BIND_KINDS = frozenset({
    "OBJECT_PRESENT", "OBJECT_ATTRIBUTE", "OBJECT_TRACK",
})
RELATE_KINDS = frozenset({
    "OBJECT_MOTION", "COLLISION", "ENTRY", "EXIT", "EVENT_ORDER",
    "CAUSAL_ANCESTOR",
})


def _entity_atoms(probe: PredicateProbe) -> frozenset[str]:
    atoms = set()
    for entity in probe.entity_refs:
        normalized = re.sub(r"[^a-z0-9]+", " ", entity.lower()).strip()
        if normalized:
            atoms.add(normalized)
    return frozenset(atoms)


def probes_share_bound_entity(left: PredicateProbe, right: PredicateProbe) -> bool:
    """Exact target-native entity identity; no source semantic dictionary."""

    return bool(_entity_atoms(left) & _entity_atoms(right))


def _role(probe: PredicateProbe) -> str:
    if probe.target_event_role:
        return probe.target_event_role
    return "BIND" if probe.predicate_kind in BIND_KINDS else "RELATE"


def _best(
    candidates: Sequence[int], features: Sequence[Sequence[float]],
) -> int:
    if not candidates:
        raise ValueError("typed IR has no admissible target-native probe")
    return max(candidates, key=lambda index: (features[index][2], features[index][1], -index))


def _first_probe(
    condition: str,
    probes: Sequence[PredicateProbe],
    features: Sequence[Sequence[float]],
    *,
    random_index: int,
) -> int:
    bind = [i for i, probe in enumerate(probes) if _role(probe) == "BIND"]
    relate = [i for i, probe in enumerate(probes) if _role(probe) == "RELATE"]
    if condition == "target_native_expected_accuracy":
        return _best(list(range(len(probes))), features)
    if condition == "deterministic_random_probe":
        return random_index % len(probes)
    if condition == "reversed_relate_bind_ir":
        eligible = [
            i for i in relate
            if any(probes_share_bound_entity(probes[i], probes[j]) for j in bind)
        ]
        return _best(eligible or relate, features)
    eligible = [
        i for i in bind
        if any(probes_share_bound_entity(probes[i], probes[j]) for j in relate)
    ]
    return _best(eligible or bind, features)


def _second_probe(
    condition: str,
    first: int,
    probes: Sequence[PredicateProbe],
    features: Sequence[Sequence[float]],
    *,
    random_index: int,
    bind_guard_observed: bool,
) -> int:
    remaining = [index for index in range(len(probes)) if index != first]
    if condition == "target_native_expected_accuracy":
        return _best(remaining, features)
    if condition == "deterministic_random_probe":
        return remaining[random_index % len(remaining)]
    if condition == "reversed_relate_bind_ir":
        pool = [
            index for index in remaining
            if _role(probes[index]) == "BIND"
            and probes_share_bound_entity(probes[first], probes[index])
        ]
    elif condition == "wrong_guard_bind_relate_ir":
        pool = [
            index for index in remaining
            if _role(probes[index]) == "RELATE"
            and not probes_share_bound_entity(probes[first], probes[index])
        ]
    elif condition == "node_only_bind_bind_ir":
        pool = [
            index for index in remaining
            if _role(probes[index]) == "BIND"
        ]
    elif bind_guard_observed:
        pool = [
            index for index in remaining
            if probes[index].predicate_kind in RELATE_KINDS
            and probes_share_bound_entity(probes[first], probes[index])
        ]
    else:
        pool = [
            index for index in remaining if _role(probes[index]) == "BIND"
        ]
    typed = [
        index for index in remaining
        if _role(probes[index]) == (
            "BIND" if condition in {"reversed_relate_bind_ir", "node_only_bind_bind_ir"}
            or (condition == "authentic_bind_relate_ir" and not bind_guard_observed)
            else "RELATE"
        )
    ]
    return _best(pool or typed or remaining, features)


def evaluate_typed_bind_relate_transfer(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
) -> dict[str, object]:
    particles, probes = world_model.particles, world_model.probes
    if len(probes) < 2 or set(probe_receipts) != {probe.probe_id for probe in probes}:
        raise ValueError("typed transfer requires at least two matched probe receipts")
    initial = initial_state(particles, probes, max_tests=2)
    _, _, answer_space = source_compatible_action_features(
        particles, initial, probes, max_tests=2,
    )
    initial_belief = answer_belief(particles, initial, answer_space=answer_space)
    baseline = answer_space[int(np.argmax(initial_belief))]
    stable = int.from_bytes(sample_id.encode("utf-8"), "little")
    conditions = {}
    for offset, condition in enumerate(TYPED_VIDEO_CONDITIONS):
        state = initial
        tests, _, _ = source_compatible_action_features(
            particles, state, probes, max_tests=2, answer_space=answer_space,
        )
        first = _first_probe(
            condition, probes, tests, random_index=stable + offset,
        )
        state = apply_probe_receipt(
            particles, state, probes, probe_receipts[probes[first].probe_id],
        )
        first_observed = bool(
            probe_receipts[probes[first].probe_id].observed_true
        )
        tests, _, _ = source_compatible_action_features(
            particles, state, probes, max_tests=2, answer_space=answer_space,
        )
        second = _second_probe(
            condition, first, probes, tests, random_index=stable + 17 * offset,
            bind_guard_observed=first_observed,
        )
        state = apply_probe_receipt(
            particles, state, probes, probe_receipts[probes[second].probe_id],
        )
        posterior = answer_belief(particles, state, answer_space=answer_space)
        committed = answer_space[int(np.argmax(posterior))]
        conditions[condition] = {
            "selected_probe_ids": [probes[first].probe_id, probes[second].probe_id],
            "predicate_kinds": [
                probes[first].predicate_kind, probes[second].predicate_kind,
            ],
            "shared_entity_guard": probes_share_bound_entity(
                probes[first], probes[second],
            ),
            "first_receipt_observed_true": first_observed,
            "edge_traversed": (
                _role(probes[first]) == "BIND"
                and first_observed
                and _role(probes[second]) == "RELATE"
                and probes_share_bound_entity(probes[first], probes[second])
            ),
            "guard_obeyed": (
                condition != "authentic_bind_relate_ir"
                or (
                    first_observed
                    and _role(probes[second]) == "RELATE"
                    and probes_share_bound_entity(probes[first], probes[second])
                )
                or (not first_observed and _role(probes[second]) == "BIND")
            ),
            "committed_answer": committed,
            "correct": committed == gold_answer,
            "gold_probability_after": (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
        }
    oracle_rows = []
    for first in range(len(probes)):
        for second in range(len(probes)):
            if first == second:
                continue
            state = apply_probe_receipt(
                particles, initial, probes, probe_receipts[probes[first].probe_id],
            )
            state = apply_probe_receipt(
                particles, state, probes, probe_receipts[probes[second].probe_id],
            )
            posterior = answer_belief(particles, state, answer_space=answer_space)
            gold_probability = (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            )
            oracle_rows.append((gold_probability, first, second, posterior))
    _, first, second, oracle_belief = max(oracle_rows, key=lambda row: row[0])
    oracle_answer = answer_space[int(np.argmax(oracle_belief))]
    authentic = conditions["authentic_bind_relate_ir"]
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "answer_space": list(answer_space),
        "baseline_answer": baseline,
        "baseline_correct": baseline == gold_answer,
        "conditions": conditions,
        "oracle_probe_ids": [probes[first].probe_id, probes[second].probe_id],
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "authentic_action_contrast": authentic["selected_probe_ids"] != conditions[
            "target_native_expected_accuracy"
        ]["selected_probe_ids"],
        "authentic_guard_obeyed": bool(authentic["guard_obeyed"]),
        "authentic_edge_traversed": bool(authentic["edge_traversed"]),
    }


__all__ = [
    "BIND_KINDS", "RELATE_KINDS", "TYPED_VIDEO_CONDITIONS",
    "evaluate_typed_bind_relate_transfer", "probes_share_bound_entity",
]
