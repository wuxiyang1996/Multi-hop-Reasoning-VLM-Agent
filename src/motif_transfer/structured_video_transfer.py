"""Shared matched-fork protocol for structured video benchmarks.

The neural target model proposes world/dynamics particles and typed predicate
probes from question text plus a low-bandwidth video scout.  Matched probe
receipts are then collected without asking for the final answer again.  This
module validates those target-native objects and evaluates frozen source
TEST-value models on the anonymous nine-dimensional interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from .video_dynamics_mdp import (
    PREDICATE_KINDS,
    PredicateProbe,
    PredicateProbeReceipt,
    WorldParticle,
    answer_belief,
    apply_probe_receipt,
    initial_state,
    source_compatible_action_features,
)


FIXED_TEST_CONDITIONS = (
    "target_native_information_gain",
    "target_native_expected_accuracy",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
    "deterministic_random_probe",
)


@dataclass(frozen=True)
class ParsedTargetWorldModel:
    particles: tuple[WorldParticle, ...]
    particle_summaries: tuple[str, ...]
    probes: tuple[PredicateProbe, ...]
    probe_rationales: tuple[str, ...]


def _answer_is_valid(
    answer: str,
    *,
    valid_answers: Sequence[str] | None,
    binary_vector_length: int | None,
) -> bool:
    if valid_answers is not None:
        return answer in set(map(str, valid_answers))
    if binary_vector_length is not None:
        return (
            len(answer) == binary_vector_length
            and set(answer) <= {"0", "1"}
        )
    raise ValueError("an explicit native answer contract is required")


def parse_target_world_model(
    payload: Mapping[str, Any],
    *,
    duration_seconds: float,
    particle_count: int,
    probe_count: int,
    valid_answers: Sequence[str] | None = None,
    binary_vector_length: int | None = None,
) -> ParsedTargetWorldModel:
    """Fail closed on malformed or answer-reprompt-like target outputs."""

    if duration_seconds <= 0 or particle_count < 2 or probe_count < 2:
        raise ValueError("invalid duration or frozen model cardinalities")
    raw_particles = list(payload.get("world_particles") or ())
    raw_probes = list(payload.get("typed_probes") or ())
    if len(raw_particles) != particle_count:
        raise ValueError("target model did not return the frozen particle count")
    if len(raw_probes) != probe_count:
        raise ValueError("target model did not return the frozen probe count")

    particles = []
    summaries = []
    for index, row in enumerate(raw_particles):
        answer = str(row.get("native_answer") or "").strip().upper()
        if not _answer_is_valid(
            answer,
            valid_answers=valid_answers,
            binary_vector_length=binary_vector_length,
        ):
            raise ValueError(f"particle has invalid native answer: {answer!r}")
        particle = WorldParticle(
            particle_id=f"W{index}",
            native_answer=answer,
            prior_weight=float(row.get("prior_weight", 0.0)),
        )
        particle.validate()
        summary = str(row.get("event_graph_summary") or "").strip()
        if not summary:
            raise ValueError("world particle requires an event/dynamics summary")
        particles.append(particle)
        summaries.append(summary)
    if len({particle.native_answer for particle in particles}) < 2:
        raise ValueError("world particles must express answer uncertainty")

    probes = []
    rationales = []
    for index, row in enumerate(raw_probes):
        kind = str(row.get("predicate_kind") or "").strip().upper()
        if kind not in PREDICATE_KINDS:
            raise ValueError(f"target model emitted unsupported predicate: {kind}")
        window = list(row.get("window_fraction") or ())
        if len(window) != 2:
            raise ValueError("typed probe needs a two-value normalized window")
        start_fraction = float(window[0])
        end_fraction = float(window[1])
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError(
                "probe window_fraction must satisfy 0 <= start < end <= 1; "
                f"got [{start_fraction}, {end_fraction}]"
            )
        start = start_fraction * duration_seconds
        end = end_fraction * duration_seconds
        likelihood = tuple(map(float, row.get("true_probability_by_particle") or ()))
        probe = PredicateProbe(
            probe_id=f"P{index}",
            predicate_kind=kind,
            entity_refs=tuple(map(str, row.get("entity_refs") or ())),
            start_sec=start,
            end_sec=end,
            latent_true_probability_by_particle=likelihood,
            target_tool="sample_frames",
            expected_sensor_reliability=float(
                row.get("expected_sensor_reliability", 0.0)
            ),
        )
        probe.validate(len(particles))
        rationale = str(row.get("rationale") or "").strip()
        if not rationale:
            raise ValueError("typed probe requires a target-native rationale")
        probes.append(probe)
        rationales.append(rationale)
    if len({
        (probe.predicate_kind, probe.entity_refs, probe.start_sec, probe.end_sec)
        for probe in probes
    }) != len(probes):
        raise ValueError("target model emitted duplicate typed probes")
    return ParsedTargetWorldModel(
        tuple(particles), tuple(summaries), tuple(probes), tuple(rationales),
    )


def parse_typed_probe_receipt(
    payload: Mapping[str, Any],
    *,
    probe: PredicateProbe,
    evidence_sha256: Sequence[str],
) -> PredicateProbeReceipt:
    value = payload.get("observed_true")
    if not isinstance(value, bool):
        raise ValueError("typed grounder must return a boolean observed_true")
    receipt = PredicateProbeReceipt(
        probe_id=probe.probe_id,
        predicate_kind=probe.predicate_kind,
        entity_refs=probe.entity_refs,
        start_sec=probe.start_sec,
        end_sec=probe.end_sec,
        observed_true=value,
        sensor_reliability=float(payload.get("sensor_reliability", 0.0)),
        evidence_sha256=tuple(map(str, evidence_sha256)),
    )
    receipt.validate_against(probe)
    return receipt


def _stable_index(sample_id: str, count: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % count


def evaluate_fixed_test_budget(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
    source_models: Mapping[str, Any],
    test_budget: int,
) -> dict[str, Any]:
    """Evaluate sequential typed-probe selection at an equal fixed budget."""

    particles = world_model.particles
    probes = world_model.probes
    if not 1 <= test_budget <= len(probes):
        raise ValueError("test_budget must be between one and the probe count")
    if set(probe_receipts) != {probe.probe_id for probe in probes}:
        raise ValueError("matched receipts must cover every frozen typed probe")
    initial = initial_state(particles, probes, max_tests=test_budget)
    _, _, answer_space = source_compatible_action_features(
        particles, initial, probes, max_tests=test_budget,
    )
    initial_belief = answer_belief(particles, initial, answer_space=answer_space)
    baseline_index = int(np.argmax(initial_belief))
    baseline_answer = answer_space[baseline_index]

    conditions: dict[str, Any] = {}
    for condition in FIXED_TEST_CONDITIONS:
        state = initial
        trajectory: list[int] = []
        for step in range(test_budget):
            tests, _, _ = source_compatible_action_features(
                particles, state, probes, max_tests=test_budget,
                answer_space=answer_space,
            )
            available = [index for index in range(len(probes)) if index not in trajectory]
            if condition == "target_native_information_gain":
                index = max(available, key=lambda value: tests[value][1])
            elif condition == "target_native_expected_accuracy":
                index = max(available, key=lambda value: tests[value][2])
            elif condition == "deterministic_random_probe":
                index = available[_stable_index(f"{sample_id}|{step}", len(available))]
            else:
                means, _ = source_models[condition].predict(tests)
                index = max(available, key=lambda value: means[value])
            trajectory.append(index)
            probe = probes[index]
            state = apply_probe_receipt(
                particles, state, probes, probe_receipts[probe.probe_id],
            )
        posterior = answer_belief(
            particles, state, answer_space=answer_space,
        )
        committed = answer_space[int(np.argmax(posterior))]
        conditions[condition] = {
            "selected_probe_ids": [probes[index].probe_id for index in trajectory],
            "predicate_kinds": [probes[index].predicate_kind for index in trajectory],
            "committed_answer": committed,
            "correct": committed == gold_answer,
            "gold_probability_before": (
                float(initial_belief[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
            "gold_probability_after": (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
        }

    oracle_rows = []
    for trajectory in itertools.permutations(range(len(probes)), test_budget):
        state = initial
        for index in trajectory:
            probe = probes[index]
            state = apply_probe_receipt(
                particles, state, probes, probe_receipts[probe.probe_id],
            )
        posterior = answer_belief(particles, state, answer_space=answer_space)
        probability = (
            float(posterior[answer_space.index(gold_answer)])
            if gold_answer in answer_space else 0.0
        )
        oracle_rows.append((probability, trajectory, posterior))
    _, oracle_trajectory, oracle_belief = max(oracle_rows, key=lambda row: row[0])
    oracle_answer = answer_space[int(np.argmax(oracle_belief))]
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "answer_space": list(answer_space),
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == gold_answer,
        "conditions": conditions,
        "test_budget": test_budget,
        "oracle_probe_ids": [probes[index].probe_id for index in oracle_trajectory],
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "authentic_action_contrast": (
            conditions["authentic_source_plus_target"]["selected_probe_ids"]
            != conditions["target_native_expected_accuracy"]["selected_probe_ids"]
        ),
    }


def evaluate_fixed_one_test(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
    source_models: Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible one-TEST specialization."""

    return evaluate_fixed_test_budget(
        sample_id=sample_id,
        gold_answer=gold_answer,
        world_model=world_model,
        probe_receipts=probe_receipts,
        source_models=source_models,
        test_budget=1,
    )


__all__ = [
    "FIXED_TEST_CONDITIONS",
    "ParsedTargetWorldModel",
    "evaluate_fixed_test_budget",
    "evaluate_fixed_one_test",
    "parse_target_world_model",
    "parse_typed_probe_receipt",
]
